import numpy as np
from numpy.linalg import matrix_power
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv
import os

#Randomizer
SEED = 42
rng = np.random.default_rng(SEED)

#Global Parameters
N_max = 100
K = 100
K_IRB = 200  # more sequences for IRB to tighten the bound
e_max = 0.05
N_shots = 1024
mu = 0.005
sigma = 0.001


# Per-gate depolarizing noise parameters (mu, sigma)
# Physically motivated: pi gates noisier than pi/2, virtual gates near-zero
# Tripathi Section 4.4 notes ~80% pi/2 and ~20% pi in a typical Clifford set
GATE_NOISE_TABLE = {
    "I":   (0.001, 0.0002),   # virtual — essentially free
    "X":   (0.015, 0.002),    # pi pulse
    "Y":   (0.015, 0.002),    # pi pulse
    "Z":   (0.001, 0.0002),   # virtual gate
    "H":   (0.008, 0.001),    # pi/2 + pi/2 decomposition
    "S":   (0.006, 0.001),    # pi/2 pulse
    "Sd":  (0.006, 0.001),    # pi/2 pulse
}
GATE_NOISE_DEFAULT_MU    = 0.010  # for remaining 17 Cliffords (multi-pulse)
GATE_NOISE_DEFAULT_SIGMA = 0.002

# DB Parameters (from Tripathi Paper)
DB_gate_time = 88e-9
DB_T1 = 23.36e-6
DB_T2 = 44.13e-6
DB_d_theta = np.radians(0.398)
DB_d_phi = np.radians(0.426)

DB_Num_repeats = 800
DB_shots = 1024


#Outputs
CSV_path = "rb_sequences.csv"
IRB_CSV_path = "irb_sequences.csv"

#Pauli Matrices Creation
I_2 = np.eye(2, dtype = complex)
X = np.array([[0, 1], [1, 0]], dtype = complex)
Y = np.array([[0, -1j], [1j, 0]], dtype = complex)
Z = np.array([[1, 0], [0, -1]], dtype = complex)
PAULISET  = [X, Y, Z]

#State Vectors/Density Matrices
ket_0 = np.array([1, 0], dtype = complex) # |0>
ket_1 = np.array([0, 1], dtype = complex) # |1>
ket_p = np.array([1, 1], dtype = complex) / np.sqrt(2) # |+>
rho_0 = np.outer(ket_0, ket_0.conj()) # |0><0|
rho_1 = np.outer(ket_1, ket_1.conj()) # |1><1|
rho_p = np.outer(ket_p, ket_p.conj()) # |+><+|

#Single-Qubit Clifford Group Functions
def u(theta, n_x, n_y, n_z):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return c * I_2 - 1j * s * (n_x * X + n_y * Y + n_z * Z)

def matrix_key(U):
    flat = U.flatten()
    # Find first element with non-negligible magnitude
    for val in flat:
        if abs(val) > 1e-9:
            phase = val / abs(val)
            break
    else:
        phase = 1.0
    canonical = (U / phase).round(6)
    return tuple(canonical.real.flatten()) + tuple(canonical.imag.flatten())

def clifford_group():
    H = (X + Z) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype = complex)
    generators = [H, S, H.conj().T, S.conj().T]

    seen = {}
    queue = [I_2.copy()]
    seen[matrix_key(I_2)] = I_2.copy()

    while queue:
        g = queue.pop(0)
        for generator in generators:
            child = generator @ g
            k = matrix_key(child)
            if k not in seen:
                seen[k] = child
                queue.append(child)
    
    cliffords = list(seen.values())
    assert len(cliffords) == 24
    return cliffords

CLIFFORD_SET = clifford_group()
NUM_CLIFFORDS = len(CLIFFORD_SET)


#Label Mapping To Each Clifford Gate
def clifford_labels():
    H_ = (X + Z) / np.sqrt(2)
    S_ = np.array([[1, 0], [0,  1j]], dtype=complex)
    Sd_= np.array([[1, 0], [0, -1j]], dtype=complex)

    named = {
        matrix_key(I_2) : "I",
        matrix_key(X)   : "X",
        matrix_key(Y)   : "Y",
        matrix_key(Z)   : "Z",
        matrix_key(H_)  : "H",
        matrix_key(S_)  : "S",
        matrix_key(Sd_) : "Sd",
    }

    paulis = {"X": X, "Y": Y, "Z": Z}

    def gate_signature(U):
        images = []
        for axis, P in paulis.items():
            img = U @ P @ U.conj().T    
            for sign in [1, -1]:
                for name, Q in paulis.items():
                    if np.allclose(img, sign * Q, atol=1e-6):
                        images.append(("+" if sign == 1 else "-") + name)
                        break
        return "".join(images) 

    label_map = {}
    for gate in CLIFFORD_SET:
        k = matrix_key(gate)
        if k in named:
            label_map[k] = named[k]
        else:
            label_map[k] = gate_signature(gate)

    assert len(label_map) == 24, f"Expected 24 Cliffords, got {len(label_map)}"
    return label_map

CLIFFORD_LABEL_SET = clifford_labels()

def clifford_names(G):
    return CLIFFORD_LABEL_SET.get(matrix_key(G), "C?")


def find_clifford_inverse(U_cumul):
    k = matrix_key(U_cumul)
    U_exact = None
    for gate in CLIFFORD_SET:
        if matrix_key(gate) == k:
            U_exact = gate
            break
    if U_exact is None:
        raise ValueError("U_cumul is not in the Clifford set — numerical drift?")
    for gate in CLIFFORD_SET:
        product = gate @ U_exact
        if np.allclose(product, np.eye(2) * product[0,0], atol=1e-6) and \
           abs(abs(product[0,0]) - 1.0) < 1e-6:
            return gate, clifford_names(gate)
    raise ValueError("Inverse not found in Clifford set")

#Error Channel (Noise)
def depolarize(rho, q):
    if q == 0:
        return rho
    noise = (1 - q) * rho
    for P in PAULISET:
        noise += (q / 3) * (P @ rho @ P.conj().T)
    return noise

#Noisy Clifford Sequence/Survival Probability Computation
# Set to True to use single-gate lookup inversion,
# set to False to use the original element-wise inversion

def rb_sequence(n, e_max, rho_in):
    rho = rho_in.copy()
    gate_sequence = []

    for _ in range(n):
        idx = rng.integers(NUM_CLIFFORDS)
        G = CLIFFORD_SET[idx]
        gate_sequence.append(G)
        rho = G @ rho @ G.conj().T

        # Gate-independent depolarizing noise (standard RB, Tripathi Eq. 7)
        q = float(np.clip(rng.normal(mu, sigma), 0, None))
        rho = depolarize(rho, q)

    U_cumul = I_2.copy()
    for G in gate_sequence:
        U_cumul = G @ U_cumul

    C_inverse, inv_name = find_clifford_inverse(U_cumul)
    rho = C_inverse @ rho @ C_inverse.conj().T

    survival = np.real(np.trace(rho_0 @ rho))
    survival = float(np.clip(survival, 0.0, 1.0))

    if N_shots > 0:
        counts = rng.binomial(N_shots, survival)
        survival = counts / N_shots

    gate_names = [clifford_names(G) for G in gate_sequence]
    return survival, gate_names, inv_name


#Interleaved RB Sequence
def irb_sequence(n, e_max, rho_in, target_gate):
    rho = rho_in.copy()
    gate_sequence = []

    target_label = clifford_names(target_gate)
    target_mu, target_sigma = GATE_NOISE_TABLE.get(
        target_label, (GATE_NOISE_DEFAULT_MU, GATE_NOISE_DEFAULT_SIGMA)
    )

    for _ in range(n):
        # Random Clifford — same gate-independent noise as standard RB
        idx = rng.integers(NUM_CLIFFORDS)
        G = CLIFFORD_SET[idx]
        gate_sequence.append(G)
        rho = G @ rho @ G.conj().T
        q = float(np.clip(rng.normal(mu, sigma), 0, None))
        rho = depolarize(rho, q)

        # Target gate — gate-specific noise from table
        gate_sequence.append(target_gate)
        rho = target_gate @ rho @ target_gate.conj().T
        q = float(np.clip(rng.normal(target_mu, target_sigma), 0, None))
        rho = depolarize(rho, q)

    U_cumul = I_2.copy()
    for G in gate_sequence:
        U_cumul = G @ U_cumul

    C_inverse, inv_name = find_clifford_inverse(U_cumul)
    rho = C_inverse @ rho @ C_inverse.conj().T

    survival = np.real(np.trace(rho_0 @ rho))
    survival = float(np.clip(survival, 0.0, 1.0))

    if N_shots > 0:
        counts = rng.binomial(N_shots, survival)
        survival = counts / N_shots

    gate_names = [clifford_names(G) for G in gate_sequence]
    return survival, gate_names, inv_name

#RB Loop Data Collection
def collect_data(n_max, k, e_max, rho_in):
    lengths = np.arange(1, n_max + 1)
    p_avg = np.zeros(n_max)
    p_std = np.zeros(n_max)
    sequence_record = []

    for i, n in enumerate(lengths):
        survivals = []
        for run_idx in range(k):
            survival, gate_names, inv_name = rb_sequence(n, e_max, rho_in)
            survivals.append(survival)

            record = {
                "sequence_length": int(n),
                "run_index": run_idx,
                "survival_probability": round(survival, 6),
                "inversion_gate": inv_name
            }
            for gate_position, name in enumerate(gate_names, start=1):
                record[f"gate_{gate_position}"] = name

            sequence_record.append(record)

        p_avg[i] = np.mean(survivals)
        p_std[i] = np.std(survivals, ddof=1)
        if (i + 1) % 10 == 0 or n == n_max:
            print(f" n = {n:3d} P_avg = {p_avg[i]:.4f} +- {p_std[i]:.4f}")

    return lengths, p_avg, p_std, sequence_record

def collect_irb_data(n_max, k, e_max, rho_in, target_gate):
    lengths = np.arange(1, n_max + 1)
    p_avg = np.zeros(n_max)
    p_std = np.zeros(n_max)
    sequence_record = []

    for i, n in enumerate(lengths):
        survivals = []
        for run_idx in range(k):
            survival, gate_names, inv_name = irb_sequence(n, e_max, rho_in, target_gate)
            survivals.append(survival)

            record = {
                "sequence_length": int(n),
                "run_index": run_idx,
                "survival_probability": round(survival, 6),
                "inversion_gate": inv_name
            }
            for gate_position, name in enumerate(gate_names, start=1):
                record[f"gate_{gate_position}"] = name
            sequence_record.append(record)

        p_avg[i] = np.mean(survivals)
        p_std[i] = np.std(survivals, ddof=1)
        if (i + 1) % 10 == 0 or n == n_max:
            print(f" [IRB] n = {n:3d} P_avg = {p_avg[i]:.4f} +- {p_std[i]:.4f}")

    return lengths, p_avg, p_std, sequence_record  # ← now returned

#Exponential Fit Functionality
def rb_mod(n, A, p, B):
    return A * p ** n + B

def fit_rb_func(lengths, p_avg, p_std):
    p_0 = [0.5, 0.9, 0.5]
    bounds = ([0, 0, 0], [1, 1, 1])
    fit_sigma = p_std if np.all(p_std > 0) else None
    
    try:
        popt, pcov = curve_fit(
            rb_mod, lengths, p_avg,
            p0 = p_0, bounds = bounds,
            sigma = fit_sigma, absolute_sigma = True,
            maxfev = 10000
        )
    except RuntimeError as e:
        print(f" Curve fitting protocol did not converge: {e}")
        popt = np.array(p_0)
        pcov = np.full((3, 3), np.nan)
    
    return popt, pcov

#Clifford Gate Infidelity Conversion
def gate_infidelity_convert(p, D = 2):
    return (D - 1) / D * (1 - p)

def irb_gate_infidelity(p, p_inter, D=2):
    r_gate = ((D - 1) / D) * (1 - p_inter / p)
    return r_gate

def irb_error_bound(p, p_inter, D=2):
    # Eq. 21 from Tripathi — bounds on the true gate infidelity
    term1 = (D - 1) * (abs(p - p_inter / p) + (1 - p)) / D
    term2 = (2 * (D**2 - 1) * (1 - p)) / (p * D**2)
    term3 = (4 * np.sqrt(1 - p) * np.sqrt(D**2 - 1)) / p
    E = min(term1, term2 + term3)
    return E

#Export Sequences to CSV
def export_gate_sequences(sequence_record, path):
    if not sequence_record:
        print(" CSV sequence_record is empty")
        return
    
    max_gates = max(
        sum(1 for k in rec if k.startswith("gate_"))
        for rec in sequence_record
    )

    gate_columns = [f"gate_{i}" for i in range(1, max_gates + 1)]
    field_names = ["sequence_length", "run_index", "survival_probability", "inversion_gate"] + gate_columns

    with open(path, "w", newline = "") as fh:
        writer = csv.DictWriter(fh, fieldnames = field_names, extrasaction = "ignore")
        writer.writeheader()
        for rec in sequence_record:
            row = {col: rec.get(col, "") for col in field_names}
            writer.writerow(row)
    
    num_rows = len(sequence_record)
    size_kb = os.path.getsize(path) / 1024
    print(f"\n [CSV] Written {num_rows} sequences → {path}  ({size_kb:.1f} KB)")
    print(f"  [CSV] Columns: sequence_length, run_index, survival_prob, "
          f"gate_1 … gate_{max_gates}")
    print(f"  [CSV] Gate_1 … gate_<m> are random Clifford gates; "
          f"gate_<m+1> is the recovery gate.")
    
# DB Simulator
def _Rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype = complex)

def _Ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype = complex)

def _Rz(phi):
    return np.array([[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]], dtype = complex)

def noise_gate(best_angle: float, axis: str, d_theta: float, d_phi: float) -> np.ndarray:
    if axis == 'x':
        R = _Rx(best_angle + d_theta)
    elif axis == 'y':
        R = _Ry(best_angle + d_theta)
    else:
        raise ValueError(f"Unknown axis: {axis}")
    return R @ _Rz(d_phi)    # was: _Rz(d_phi) @ R

def db_gates_construct(d_theta: float, d_phi: float):
    X_plus = noise_gate(+np.pi, 'x', d_theta, d_phi)
    X_minus = noise_gate(-np.pi, 'x', -d_theta, -d_phi)
    Y_plus = noise_gate(+np.pi, 'y', d_theta, d_phi)
    Y_minus = noise_gate(-np.pi, 'y', -d_theta, -d_phi)

    return {
        "XX": (X_plus, X_plus),
        "XXd": (X_plus, X_minus),
        "YY": (Y_plus, Y_plus),
        "YYd": (Y_plus, Y_minus),
        "YYdY": (Y_minus, Y_plus)
    }

# Lindblad Theory
def _vec(rho: np.ndarray) -> np.ndarray:
    return rho.flatten(order = 'F')

def _unvec(v: np.ndarray) -> np.ndarray:
    return v.reshape(2, 2, order = 'F')

def _superoperator_commutator(H: np.ndarray) -> np.ndarray:
    I = np.eye(2, dtype = complex)
    out = -1j * (np.kron(I, H) - np.kron(H.T, I))
    return out

def _superoperator_lindblad(L: np.ndarray) -> np.ndarray:
    I = np.eye(2, dtype = complex)
    LdL = L.conj().T @ L
    out = (np.kron(L.conj(), L) - 0.5 * np.kron(I, LdL) - 0.5 * np.kron(LdL.T, I))
    return out

def free_evol_prop(dt: float, T_1: float, T_2: float) -> np.ndarray:
    gamma_1 = 1.0 / T_1
    T_phi = 2 * T_1 * T_2 / max(2 * T_1 - T_2, 1e-30)
    gamma_phi = 1.0 / T_phi

    sigma_minus = np.array([[0, 1], [0, 0]], dtype = complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype = complex)

    Lindblad_tot = (_superoperator_lindblad(sigma_minus) * gamma_1 + _superoperator_lindblad(sigma_z / np.sqrt(2)) * gamma_phi)

    from scipy.linalg import expm
    out = expm(Lindblad_tot * dt)
    return out

def gate_propagator(U: np.ndarray, dt: float, T_1: float, T_2: float) -> np.ndarray:
    U_sup = np.kron(U.conj(), U)
    E_open = free_evol_prop(dt, T_1, T_2)
    out = E_open @ U_sup
    return out

# Fidelity Simulator
def db_simulate_sequence(gate_pair: tuple, rho_init: np.ndarray, num_reps: int, t_g: float, T_1: float, T_2: float, shots: int = 0) -> np.ndarray:
    U_1, U_2 = gate_pair
    E_1 = gate_propagator(U_1, t_g, T_1, T_2)
    E_2 = gate_propagator(U_2, t_g, T_1, T_2)

    E_pair = E_2 @ E_1

    rho_vector = _vec(rho_init)
    rho_measured = _vec(rho_init).conj()

    fidelities = np.zeros(num_reps)
    rho_present = rho_vector.copy()

    for n in range(num_reps):
        rho_present = E_pair @ rho_present
        rho_d_m = _unvec(rho_present)
        F = float(np.real(np.trace(rho_init @ rho_d_m)))
        F = np.clip(F, 0.0, 1.0)

        if shots > 0:
            F = rng.binomial(shots, F) / shots
        
        fidelities[n] = F
    
    return fidelities

def db_simulate_T_1(num_points: int, t_g: float, T_1: float, T_2: float, shots: int = 0) -> tuple:
    from scipy.linalg import expm
    
    E_free = free_evol_prop(t_g, T_1, T_2)
    rho_vector = _vec(rho_1)
    measure_d_m = rho_1

    fidelities = np.zeros(num_points)
    rho_present = rho_vector.copy()

    for n in range(num_points):
        rho_present = E_free @ rho_present
        rho_d_m = _unvec(rho_present)
        F = float(np.real(np.trace(measure_d_m @ rho_d_m)))
        F = np.clip(F, 0.0, 1.0)

        if shots > 0:
            F = rng.binomial(shots, F) / shots
        
        fidelities[n] = F

    times = np.arange(1, num_points + 1) * t_g
    return times, fidelities

# DB Phenomenological Fit
def db_fidelity(t, a, T_D, omega):
    out = 0.5 * (1 + a) + 0.5 * (1 - a) * np.exp(-t / T_D) * np.cos(2 * omega * t)
    return out

def db_t_1_model(t, T_1, a):
    out = 0.5 * (1 + a) + 0.5 * (1 - a) * np.exp(-t / T_1)
    return out

def fit_db_sequence(times: np.ndarray, fidelities: np.ndarray, fix_omega: bool = False, omega_predict: float = 0.0, label: str = "") -> dict:
    if fix_omega:
        def model(t, a, T_D):
            out = db_fidelity(t, a, T_D, 0.0)
            return out
        p_0 = [0.0, times[-1] / 3]
        bounds = ([-1, 1e-12], [1, np.inf])
        try:
            popt, pcov = curve_fit(model, times, fidelities, p0 = p_0, bounds = bounds, maxfev = 20000)
            perr = np.sqrt(np.diag(pcov))
            out = dict(a = popt[0], T_D = popt[1], omega = 0.0, a_err = perr[0], T_D_err = perr[1], omega_err = 0.0, label = label)
            return out
        except Exception as e:
            print(f" DB fit failed: {e}")
            out = dict(a = 0.0, T_D = np.nan, omega = 0.0, a_err = np.nan, T_D_err = np.nan, omega_err = np.nan, label = label)
            return out
        
    else:
        d_c = np.mean(fidelities)
        residual = fidelities - d_c
        if np.max(np.abs(residual)) > 1e-6:
            frequencies = np.fft.rfftfreq(len(times), d = (times[1] - times[0]))
            spectrum = np.abs(np.fft.rfft(residual))
            omega_fft = 2 * np.pi * frequencies[np.argmax(spectrum[1:]) + 1]
        else:
            omega_fft = 0.0
        
        if omega_predict > 0 and (omega_fft == 0.0 or abs(omega_fft - omega_predict) > 2 * omega_predict):
            omega_init = omega_predict
        else:
            omega_init = omega_fft if omega_fft > 0 else 1.0
        
        p_0 = [0.0, times[-1] / 3, omega_init]
        bounds = ([-1, 1e-12, 0.0], [1, np.inf, np.inf])

        try:
            popt, pcov = curve_fit(db_fidelity, times, fidelities, p0 = p_0, bounds = bounds, maxfev = 50000)
            perr = np.sqrt(np.diag(pcov))
            out = dict(a = popt[0], T_D = popt[1], omega = popt[2], a_err = perr[0], T_D_err = perr[1], omega_err = perr[2], label = label)
            return out
        except Exception as e:
            print(f"DB Fit Failed: {e}")
            out = dict(a = 0.0, T_D = np.nan, omega = np.nan, a_err = np.nan, T_D_err = np.nan, omega_err = np.nan, label = label)
            return out

# DB Parameter Extraction
def db_extract(fit_T_1: dict, fit_XX: dict, fit_YY: dict, fit_XXd: dict, t_g: float) -> dict:
    T_1 = fit_T_1["T_D"]
    T_2 = fit_XX["T_D"]
    d_theta = 2 * fit_YY["omega"] * t_g
    d_phi = fit_XXd["omega"] * 2 * t_g

    denominator = max(2 * T_1 - T_2, 1e-30)
    T_phi = 2 * T_1 * T_2 / denominator

    out = dict(T1 = T_1, T2 = T_2, Tphi = T_phi, delta_theta = d_theta, delta_phi = d_phi)
    return out

def compute_lindblad_fidelity(times, propagators, rho_init, measure_state):
    E_step = np.eye(4, dtype=complex)
    for E in propagators:
        E_step = E @ E_step
 
    rho_vec = _vec(rho_init)
    fidelities = np.zeros(len(times))
 
    rho_current = rho_vec.copy()
    for i in range(len(times)):
        rho_current = E_step @ rho_current
        rho_dm = _unvec(rho_current)
        F = float(np.real(np.trace(measure_state @ rho_dm)))
        fidelities[i] = float(np.clip(F, 0.0, 1.0))
 
    return fidelities
 

# Run DB
def run_db(T_1: float = DB_T1, T_2: float = DB_T2, t_g: float = DB_gate_time, d_theta: float = DB_d_theta, d_phi: float = DB_d_phi, num_reps: int = DB_Num_repeats, shots: int = DB_shots) -> dict:
    print("  DETERMINISTIC BENCHMARKING  (Tripathi et al. 2025 §4)")
    print("═" * 60)
    print(f"  T1          = {T_1 * 1e6:.2f} µs")
    print(f"  T2          = {T_2 * 1e6:.2f} µs")
    print(f"  δθ (input)  = {np.degrees(d_theta):.4f}°")
    print(f"  δφ (input)  = {np.degrees(d_phi):.4f}°")
    print(f"  t_g         = {t_g * 1e9:.1f} ns")
    print(f"  n_reps      = {num_reps}")
    print(f"  shots       = {shots}")
    print()

    gates = db_gates_construct(d_theta, d_phi)
    n_arr = np.arange(1, num_reps + 1)
    times = n_arr * 2 * t_g

    print("  [Step 1] Free evolution |1⟩ → T1")
    t1_times, f_T1 = db_simulate_T_1(num_reps, t_g, T_1, T_2, shots=shots)
    fit_t1 = fit_db_sequence(t1_times, f_T1, fix_omega=True, label="T1/free")
    print(f"           T1 fit = {fit_t1['T_D'] * 1e6:.2f} µs  "
          f"(input {T_1 * 1e6:.2f} µs)")
    
    print("  [Step 2] XX;|+⟩ → T2  (insensitive to coherent errors)")
    f_XX = db_simulate_sequence(gates["XX"], rho_p, num_reps, t_g, T_1, T_2, shots)
    fit_XX = fit_db_sequence(times, f_XX, fix_omega=True, label="XX")
    print(f"           T2 fit = {fit_XX['T_D'] * 1e6:.2f} µs  "
          f"(input {T_2 * 1e6:.2f} µs)")
    
    print("  [Step 3] YY;|+⟩ → δθ  (rotation error)")
    f_YY = db_simulate_sequence(gates["YY"], rho_p, num_reps, t_g, T_1, T_2, shots)
    fit_YY = fit_db_sequence(times, f_YY, fix_omega=False, label="YY")
    delta_theta_fit = 2 * fit_YY["omega"] * t_g
    print(f"           ω     = {fit_YY['omega']:.2f} rad/s  →  "
          f"δθ fit = {np.degrees(delta_theta_fit):.4f}°  "
          f"(input {np.degrees(d_theta):.4f}°)")
    
    print("  [Step 4] X̄X;|+⟩ → δφ  (phase error)")
    f_XXbar = db_simulate_sequence(gates["XXd"], rho_p, num_reps, t_g, T_1, T_2, shots)
    omega_prior = d_phi / (2 * t_g)   # use input δφ as warm-start (physical prior)
    fit_XXbar = fit_db_sequence(times, f_XXbar, fix_omega=False,
                                omega_predict = omega_prior, label="XXbar")
    delta_phi_fit = fit_XXbar["omega"] * 2 * t_g 
    print(f"           ω     = {fit_XXbar['omega']:.2f} rad/s  →  "
          f"δφ fit = {np.degrees(delta_phi_fit):.4f}°  "
          f"(input {np.degrees(d_phi):.4f}°)")
    
    params = db_extract(fit_t1, fit_XX, fit_YY, fit_XXbar, t_g)
    print()
    print("  ── Extracted Parameters ──────────────────────────────────")
    print(f"  T1   = {params['T1'] * 1e6:.2f} µs")
    print(f"  T2   = {params['T2'] * 1e6:.2f} µs")
    print(f"  Tφ   = {params['Tphi'] * 1e6:.2f} µs  (pure dephasing)")
    print(f"  δθ   = {np.degrees(params['delta_theta']):.4f}°")
    print(f"  δφ   = {np.degrees(params['delta_phi']):.4f}°")

    print()
    print("  [Test 1] ȲY and YȲ on |+⟩  (T1 asymmetry probe)")
    f_YYdY = db_simulate_sequence(gates["YYdY"], rho_p, num_reps, t_g, T_1, T_2, shots)
    f_YYd = db_simulate_sequence(gates["YYd"], rho_p, num_reps, t_g, T_1, T_2, shots)
    print(f"           ȲY saturation  ≈ {f_YYdY[-10:].mean():.4f}  "
          f"(excited hemisphere, decays faster)")
    print(f"           YȲ saturation  ≈ {f_YYd[-10:].mean():.4f}  "
          f"(ground hemisphere,  decays slower)")
    
    return dict(
        times      = times,
        t1_times   = t1_times,
        f_T1       = f_T1,
        f_XX       = f_XX,
        f_YY       = f_YY,
        f_XXbar    = f_XXbar,
        f_YYbar    = f_YYdY,
        f_YbarY    = f_YYd,
        fit_t1     = fit_t1,
        fit_XX     = fit_XX,
        fit_YY     = fit_YY,
        fit_XXbar  = fit_XXbar,
        params     = params,
        shots      = shots,
        # Input truth values for comparison
        truth      = dict(T1=T_1, T2=T_2, delta_theta=d_theta,
                          delta_phi=d_phi, tg=t_g)
    )

    
#Visualizations
def plot_rb(lengths, p_avg, p_std, popt, r_C, e_max, save_path = None):
    A, p, B = popt
    n_fine = np.linspace(1, lengths[-1], 400)
    fit_curve = rb_mod(n_fine, A, p, B)

    fig, ax = plt.subplots(figsize = (10, 10))
    fig.patch.set_facecolor("green")
    ax.set_facecolor("green")

    ax.errorbar(
        lengths, p_avg, yerr = p_std,
        fmt = "o", color = "red", markersize = 4,
        elinewidth = 0.8, capsize = 3, capthick = 0.8,
        label = f"$P_{{avg}}(n)$ ({K} sequences / length)", 
        zorder = 3
    )
    
    ax.plot(
        n_fine, fit_curve,
        color = "yellow", linewidth = 2,
        label = (
            f"Fit: $A p^n + B$ \n"
            f"$A = {A:.3f}$, $p = {p:.4f}$, $B = {B:.3f}$ \n"
            f"$r_C = (1 - p) / 2 = {r_C * 100:.3f}$"
        ),
        zorder = 4
    )

    ax.axhline(A + B, color = "black")
    ax.axhline(B, color = "orange")
    ax.text(lengths[-1] * 0.98, A + B + 0.01, "$A+B$ (no error)")
    ax.text(lengths[-1] * 0.98, B + 0.01, "$B$ (full depolarization)")

    ax.set_xlabel("Sequence Length $n$")
    ax.set_ylabel("Survival Probability $P_{avg}(n)$")
    ax.set_title(f"Single Qubit Randomized Benchmarking Simulation \n"
    f"$\\mu = {mu}$, $\\sigma = {sigma}$, $K = {K}$, shots / sequence = {N_shots if N_shots else 'inf'}",
    color = "white"
    )

    ax.tick_params(colors = "white")
    for spine in ax.spines.values():
        spine.set_edgecolor("purple")
    ax.grid(True, linestyle = "--")
    ax.set_xlim(0, lengths[-1] + 1)
    ax.set_ylim(-0.02, 1.05)
    legend = ax.legend(facecolor = "blue", edgecolor = "blue", labelcolor = "white", loc = "upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi = 150, bbox_inches = "tight", facecolor = fig.get_facecolor())
        print(f"\nPlot saved to {save_path}")
    
    plt.show()

def plot_irb_single(lengths, p_avg_rb, p_std_rb, popt_rb,
                    lengths_irb, p_avg_irb, p_std_irb, popt_irb,
                    r_C, r_gate, target_name, e_max, save_path=None):
    A_rb, p_rb, B_rb = popt_rb
    A_irb, p_irb, B_irb = popt_irb
    n_fine = np.linspace(1, lengths[-1], 400)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("green")
    ax.set_facecolor("green")

    ax.errorbar(lengths, p_avg_rb, yerr=p_std_rb,
                fmt="o", color="red", markersize=4,
                elinewidth=0.8, capsize=3, capthick=0.8,
                label=f"Standard RB ($r_C = {r_C*100:.3f}$%)", zorder=4)
    ax.plot(n_fine, rb_mod(n_fine, A_rb, p_rb, B_rb),
            color="yellow", linewidth=2,
            label=f"RB fit: $p = {p_rb:.5f}$", zorder=4)

    ax.errorbar(lengths_irb, p_avg_irb, yerr=p_std_irb,
                fmt="s", color="cyan", markersize=4,
                elinewidth=0.8, capsize=3, capthick=0.8,
                label=f"IRB — {target_name} ($r_G = {r_gate*100:.3f}$%)", zorder=4)
    ax.plot(n_fine, rb_mod(n_fine, A_irb, p_irb, B_irb),
            color="magenta", linewidth=2, linestyle="--",
            label=f"IRB fit: $p_{{inter}} = {p_irb:.5f}$", zorder=4)

    ax.set_xlabel("Sequence Length $n$")
    ax.set_ylabel("Survival Probability $P_{avg}(n)$")
    ax.set_title(
        f"Standard RB vs Interleaved RB — Gate: {target_name}\n"
        f"$K = {K}$, shots = {N_shots}, per-gate depolarizing noise",
        color="white"
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("purple")
    ax.grid(True, linestyle="--")
    ax.set_xlim(0, lengths[-1] + 1)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(facecolor="blue", edgecolor="blue", labelcolor="white", loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\nPlot saved to {save_path}")
    plt.show()

def plot_db(db_results: dict, save_path: str = None):
    times = db_results["times"]
    T_1_times = db_results["t1_times"]
    params = db_results["params"]
    truth = db_results["truth"]
    t_g = truth["tg"]

    fit_T_1 = db_results["fit_t1"]
    fit_XX = db_results["fit_XX"]
    fit_YY = db_results["fit_YY"]
    fit_XXd = db_results["fit_XXbar"]

    T_us = times * 1e6
    T_1_us = T_1_times * 1e6

    T_fine = np.linspace(times[0], times[-1], 1000)
    T_1_fine = np.linspace(T_1_times[0], T_1_times[-1], 1000)
    T_us_fine = T_fine * 1e6
    T_1_us_fine = T_1_fine * 1e6

    def curve_T_1(t):
        return db_fidelity(t, fit_T_1["a"], fit_T_1["T_D"], 0.0)

    def curve_XX(t):
        return db_fidelity(t, fit_XX["a"], fit_XX["T_D"], 0.0)
    
    def curve_YY(t):
        return db_fidelity(t, fit_YY["a"], fit_YY["T_D"], fit_YY["omega"])
    
    def curve_XXd(t):
        return db_fidelity(t, fit_XXd["a"], fit_XXd["T_D"], fit_XXd["omega"])
    
    fig, axes = plt.subplots(2, 3, figsize = (10, 10))
    fig.patch.set_facecolor("blue")
    for ax in axes.flat:
        ax.set_facecolor("red")
        ax.tick_params(colors = "white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("purple")
        ax.grid(True, linestyle = "--", color = "purple", alpha = 0.7)
        ax.set_ylim(-0.05, 1.05)
    
    marker_kw = dict(fmt = "o", markersize = 3.5, elinewidth = 0.7, capsize = 2.5, capthick = 0.7, alpha = 0.85)

    ax = axes[0, 0]
    ax.plot(T_1_us, db_results["f_T1"], "o", color = "red", markersize = 3.5, alpha = 0.8, label = "Free; |1> (data)")
    ax.plot(T_1_us_fine, curve_T_1(T_1_fine), color="#ff9999", lw=1.8,
            label=f"Fit  T₁ = {params['T1']*1e6:.2f} µs")
    ax.axhline(0.5, color="#888899", lw=0.8, ls=":")
    ax.set_xlabel("Evolution time (µs)")
    ax.set_ylabel("Fidelity F̃")
    ax.set_title(f"Step 1 — T₁ Measurement\nT₁ = {params['T1']*1e6:.2f} µs  "
                 f"[truth {truth['T1']*1e6:.2f} µs]")
    ax.legend(facecolor="#1a1a33", edgecolor="#555577",
              labelcolor="white", fontsize=8)
    
    ax = axes[0, 1]
    ax.plot(T_us, db_results["f_XX"], "o", color="#4ecdc4",
            markersize=3.5, alpha=0.8, label="XX;|+⟩ (data)")
    ax.plot(T_us_fine, curve_XX(T_fine), color="#80eeea", lw=1.8,
            label=f"Fit  T₂ = {params['T2']*1e6:.2f} µs")
    ax.axhline(0.5, color="#888899", lw=0.8, ls=":")
    ax.set_xlabel("Evolution time (µs)")
    ax.set_title(f"Step 2 — T₂ via XX;|+⟩\nT₂ = {params['T2']*1e6:.2f} µs  "
                 f"[truth {truth['T2']*1e6:.2f} µs]")
    ax.legend(facecolor="#1a1a33", edgecolor="#555577",
              labelcolor="white", fontsize=8)
    
    ax = axes[0, 2]
    ax.plot(T_us, db_results["f_YY"], "o", color="#f7dc6f",
            markersize=3.5, alpha=0.8, label="YY;|+⟩ (data)")
    ax.plot(T_us_fine, curve_YY(T_fine), color="#fde98d", lw=1.8,
            label=f"Fit  δθ = {np.degrees(params['delta_theta']):.4f}°")
    ax.set_xlabel("Evolution time (µs)")
    ax.set_title(f"Step 3 — δθ via YY;|+⟩\n"
                 f"δθ = {np.degrees(params['delta_theta']):.4f}°  "
                 f"[truth {np.degrees(truth['delta_theta']):.4f}°]")
    ax.legend(facecolor="#1a1a33", edgecolor="#555577",
              labelcolor="white", fontsize=8)
    
    ax = axes[1, 0]
    ax.plot(T_us, db_results["f_XXbar"], "o", color="#bb8fce",
            markersize=3.5, alpha=0.8, label="X̄X;|+⟩ (data)")
    ax.plot(T_us_fine, curve_XXd(T_fine), color="#d2b4de", lw=1.8,
            label=f"Fit  δφ = {np.degrees(params['delta_phi']):.4f}°")
    ax.set_xlabel("Evolution time (µs)")
    ax.set_ylabel("Fidelity F̃")
    ax.set_title(f"Step 4 — δφ via X̄X;|+⟩\n"
                 f"δφ = {np.degrees(params['delta_phi']):.4f}°  "
                 f"[truth {np.degrees(truth['delta_phi']):.4f}°]")
    ax.legend(facecolor="#1a1a33", edgecolor="#555577",
              labelcolor="white", fontsize=8)
    
    ax = axes[1, 1]
    ax.plot(T_us, db_results["f_YYbar"], "o", color="#e74c3c",
            markersize=3.5, alpha=0.8, label="ȲY;|+⟩  (excited ↓ faster)")
    ax.plot(T_us, db_results["f_YbarY"], "s", color="#2ecc71",
            markersize=3.5, alpha=0.8, label="YȲ;|+⟩  (ground ↓ slower)")
    ax.axhline(0.5, color="#888899", lw=0.8, ls=":")
    ax.set_xlabel("Evolution time (µs)")
    ax.set_title("Test 1 — T₁ Asymmetry\nȲY vs YȲ on |+⟩")
    ax.legend(facecolor="#1a1a33", edgecolor="#555577",
              labelcolor="white", fontsize=8)
    
    ax = axes[1, 2]
    ax.axis("off")
    table_data = [
        ["Parameter", "Extracted", "Input (truth)", "Error"],
        ["T₁ (µs)",
         f"{params['T1']*1e6:.3f}",
         f"{truth['T1']*1e6:.3f}",
         f"{(params['T1']-truth['T1'])/truth['T1']*100:+.2f}%"],
        ["T₂ (µs)",
         f"{params['T2']*1e6:.3f}",
         f"{truth['T2']*1e6:.3f}",
         f"{(params['T2']-truth['T2'])/truth['T2']*100:+.2f}%"],
        ["Tφ (µs)",
         f"{params['Tphi']*1e6:.3f}", "—", "—"],
        ["δθ (°)",
         f"{np.degrees(params['delta_theta']):.4f}",
         f"{np.degrees(truth['delta_theta']):.4f}",
         f"{(params['delta_theta']-truth['delta_theta'])/truth['delta_theta']*100:+.2f}%"],
        ["δφ (°)",
         f"{np.degrees(params['delta_phi']):.4f}",
         f"{np.degrees(truth['delta_phi']):.4f}",
         f"{(params['delta_phi']-truth['delta_phi'])/truth['delta_phi']*100:+.2f}%"],
    ]

    col_colors = [["#1a2a4a"] * 4]
    cell_colors = [["#1a2a4a"] * 4] + [["#111128"] * 4] * (len(table_data) - 1)
 
    table = ax.table(cellText=table_data[1:],
                     colLabels=table_data[0],
                     loc="center",
                     cellLoc="center",
                     cellColours=cell_colors[1:],
                     colColours=["#1a2a4a"] * 4)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.8)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#444466")
        cell.get_text().set_color("white")
    
    ax.set_title("DB Extracted vs Truth", color="white", fontsize=10, pad=12)

    n_shots_used = db_results.get("shots", DB_shots)
    fig.suptitle(
        "Deterministic Benchmarking  —  Tripathi et al. 2025 §4\n"
        f"t_g = {truth['tg']*1e9:.0f} ns  |  shots = {n_shots_used}",
        color="white", fontsize=13, y=1.01)
 
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\nDB plot saved to {save_path}")
    plt.show()

def plot_db_figure6(db_results: dict, save_path: str = None):
    truth   = db_results["truth"]
    params  = db_results["params"]
    times   = db_results["times"]       # 2·n·tg  for gate sequences
    t1_times = db_results["t1_times"]   # n·tg    for free evolution
 
    T_1  = params["T1"]
    T_2  = params["T2"]
    t_g  = truth["tg"]
    d_theta = truth["delta_theta"]
    d_phi   = truth["delta_phi"]
 
    fit_t1    = db_results["fit_t1"]
    fit_XX    = db_results["fit_XX"]
    fit_YY    = db_results["fit_YY"]
    fit_XXbar = db_results["fit_XXbar"]
 
    T1_fine   = np.linspace(t1_times[0], t1_times[-1], 800)
    T_fine    = np.linspace(times[0],    times[-1],    800)
    T1_us_fine = T1_fine * 1e6
    T_us_fine  = T_fine  * 1e6
    T1_us      = t1_times * 1e6
    T_us       = times    * 1e6
 
    def analytic(t, fit):
        return db_fidelity(t, fit["a"], fit["T_D"], fit["omega"])
 
    curve_T1    = analytic(T1_fine,  fit_t1)
    curve_XX    = analytic(T_fine,   fit_XX)
    curve_YY    = analytic(T_fine,   fit_YY)
    curve_XXbar = analytic(T_fine,   fit_XXbar)
 
    gates = db_gates_construct(d_theta, d_phi)
 
    E_free = free_evol_prop(t_g, T_1, T_2)
    lind_T1 = compute_lindblad_fidelity(t1_times, [E_free], rho_1, rho_1)
 
    def lind_seq(key, rho_in, meas):
        U1, U2 = gates[key]
        E1 = gate_propagator(U1, t_g, T_1, T_2)
        E2 = gate_propagator(U2, t_g, T_1, T_2)
        return compute_lindblad_fidelity(times, [E1, E2], rho_in, meas)
 
    lind_XX    = lind_seq("XX",   rho_p, rho_p)
    lind_YY    = lind_seq("YY",   rho_p, rho_p)
    lind_XXbar = lind_seq("XXd",  rho_p, rho_p)
    lind_YYbar = lind_seq("YYdY", rho_p, rho_p)   # ȲY  (excited hem.)
    lind_YbarY = lind_seq("YYd",  rho_p, rho_p)   # YȲ  (ground hem.)
 
    seq_styles = [
        # (label,            x_us,   data,                  color,     marker,
        #  analytic_x,       analytic_y,    lind_x,    lind_y)
        ("Free; |1⟩",        T1_us,  db_results["f_T1"],   "#4dffb4", "o",
         T1_us_fine, curve_T1,    T1_us, lind_T1),
        ("XX; |+⟩",          T_us,   db_results["f_XX"],   "#4dc3ff", "^",
         T_us_fine,  curve_XX,    T_us,  lind_XX),
        ("YY; |+⟩",          T_us,   db_results["f_YY"],   "#ffd700", "v",
         T_us_fine,  curve_YY,    T_us,  lind_YY),
        ("X̄X; |+⟩",         T_us,   db_results["f_XXbar"],"#ff7043", "s",
         T_us_fine,  curve_XXbar, T_us,  lind_XXbar),
        ("ȲY; |+⟩",          T_us,   db_results["f_YYbar"],"#ce93d8", "D",
         T_us,       db_results["f_YYbar"], T_us, lind_YYbar),
        ("YȲ; |+⟩",          T_us,   db_results["f_YbarY"],"#a5d6a7", "P",
         T_us,       db_results["f_YbarY"], T_us, lind_YbarY),
    ]
 
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
 
    first_data = True   # for legend grouping headers (proxy artists)
    for (label, x_d, data, color, mkr,
         x_a,   a_curve, x_l, l_curve) in seq_styles:
 
        # empirical data – symbols
        ax.plot(x_d, data, mkr,
                color=color, markersize=4, alpha=0.85,
                markeredgewidth=0.3, markeredgecolor="white",
                label=label, zorder=4)
 
        # analytical fit – solid line
        ax.plot(x_a, a_curve, "-",
                color=color, linewidth=1.6, alpha=0.9, zorder=3)
 
        # Lindblad numerical – dashed line
        ax.plot(x_l, l_curve, "--",
                color=color, linewidth=1.3, alpha=0.75, zorder=2)
 
    from matplotlib.lines import Line2D
    proxy_data   = Line2D([0],[0], ls="none",  marker="o", color="white",
                          markersize=5, label="Simulated data (symbols)")
    proxy_fit    = Line2D([0],[0], ls="-",    color="white", lw=1.6,
                          label="Analytical fit  (solid)")
    proxy_lind   = Line2D([0],[0], ls="--",   color="white", lw=1.3,
                          label="Lindblad model (dashed)")
 
    # collect sequence legend handles then append style proxies
    seq_handles, seq_labels = ax.get_legend_handles_labels()
    all_handles = seq_handles + [proxy_data, proxy_fit, proxy_lind]
    all_labels  = seq_labels  + ["Simulated data (symbols)",
                                  "Analytical fit  (solid)",
                                  "Lindblad model (dashed)"]
 
    leg = ax.legend(all_handles, all_labels,
                    ncol=2, fontsize=8,
                    facecolor="#1c2230", edgecolor="#444466",
                    labelcolor="white",
                    loc="upper right")
 
    ax.set_xlabel("Evolution time (µs)", color="white", fontsize=12)
    ax.set_ylabel("Fidelity  F̃", color="white", fontsize=12)
    ax.set_title(
        "Deterministic Benchmarking — Figure 6 overlay\n"
        f"T₁ = {T_1*1e6:.2f} µs · T₂ = {T_2*1e6:.2f} µs · "
        f"δθ = {np.degrees(d_theta):.3f}° · δφ = {np.degrees(d_phi):.3f}°",
        color="white", fontsize=11, pad=10
    )
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(left=0)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(True, linestyle="--", color="#2a3550", alpha=0.8)
 
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\nFigure-6 overlay saved to {save_path}")
 
    plt.show()
 
 



def main():
    print("Single-Qubit Randomized Benchmarking")
    print(f"Clifford Group Size: {NUM_CLIFFORDS}")
    print(f"Initial State: |0>")
    print(f"Max Sequence Length: {N_max}")
    print(f"Sequences / Length: {K}")
    print(f"RB Noise: mu = {mu}, sigma = {sigma}")
    print(f"Measurement Shots: {N_shots if N_shots else 'exact (inf)'}")

    # --- Standard RB ---
    print("\n--- Standard RB ---")
    lengths, p_avg, p_std, sequence_record = collect_data(N_max, K, e_max, rho_0)
    export_gate_sequences(sequence_record, CSV_path)
    popt_rb, pcov_rb = fit_rb_func(lengths, p_avg, p_std)
    A_rb, p_rb, B_rb = popt_rb
    perr_rb = np.sqrt(np.diag(pcov_rb)) if not np.any(np.isnan(pcov_rb)) else [np.nan]*3
    r_C = gate_infidelity_convert(p_rb)
    r_C_err = perr_rb[1] / 2

    print("Standard RB Results")
    print(f"A = {A_rb:.4f} +- {perr_rb[0]:.4f}")
    print(f"p = {p_rb:.5f} +- {perr_rb[1]:.5f}")
    print(f"B = {B_rb:.4f} +- {perr_rb[2]:.4f}")
    print(f"r_C = {r_C*100:.4f}% +- {r_C_err*100:.4f}%")

    plot_rb(lengths, p_avg, p_std, popt_rb, r_C, e_max,
            save_path="rb_single_qubit.png")

# --- Interleaved RB: user-selected target gate ---
    # Choose from any gate name in CLIFFORD_LABEL_SET, e.g.:
    # "I", "X", "Y", "Z", "H", "S", "Sd"
    # or signature names like "+X+Z-Y", "-Y+Z-X", etc.
    # Run print(set(CLIFFORD_LABEL_SET.values())) to see all 24 names
    TARGET_GATE_NAME = "X"

    # Resolve name to matrix
    name_to_mat = {}
    for gate in CLIFFORD_SET:
        label = CLIFFORD_LABEL_SET[matrix_key(gate)]
        name_to_mat[label] = gate
    if TARGET_GATE_NAME not in name_to_mat:
        print(f"\nUnknown gate '{TARGET_GATE_NAME}'. Available gates:")
        print(sorted(set(CLIFFORD_LABEL_SET.values())))
    else:
        target_gate = name_to_mat[TARGET_GATE_NAME]
        print(f"\n--- Interleaved RB (target gate: {TARGET_GATE_NAME}) ---")

        lengths_irb, p_avg_irb, p_std_irb, irb_sequence_record = collect_irb_data(N_max, K_IRB, e_max, rho_0, target_gate)
        export_gate_sequences(irb_sequence_record, IRB_CSV_path)
        
        popt_irb, pcov_irb = fit_rb_func(lengths_irb, p_avg_irb, p_std_irb)
        A_irb, p_irb, B_irb = popt_irb
        perr_irb = (np.sqrt(np.diag(pcov_irb))
                    if not np.any(np.isnan(pcov_irb))
                    else [np.nan]*3)

        r_gate = irb_gate_infidelity(p_rb, p_irb)
        E = irb_error_bound(p_rb, p_irb)
        r_gate_err = perr_irb[1] / (2 * p_rb)

        print(f"p_inter = {p_irb:.5f} +- {perr_irb[1]:.5f}")
        print(f"r_G     = {r_gate*100:.4f}% +- {r_gate_err*100:.4f}%")
        r_G_lower = max(r_gate - E, 0.0)
        r_G_upper = min(r_gate + E, 1.0)
        flag = " [lower bound clamped]" if r_gate <= E else ""
        print(f"E bound = {E*100:.4f}%  =>  "
              f"r_G in [{r_G_lower*100:.4f}%, {r_G_upper*100:.4f}%]{flag}")

        plot_irb_single(lengths, p_avg, p_std, popt_rb,
                        lengths_irb, p_avg_irb, p_std_irb, popt_irb,
                        r_C, r_gate, TARGET_GATE_NAME, e_max,
                        save_path=f"irb_{TARGET_GATE_NAME}.png")

        db_results = run_db(
        T_1          = DB_T1,
        T_2          = DB_T2,
        t_g          = DB_gate_time,
        d_theta = DB_d_theta,
        d_phi   = DB_d_phi,
        num_reps      = DB_Num_repeats,
        shots       = DB_shots,
    )
    plot_db(db_results, save_path="db_results.png")

    plot_db_figure6(db_results, save_path="db_figure6.png")

#Run Main
if __name__ == "__main__":
    main()
