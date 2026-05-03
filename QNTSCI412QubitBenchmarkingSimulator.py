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
rho_0 = np.outer(ket_0, ket_0.conj()) # |0><0|

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

#Run Main
if __name__ == "__main__":
    main()
