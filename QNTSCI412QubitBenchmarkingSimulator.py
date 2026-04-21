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
K = 50
e_max = 0.05
N_shots = 1024

#Outputs
CSV_path = "rb_sequences.csv"

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
    idx = np.argmax(np.abs(flat))
    phase = flat[idx] / abs(flat[idx])
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


#Error Channel (Noise)
def depolarize(rho, q):
    if q == 0:
        return rho
    noise = (1 - q) * rho
    for P in PAULISET:
        noise += (q / 3) * (P @ rho @ P.conj().T)
    return noise

#Noisy Clifford Sequence/Survival Probability Computation
def rb_sequence(n, e_max, rho_in):
    rho = rho_in.copy()
    U_cumul = I_2.copy()
    gate_names = []

    for _ in range(n):
        idx = rng.integers(NUM_CLIFFORDS)
        G = CLIFFORD_SET[idx]

        gate_names.append(clifford_names(G))

        rho = G @ rho @ G.conj().T

        q = rng.uniform(0, e_max)
        rho = depolarize(rho, q)

        U_cumul = G @ U_cumul

    C_inverse = U_cumul.conj().T
    rho = C_inverse @ rho @ C_inverse.conj().T

    gate_names.append(clifford_names(C_inverse))

    survival = np.real(np.trace(rho_0 @ rho))
    survival = float(np.clip(survival, 0.0, 1.0))

    if N_shots > 0:
        counts = rng.binomial(N_shots, survival)
        survival = counts / N_shots

    return survival, gate_names

#RB Loop Data Collection
def collect_data(n_max, k, e_max, rho_in):
    lengths = np.arange(1, n_max + 1)
    p_avg = np.zeros(n_max)
    p_std = np.zeros(n_max)
    sequence_record = []

    for i, n in enumerate(lengths):
        survivals = []
        for run_idx in range(k):
            survival, gate_names = rb_sequence(n, e_max, rho_in)
            survivals.append(survival)

            record = {
                "sequence_length": int(n),
                "run_index": run_idx,
                "survival_probability": round(survival, 6),
            }

            for gate_position, name in enumerate(gate_names, start = 1):
                record[f"gate_{gate_position}"] = name
            
            sequence_record.append(record)
        
        p_avg[i] = np.mean(survivals)
        p_std[i] = np.std(survivals, ddof = 1)
        if (i + 1) % 10 == 0 or n == n_max:
            print(f" n = {n:3d} P_avg = {p_avg[i]:.4f} +- {p_std[i]:.4f}")
    
    return lengths, p_avg, p_std, sequence_record

#Exponential Fit Functionality
def rb_mod(n, A, p, B):
    return A * p ** n + B

def fit_rb_func(lengths, p_avg, p_std):
    p_0 = [0.5, 0.9, 0.5]
    bounds = ([0, 0, 0], [1, 1, 1])
    sigma = p_std if np.all(p_std > 0) else None
    
    try:
        popt, pcov = curve_fit(
            rb_mod, lengths, p_avg,
            p0 = p_0, bounds = bounds,
            sigma = sigma, absolute_sigma = True,
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
    field_names = ["sequence_length", "run_index", "survival_probability"] + gate_columns

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
    ax.set_title(
        f"Single Qubit Randomized Benchmarking Simulation \n"
        f"$\\epsilon_{{max}} = {e_max}$, $K = {K}$, shots / sequence = {N_shots if N_shots else 'inf'}",
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


#Main Executable
def main():
    print("Single-Qubit Randomized Benchmarking")
    print(f"Clifford Group Size: {NUM_CLIFFORDS}")
    print(f"Initial State: |0>")
    print(f"Max Sequence Length: {N_max}")
    print(f"Sequences / Length: {K}")
    print(f"Max Gate Error: {e_max}")
    print(f"Measurement Shots: {N_shots if N_shots else 'exact (inf)'}")

    lengths, p_avg, p_std, sequence_record = collect_data(N_max, K, e_max, rho_0)
    export_gate_sequences(sequence_record, CSV_path)
    popt, pcov = fit_rb_func(lengths, p_avg, p_std)
    A, p, B = popt
    perr = np.sqrt(np.diag(pcov)) if not np.any(np.isnan(pcov)) else [np.nan] * 3
    r_C = gate_infidelity_convert(p)
    r_C_err = perr[1] / 2

    print("Randomized Benchmarking Results")
    print(f"A (SPAM Amplitude) = {A:.4f} +- {perr[0]:.4f}")
    print(f"p (decay parameter) = {p:.5f} +- {perr[1]:.5f}")
    print(f"B (SPAM Offset) = {B:.4f} +- {perr[2]:.4f}")
    print(f"SPAM Error = 1 - (A + B) = {1 - (A + B):.4f}")
    print(f"\n Average Gate Infidelity r_C = (1 - p) / 2")
    print(f" r_C = {r_C * 100 :.4f}% +- {r_C_err * 100:.4f}%")

    plot_rb(lengths, p_avg, p_std, popt, r_C, e_max, save_path = "rb_single_qubit.png")


#Run Main
if __name__ == "__main__":
    main()
