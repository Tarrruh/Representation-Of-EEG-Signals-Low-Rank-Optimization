import argparse
import os
import json
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import mne
import cvxpy as cp
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def second_order_diff_matrix(N: int) -> sp.csr_matrix:
    data = []
    rows = []
    cols = []
    for i in range(N - 2):
        rows.extend([i, i, i])
        cols.extend([i, i+1, i+2])
        data.extend([1.0, -2.0, 1.0])
    Omega = sp.csr_matrix((data, (rows, cols)), shape=(N - 2, N), dtype=float)
    return Omega


def project_affine_measurement(Z: np.ndarray, Phi: np.ndarray, Y: np.ndarray) -> np.ndarray:
    PhiPhiT_inv = la.pinv(Phi @ Phi.T)
    correction = Phi.T @ (PhiPhiT_inv @ (Phi @ Z - Y))
    X_proj = Z - correction
    return X_proj


def compute_metrics(X_true: np.ndarray, X_rec: np.ndarray) -> dict:

    Xt = X_true / (la.norm(X_true, ord='fro') + 1e-12)
    Xr = X_rec / (la.norm(X_rec, ord='fro') + 1e-12)

    mse = la.norm(Xr - Xt, ord='fro')**2 / X_true.size

    num = np.vdot(X_true.ravel(), X_rec.ravel()).real
    den = (la.norm(X_true, ord='fro') * la.norm(X_rec, ord='fro') + 1e-12)
    mcc = num / den

    return {"MSE": float(mse), "MCC": float(mcc)}


def sclr_cvxpy(Phi: np.ndarray, Y: np.ndarray, Omega: sp.csr_matrix,
               alpha: float = 1.0, beta: float = 1.0, verbose: bool = True) -> np.ndarray:

    N = Phi.shape[1]
    R = Y.shape[1]
    X = cp.Variable((N, R))
    Omega_cvx = cp.Constant(Omega.toarray())

    obj = alpha * cp.norm1(cp.vec(Omega_cvx @ X, order='F')) + beta * cp.normNuc(X)
    constraints = [Phi @ X == Y]
    prob = cp.Problem(cp.Minimize(obj), constraints)

    try_solvers = ["MOSEK", "SCS", "OSQP", "ECOS"]
    last_status = None
    for s in try_solvers:
        try:
            prob.solve(solver=s, verbose=verbose, max_iters=20000)
            last_status = prob.status
            if X.value is not None and last_status in ["optimal", "optimal_inaccurate"]:
                break
        except Exception as e:
            last_status = f"failed_with_{s}: {e}"

    if X.value is None:
        raise RuntimeError(f"CVXPY failed to solve SCLR. Last status: {last_status}")
    return np.array(X.value, dtype=float)


def main():
    parser = argparse.ArgumentParser(description="EEG SCLR (convex relaxation) reconstruction demo")
    parser.add_argument("--edf_path", type=str, required=True, help="Path to the EDF file")
    parser.add_argument("--channels", type=str, default="Fp1-F7,F7-T7,T7-P7,P7-O1", help="MNE picks by channel names")
    parser.add_argument("--segment_start", type=float, default=0.0, help="Start of the EEG segment in seconds")
    parser.add_argument("--segment_len", type=float, default=10.0, help="Length of the EEG segment in seconds")
    parser.add_argument("--ssr", type=float, default=0.35, help="Subsampling ratio M/N")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for cosparsity term")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for nuclear norm term")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Phi")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Directory to save outputs")
    parser.add_argument("--plot", action="store_true", help="Generate and save plots of results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading EDF: {args.edf_path}")
    raw = mne.io.read_raw_edf(args.edf_path, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True)
    ch_names_requested = [c.strip() for c in args.channels.split(",") if c.strip()]

    existing = [c for c in ch_names_requested if c in raw.ch_names]
    if len(existing) == 0:
        raise ValueError("None of the requested channels were found in this EDF. "
                         f"Requested: {ch_names_requested}\nAvailable example: {raw.ch_names[:10]}")
    raw.pick(existing)

    sfreq = raw.info["sfreq"]
    if abs(sfreq - 256.0) > 1e-6:
        print(f"WARNING: Sampling frequency is {sfreq} Hz (expected 256 Hz for CHB-MIT). Proceeding anyway.")

    start_s = float(args.segment_start)
    dur_s = float(args.segment_len)
    start_sample = int(start_s * sfreq)
    stop_sample = int((start_s + dur_s) * sfreq)

    data, times = raw.get_data(start=start_sample, stop=stop_sample, return_times=True)
    X_true = data.T
    N, R = X_true.shape
    print(f"Segment shape: N={N} samples, R={R} channels")

    X_true_norm = X_true / (la.norm(X_true, ord='fro') + 1e-12)

    rng = np.random.default_rng(args.seed)
    M = int(np.ceil(args.ssr * N))
    Phi = rng.standard_normal((M, N)) / np.sqrt(M) 
    Y = Phi @ X_true_norm

    Omega = second_order_diff_matrix(N)

    print("Solving convex SCLR...")
    X_rec = sclr_cvxpy(Phi, Y, Omega, alpha=args.alpha, beta=args.beta, verbose=False)

    X_rec = project_affine_measurement(X_rec, Phi, Y)

    metrics = compute_metrics(X_true_norm, X_rec)
    print("Reconstruction metrics:")
    print(json.dumps(metrics, indent=2))

    np.save(out_dir / "X_true.npy", X_true_norm)
    np.save(out_dir / "X_rec.npy", X_rec)
    np.save(out_dir / "Phi.npy", Phi)
    np.save(out_dir / "Y.npy", Y)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if args.plot:
        t = np.arange(N) / sfreq
        for ch in range(min(R, 4)):
            plt.figure()
            plt.plot(t, X_true_norm[:, ch], label="True (norm)")
            plt.plot(t, X_rec[:, ch], label="Reconstructed")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (a.u.)")
            plt.title(f"Channel {existing[ch]}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"waveform_ch{ch}.png")
            plt.close()

        s_true = la.svdvals(X_true_norm)
        s_rec  = la.svdvals(X_rec)
        plt.figure()
        plt.semilogy(s_true, marker="o", label="True")
        plt.semilogy(s_rec, marker="x", label="Reconstructed")
        plt.xlabel("Index")
        plt.ylabel("Singular value (log)")
        plt.title("Singular values")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "singular_values.png")
        plt.close()

    print(f"Done. Outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
