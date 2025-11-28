import subprocess
import sys
from pathlib import Path
from datetime import datetime
import csv

import numpy as np

# Import shared utilities
from common import load_config, make_path, SCRIPT_DIR, CONFIG_PATH

# Try importing both libraries
try:
    import iisignature
    HAS_IISIG = True
except ImportError:
    HAS_IISIG = False
    print("Warning: iisignature not available", file=sys.stderr)

try:
    import pysiglib
    HAS_PYSIGLIB = True
except ImportError:
    HAS_PYSIGLIB = False
    print("Warning: pysiglib not available", file=sys.stderr)

# -------- call Julia helper --------

def julia_signature(N: int, d: int, m: int, path_kind: str, operation: str) -> np.ndarray:
    # Ensure we use the sigcheck.jl in the same folder
    sigcheck_script = SCRIPT_DIR / "sigcheck.jl"
    if not sigcheck_script.exists():
        raise RuntimeError(f"sigcheck.jl not found at {sigcheck_script}")

    cmd = [
        "julia",
        "--startup-file=no",
        "--project=.",
        str(sigcheck_script),
        str(N),
        str(d),
        str(m),
        path_kind,
        operation,
    ]

    # Run from repo root (parent of benchmark folder) so Project.toml is found
    repo_root = SCRIPT_DIR.parent
    
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        print("Julia stdout:\n", result.stdout, file=sys.stderr)
        print("Julia stderr:\n", result.stderr, file=sys.stderr)
        raise RuntimeError(f"Julia sigcheck failed with code {result.returncode}")

    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        print("Julia stderr:\n", result.stderr, file=sys.stderr)
        raise RuntimeError("Julia sigcheck produced no numeric output")
    
    # Parse the last line
    line = lines[-1]
    try:
        values = np.fromstring(line, sep=" ", dtype=float)
    except Exception as e:
        print(f"Parse error on line: {line[:100]}...", file=sys.stderr)
        raise e

    return values

# -------- Python signature calculations --------

def iisignature_compute(path: np.ndarray, m: int, operation: str, logsig_method: str) -> np.ndarray:
    if not HAS_IISIG:
        return None
    
    d = path.shape[1]
    if hasattr(iisignature, "_basis_cache"):
        iisignature._basis_cache.clear()
    
    if operation == "signature":
        return np.asarray(iisignature.sig(path, m), dtype=float).ravel()
    else:  # logsignature
        if d < 2:
            return None
        basis = iisignature.prepare(d, m, logsig_method)
        return np.asarray(iisignature.logsig(path, basis, logsig_method), dtype=float).ravel()

def pysiglib_compute(path: np.ndarray, m: int, operation: str) -> np.ndarray:
    if not HAS_PYSIGLIB:
        return None
    
    if operation != "signature":
        return None
    
    try:
        # pysiglib expects (Batch, Length, Dim)
        path_batch = path[np.newaxis, :, :]
        
        # Compute
        sig = np.asarray(pysiglib.signature(path_batch, degree=m), dtype=float).ravel()
        
        # FIX: pysiglib includes level 0 (scalar 1.0) at index 0.
        # iisignature and ChenSignatures.jl do not. We slice it off.
        if sig.size > 0:
            return sig[1:]
        return sig
        
    except Exception as e:
        print(f"pysiglib error for m={m}, op={operation}: {e}", file=sys.stderr)
        return None

# -------- main comparison loop --------

def compare_signatures():
    cfg = load_config(CONFIG_PATH)
    
    # Updated defaults for a "Meaningful" check
    Ns = cfg.get("Ns", [50, 100, 4000])
    Ds = cfg.get("Ds", [2, 3, 5])
    Ms = cfg.get("Ms", [4, 6, 8])
    path_kind = cfg.get("path_kind", "sin")
    runs_dir = SCRIPT_DIR / cfg.get("runs_dir", "runs")
    logsig_method = cfg.get("logsig_method", "O")
    operations = cfg.get("operations", ["signature", "logsignature"])
    
    runs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = runs_dir / f"signature_compare_{ts}.csv"

    fieldnames = [
        "N", "d", "m", "path_kind", "operation",
        "python_library", "len_sig",
        "max_abs_diff", "l2_diff", "rel_l2_diff", "status"
    ]

    print(f"Comparing signatures...")
    print(f"  operations:  {operations}")
    print(f"  path_kind:   {path_kind} (Using non-linear path for rigour)")
    print(f"  iisignature: {'available' if HAS_IISIG else 'NOT AVAILABLE'}")
    print(f"  pysiglib:    {'available' if HAS_PYSIGLIB else 'NOT AVAILABLE'}")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for N in Ns:
            for d in Ds:
                for m in Ms:
                    # Skip insanely large outputs for validation (d^m > 2 million elements)
                    # 5^8 is ~390k (fine), but 5^10 is 9M.
                    total_els = (d**(m+1) - d) // (d - 1)
                    if total_els > 2_000_000:
                        print(f"Skipping N={N} d={d} m={m} (Size {total_els} too large for quick check)")
                        continue

                    path = make_path(d, N, path_kind)

                    for op in operations:
                        if op == "logsignature" and d < 2:
                            continue

                        print(f"Comparing {op} for N={N}, d={d}, m={m}, kind={path_kind}...")

                        # 1. Julia calculation (reference)
                        try:
                            sig_jl = julia_signature(N, d, m, path_kind, op)
                        except Exception as e:
                            print(f"  Julia Failed: {e}")
                            continue

                        # 2. Compare against iisignature
                        if HAS_IISIG:
                            sig_iisig = iisignature_compute(path, m, op, logsig_method)
                            if sig_iisig is not None:
                                if sig_iisig.shape != sig_jl.shape:
                                    print(f"  iisig: SIZE MISMATCH jl={sig_jl.shape} py={sig_iisig.shape}")
                                    status = "shape_mismatch"
                                    max_abs = l2 = rel_l2 = -1.0
                                else:
                                    diff = sig_jl - sig_iisig
                                    max_abs = float(np.max(np.abs(diff)))
                                    l2 = float(np.linalg.norm(diff))
                                    norm_ref = float(np.linalg.norm(sig_jl))
                                    rel_l2 = l2 / norm_ref if norm_ref > 1e-15 else 0.0
                                    
                                    # Loose tolerance for high degree signatures on float64
                                    status = "OK" if (rel_l2 < 1e-7 or max_abs < 1e-12) else "FAIL"
                                    print(f"  iisig:    len={len(sig_jl)}, diff={max_abs:.1e}, rel={rel_l2:.1e} -> {status}")

                                writer.writerow({
                                    "N": N, "d": d, "m": m,
                                    "path_kind": path_kind,
                                    "operation": op,
                                    "python_library": "iisignature",
                                    "len_sig": len(sig_jl),
                                    "max_abs_diff": max_abs,
                                    "l2_diff": l2,
                                    "rel_l2_diff": rel_l2,
                                    "status": status,
                                })

                        # 3. Compare against pysiglib
                        if HAS_PYSIGLIB:
                            sig_pysig = pysiglib_compute(path, m, op)
                            if sig_pysig is not None:
                                if sig_pysig.shape != sig_jl.shape:
                                    print(f"  pysiglib: SIZE MISMATCH jl={sig_jl.shape} py={sig_pysig.shape}")
                                    status = "shape_mismatch"
                                    max_abs = l2 = rel_l2 = -1.0
                                else:
                                    diff = sig_jl - sig_pysig
                                    max_abs = float(np.max(np.abs(diff)))
                                    l2 = float(np.linalg.norm(diff))
                                    norm_ref = float(np.linalg.norm(sig_jl))
                                    rel_l2 = l2 / norm_ref if norm_ref > 1e-15 else 0.0
                                    
                                    status = "OK" if rel_l2 < 1e-8 else "FAIL"
                                    print(f"  pysiglib: len={len(sig_jl)}, diff={max_abs:.1e}, rel={rel_l2:.1e} -> {status}")

                                writer.writerow({
                                    "N": N, "d": d, "m": m,
                                    "path_kind": path_kind,
                                    "operation": op,
                                    "python_library": "pysiglib",
                                    "len_sig": len(sig_jl),
                                    "max_abs_diff": max_abs,
                                    "l2_diff": l2,
                                    "rel_l2_diff": rel_l2,
                                    "status": status,
                                })

    print(f"\nSignature comparison CSV written to: {out_csv}")
    return out_csv

if __name__ == "__main__":
    compare_signatures()