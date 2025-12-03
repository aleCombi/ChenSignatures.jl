"""
Compare ChenSignatures against pySigLib for correctness and performance.

IMPORTANT: Different array layout conventions:
- ChenSignatures: (N, d, B) where N=length, d=dimension, B=batch_size
- pySigLib: (B, N, d) where B=batch_size, N=length, d=dimension
"""

import time
import numpy as np
import chen

try:
    from pysiglib import signature as pysiglib_signature
    PYSIGLIB_AVAILABLE = True
except ImportError:
    PYSIGLIB_AVAILABLE = False
    print("WARNING: pySigLib not available, skipping comparison")


def generate_paths_chen(B: int, L: int, d: int, dtype=np.float64) -> np.ndarray:
    """Generate paths in ChenSignatures format: (L, d, B)"""
    return np.random.randn(L, d, B).astype(dtype)


def chen_to_pysiglib(paths: np.ndarray) -> np.ndarray:
    """
    Convert ChenSignatures format (L, d, B) to pySigLib format (B, L, d).

    Args:
        paths: Array of shape (L, d, B)

    Returns:
        Array of shape (B, L, d)
    """
    # (L, d, B) -> (B, L, d)
    return np.transpose(paths, (2, 0, 1))


def pysiglib_to_chen(result: np.ndarray) -> np.ndarray:
    """
    Convert pySigLib result format (B, sig_len) to ChenSignatures format (sig_len, B).

    Args:
        result: Array of shape (B, sig_len)

    Returns:
        Array of shape (sig_len, B)
    """
    return result.T


def benchmark_and_compare(B: int, L: int, d: int, N: int, n_trials: int = 10):
    """
    Benchmark both implementations and check for correctness.

    Args:
        B: Batch size
        L: Path length
        d: Dimension
        N: Truncation level
        n_trials: Number of benchmark trials
    """
    print(f"\n{'='*80}")
    print(f"Test case: B={B}, L={L}, d={d}, N={N}")
    print(f"{'='*80}")

    # Generate test data in ChenSignatures format (use float32 for pySigLib compatibility)
    np.random.seed(42)
    paths_chen = generate_paths_chen(B, L, d, dtype=np.float32)

    # Convert to pySigLib format
    paths_pysig = chen_to_pysiglib(paths_chen).astype(np.float32)

    print(f"\nArray shapes:")
    print(f"  ChenSignatures input: {paths_chen.shape} (L, d, B)")
    print(f"  pySigLib input:       {paths_pysig.shape} (B, L, d)")

    # =========================================================================
    # Test ChenSignatures
    # =========================================================================
    print(f"\n{'-'*80}")
    print("ChenSignatures (threaded):")
    print(f"{'-'*80}")

    # Warmup
    _ = chen.sig(paths_chen, N, threaded=True)

    # Benchmark
    chen_times = []
    for i in range(n_trials):
        t0 = time.perf_counter()
        result_chen = chen.sig(paths_chen, N, threaded=True)
        t1 = time.perf_counter()
        chen_times.append(t1 - t0)

    chen_min = np.min(chen_times)
    chen_mean = np.mean(chen_times)
    chen_std = np.std(chen_times)

    print(f"  Output shape: {result_chen.shape} (sig_len, B)")
    print(f"  Min time:  {chen_min:.4f}s")
    print(f"  Mean time: {chen_mean:.4f}s")
    print(f"  Std time:  {chen_std:.4f}s")

    # =========================================================================
    # Test pySigLib
    # =========================================================================
    if not PYSIGLIB_AVAILABLE:
        print("\npySigLib not available - skipping")
        return

    print(f"\n{'-'*80}")
    print("pySigLib (CPU, parallel with n_jobs=-1):")
    print(f"{'-'*80}")

    # Warmup
    _ = pysiglib_signature(paths_pysig, N, n_jobs=-1)

    # Benchmark
    pysig_times = []
    for i in range(n_trials):
        t0 = time.perf_counter()
        result_pysig = pysiglib_signature(paths_pysig, N, n_jobs=-1)
        t1 = time.perf_counter()
        pysig_times.append(t1 - t0)

    pysig_min = np.min(pysig_times)
    pysig_mean = np.mean(pysig_times)
    pysig_std = np.std(pysig_times)

    print(f"  Output shape: {result_pysig.shape} (B, sig_len)")
    print(f"  Min time:  {pysig_min:.4f}s")
    print(f"  Mean time: {pysig_mean:.4f}s")
    print(f"  Std time:  {pysig_std:.4f}s")

    # =========================================================================
    # Compare results
    # =========================================================================
    print(f"\n{'-'*80}")
    print("Correctness check:")
    print(f"{'-'*80}")

    # Convert pySigLib result to ChenSignatures format for comparison
    result_pysig_transposed = pysiglib_to_chen(result_pysig)

    print(f"  ChenSignatures result shape: {result_chen.shape}")
    print(f"  pySigLib result shape (transposed): {result_pysig_transposed.shape}")

    # pySigLib includes constant term (1) as first element, ChenSignatures doesn't
    # Remove the constant term from pySigLib result
    result_pysig_no_const = result_pysig_transposed[1:, :]

    print(f"  pySigLib without constant: {result_pysig_no_const.shape}")

    # Check if results match
    max_diff = np.max(np.abs(result_chen - result_pysig_no_const))
    rel_diff = max_diff / (np.max(np.abs(result_chen)) + 1e-10)

    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Max relative difference: {rel_diff:.2e}")

    # Use a more lenient tolerance for numerical differences (float32 has ~7 digits precision)
    if np.allclose(result_chen, result_pysig_no_const, rtol=1e-5, atol=1e-2):
        print(f"  [OK] Results match (within tolerance)")
    else:
        print(f"  [FAIL] Results differ significantly!")
        print(f"    Sample ChenSignatures values: {result_chen[:5, 0]}")
        print(f"    Sample pySigLib values:       {result_pysig_no_const[:5, 0]}")

    # =========================================================================
    # Performance comparison
    # =========================================================================
    print(f"\n{'-'*80}")
    print("Performance comparison:")
    print(f"{'-'*80}")

    speedup = pysig_min / chen_min
    print(f"  pySigLib min time:       {pysig_min:.4f}s")
    print(f"  ChenSignatures min time: {chen_min:.4f}s")

    if speedup > 1:
        print(f"  ChenSignatures is {speedup:.2f}x FASTER than pySigLib")
    else:
        print(f"  pySigLib is {1/speedup:.2f}x FASTER than ChenSignatures")


def main():
    """Run comparison benchmarks."""
    print("="*80)
    print("ChenSignatures vs pySigLib Comparison")
    print("="*80)

    if not PYSIGLIB_AVAILABLE:
        print("\nERROR: pySigLib not available")
        print("Install with: pip install pysiglib")
        return

    # Test cases from the reference paper
    test_cases = [
        (128, 256, 4, 6),    # Small case
        (128, 512, 8, 5),    # Medium case
        (128, 1024, 16, 4),  # Large case
    ]

    for B, L, d, N in test_cases:
        benchmark_and_compare(B, L, d, N, n_trials=10)

    print("\n" + "="*80)
    print("Comparison complete!")
    print("="*80)


if __name__ == "__main__":
    main()
