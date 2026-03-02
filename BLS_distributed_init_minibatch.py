#!/usr/bin/env python3
"""
final.py — Blockwise MiniBatch k-means on KDDCup'99 with Dask

Implements a scalable k-means pipeline. The core ideas are:

1) Distributed data (Dask Array):
   - CSV is read with `dask.dataframe.read_csv(...)` using a configurable `blocksize`.
   - Categorical columns are one-hot encoded.
   - Numeric columns are cast to float32 and optionally standardized.
   - The result is converted to a 2D `dask.array.Array` and rechunked so that:
       * rows are split into manageable chunks,
       * all features for a row are contiguous.

2) Initialization:
   - `init="kmeans||"` uses the scalable k-means|| seeding algorithm
     (Bahmani et al., "Scalable K-means++", VLDB 2012).
   - k-means|| samples candidate centers in a few distributed rounds, then
     reclusters the smaller candidate set locally with weights.
   - `init="random"` is available as a baseline.

3) MiniBatch updates, sampled "blockwise":
   - Naively sampling random indices from a huge Dask array can be slow and can
     introduce bias if chunks have different sizes or if you repeatedly hit the same
     chunks due to access patterns.
   - We implemented first a standard minibatch to validate our results and noticed that
     the blockwise implementation was 20% faster, so we sticked to it.  
   - blockwise_minibatch_dask:
       * allocates how many rows to draw from each chunk, proportional to its size,
       * samples rows within each chunk, without replacement,
       * concatenates the sampled pieces into a minibatch Dask Array.
     This approximates uniform sampling over the full dataset while keeping the
     work local to workers.

4) Technical speedups:
   - Distances are computed using the standard identity:
       ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
   - Most heavy work happens on workers; the driver only aggregates small
     (k x d) sums/counts per iteration.
   - A simple "empty cluster rescue" prevents dead clusters during minibatch training.
   - Logging is JSONL so runs can be diagnosed and plotted after the fact.
"""

import os
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import dask
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client, wait
# --- add near the top with imports ---
from dask_ml.cluster import KMeans as DaskKMeans

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===========================
# Logging + run dir helpers
# ===========================
@dataclass
class RunLogger:
    run_dir: str
    metrics_path: str
    t0: float

    def log(self, event: str, step: int, **fields):
        rec = {
            "ts": time.time(),
            "t_rel_s": time.time() - self.t0,
            "event": event,
            "step": int(step),
            **fields,
        }
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        # line-buffered append; newline-delimited JSONL
        with open(self.metrics_path, "a", encoding="utf-8", buffering=1) as f:
            f.write(json.dumps(rec) + "\n")


def make_run_dir(base_dir: str, params: Dict[str, Any]) -> str:
    """Create a unique run directory under base_dir (absolute path)."""
    base_dir = os.path.abspath(base_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = (
        f'k{params.get("k")}_bs{params.get("batch_size")}_blk{params.get("blocksize")}'
        f'_mc{params.get("ll_max_candidates")}_seed{params.get("seed",0)}'
    )
    run_dir = os.path.join(base_dir, f"run_{stamp}_{tag}")
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    return run_dir


def start_logger(base_dir: str, params: Dict[str, Any]) -> RunLogger:
    run_dir = make_run_dir(base_dir, params)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, sort_keys=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    return RunLogger(run_dir=run_dir, metrics_path=metrics_path, t0=time.time())


# ============================================================
# Plot helpers
# ============================================================
def save_line_plot(values, path, title, xlabel, ylabel):
    plt.figure()
    plt.plot(np.arange(1, len(values) + 1), values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def save_bar_plot(keys, vals, path, title, xlabel, ylabel):
    plt.figure()
    plt.bar([str(k) for k in keys], vals)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def load_metrics_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_run_diagnostics(run_dir: str):
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    out_dir = os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    rows = load_metrics_jsonl(metrics_path)
    if not rows:
        return

    # --- kmeans|| rounds ---
    km = [r for r in rows if r.get("event") == "kmeansll_round"]
    if km:
        rounds = [r["step"] for r in km]
        phi = [r.get("phi", np.nan) for r in km]
        expected = [r.get("expected_samples", np.nan) for r in km]
        totalC = [r.get("total_candidates", np.nan) for r in km]

        plt.figure()
        plt.plot(rounds, phi)
        plt.xlabel("round")
        plt.ylabel("phi")
        plt.title("k-means||: phi by round")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "kmeansll_phi.png"), dpi=160)
        plt.close()

        plt.figure()
        plt.plot(rounds, expected)
        plt.xlabel("round")
        plt.ylabel("expected samples")
        plt.title("k-means||: expected samples by round")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "kmeansll_expected_samples.png"), dpi=160)
        plt.close()

        plt.figure()
        plt.plot(rounds, totalC)
        plt.xlabel("round")
        plt.ylabel("total candidates |C|")
        plt.title("k-means||: total candidates by round")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "kmeansll_total_candidates.png"), dpi=160)
        plt.close()

    # --- minibatch iters ---
    mb = [r for r in rows if r.get("event") == "minibatch_iter" and r.get("inertia_batch") is not None]
    if mb:
        iters = [r["step"] for r in mb]
        inertia = [r["inertia_batch"] for r in mb]
        it_time = [r.get("time_s", np.nan) for r in mb]

        plt.figure()
        plt.plot(iters, inertia)
        plt.xlabel("iter")
        plt.ylabel("batch inertia")
        plt.title("MiniBatch: inertia (batch)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "minibatch_inertia.png"), dpi=160)
        plt.close()

        plt.figure()
        plt.plot(iters, it_time)
        plt.xlabel("iter")
        plt.ylabel("iter time (s)")
        plt.title("MiniBatch: iteration time")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "minibatch_iter_time.png"), dpi=160)
        plt.close()


# ============================================================
# Distance helpers
# ============================================================
# We compute squared Euclidean distances in a vectorized way.
#
# Key identity (avoids explicit broadcasting over d in Python):
#   ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
#
# - `pairwise_sq_dists(...)` operates on a *NumPy block* (already in worker memory).
# - `pairwise_sq_dists_dask(...)` builds a Dask graph that keeps computations distributed.
#
# We keep float32 for speed / memory.
def pairwise_sq_dists(X_block, C_np, c2=None):
    """Compute squared distances from rows in X_block to centers C_np."""
    Xb = np.asarray(X_block, dtype=np.float32, order="C")
    Cn = np.asarray(C_np, dtype=np.float32, order="C")

    x2 = np.sum(Xb * Xb, axis=1, keepdims=True, dtype=np.float32)  # (n,1)
    if c2 is None:
        c2v = np.sum(Cn * Cn, axis=1, dtype=np.float32)  # (k,)
    else:
        c2v = np.asarray(c2, dtype=np.float32).reshape(-1)
    return x2 + c2v.reshape((1, -1)) - 2.0 * (Xb @ Cn.T)


def pairwise_sq_dists_dask(X: da.Array, C_np: np.ndarray, c2=None) -> da.Array:
    """Dask version of pairwise_sq_dists (no materialization on driver)."""
    Xf = X.astype(np.float32, copy=False)
    x2 = da.sum(Xf * Xf, axis=1, dtype=np.float32).reshape((-1, 1))  # (n,1)
    if c2 is None:
        c2v = np.sum(np.asarray(C_np, dtype=np.float32) ** 2, axis=1, dtype=np.float32)  # (k,)
    else:
        c2v = np.asarray(c2, dtype=np.float32).reshape(-1)
    xc = Xf.dot(np.asarray(C_np, dtype=np.float32).T)  # (n,k) dask
    return x2 + c2v[None, :] - 2.0 * xc


# =============================
# Blockwise minibatch sampling 
# =============================
# Goal: draw a minibatch of `batch_size` rows that is approximately uniform over
# the entire dataset, while keeping I/O local and avoiding driver-side materialization.
#
# Our strategy:
#  1) Compute the row-count of each chunk along axis=0: `Xp.chunks[0]`.
#  2) Allocate an integer number of samples per chunk proportional to its size.
#     We use a multinomial draw and then fix any leftovers without exceeding capacity.
#  3) For each chunk, sample `s` rows *within that chunk* (no replacement).
#     This happens on workers via `dask.delayed`, so blocks never come back to the driver.
#  4) Concatenate sampled pieces to form the minibatch.
#
# This gives us good enough uniformity for MiniBatch k-means while staying fast.
# The quality of the algorithm has been confronted with sklearn KMeans.
def _allocate_block_samples_exact(total: int, block_sizes: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Compute how many samples to draw from each Dask chunk.

    Parameters
    ----------
    total:
        Total number of rows in the minibatch.
    block_sizes:
        1D array with the number of rows in each chunk along axis=0.
    rng:
        RandomState.

    Returns
    -------
    alloc:
        Integer array, same length as `block_sizes`, where alloc[i] is the number of rows
        to sample from block i.

    Notes
    -----
    - We start with a multinomial draw with probabilities proportional to block sizes.
    - We cap allocations by block capacity (cannot sample more than the block contains).
    - If capping creates "leftover" samples, we redistribute them across blocks that still
      have remaining capacity.

    This will keep the minibatch close to uniform over the full dataset while respecting the
    physical chunk boundaries in Dask.
    """
    cap = np.asarray(block_sizes, dtype=np.int64).copy()
    cap[cap < 0] = 0
    if total <= 0 or cap.sum() <= 0:
        return np.zeros_like(cap)

    p = cap / cap.sum()
    alloc = rng.multinomial(int(total), p)
    alloc = np.minimum(alloc, cap)

    leftover = int(total - alloc.sum())
    if leftover <= 0:
        return alloc

    rem = cap - alloc
    while leftover > 0 and rem.sum() > 0:
        # As before, distribute over remaining blocks
        p2 = rem / rem.sum()
        add = rng.multinomial(leftover, p2)
        add = np.minimum(add, rem)
        alloc += add
        leftover = int(total - alloc.sum())
        rem = cap - alloc
        if add.sum() == 0 and leftover > 0:
            idxs = np.flatnonzero(rem > 0)
            take = min(leftover, idxs.size)
            alloc[idxs[:take]] += 1
            leftover = int(total - alloc.sum())
            rem = cap - alloc

    return alloc


def blockwise_minibatch_dask(Xp: da.Array, batch_size: int, seed: int, iteration: int) -> da.Array:
    """
    Build a minibatch as a Dask Array by sampling within each row-chunk.

    The returned array is lazy (Dask graph). It is typically computed later when
    distances / aggregations are executed.

    Implementation details:
    - Sampling is without replacement inside each chunk to avoid duplicates.
    - We seed per (iteration, chunk) so runs are reproducible and sampling differs per step.
    """
    assert Xp.ndim == 2
    n = int(Xp.shape[0])
    d = int(Xp.shape[1])
    m = min(int(batch_size), n)
    if m <= 0:
        return Xp[:0]

    rng = np.random.RandomState(int(seed) + 10_000 * int(iteration + 1))
    block_sizes = np.array([int(c) for c in Xp.chunks[0]], dtype=np.int64)
    alloc = _allocate_block_samples_exact(m, block_sizes, rng)

    def _sample_rows(block: np.ndarray, s: int, seed_local: int) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32, order="C")
        if s <= 0 or block.shape[0] == 0:
            return block[:0]
        rs = np.random.RandomState(int(seed_local))
        idx = rs.choice(block.shape[0], size=int(s), replace=False)
        return block[idx]

    pieces = []
    for bi, s in enumerate(alloc.tolist()):
        if s <= 0:
            continue
        x_del = Xp.blocks[bi, 0].to_delayed().item()
        seed_local = int(seed) + 1_000_000 * int(iteration + 1) + int(bi)
        samp_del = dask.delayed(_sample_rows)(x_del, int(s), seed_local)
        pieces.append(da.from_delayed(samp_del, shape=(int(s), d), dtype=np.float32))

    if not pieces:
        return Xp[:0]

    B = da.concatenate(pieces, axis=0)
    return B.rechunk((min(50_000, m), d))


# ============================================================
# Init: k-means|| (Bahmani et al.) + weighted recluster
# ============================================================
# k-means|| is a parallel-friendly approximation to k-means++ seeding.
#
# High-level algorithm (distributed part):
#   - Start with 1 random center C.
#   - Maintain for each point x its current D(x) = distance^2 to nearest center in C.
#   - For r rounds:
#       * phi = sum_x D(x)
#       * sample each point independently with probability p(x) = min(l * D(x) / phi, 1)
#         (l is an oversampling factor)
#       * add sampled points to candidate set C
#       * update D(x) with distances to new candidates
#
# After r rounds, we have a candidate set C that is much smaller than n.
# We then:
#   - assign each original point to its nearest candidate center,
#   - count how many points map to each candidate => weights w,
#   - run a local weighted k-means on (C, w) to obtain exactly k initial centers.
def local_weighted_kmeans(C: np.ndarray, w: np.ndarray, k: int, max_iters: int, tol: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    m, d = C.shape
    w = w.reshape(-1)

    idx0 = rng.choice(m, p=(w / w.sum())) if w.sum() > 0 else rng.randint(0, m)
    centers = [C[idx0].copy()]
    D2 = np.sum((C - centers[0]) ** 2, axis=1)

    for _ in range(1, k):
        probs = w * D2
        s = probs.sum()
        idx = rng.randint(0, m) if (not np.isfinite(s) or s <= 0) else rng.choice(m, p=(probs / s))
        centers.append(C[idx].copy())
        D2 = np.minimum(D2, np.sum((C - centers[-1]) ** 2, axis=1))

    centers = np.vstack(centers)

    for _ in range(max_iters):
        d2 = ((C[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        lbl = d2.argmin(axis=1)

        new_centers = centers.copy()
        for j in range(k):
            mask = (lbl == j)
            if not np.any(mask):
                continue
            ww = w[mask]
            new_centers[j] = (C[mask] * ww[:, None]).sum(axis=0) / max(ww.sum(), 1.0)

        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < tol:
            break
    return centers


def kmeans_init_parallel_ll_optimized(
    X: da.Array,
    k: int,
    l: Optional[float] = None,
    r: int = 5,
    max_candidates: int = 20000,
    seed: int = 0,
    logger: Optional[RunLogger] = None,
    run_dir: Optional[str] = None,
    i = 1,
    outdir = 'kdd_results_different_init',
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """k-means|| initialization for Dask (Bahmani et al.). Physics unchanged."""
    assert X.ndim == 2
    n, d = X.shape
    rng = np.random.RandomState(seed)
    n_int = int(n)

    if l is None:
        l = 2 * k

    t_tot = time.time()

    idx0 = int(rng.randint(0, n_int))
    C = X[idx0:idx0 + 1].compute().astype(np.float32, copy=False)
    if logger is not None:
        logger.log("kmeansll_init", 0, idx0=int(idx0), d=int(C.shape[1]), k=int(k))

    Xp = X.rechunk({1: -1}).persist()

    def _min_d2_from_C_block(X_block, C_np):
        d2 = pairwise_sq_dists(X_block, C_np)
        return np.min(d2, axis=1)

    min_d2 = da.map_blocks(_min_d2_from_C_block, Xp, C, dtype=np.float32, drop_axis=1)

    def _unwrap(x, name="block"):
        if isinstance(x, tuple):
            if len(x) == 1:
                return x[0]
            arrs = [it for it in x if isinstance(it, np.ndarray) and it.ndim >= 1]
            if len(arrs) == 1:
                return arrs[0]
            if len(arrs) >= 2:
                if all(a.ndim == 2 for a in arrs) and all(a.shape[0] == arrs[0].shape[0] for a in arrs):
                    return np.concatenate(arrs, axis=1)
                return np.concatenate(arrs, axis=0)
            raise TypeError(f"{name}: tuple with no usable ndarray parts: {[type(i).__name__ for i in x]}")
        return x

    def _sample_block(X_block, probs_block, seed_block):
        X_block = _unwrap(X_block, "X_block")
        probs_block = _unwrap(probs_block, "probs_block")
        probs_block = np.asarray(probs_block, dtype=np.float32).reshape(-1)

        rs = np.random.RandomState(int(seed_block))
        u = rs.rand(X_block.shape[0])
        mask = u < probs_block
        return np.asarray(X_block, dtype=np.float32)[mask]

    total_candidates = int(C.shape[0])
    prev_phi = None
    phi_history = []  # optional, for debugging/logging

    for t in range(int(r)):

        phi = float(min_d2.sum().compute())
        if phi <= 0:
            phi = 1.0
            
        # ---- early stopping on relative change in phi ----
        if prev_phi is not None:
            rel_change = abs(phi - prev_phi) / max(abs(prev_phi), 1e-12)
            if logger is not None:
                logger.log("kmeansll_earlystop_check", t + 1, phi=float(phi), prev_phi=float(prev_phi), rel_change=float(rel_change))
            if verbose:
                print(f"[kmeans||] rel_change(phi)={rel_change:.3e}")
            if rel_change < 1e-4:
                if logger is not None:
                    logger.log("kmeansll_earlystop", t + 1, phi=float(phi), prev_phi=float(prev_phi), rel_change=float(rel_change))
                if verbose:
                    print(f"[kmeans||] early stop at round {t+1}: rel_change(phi)={rel_change:.3e} < 1e-3")
                    np.savetxt(f'{outdir}/init_time_l_{l}_k_{k}_rounds_{r}_{i}.txt', np.array([time.time() - t_tot]))
                    np.savetxt(f'{outdir}/init_phi_l_{l}_k_{k}_rounds_{r}_{i}.txt', np.array([phi_history]))

                break

        prev_phi = phi
        phi_history.append(phi)
        probs = da.minimum((l * min_d2) / phi, 1.0)

        # Debug: expected sample count = sum(p)
        expected = float(probs.sum().compute())

        if logger is not None:
            logger.log("kmeansll_round", t + 1, phi=float(phi), expected_samples=float(expected), total_candidates=int(total_candidates))

        if verbose:
            print(f"[kmeans||] round {t+1}/{r}: phi={phi:.6g}, E[samples]={expected:.2f}, current|C|={total_candidates}")

        block_seeds = [seed + 1000 * (t + 1) + bi for bi in range(probs.numblocks[0])]
        X_blocks = [Xp.blocks[i, 0].to_delayed().item() for i in range(Xp.numblocks[0])]
        P_blocks = [probs.blocks[i].to_delayed().item() for i in range(probs.numblocks[0])]

        sampled_lists = [dask.delayed(_sample_block)(x_del, p_del, block_seeds[bi])
                         for bi, (x_del, p_del) in enumerate(zip(X_blocks, P_blocks))]

        sampled_blocks = dask.compute(*sampled_lists)
        nonempty = [b for b in sampled_blocks if b is not None and len(b) > 0]
        if not nonempty:
            continue

        C_new = np.vstack(nonempty).astype(np.float32, copy=False)

        remaining = max(0, int(max_candidates) - int(total_candidates))
        if remaining <= 0:
            break
        if C_new.shape[0] > remaining:
            take = rng.choice(C_new.shape[0], size=remaining, replace=False)
            C_new = C_new[take]

        C = np.vstack([C, C_new]).astype(np.float32, copy=False)
        total_candidates = int(C.shape[0])

        min_d2_new = da.map_blocks(_min_d2_from_C_block, Xp, C_new, dtype=np.float32, drop_axis=1)
        min_d2 = da.minimum(min_d2, min_d2_new)

        if total_candidates >= int(max_candidates):
            break

    def _labels_block(X_block, C_np):
        d2 = pairwise_sq_dists(X_block, C_np)
        return np.argmin(d2, axis=1).astype(np.int64)

    labels = da.map_blocks(_labels_block, Xp, C, dtype=np.int64, drop_axis=1)

    def _bincount_block(lbl_block, K):
        return np.bincount(np.asarray(lbl_block, dtype=np.int64), minlength=K).astype(np.int64)

    Kcand = int(C.shape[0])
    counts_blocks = [dask.delayed(_bincount_block)(lb, Kcand) for lb in labels.to_delayed().flatten()]
    counts = np.sum(dask.compute(*counts_blocks), axis=0).astype(np.int64, copy=False)

    if logger is not None:
        logger.log("kmeansll_weighting", 0, candidates=int(Kcand), total_weight=int(counts.sum()))
    if verbose:
        print(f"[kmeans||] weighting done: candidates={Kcand}, total_weight={counts.sum()}")

    centers = local_weighted_kmeans(C, counts, k=k, max_iters=30, tol=1e-6, seed=seed).astype(np.float32, copy=False)

    if logger is not None:
        logger.log("kmeansll_recluster", 0, time_s=float(time.time() - t_tot), centers_k=int(k))

    np.savetxt(f'{outdir}/init_time_l_{l}_k_{k}_rounds_{r}_{i}.txt', np.array([time.time() - t_tot]))
    np.savetxt(f'{outdir}/init_phi_l_{l}_k_{k}_rounds_{r}_{i}.txt', np.array([phi_history]))

    if run_dir:
        try:
            os.makedirs(run_dir, exist_ok=True)
            np.savetxt(os.path.join(run_dir, "kmeansll_centers.txt"), centers)
            np.savetxt(os.path.join(run_dir, "kmeansll_candidates.txt"), C)
            np.savetxt(os.path.join(run_dir, "kmeansll_weights.txt"), counts)
            np.savetxt(os.path.join(run_dir, "init_time.txt"), np.array([time.time() - t_tot]))
        except Exception:
            pass

    return centers, C, counts


# ============================================================
# MiniBatch k-means (uses blockwise_minibatch_dask)
# ============================================================
# This implements the standard MiniBatch k-means update:
#   C_j <- (1 - eta) * C_j + eta * mean(B_j)
# where B_j are minibatch points assigned to cluster j, and eta is a learning rate.
#
# We choose eta using a cumulative count per cluster:
#   eta = n_j / (N_j + n_j)
# where n_j is the number of minibatch points assigned to cluster j this iteration,
# and N_j is the total number of points ever assigned to j so far.
# This makes eta naturally decrease over time, avoids bias over clusters.
#
# Implementation detail:
# - We compute labels and (sum, count) per cluster on workers for each minibatch block.
# - The driver only aggregates k x d sums and k counts.
def minibatch_kmeans_dask(
    X: da.Array,
    k: int,
    batch_size: int = 2048,
    max_iters: int = 100,
    tol: float = 1e-4,
    init: str = "kmeans||",
    ll_l: Optional[float] = None,
    ll_rounds: int = 5,
    ll_max_candidates: Optional[int] = 5000,
    seed: int = 0,
    logger: Optional[RunLogger] = None,
    i = 1,
    outdir= 'kdd_results',
    verbose: bool = True,
    init_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    assert X.ndim == 2
    n, d = X.shape

    # To avoid clusters never being assigned points
    reassign_empty_every = 5          # iterations

    rng = np.random.RandomState(seed)
    Xp = X.astype(np.float32, copy=False).rechunk({1: -1}).persist()

    init_time = time.time()
    if init == "kmeans||":
        if verbose:
            print(f"[minibatch] init with k-means|| (k={k}, l={ll_l}, rounds={ll_rounds}, max_candidates={ll_max_candidates})")
        C, _, _ = kmeans_init_parallel_ll_optimized(
            Xp,
            k=k,
            l=ll_l,
            r=ll_rounds,
            max_candidates=ll_max_candidates if ll_max_candidates is not None else 20000,
            seed=seed,
            logger=logger,
            outdir = outdir,
            i = i,
            run_dir=(logger.run_dir if logger is not None else None),
            verbose=verbose,
        )
    else:
        if verbose:
            print(f"[minibatch] init with random sample (k={k})")
        n_int = int(n)
        idx0 = rng.choice(n_int, size=min(k, n_int), replace=False)
        C = Xp[idx0].compute().astype(np.float32, copy=False)
        
    print(f'Init time: {time.time() - init_time}')
    time_in = time.time() - init_time
    np.savetxt(f'{outdir}/init_time_l_{ll_l}_k_{k}_rounds_{ll_rounds}.txt', np.array([time_in]))
    #return None, None, None, None # uncomment for init runs
        
    # Used for analysis and debugging    
    def _full_inertia(C_np):
        c2_full = np.sum(np.asarray(C_np, dtype=np.float32) ** 2, axis=1, dtype=np.float32)
        d2_full = pairwise_sq_dists_dask(Xp, C_np, c2=c2_full)
        return da.min(d2_full, axis=1).sum()    

    half_iter = max(1, int(max_iters) // 2)
    full_inertia_targets = {half_iter, int(max_iters)}
    computed_full_inertia_iters = set()

    C = np.asarray(C, dtype=np.float32, order="C")
    if verbose:
        print(f"[minibatch] init done: C.shape={C.shape}")

    counts_global = np.zeros(k, dtype=np.int64)

    def _agg_block(B_block, lbl_block, k_):
        Bb = np.asarray(B_block, dtype=np.float32)
        lb = np.asarray(lbl_block, dtype=np.int64)
        if Bb.size == 0:
            d_ = int(Bb.shape[1]) if Bb.ndim == 2 else 0
            return np.zeros((k_, d_), dtype=np.float64), np.zeros(k_, dtype=np.int64)

        cnts = np.bincount(lb, minlength=k_).astype(np.int64, copy=False)
        sums = np.zeros((k_, Bb.shape[1]), dtype=np.float64)
        np.add.at(sums, lb, Bb.astype(np.float64, copy=False))
        return sums, cnts

    shifts: List[float] = []
    inertias_batch: List[float] = []
    inertias_full = []  # (iter, inertia_full) pairs
    count_history: List[np.ndarray] = []

    # --- Epoch Tracking Variables ---
    epoch_inertia_accum = 0.0  # Sum of batch inertias in current epoch
    epoch_batch_count = 0      # Number of batches in current epoch
    prev_epoch_avg = None      # Average inertia of the PREVIOUS epoch

    for it in range(int(max_iters)):
        it_t0 = time.time()
        old = C.copy()

        B = blockwise_minibatch_dask(Xp, batch_size=batch_size, seed=seed, iteration=it)
        m = int(B.shape[0])

        c2 = np.sum(C * C, axis=1, dtype=np.float32)
        d2 = pairwise_sq_dists_dask(B, C, c2=c2)
        lbl = da.argmin(d2, axis=1).astype(np.int64)

        B_del = B.to_delayed().ravel()
        L_del = lbl.to_delayed().ravel()
        aggs = [dask.delayed(_agg_block)(bd, ld, k) for bd, ld in zip(B_del, L_del)]
        agg_res = dask.compute(*aggs)

        sums_total = np.zeros((k, int(d)), dtype=np.float64)
        counts = np.zeros(k, dtype=np.int64)
        for s, c in agg_res:
            if s.size:
                sums_total += s
            counts += c
        
        # --- Center update (MiniBatch k-means) ---
        # We update each center using the mean of points assigned to it in the minibatch.
        # The learning rate eta is derived from `counts_global` so it decays as the cluster
        # receives more assignments.
        for j in range(k):
            nj = int(counts[j])
            if nj == 0:
                continue

            prev = counts_global[j]
            counts_global[j] = prev + nj
            eta = 1.0 if prev == 0 else (nj / counts_global[j])
            C[j] = ((1.0 - eta) * C[j] + eta * (sums_total[j] / nj)).astype(np.float32, copy=False)   

        count_history.append(counts_global.copy())

        shift = float(np.linalg.norm(C - old))
        shifts.append(shift)

        c2_post = np.sum(C * C, axis=1, dtype=np.float32)
        inertia_b = float(da.min(pairwise_sq_dists_dask(B, C, c2=c2_post), axis=1).sum().compute())
        inertias_batch.append(inertia_b)

        # --- Accumulate for Epoch Check ---
        epoch_inertia_accum += inertia_b
        epoch_batch_count += 1
        # ----------------------------------

        inertia_f = None
        if (it + 1) in full_inertia_targets:
            inertia_f = float(_full_inertia(C).compute())
            inertias_full.append((it + 1, inertia_f))
            computed_full_inertia_iters.add(it + 1)
            print(f"[minibatch] full inertia at iter {it+1}: inertia_full={inertia_f:.4g}")

        it_time = time.time() - it_t0

        if verbose and ((it + 1) % max(1, max_iters // 10) == 0 or it == 0):
            print(
                f"[minibatch] iter {it+1}/{max_iters}: batch={m}, inertia(batch)={inertia_b:.4g}, "
                f"counts_global_min={counts_global.min()}, shift={shift:.4g}, time={it_time:.3f}s"
            )

        if logger is not None:
            if (it + 1) in full_inertia_targets:
                logger.log("minibatch_iter", 
                           it + 1, batch=int(m), 
                           inertia_batch=float(inertia_b), 
                           inertia_full=float(inertia_f), 
                           shift=float(shift), 
                           counts_global_min=int(counts_global.min()), 
                           time_s=float(it_time))
            else:
                logger.log(
                    "minibatch_iter",
                    it + 1,
                    batch=int(m),
                    inertia_batch=float(inertia_b),
                    shift=float(shift),
                    counts_global_min=int(counts_global.min()),
                    time_s=float(it_time),
                )
            # --- FULL-INERTIA CONVERGENCE CHECK ---
            full_check_every = 1  

            if (it + 1) % full_check_every == 0:
                inertia_f = float(_full_inertia(C).compute())
                inertias_full.append((it + 1, inertia_f))

                if verbose:
                    print(f"[minibatch] full inertia check @ iter {it+1}: inertia_full={inertia_f:.6g}")

                if logger is not None:
                    logger.log(
                        "minibatch_full_inertia",
                        it + 1,
                        inertia_full=float(inertia_f),
                        counts_global_min=int(counts_global.min()),
                        shift=float(shift),
                        time_s=float(it_time),
                    )

                # Compare to previous full inertia (relative improvement)
                if len(inertias_full) >= 2:
                    prev_f = inertias_full[-2][1]
                    # Guard against divide-by-zero / tiny prev values
                    denom = prev_f if prev_f != 0.0 else 1.0
                    rel_drop_full = (prev_f - inertia_f) / denom

                    if verbose:
                        print(f"[minibatch] rel_drop_full={rel_drop_full:.6g} (tol={tol})")

                    if (abs(rel_drop_full) < float(tol)) & (inertia_f < 10): # for debugging and analysis
                        print(
                            f"[minibatch] converged at iter {it+1}: "
                            f"full-inertia drop={rel_drop_full:.6g} < tol={tol}"
                        )

                        if outdir is not None:
                            np.savetxt(f"{outdir}/minibatch_final_inertia_{init}_k_{k}_{i}.txt", np.array([inertia_f], dtype=np.float64))

                        # If you want a final log event
                        if logger is not None:
                            logger.log("minibatch_converged", it + 1, inertia_full=float(inertia_f))

                        break

    return C, np.array(shifts, dtype=np.float64), np.array(inertias_batch, dtype=np.float64), count_history


# ================================
# KDDCup loader + preprocessing
# ================================
KDD_COLS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
    "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
    "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label"
]
KDD_CATEGORICAL = ["protocol_type", "service", "flag"]
KDD_LABEL = "label"


def load_kddcup_as_dask_array(
    path: str,
    blocksize: str,
    one_hot: bool = True,
    standardize: bool = True,
    drop_label: bool = True,
):
    """
    Docstring for load_kddcup_as_dask_array
    
    :param path: Path to datafile
    :type path: str
    :param blocksize: Blocksize for Dask CSV reading (e.g., "64MB", "256MB")
    :type blocksize: str
    :param one_hot: Whether to one-hot encode categorical columns
    :type one_hot: bool
    :param standardize: Whether to standardize features (mean=0, std=1)
    :type standardize: bool
    :param drop_label: Whether to drop the label column (if present)
    :type drop_label: bool

    Loads the KDDCup dataset from a CSV file into a Dask Array, with optional preprocessing steps.
    """
    print(f"[data] reading: {path}")
    ddf = dd.read_csv(
        path,
        names=KDD_COLS,
        header=None,
        compression="infer",
        blocksize=blocksize,
        assume_missing=True,
        on_bad_lines="skip",
        engine="python",
    )
    print(f"[data] read done: {len(ddf)} rows, {len(ddf.columns)} columns")

    if drop_label and KDD_LABEL in ddf.columns:
        ddf = ddf.drop(columns=[KDD_LABEL])

    cat_cols = [c for c in KDD_CATEGORICAL if c in ddf.columns]
    for c in cat_cols:
        ddf[c] = ddf[c].astype("category")
    if cat_cols:
        ddf = ddf.categorize(columns=cat_cols)

    for c in ddf.columns:
        if c in cat_cols:
            continue
        ddf[c] = dd.to_numeric(ddf[c], errors="coerce").astype("float32")

    num_cols = [c for c in ddf.columns if c not in cat_cols]
    ddf[num_cols] = ddf[num_cols].fillna(0)

    if one_hot and cat_cols:
        ddf = dd.get_dummies(ddf, columns=cat_cols, sparse=False)

    feature_names = list(ddf.columns)
    X = ddf.to_dask_array(lengths=True).astype(np.float32)

    if standardize:
        print("[data] computing mean/std for standardization (distributed)")
        mu = X.mean(axis=0)
        var = X.var(axis=0)
        mu_v, var_v = dask.compute(mu, var)
        sigma_v = np.sqrt(var_v)
        sigma_v[sigma_v == 0] = 1.0
        X = (X - mu_v) / sigma_v

    return X, feature_names


# ============================================================
# sklearn reference (sample)
# ============================================================
def sklearn_reference_kmeans(X: da.Array, k: int, seed: int, sample_n: int, max_iter: int):
    from sklearn.cluster import KMeans
    n = int(X.shape[0])
    if sample_n <= 0 or sample_n >= n:
        sample_n = min(n, 200_000)

    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=sample_n, replace=False)
    Xs = X[idx].compute()

    print(f"[sklearn] running KMeans on sample_n={sample_n} with k={k}")
    t0 = time.time()
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto", max_iter=max_iter)
    km.fit(Xs)
    dt = time.time() - t0
    print(f"[sklearn] inertia={km.inertia_:.6g} iters={km.n_iter_} dt={dt:.2f}s")
    return km.cluster_centers_, km.inertia_, km.n_iter_, dt

def full_inertia_distributed(X: da.Array, centers: np.ndarray) -> float:
    """
    Compute full-dataset inertia (sum of squared distances to nearest center),
    using existing distributed squared-distance routine.
    """
    centers = np.asarray(centers, dtype=np.float32)
    c2 = np.sum(centers * centers, axis=1, dtype=np.float32)
    d2 = pairwise_sq_dists_dask(X, centers, c2=c2)   # (n, k)
    return float(da.min(d2, axis=1).sum().compute())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheduler", default=os.getenv("DASK_SCHEDULER_ADDRESS", "tcp://dask-scheduler:8786"))
    ap.add_argument("--data-path", required=True, help="Path to kddcup .data / .csv / .gz")
    ap.add_argument("--blocksize", default="256MB", help="Dask CSV blocksize")
    ap.add_argument("--chunk-rows", type=int, default=200_000, help="Row chunk size for Dask Array rechunking")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=200_000)
    ap.add_argument("--init", choices=["kmeans||", "random"], default="kmeans||")
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--ll-l", type=float, default=None)
    ap.add_argument("--ll-rounds", type=int, default=5)
    ap.add_argument("--ll-max-candidates", type=int, default=5000)

    ap.add_argument("--no-onehot", action="store_true", help="Disable one-hot encoding")
    ap.add_argument("--no-standardize", action="store_true", help="Disable standardization")

    ap.add_argument("--outdir", default="kdd_kmeans_out")
    ap.add_argument("--sklearn-sample", type=int, default=200_000, help="Sample size for sklearn reference (0 disables)")
    ap.add_argument("--sklearn-max-iter", type=int, default=300)

    
    ap.add_argument("--parquet-path", default=None, help="Path to save/load preprocessed parquet data")
    ap.add_argument("--preprocess-only", action="store_true", help="If set, only process CSV->Parquet and exit")

    ap.add_argument("--init-only", action="store_true", help="Stop after initialization (skip training loop)")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[dask] connecting to {args.scheduler}")
    client = Client(args.scheduler, set_as_default=True)
    info = client.scheduler_info()
    print("[dask] workers:", len(info.get("workers", {})))
    print("[dask] dashboard:", client.dashboard_link)
    
    workers = info.get("workers", {})
    if workers:
        # Get memory limit of the first worker to verify config
        first_worker = next(iter(workers.values()))
        mem_limit = first_worker.get("memory_limit", 0)
        print(f"[dask] workers: {len(workers)}, memory/worker: {mem_limit / (1024**3):.2f} GB")

    dask.config.set({"array.slicing.split_large_chunks": True})

    t_read0 = time.time()

    mode = "preprocess" if args.preprocess_only else "run"

    X, feature_names = load_kddcup_as_dask_array(
        args.data_path,
        blocksize=args.blocksize,
        one_hot=(not args.no_onehot),
        standardize=(not args.no_standardize),
        drop_label=True,
    )
    if args.preprocess_only:
        print(f"[main] Preprocessing finished in {time.time() - t_read0:.2f}s. Exiting.")
        return
        
    print(f"[data] read in time={time.time() - t_read0:.2f}s")

    n_features = int(X.shape[1])
    X = X.rechunk((args.chunk_rows, n_features)).persist()
    wait(X)
    print(f"[data] X ready: shape={X.shape} chunks={X.chunks} n_features={n_features}")

    ll_max_candidates = None if args.ll_max_candidates == 0 else args.ll_max_candidates

    params = vars(args)
    logger = start_logger(str(outdir), params)

    print(f"[run] minibatch_kmeans_dask k={args.k} max_iters={args.max_iters}")
    t0 = time.time()
    ll_l = args.ll_l * args.k

    for k in [20, 50, 100, 150]:
        args.k = k
        for i in range(1):
            for ll_l in [1]:
                ll_l_run = ll_l * args.k                
                for init in ["kmeans||", "random"]:
                    print(f"\n=== Running minibatch_kmeans_dask with init={init} ===")
                    t0 = time.time()
                    centers, shifts, inertias, count_hist = minibatch_kmeans_dask(
                        X,
                        k=args.k,
                        batch_size=args.batch_size,
                        max_iters=args.max_iters,
                        tol=args.tol,
                        seed=args.seed,
                        init=init,
                        ll_l=ll_l_run,
                        ll_rounds=args.ll_rounds,
                        ll_max_candidates=ll_max_candidates,
                        i = i,
                        logger=logger,
                        outdir=outdir,
                        verbose=True,
                    )

                dt = time.time() - t0
                print(f"[done] total time: {dt:.3f}s for init={init}")
                np.savetxt(f'{outdir}/total_time_{init}_l_{ll_l}_k_{k}_iters_{args.max_iters}_{i}.txt', [dt])

    #for k in [100]:
    #    args.k = k
    #    for ll_l in [1]:
    #        ll_l_run = ll_l * args.k
    #        print(f"\n=== Running minibatch_kmeans_dask with l={ll_l_run} ===")
    #        t0 = time.time()
    #        centers, shifts, inertias, count_hist = minibatch_kmeans_dask(
    #            X,
    #            k=args.k,
    #            batch_size=args.batch_size,
    #            max_iters=args.max_iters,
    #            tol=args.tol,
    #            seed=args.seed,
    #            init=args.init,
    #            ll_l=ll_l_run,
    #            ll_rounds=args.ll_rounds,
    #            ll_max_candidates=ll_max_candidates,
    #            i = 99,
    #            logger=logger,
    #            outdir=outdir,
    #            verbose=True,
    #        )
#
    #        dt = time.time() - t0
    #        print(f"[done] total time: {dt:.3f}s for l={ll_l_run}")
    #        np.savetxt(f'{outdir}/init_time_l_{ll_l_run}_k_{k}_{i}.txt', [dt])

    if centers is None: return # for initialization run

    try:
        plot_run_diagnostics(logger.run_dir)
    except Exception as e:
        print("[diagnostics] plot_run_diagnostics failed:", repr(e))

    np.save(outdir / f"centers_k{args.k}.npy", centers)
    np.save(outdir / "shifts.npy", shifts)
    np.save(outdir / "inertias.npy", inertias)
    np.save(outdir / "count_hist.npy", np.array(count_hist, dtype=object), allow_pickle=True)

    if len(shifts) > 0:
        save_line_plot(shifts, outdir / "centroid_shift.png",
                       "Centroid shift per iteration", "iteration", "L2 shift")
    if len(inertias) > 0:
        save_line_plot(inertias, outdir / "inertia.png",
                       "Inertia (SSE) per iteration", "iteration", "SSE")

    if len(count_hist) > 0:
        final_counts = count_hist[-1]
        save_bar_plot(list(range(args.k)), final_counts, outdir / "cluster_counts_final.png",
                      "Final cluster counts", "cluster", "count")
        

if __name__ == "__main__":
    main()
