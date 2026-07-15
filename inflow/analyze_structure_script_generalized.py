import argparse
import time
from pathlib import Path

import numpy as np
import torch

import get_st_funcs_generalized as gs


def parse_age_value_map(raw: str | None) -> dict[int, float]:
    if raw is None:
        return {}
    out: dict[int, float] = {}
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        if ":" not in p:
            raise ValueError(f"Invalid age map entry '{p}'. Expected format age:value.")
        age_s, val_s = p.split(":", 1)
        out[int(age_s.strip())] = float(val_s.strip())
    return out


def resolve_value_by_age(ages: list[int], default: float | None, overrides: dict[int, float], name: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for age in ages:
        if age in overrides:
            out[age] = float(overrides[age])
        elif default is not None:
            out[age] = float(default)
        else:
            raise ValueError(f"No {name} specified for age {age}. Provide --{name} or --{name}-by-age.")
    return out


def load_theta(models_dir: Path, tissue: str, age: int, gene_type: str, lam: float, threshold: float) -> np.ndarray:
    path = gs.model_path(models_dir, tissue, age, gene_type, lam)
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    payload = torch.load(path, map_location="cpu")
    theta = payload["theta"]
    if isinstance(theta, torch.Tensor):
        theta = theta.detach().cpu().numpy()
    theta = np.asarray(theta, dtype=np.float64).copy()
    if threshold > 0:
        theta[np.abs(theta) < threshold] = 0
    return theta


def load_names_tf(data_dir: Path, tissue: str, n_tf: int) -> np.ndarray:
    candidates = [
        data_dir / f"final_tf_names_{tissue}.npy",
        Path("data") / f"final_tf_names_{tissue}.npy",
    ]
    for path in candidates:
        if path.exists():
            names = np.load(path, allow_pickle=True)
            if len(names) == n_tf:
                return names
    return np.array([f"TF_{i}" for i in range(n_tf)], dtype=object)


def calc_all_structs(
    t,
    theta_tf_a,
    theta_tf_b,
    theta_tg_a,
    theta_tg_b,
    age_a,
    age_b,
    names_tf,
    out_dir: Path,
    factor=0.0,
    return_timings=False,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    total_t0 = time.perf_counter()

    theta_tf_a = np.asarray(theta_tf_a).copy()
    theta_tf_b = np.asarray(theta_tf_b).copy()
    theta_tg_a = np.asarray(theta_tg_a).copy()
    theta_tg_b = np.asarray(theta_tg_b).copy()

    n_tf, _ = theta_tf_a.shape

    theta_tf_a[np.abs(theta_tf_a) < factor] = 0
    theta_tf_b[np.abs(theta_tf_b) < factor] = 0
    theta_tg_a[np.abs(theta_tg_a) < factor] = 0
    theta_tg_b[np.abs(theta_tg_b) < factor] = 0

    sp_a_tf = np.count_nonzero(theta_tf_a) / theta_tf_a.size
    sp_b_tf = np.count_nonzero(theta_tf_b) / theta_tf_b.size
    sp_a_tg = np.count_nonzero(theta_tg_a) / theta_tg_a.size
    sp_b_tg = np.count_nonzero(theta_tg_b) / theta_tg_b.size

    # ---- Top 10% filtering: keep only the strongest connections ----
    top_pct = 0.1
    theta_tf_a_top = gs.keep_top_pct(theta_tf_a, pct=top_pct)
    theta_tf_b_top = gs.keep_top_pct(theta_tf_b, pct=top_pct)
    theta_tg_a_top = gs.keep_top_pct(theta_tg_a, pct=top_pct)
    theta_tg_b_top = gs.keep_top_pct(theta_tg_b, pct=top_pct)

    print(f"Top {int(top_pct*100)}% TF nonzero: {age_a}m={np.count_nonzero(theta_tf_a_top)}, {age_b}m={np.count_nonzero(theta_tf_b_top)}")
    print(f"Top {int(top_pct*100)}% TG nonzero: {age_a}m={np.count_nonzero(theta_tg_a_top)}, {age_b}m={np.count_nonzero(theta_tg_b_top)}")

    # Binary out degrees (number of connections)
    out_deg_tf_a_top = (np.abs(theta_tf_a_top) > 0).astype(float).sum(axis=1)
    out_deg_tf_b_top = (np.abs(theta_tf_b_top) > 0).astype(float).sum(axis=1)
    out_deg_tg_a_top = (np.abs(theta_tg_a_top) > 0).astype(float).sum(axis=1)
    out_deg_tg_b_top = (np.abs(theta_tg_b_top) > 0).astype(float).sum(axis=1)

    # Effective out degrees (sum of |weights|)
    out_deg_tf_a_top_eff = np.abs(theta_tf_a_top).sum(axis=1)
    out_deg_tf_b_top_eff = np.abs(theta_tf_b_top).sum(axis=1)
    out_deg_tg_a_top_eff = np.abs(theta_tg_a_top).sum(axis=1)
    out_deg_tg_b_top_eff = np.abs(theta_tg_b_top).sum(axis=1)

    # Save top-pct out degrees
    np.save(out_dir / f"out_deg_{t}_{age_a}m_tf_top{int(top_pct*100)}pct", out_deg_tf_a_top)
    np.save(out_dir / f"out_deg_{t}_{age_b}m_tf_top{int(top_pct*100)}pct", out_deg_tf_b_top)
    np.save(out_dir / f"out_deg_{t}_{age_a}m_tg_top{int(top_pct*100)}pct", out_deg_tg_a_top)
    np.save(out_dir / f"out_deg_{t}_{age_b}m_tg_top{int(top_pct*100)}pct", out_deg_tg_b_top)

    # Plot top-pct out degree distributions
    gs.plot_out_deg_top_pct(out_deg_tf_a_top, out_deg_tf_b_top, t, "tf", age_a, age_b, pct=top_pct)
    gs.plot_out_deg_top_pct(out_deg_tg_a_top, out_deg_tg_b_top, t, "tg", age_a, age_b, pct=top_pct)
    gs.plot_out_deg_top_pct(out_deg_tf_a_top_eff, out_deg_tf_b_top_eff, t, "tf", age_a, age_b, pct=top_pct, effective=True)
    gs.plot_out_deg_top_pct(out_deg_tg_a_top_eff, out_deg_tg_b_top_eff, t, "tg", age_a, age_b, pct=top_pct, effective=True)

    # Clean up top-pct copies
    del theta_tf_a_top, theta_tf_b_top, theta_tg_a_top, theta_tg_b_top

    if sp_a_tf < sp_b_tf:
        theta_tf_sparse = theta_tf_a
        theta_tf_dense = theta_tf_b
        age_tf_sparse = age_a
        age_tf_dense = age_b
        sp_sparse_tf = sp_a_tf
    else:
        theta_tf_sparse = theta_tf_b
        theta_tf_dense = theta_tf_a
        age_tf_sparse = age_b
        age_tf_dense = age_a
        sp_sparse_tf = sp_b_tf

    if sp_a_tg < sp_b_tg:
        theta_tg_sparse = theta_tg_a
        theta_tg_dense = theta_tg_b
        age_tg_sparse = age_a
        age_tg_dense = age_b
    else:
        theta_tg_sparse = theta_tg_b
        theta_tg_dense = theta_tg_a
        age_tg_sparse = age_b
        age_tg_dense = age_a

    print(f"TF sparsity: {age_a}m={sp_a_tf:.4f}, {age_b}m={sp_b_tf:.4f}")
    print(f"TG sparsity: {age_a}m={sp_a_tg:.4f}, {age_b}m={sp_b_tg:.4f}")
    print(f"TF sparse age={age_tf_sparse}m, dense age={age_tf_dense}m")
    print(f"TG sparse age={age_tg_sparse}m, dense age={age_tg_dense}m")

    t0 = time.perf_counter()
    theta_tf_sparsified, out_deg_tf_sparsified, out_deg_tf_sparsified_eff, _, sp_new_tf = gs.correct_sparsity(
        theta_tf_dense, theta_tf_sparse, n_tf, niters=50
    )
    timings["correct_sparsity_tf_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    theta_tg_sparsified, out_deg_tg_sparsified, out_deg_tg_sparsified_eff, _, _ = gs.correct_sparsity(
        theta_tg_dense, theta_tg_sparse, n_tf, niters=50
    )
    timings["correct_sparsity_tg_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    np.save(out_dir / f"out_deg_{t}_{age_tf_dense}m_tf_corrected_sp", out_deg_tf_sparsified)
    np.save(out_dir / f"out_deg_{t}_{age_tg_dense}m_tg_corrected_sp", out_deg_tg_sparsified)
    np.save(out_dir / f"out_deg_{t}_{age_tf_dense}m_tf_corrected_sp_sp_age_{age_tf_sparse}m", out_deg_tf_sparsified)
    np.save(out_dir / f"out_deg_{t}_{age_tg_dense}m_tg_corrected_sp_sp_age_{age_tg_sparse}m", out_deg_tg_sparsified)
    timings["save_out_degree_arrays_s"] = time.perf_counter() - t0

    theta_tf_sparsified_bin = (np.abs(theta_tf_sparsified) > 0).astype(float)
    theta_tg_sparsified_bin = (np.abs(theta_tg_sparsified) > 0).astype(float)
    theta_tf_sparse_bin = (np.abs(theta_tf_sparse) > 0).astype(float)
    theta_tg_sparse_bin = (np.abs(theta_tg_sparse) > 0).astype(float)

    print(f"Remaining sparse TF sparsity: target={sp_sparse_tf:.4f}, realized={sp_new_tf:.4f}")

    t0 = time.perf_counter()
    motif_df, counts_df, graphs_by_age, iter_counts_df = gs.analyze_IFL(
        theta_tf_sparsified,
        theta_tf_sparse,
        n_tf,
        niters=20,
        names_tf=names_tf,
        age_sparse=age_tf_sparse,
        age_dense=age_tf_dense,
        return_iter_counts=True,
    )
    motif_df.to_csv(out_dir / f"IFL_df_tf_{t}.csv", index=False)
    counts_df.to_csv(out_dir / f"IFL_counts_df_tf_{t}.csv", index=False)
    iter_counts_df.to_csv(out_dir / f"IFL_iter_counts_df_tf_{t}.csv", index=False)
    timings["analyze_ifl_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    motif_df, counts_df, graphs_by_age, iter_counts_df = gs.analyze_IFFL(
        theta_tf_sparsified,
        theta_tf_sparse,
        n_tf,
        niters=30,
        names_tf=names_tf,
        age_sparse=age_tf_sparse,
        age_dense=age_tf_dense,
        return_iter_counts=True,
    )
    motif_df.to_csv(out_dir / f"IFFL_df_tf_{t}.csv", index=False)
    counts_df.to_csv(out_dir / f"IFFL_counts_df_tf_{t}.csv", index=False)
    iter_counts_df.to_csv(out_dir / f"IFFL_iter_counts_df_tf_{t}.csv", index=False)
    timings["analyze_iffl_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    motif_df, counts_df, graphs_by_age = gs.analyze_3_node_feedback(
        theta_tf_sparsified, theta_tf_sparse, n_tf, niters=50, names_tf=names_tf, age_sparse=age_tf_sparse, age_dense=age_tf_dense
    )
    motif_df.to_csv(out_dir / f"3_node_feedback_df_tf_{t}.csv", index=False)
    counts_df.to_csv(out_dir / f"3_node_feedback_counts_df_tf_{t}.csv", index=False)
    timings["analyze_3_node_feedback_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    motif_df, counts_df, graphs_by_age = gs.analyze_struct_2nodes(
        theta_tf_sparsified, theta_tf_sparse, n_tf, niters=50, names_tf=names_tf, age_sparse=age_tf_sparse, age_dense=age_tf_dense
    )
    motif_df.to_csv(out_dir / f"2_node_feedback_df_{t}.csv", index=False)
    counts_df.to_csv(out_dir / f"2_node_feedback_counts_df_{t}.csv", index=False)
    timings["analyze_2_node_feedback_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    null_out_deg_tf = gs.get_null(theta_tf_a, theta_tf_b, n_tf, niters=100)
    null_out_deg_tg = gs.get_null(theta_tg_a, theta_tg_b, n_tf, niters=100)
    np.save(out_dir / f"out_deg_{t}_null_tf", null_out_deg_tf)
    np.save(out_dir / f"out_deg_{t}_null_tg", null_out_deg_tg)
    timings["null_model_s"] = time.perf_counter() - t0

    out_deg_tf_sparse = theta_tf_sparse_bin.sum(axis=1)
    out_deg_tg_sparse = theta_tg_sparse_bin.sum(axis=1)
    out_deg_tf_sparse_eff = np.abs(theta_tf_sparse).sum(axis=1)
    out_deg_tg_sparse_eff = np.abs(theta_tg_sparse).sum(axis=1)
    np.save(out_dir / f"out_deg_{t}_{age_tf_sparse}m_tf_corrected_sp_dense_age_{age_tf_dense}m", out_deg_tf_sparse)
    np.save(out_dir / f"out_deg_{t}_{age_tg_sparse}m_tg_corrected_sp_dense_age_{age_tg_dense}m", out_deg_tg_sparse)

    t0 = time.perf_counter()
    gs.plot_out_deg(
        out_deg_tf_sparsified,
        out_deg_tf_sparse,
        null_out_deg_tf,
        t,
        "tf",
        dense_age=age_tf_dense,
        sparse_age=age_tf_sparse,
        dense_sparsity=max(sp_a_tf, sp_b_tf),
        sparse_sparsity=min(sp_a_tf, sp_b_tf),
    )
    gs.plot_out_deg(
        out_deg_tg_sparsified,
        out_deg_tg_sparse,
        null_out_deg_tg,
        t,
        "tg",
        dense_age=age_tg_dense,
        sparse_age=age_tg_sparse,
        dense_sparsity=max(sp_a_tg, sp_b_tg),
        sparse_sparsity=min(sp_a_tg, sp_b_tg),
    )
    gs.plot_out_deg_eff(
        out_deg_tf_sparsified_eff,
        out_deg_tf_sparse_eff,
        t,
        "tf",
        dense_age=age_tf_dense,
        sparse_age=age_tf_sparse,
        dense_sparsity=max(sp_a_tf, sp_b_tf),
        sparse_sparsity=min(sp_a_tf, sp_b_tf),
    )
    gs.plot_out_deg_eff(
        out_deg_tg_sparsified_eff,
        out_deg_tg_sparse_eff,
        t,
        "tg",
        dense_age=age_tg_dense,
        sparse_age=age_tg_sparse,
        dense_sparsity=max(sp_a_tg, sp_b_tg),
        sparse_sparsity=min(sp_a_tg, sp_b_tg),
    )
    gs.plot_2nodes(motif_df, counts_df, graphs_by_age, t)
    timings["plotting_s"] = time.perf_counter() - t0

    timings["total_s"] = time.perf_counter() - total_t0
    print("=== Timing Summary (s) ===")
    for key in sorted(timings):
        print(f"{key}: {timings[key]:.3f}")

    if return_timings:
        return timings


def parse_args():
    p = argparse.ArgumentParser(description="Run structure analysis from trained TF/TG models.")
    p.add_argument("--tissue", required=True)
    p.add_argument("--ages", nargs=2, type=int, default=[3, 24], help="Exactly two ages to compare.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--models-dir", type=Path, default=Path("outputs/models/final"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/structure"))
    p.add_argument("--lambda", dest="lam", type=float, default=None)
    p.add_argument("--lambda-by-age", default=None, help="Comma-separated map like 3:0.7,24:0.4")
    p.add_argument("--theta-threshold", type=float, default=0.0)
    p.add_argument("--theta-threshold-by-age", default=None, help="Comma-separated map like 3:0.02,24:0.05")
    p.add_argument("--dry-run-check", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ages = sorted(set(args.ages))
    if len(ages) != 2:
        raise ValueError("--ages must contain exactly two distinct ages.")

    lam_by_age = resolve_value_by_age(ages, args.lam, parse_age_value_map(args.lambda_by_age), "lambda")
    th_by_age = resolve_value_by_age(ages, args.theta_threshold, parse_age_value_map(args.theta_threshold_by_age), "theta-threshold")

    age_a, age_b = ages
    theta_tf_a = load_theta(args.models_dir, args.tissue, age_a, "TF", lam_by_age[age_a], th_by_age[age_a])
    theta_tf_b = load_theta(args.models_dir, args.tissue, age_b, "TF", lam_by_age[age_b], th_by_age[age_b])
    theta_tg_a = load_theta(args.models_dir, args.tissue, age_a, "TG", lam_by_age[age_a], th_by_age[age_a])
    theta_tg_b = load_theta(args.models_dir, args.tissue, age_b, "TG", lam_by_age[age_b], th_by_age[age_b])
    tf_data_path = gs.data_path(args.data_dir, args.tissue, age_a, "TF")
    tf_data = np.load(tf_data_path)
    names_tf = load_names_tf(args.data_dir, args.tissue, theta_tf_a.shape[0])

    if args.dry_run_check:
        print("=== DRY RUN CHECK ===")
        print(f"tissue={args.tissue} ages={ages}")
        print(f"lambda_by_age={lam_by_age}")
        print(f"theta_threshold_by_age={th_by_age}")
        print(f"tf_data={tf_data_path} shape={tf_data.shape}")
        print(f"theta_tf[{age_a}] shape={theta_tf_a.shape}")
        print(f"theta_tf[{age_b}] shape={theta_tf_b.shape}")
        print(f"theta_tg[{age_a}] shape={theta_tg_a.shape}")
        print(f"theta_tg[{age_b}] shape={theta_tg_b.shape}")
        return

    calc_all_structs(
        t=args.tissue,
        theta_tf_a=theta_tf_a,
        theta_tf_b=theta_tf_b,
        theta_tg_a=theta_tg_a,
        theta_tg_b=theta_tg_b,
        age_a=age_a,
        age_b=age_b,
        names_tf=names_tf,
        out_dir=args.out_dir,
        factor=0.0,
    )


if __name__ == "__main__":
    main()
