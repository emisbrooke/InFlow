import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import GeneModel


def lambda_tag(lambda_value: float) -> str:
    return f"{lambda_value:g}".replace("-", "m").replace(".", "p")


def build_data_path(data_dir: Path, tissue: str, age: int, gene_type: str) -> Path:
    return data_dir / f"data_bin_filt_{tissue}_{age}m_{gene_type}.npy"


def safe_corrcoef(x: torch.Tensor) -> torch.Tensor:
    c = torch.corrcoef(x)
    return torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)


def vector_pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten()
    b = b.flatten()
    if a.numel() == 0:
        return math.nan
    a_std = a.std()
    b_std = b.std()
    if float(a_std) == 0.0 or float(b_std) == 0.0:
        return math.nan
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1].item())


def get_summary_vectors(observed: torch.Tensor, generated: torch.Tensor):
    obs = observed.to(torch.double)
    gen = generated.to(torch.double)
    mean_obs = obs.mean(dim=1)
    mean_gen = gen.mean(dim=1)

    corr_obs = safe_corrcoef(obs)
    corr_gen = safe_corrcoef(gen)
    n = corr_obs.shape[0]
    tri = torch.triu_indices(n, n, offset=1, device=corr_obs.device)
    corr_obs_upper = corr_obs[tri[0], tri[1]]
    corr_gen_upper = corr_gen[tri[0], tri[1]]
    return mean_obs, mean_gen, corr_obs_upper, corr_gen_upper


def sample_tf_gibbs(
    theta: torch.Tensor,
    m: torch.Tensor,
    tf_data: torch.Tensor,
    n_samples: int,
    int_burn: int,
    int_save: int,
    batch_size: int,
    n_chains: int,
    verbose: bool = False,
    progress_every: int = 10,
) -> torch.Tensor:
    theta = theta.to(torch.double)
    m = m.to(torch.double)
    tf_data = tf_data.to(torch.double)
    nTF, nC = tf_data.shape

    n_chain = min(max(1, n_chains), nC)
    if n_chain < n_samples:
        pass

    idxs = torch.randperm(nC)[:n_chain]
    tf_samples = tf_data[:, idxs].clone()

    n_collect = math.ceil(n_samples / n_chain)
    iters = n_collect * int_save + (int_burn - int_save)
    row_idx_all = torch.randint(nTF, (iters + 1, n_chain), device=tf_samples.device)

    stored = []
    count = 0
    n_batches = math.ceil((iters + 1) / batch_size)
    t0 = time.perf_counter()
    for batch_idx, i in enumerate(range(0, iters + 1, batch_size), start=1):
        row_idx_batch = row_idx_all[i:i + batch_size]
        for row_idxs in row_idx_batch:
            tf_samples = GeneModel.update_samples(theta, tf_samples, m, row_idxs)
            if ((count >= int_burn) and (count % int_save == 0)) or (count == int_burn):
                stored.append(tf_samples.clone())
            count += 1
        if verbose and (batch_idx % progress_every == 0 or batch_idx == n_batches):
            elapsed = time.perf_counter() - t0
            print(
                f"    TF sampling progress: batch {batch_idx}/{n_batches}, "
                f"iters_done={count}/{iters + 1}, stored={len(stored)}, elapsed={elapsed:.1f}s"
            )

    if not stored:
        raise RuntimeError("No TF samples stored; adjust int_burn/int_save.")
    out = torch.hstack(stored)
    return out[:, :n_samples]


def sample_tg_from_tf(theta: torch.Tensor, m: torch.Tensor, tf_data: torch.Tensor, n_samples: int) -> torch.Tensor:
    theta = theta.to(torch.double)
    m = m.to(torch.double)
    tf_data = tf_data.to(torch.double)
    nC = tf_data.shape[1]
    exp = torch.mm(theta.transpose(0, 1), tf_data) + m.unsqueeze(1)
    pi = torch.sigmoid(-exp)
    if n_samples <= nC:
        idxs = torch.randperm(nC)[:n_samples]
    else:
        idxs = torch.randint(0, nC, (n_samples,))
    return torch.bernoulli(pi[:, idxs]).to(torch.double)


def compute_metrics(mean_obs: torch.Tensor, mean_gen: torch.Tensor, corr_obs_upper: torch.Tensor, corr_gen_upper: torch.Tensor):
    mean_diff = mean_gen - mean_obs

    corr_diff = corr_gen_upper - corr_obs_upper

    return {
        "mean_mae": float(torch.mean(torch.abs(mean_diff)).item()),
        "mean_rmse": float(torch.sqrt(torch.mean(mean_diff ** 2)).item()),
        "mean_pearson": vector_pearson(mean_obs, mean_gen),
        "corr_mae": float(torch.mean(torch.abs(corr_diff)).item()),
        "corr_rmse": float(torch.sqrt(torch.mean(corr_diff ** 2)).item()),
        "corr_pearson_upper": vector_pearson(corr_obs_upper, corr_gen_upper),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Compare model-generated vs observed summary statistics.")
    p.add_argument("--models-dir", type=Path, default=Path("outputs/models"))
    p.add_argument("--model-paths", nargs="*", default=None)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-csv", type=Path, default=Path("outputs/model_stats.csv"))
    p.add_argument("--n-samples", type=int, default=0, help="0 means use observed cell count.")
    p.add_argument("--tf-burn", type=int, default=10000)
    p.add_argument("--tf-save-interval", type=int, default=1000)
    p.add_argument("--tf-batch-size", type=int, default=1000)
    p.add_argument("--tf-n-chains", type=int, default=256)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--progress-every", type=int, default=10, help="Print TF sampling progress every N batches when --verbose is set.")
    p.add_argument("--plot-dir", type=Path, default=None, help="If set, save per-model mean/corr scatter plots to this directory.")
    p.add_argument("--corr-plot-max-pairs", type=int, default=20000, help="Max number of correlation pairs to plot per model.")
    p.add_argument("--plot-dpi", type=int, default=150)
    p.add_argument("--plot-seed", type=int, default=0)
    p.add_argument("--checkpoint-path", type=Path, default=Path("outputs/model_stats.checkpoint.jsonl"))
    p.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoint file and recompute all models.")
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args()


def resolve_model_paths(args):
    if args.model_paths:
        return [Path(x) for x in args.model_paths]
    return sorted(args.models_dir.glob("model_*.pt"))


def infer_identity_from_filename(path: Path):
    # model_{tissue}_{age}m_{gene_type}_lam{tag}.pt
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 5:
        return None
    try:
        tissue = parts[1]
        age = int(parts[2].removesuffix("m"))
        gene_type = parts[3].upper()
    except Exception:
        return None
    return tissue, age, gene_type


def load_checkpoint_rows(path: Path):
    rows = []
    completed = set()
    if not path.exists():
        return rows, completed

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "model_path" not in row:
                continue
            rows.append(row)
            completed.add(str(row["model_path"]))
    return rows, completed


def append_checkpoint_row(path: Path, row: dict):
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def maybe_save_plot(
    plot_path: Path,
    mean_obs: torch.Tensor,
    mean_gen: torch.Tensor,
    corr_obs_upper: torch.Tensor,
    corr_gen_upper: torch.Tensor,
    corr_plot_max_pairs: int,
    plot_seed: int,
    plot_dpi: int,
    title_prefix: str,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plotting skipped: matplotlib is not installed.")
        return

    mean_obs_np = mean_obs.detach().cpu().numpy()
    mean_gen_np = mean_gen.detach().cpu().numpy()
    corr_obs_np = corr_obs_upper.detach().cpu().numpy()
    corr_gen_np = corr_gen_upper.detach().cpu().numpy()

    rng = np.random.default_rng(plot_seed)
    n_corr = corr_obs_np.shape[0]
    if n_corr > corr_plot_max_pairs:
        sel = rng.choice(n_corr, size=corr_plot_max_pairs, replace=False)
        corr_obs_np = corr_obs_np[sel]
        corr_gen_np = corr_gen_np[sel]

    mean_pcc = np.corrcoef(mean_obs_np, mean_gen_np)[0, 1] if mean_obs_np.size > 1 else np.nan
    corr_pcc = np.corrcoef(corr_obs_np, corr_gen_np)[0, 1] if corr_obs_np.size > 1 else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(mean_obs_np, mean_gen_np, s=10, alpha=0.6)
    axes[0].set_xlabel("Observed gene mean")
    axes[0].set_ylabel("Generated gene mean")
    axes[0].set_title(f"Mean vs Mean (PCC={mean_pcc:.3f})")
    axes[0].grid(alpha=0.2)

    axes[1].scatter(corr_obs_np, corr_gen_np, s=6, alpha=0.4)
    axes[1].set_xlabel("Observed corr (upper triangle)")
    axes[1].set_ylabel("Generated corr (upper triangle)")
    axes[1].set_title(f"Corr vs Corr (PCC={corr_pcc:.3f}, n={corr_obs_np.size})")
    axes[1].grid(alpha=0.2)

    fig.suptitle(title_prefix)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=plot_dpi)
    plt.close(fig)


def main():
    args = parse_args()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        device = torch.device("cuda:0")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        print(f"CUDA device index={idx}, name={name}, total_mem={total_gb:.1f} GB")

    paths = resolve_model_paths(args)
    if not paths:
        raise FileNotFoundError("No model files found.")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    completed = set()
    if not args.no_resume:
        rows, completed = load_checkpoint_rows(args.checkpoint_path)
        if completed:
            print(f"Loaded checkpoint rows={len(rows)} from {args.checkpoint_path}")
    if args.plot_dir is not None:
        args.plot_dir.mkdir(parents=True, exist_ok=True)

    for model_idx, path in enumerate(paths, start=1):
        path_str = str(path)
        if path_str in completed:
            print(f"[{model_idx}/{len(paths)}] SKIP (checkpoint) {path}")
            continue
        try:
            model_t0 = time.perf_counter()
            print(f"[{model_idx}/{len(paths)}] START {path}")

            load_t0 = time.perf_counter()
            payload = torch.load(path, map_location="cpu")
            meta = payload.get("meta", {})
            identity = (
                meta.get("tissue"),
                meta.get("age_m"),
                meta.get("gene_type"),
            )
            if any(x is None for x in identity):
                inferred = infer_identity_from_filename(path)
                if inferred is None:
                    raise ValueError(f"Could not infer tissue/age/gene_type for {path}")
                tissue, age, gene_type = inferred
            else:
                tissue, age, gene_type = identity[0], int(identity[1]), str(identity[2]).upper()

            obs_path = build_data_path(args.data_dir, tissue, age, gene_type)
            if not obs_path.exists():
                raise FileNotFoundError(f"Observed data not found: {obs_path}")
            observed = torch.as_tensor(np.load(obs_path), dtype=torch.double, device=device)
            target_samples = args.n_samples if args.n_samples > 0 else observed.shape[1]
            if device.type == "cuda":
                torch.cuda.synchronize()
            load_dt = time.perf_counter() - load_t0

            theta = payload["theta"].to(dtype=torch.double, device=device)
            m = payload["m"].to(dtype=torch.double, device=device)

            sample_t0 = time.perf_counter()
            if gene_type == "TF":
                tf_data = observed
                generated = sample_tf_gibbs(
                    theta=theta,
                    m=m,
                    tf_data=tf_data,
                    n_samples=target_samples,
                    int_burn=args.tf_burn,
                    int_save=args.tf_save_interval,
                    batch_size=args.tf_batch_size,
                    n_chains=args.tf_n_chains,
                    verbose=args.verbose,
                    progress_every=max(1, args.progress_every),
                )
            elif gene_type == "TG":
                tf_path = build_data_path(args.data_dir, tissue, age, "TF")
                if not tf_path.exists():
                    raise FileNotFoundError(f"TF conditioning data not found for TG model: {tf_path}")
                tf_data = torch.as_tensor(np.load(tf_path), dtype=torch.double, device=device)
                generated = sample_tg_from_tf(theta, m, tf_data, target_samples)
            else:
                raise ValueError(f"Unsupported gene_type: {gene_type}")
            if device.type == "cuda":
                torch.cuda.synchronize()
            sample_dt = time.perf_counter() - sample_t0

            metric_t0 = time.perf_counter()
            mean_obs, mean_gen, corr_obs_upper, corr_gen_upper = get_summary_vectors(observed, generated)
            metrics = compute_metrics(mean_obs, mean_gen, corr_obs_upper, corr_gen_upper)
            if device.type == "cuda":
                torch.cuda.synchronize()
            metric_dt = time.perf_counter() - metric_t0

            if args.plot_dir is not None:
                plot_path = args.plot_dir / f"{path.stem}_stats.png"
                maybe_save_plot(
                    plot_path=plot_path,
                    mean_obs=mean_obs,
                    mean_gen=mean_gen,
                    corr_obs_upper=corr_obs_upper,
                    corr_gen_upper=corr_gen_upper,
                    corr_plot_max_pairs=max(1000, args.corr_plot_max_pairs),
                    plot_seed=args.plot_seed,
                    plot_dpi=args.plot_dpi,
                    title_prefix=f"{tissue} {age}m {gene_type} lambda={meta.get('lambda', 'NA')}",
                )
            row = {
                "model_path": str(path),
                "tissue": tissue,
                "age_m": age,
                "gene_type": gene_type,
                "lambda": meta.get("lambda", math.nan),
                "steps_run": meta.get("steps_run", math.nan),
                "final_penalized_obj": payload.get("final_penalized_obj", math.nan),
                "n_genes": int(observed.shape[0]),
                "n_cells_observed": int(observed.shape[1]),
                "n_cells_generated": int(generated.shape[1]),
            }
            row.update(metrics)
            rows.append(row)
            append_checkpoint_row(args.checkpoint_path, row)
            completed.add(path_str)
            total_dt = time.perf_counter() - model_t0
            print(
                f"[{model_idx}/{len(paths)}] OK {path} | "
                f"load={load_dt:.2f}s sample={sample_dt:.2f}s metrics={metric_dt:.2f}s total={total_dt:.2f}s"
            )

            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            print(f"FAIL {path}: {exc}")
            if args.fail_fast:
                raise

    if not rows:
        raise RuntimeError("No successful model evaluations.")

    fieldnames = list(rows[0].keys())
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
