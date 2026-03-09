import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import calc_MI_funcs as MI


def lambda_tag(lambda_value: float) -> str:
    return f"{lambda_value:g}".replace("-", "m").replace(".", "p")


def threshold_tag(threshold: float) -> str:
    return f"{threshold:g}".replace("-", "m").replace(".", "p")


def data_path(data_dir: Path, tissue: str, age: int, gene_type: str) -> Path:
    # New naming convention used in final droplet-union datasets.
    preferred = data_dir / f"{tissue}_{gene_type.lower()}_data_binary_{age}m_droplet_union.npy"
    if preferred.exists():
        return preferred
    # Backward-compat fallback.
    legacy = data_dir / f"data_bin_filt_{tissue}_{age}m_{gene_type.upper()}.npy"
    return legacy


def model_path(models_dir: Path, tissue: str, age: int, gene_type: str, lam: float) -> Path:
    return models_dir / f"model_{tissue}_{age}m_{gene_type.upper()}_lam{lambda_tag(lam)}.pt"


def parse_age_value_map(raw: str | None) -> dict[int, float]:
    if raw is None:
        return {}
    out: dict[int, float] = {}
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        if ":" not in p:
            raise ValueError(f"Invalid age map entry '{p}'. Expected format age:value (e.g. 3:0.7).")
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


def parse_args():
    p = argparse.ArgumentParser(description="Run MI swap experiments across source TF ages and target TG ages.")
    p.add_argument("--tissue", required=True)
    p.add_argument("--ages", nargs="+", required=True, type=int, help="Ages to include in swap matrix, e.g. 3 24")
    p.add_argument("--source-ages", nargs="+", type=int, default=None, help="Optional subset of source TF ages.")
    p.add_argument("--target-ages", nargs="+", type=int, default=None, help="Optional subset of target TG ages.")
    p.add_argument("--lambda", dest="lam", required=False, type=float)
    p.add_argument(
        "--lambda-by-age",
        default=None,
        help="Comma-separated per-age map, e.g. 3:0.7,24:0.4. Overrides --lambda by age.",
    )
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--n-tf-samples", type=int, default=30000)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--models-dir", type=Path, default=Path("outputs/models/final"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/mi_swaps"))
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--int-burn", type=int, default=200000)
    p.add_argument("--int-save", type=int, default=500)
    p.add_argument("--mc-batch-size", type=int, default=500)
    p.add_argument("--reuse-tf-samples-per-iter", action="store_true", help="Generate source TF samples once per (iter, src_age) and reuse across dst ages.")
    p.add_argument("--theta-threshold", type=float, default=0.0, help="Hard-threshold |theta| below this value to zero before MI.")
    p.add_argument(
        "--theta-threshold-by-age",
        default=None,
        help="Comma-separated per-age map, e.g. 3:0.02,24:0.05. Overrides --theta-threshold by age.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling.")
    p.add_argument("--dry-run-check", action="store_true", help="Print resolved inputs (files/shapes) and exit without computing MI.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but CUDA is not available.")
        return torch.device("cuda:0")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_payload(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    return torch.load(path, map_location="cpu")


def apply_theta_threshold(theta: torch.Tensor, threshold: float) -> torch.Tensor:
    if threshold <= 0:
        return theta
    out = theta.clone()
    out[torch.abs(out) < threshold] = 0
    return out


def nonzero_fraction(theta: torch.Tensor) -> float:
    total = theta.numel()
    if total == 0:
        return 0.0
    return float((theta != 0).sum().item() / total)


def main():
    args = parse_args()
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    use_gpu = device.type == "cuda"
    if args.verbose:
        print(f"Using device: {device}")
        if use_gpu:
            print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    ages = sorted(set(args.ages))
    source_ages = sorted(set(args.source_ages)) if args.source_ages is not None else ages
    target_ages = sorted(set(args.target_ages)) if args.target_ages is not None else ages
    missing_source = sorted(set(source_ages) - set(ages))
    missing_target = sorted(set(target_ages) - set(ages))
    if missing_source or missing_target:
        raise ValueError(
            f"source/target ages must be subsets of --ages. "
            f"missing_source={missing_source}, missing_target={missing_target}"
        )

    lam_by_age = resolve_value_by_age(
        ages=ages,
        default=args.lam,
        overrides=parse_age_value_map(args.lambda_by_age),
        name="lambda",
    )
    th_by_age = resolve_value_by_age(
        ages=ages,
        default=args.theta_threshold,
        overrides=parse_age_value_map(args.theta_threshold_by_age),
        name="theta-threshold",
    )

    # Preload required model parameters and data
    tf_model = {}
    tg_model = {}
    tf_data = {}
    tg_data = {}
    tf_model_paths = {}
    tg_model_paths = {}
    tf_data_paths = {}
    tg_data_paths = {}
    for age in ages:
        lam_age = lam_by_age[age]
        th_age = th_by_age[age]
        tf_model_path = model_path(args.models_dir, args.tissue, age, "TF", lam_age)
        tg_model_path = model_path(args.models_dir, args.tissue, age, "TG", lam_age)
        tf_model_paths[age] = tf_model_path
        tg_model_paths[age] = tg_model_path
        tf_payload = load_payload(tf_model_path)
        tg_payload = load_payload(tg_model_path)

        theta_tf_raw = tf_payload["theta"]
        theta_tg_raw = tg_payload["theta"]
        theta_tf = apply_theta_threshold(theta_tf_raw, th_age)
        theta_tg = apply_theta_threshold(theta_tg_raw, th_age)
        tf_model[age] = {
            "theta": theta_tf,
            "m": tf_payload["m"],
            "lambda": lam_age,
            "theta_threshold": th_age,
            "theta_nz_frac_raw": nonzero_fraction(theta_tf_raw),
            "theta_nz_frac_used": nonzero_fraction(theta_tf),
        }
        tg_model[age] = {
            "theta": theta_tg,
            "m": tg_payload["m"],
            "lambda": lam_age,
            "theta_threshold": th_age,
            "theta_nz_frac_raw": nonzero_fraction(theta_tg_raw),
            "theta_nz_frac_used": nonzero_fraction(theta_tg),
        }

        tf_data_path = data_path(args.data_dir, args.tissue, age, "TF")
        tg_data_path = data_path(args.data_dir, args.tissue, age, "TG")
        tf_data_paths[age] = tf_data_path
        tg_data_paths[age] = tg_data_path
        tf_np = np.load(tf_data_path)
        tg_np = np.load(tg_data_path)
        if tf_np.shape[1] == 0 or tg_np.shape[1] == 0:
            raise RuntimeError(
                f"Empty input matrix for tissue={args.tissue}, age={age}: "
                f"tf shape={tf_np.shape}, tg shape={tg_np.shape}"
            )
        tf_data[age] = tf_np
        tg_data[age] = tg_np

    if args.dry_run_check:
        print("=== DRY RUN CHECK ===")
        print(f"tissue={args.tissue} ages={ages}")
        print(f"lambda_by_age={lam_by_age}")
        print(f"theta_threshold_by_age={th_by_age}")
        print(f"source_ages={source_ages} target_ages={target_ages}")
        for age in ages:
            print(f"[age={age}m]")
            print(f"  tf_model: {tf_model_paths[age]}")
            print(f"  tg_model: {tg_model_paths[age]}")
            print(f"  tf_data:  {tf_data_paths[age]} shape={tf_data[age].shape}")
            print(f"  tg_data:  {tg_data_paths[age]} shape={tg_data[age].shape}")
            print(f"  theta_tf shape={tuple(tf_model[age]['theta'].shape)} m_tf shape={tuple(tf_model[age]['m'].shape)}")
            print(f"  theta_tg shape={tuple(tg_model[age]['theta'].shape)} m_tg shape={tuple(tg_model[age]['m'].shape)}")
            print(
                f"  tf theta nz frac raw/used: "
                f"{tf_model[age]['theta_nz_frac_raw']:.4f}/{tf_model[age]['theta_nz_frac_used']:.4f}"
            )
            print(
                f"  tg theta nz frac raw/used: "
                f"{tg_model[age]['theta_nz_frac_raw']:.4f}/{tg_model[age]['theta_nz_frac_used']:.4f}"
            )
        print("Planned swaps:")
        for src_age in source_ages:
            for dst_age in target_ages:
                print(f"  src={src_age}m -> dst={dst_age}m")
        return

    for src_age in source_ages:
        for dst_age in target_ages:
            src_lam = tf_model[src_age]["lambda"]
            dst_lam = tg_model[dst_age]["lambda"]
            src_th = tf_model[src_age]["theta_threshold"]
            dst_th = tg_model[dst_age]["theta_threshold"]
            out_base = (
                f"mi_{args.tissue}_{src_age}m_in_{dst_age}m_"
                f"lamTF{lambda_tag(src_lam)}_lamTG{lambda_tag(dst_lam)}_"
                f"thTF{threshold_tag(src_th)}_thTG{threshold_tag(dst_th)}"
            )
            out_mi = args.out_dir / f"{out_base}.npy"
            out_entropy = args.out_dir / f"entropy_{args.tissue}_{src_age}m_in_{dst_age}m_lamTF{lambda_tag(src_lam)}_lamTG{lambda_tag(dst_lam)}_thTF{threshold_tag(src_th)}_thTG{threshold_tag(dst_th)}.npy"
            out_summary = args.out_dir / f"{out_base}_summary.json"
            if out_mi.exists() and out_entropy.exists() and (not args.overwrite):
                print(f"SKIP existing {out_base}")
                continue

            nG = tg_data[dst_age].shape[0]
            mi_runs = torch.zeros(args.iters, nG, dtype=torch.float32)
            ent_runs = torch.zeros(args.iters, nG, dtype=torch.float32)

            for i in range(args.iters):
                iter_t0 = time.time()
                samples_tf = None
                if args.reuse_tf_samples_per_iter:
                    src_theta_tf = torch.as_tensor(tf_model[src_age]["theta"], dtype=torch.float32, device=device)
                    src_m_tf = torch.as_tensor(tf_model[src_age]["m"], dtype=torch.float32, device=device)
                    src_sigma_tf = torch.as_tensor(tf_data[src_age], dtype=torch.float32, device=device)
                    samples_tf = MI.generate_samples_mc(
                        theta=src_theta_tf,
                        tf_samples=src_sigma_tf,
                        m=src_m_tf,
                        int_burn=args.int_burn,
                        nSamples=args.n_tf_samples,
                        int_save=args.int_save,
                        add=False,
                        batch_size=args.mc_batch_size,
                        verbose=args.verbose,
                    ).detach().cpu().numpy()

                mi, ent = MI.calc_MI(
                    sigma_tf=tf_data[src_age],
                    theta_tf=tf_model[src_age]["theta"],
                    m_tf=tf_model[src_age]["m"],
                    sigma_tg=tg_data[dst_age],
                    theta_tg=tg_model[dst_age]["theta"],
                    m_tg=tg_model[dst_age]["m"],
                    age_tf=src_age,
                    age_tg=dst_age,
                    t=args.tissue,
                    lam=src_lam,
                    n_tf_samples=args.n_tf_samples,
                    use_gpu_if_avail=use_gpu,
                    samples_tf=samples_tf,
                    int_burn=args.int_burn,
                    int_save=args.int_save,
                    mc_batch_size=args.mc_batch_size,
                    verbose=args.verbose,
                )

                mi_runs[i] = mi.detach().cpu().to(torch.float32)
                ent_runs[i] = ent.detach().cpu().to(torch.float32)
                print(
                    f"[{args.tissue}] {src_age}m -> {dst_age}m iter {i + 1}/{args.iters} "
                    f"done in {time.time() - iter_t0:.1f}s"
                )

            np.save(out_mi, mi_runs.numpy())
            np.save(out_entropy, ent_runs.numpy())

            summary = {
                "tissue": args.tissue,
                "source_age_m": src_age,
                "target_age_m": dst_age,
                "source_tf_lambda": src_lam,
                "target_tg_lambda": dst_lam,
                "iters": args.iters,
                "n_genes": int(nG),
                "n_tf_samples": args.n_tf_samples,
                "source_tf_theta_threshold": src_th,
                "target_tg_theta_threshold": dst_th,
                "device": str(device),
                "source_tf_theta_nonzero_fraction": tf_model[src_age]["theta_nz_frac_used"],
                "target_tg_theta_nonzero_fraction": tg_model[dst_age]["theta_nz_frac_used"],
                "mean_mi_per_gene_mean": float(mi_runs.mean(dim=0).mean().item()),
                "mean_mi_per_gene_std": float(mi_runs.mean(dim=0).std().item()),
                "mean_entropy_per_gene_mean": float(ent_runs.mean(dim=0).mean().item()),
            }
            with out_summary.open("w") as f:
                json.dump(summary, f, indent=2)
            print(f"WROTE {out_mi}")
            print(f"WROTE {out_entropy}")
            print(f"WROTE {out_summary}")

    print(f"Finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
