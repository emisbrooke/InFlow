import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import calc_MI_funcs as MI


def lambda_tag(lambda_value: float) -> str:
    return f"{lambda_value:g}".replace("-", "m").replace(".", "p")


def data_path(data_dir: Path, tissue: str, age: int, gene_type: str) -> Path:
    return data_dir / f"data_bin_filt_{tissue}_{age}m_{gene_type.upper()}.npy"


def model_path(models_dir: Path, tissue: str, age: int, gene_type: str, lam: float) -> Path:
    return models_dir / f"model_{tissue}_{age}m_{gene_type.upper()}_lam{lambda_tag(lam)}.pt"


def parse_args():
    p = argparse.ArgumentParser(description="Run MI swap experiments across source TF ages and target TG ages.")
    p.add_argument("--tissue", required=True)
    p.add_argument("--ages", nargs="+", required=True, type=int, help="Ages to include in swap matrix, e.g. 3 24")
    p.add_argument("--lambda", dest="lam", required=True, type=float)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--n-tf-samples", type=int, default=30000)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--models-dir", type=Path, default=Path("outputs/models"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/mi_swaps"))
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--int-burn", type=int, default=200000)
    p.add_argument("--int-save", type=int, default=500)
    p.add_argument("--mc-batch-size", type=int, default=500)
    p.add_argument("--reuse-tf-samples-per-iter", action="store_true", help="Generate source TF samples once per (iter, src_age) and reuse across dst ages.")
    p.add_argument("--theta-threshold", type=float, default=0.0, help="Hard-threshold |theta| below this value to zero before MI.")
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
        tf_model_path = model_path(args.models_dir, args.tissue, age, "TF", args.lam)
        tg_model_path = model_path(args.models_dir, args.tissue, age, "TG", args.lam)
        tf_model_paths[age] = tf_model_path
        tg_model_paths[age] = tg_model_path
        tf_payload = load_payload(tf_model_path)
        tg_payload = load_payload(tg_model_path)

        theta_tf_raw = tf_payload["theta"]
        theta_tg_raw = tg_payload["theta"]
        theta_tf = apply_theta_threshold(theta_tf_raw, args.theta_threshold)
        theta_tg = apply_theta_threshold(theta_tg_raw, args.theta_threshold)
        tf_model[age] = {
            "theta": theta_tf,
            "m": tf_payload["m"],
            "theta_nz_frac_raw": nonzero_fraction(theta_tf_raw),
            "theta_nz_frac_used": nonzero_fraction(theta_tf),
        }
        tg_model[age] = {
            "theta": theta_tg,
            "m": tg_payload["m"],
            "theta_nz_frac_raw": nonzero_fraction(theta_tg_raw),
            "theta_nz_frac_used": nonzero_fraction(theta_tg),
        }

        tf_data_path = data_path(args.data_dir, args.tissue, age, "TF")
        tg_data_path = data_path(args.data_dir, args.tissue, age, "TG")
        tf_data_paths[age] = tf_data_path
        tg_data_paths[age] = tg_data_path
        tf_np = np.load(tf_data_path)
        tg_np = np.load(tg_data_path)
        tf_data[age] = tf_np
        tg_data[age] = tg_np

    if args.dry_run_check:
        print("=== DRY RUN CHECK ===")
        print(f"tissue={args.tissue} lambda={args.lam} ages={ages}")
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
        for src_age in ages:
            for dst_age in ages:
                print(f"  src={src_age}m -> dst={dst_age}m")
        return

    for src_age in ages:
        for dst_age in ages:
            out_base = f"mi_{args.tissue}_{src_age}m_in_{dst_age}m_lam{lambda_tag(args.lam)}"
            out_mi = args.out_dir / f"{out_base}.npy"
            out_entropy = args.out_dir / f"entropy_{args.tissue}_{src_age}m_in_{dst_age}m_lam{lambda_tag(args.lam)}.npy"
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
                    lam=args.lam,
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
                "lambda": args.lam,
                "iters": args.iters,
                "n_genes": int(nG),
                "n_tf_samples": args.n_tf_samples,
                "theta_threshold": args.theta_threshold,
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
