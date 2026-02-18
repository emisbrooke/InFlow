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


def main():
    args = parse_args()
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
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
    for age in ages:
        tf_model_path = model_path(args.models_dir, args.tissue, age, "TF", args.lam)
        tg_model_path = model_path(args.models_dir, args.tissue, age, "TG", args.lam)
        tf_payload = load_payload(tf_model_path)
        tg_payload = load_payload(tg_model_path)

        tf_model[age] = {"theta": tf_payload["theta"], "m": tf_payload["m"]}
        tg_model[age] = {"theta": tg_payload["theta"], "m": tg_payload["m"]}

        tf_np = np.load(data_path(args.data_dir, args.tissue, age, "TF"))
        tg_np = np.load(data_path(args.data_dir, args.tissue, age, "TG"))
        tf_data[age] = tf_np
        tg_data[age] = tg_np

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
                "device": str(device),
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
