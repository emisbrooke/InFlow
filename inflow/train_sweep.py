import argparse
import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import TFModel


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def build_data_path(data_dir: Path, tissue: str, age: int, gene_type: str) -> Path:
    return data_dir / f"data_bin_filt_{tissue}_{age}m_{gene_type}.npy"


def lambda_tag(lambda_value: float) -> str:
    return f"{lambda_value:g}".replace("-", "m").replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser(description="Train InFlow models over tissue/age/gene-type/lambda sweeps.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/models"))
    parser.add_argument("--tissues", nargs="+", required=True)
    parser.add_argument("--ages", nargs="+", required=True, type=int)
    parser.add_argument("--gene-types", nargs="+", default=["TF", "TG"])
    parser.add_argument("--lambdas", nargs="+", required=True, type=float)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--n-restarts", type=int, default=1, help="Train each configuration multiple times and keep the best run.")
    parser.add_argument("--optimizer", choices=["adam", "yogi", "sgd"], default="adam")
    parser.add_argument("--eta", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--min-delta-ll", type=float, default=50.0)
    parser.add_argument("--log-int", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dtype", choices=list(DTYPE_MAP.keys()), default="float16")
    parser.add_argument("--save-loss", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_training_arrays(data_dir: Path, tissue: str, age: int, gene_type: str):
    gt = gene_type.upper()
    data_path = build_data_path(data_dir, tissue, age, gt)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    data = np.load(data_path)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D matrix in {data_path}, got shape {data.shape}")

    if gt == "TF":
        tf_data = data
    else:
        tf_path = build_data_path(data_dir, tissue, age, "TF")
        if not tf_path.exists():
            raise FileNotFoundError(f"Missing TF reference file for TG training: {tf_path}")
        tf_data = np.load(tf_path)
        if tf_data.ndim != 2:
            raise ValueError(f"Expected 2D matrix in {tf_path}, got shape {tf_data.shape}")

    return tf_data, data, data_path


def run_one_job(args, tissue: str, age: int, gene_type: str, lambda_value: float):
    tf_data, data, data_path = load_training_arrays(args.data_dir, tissue, age, gene_type)
    optimizer = "sgd" if args.optimizer == "sgd" else args.optimizer
    best_payload = None
    best_obj = -math.inf
    best_restart = -1

    for restart_idx in range(args.n_restarts):
        torch.manual_seed(args.seed + restart_idx)
        np.random.seed(args.seed + restart_idx)

        model = TFModel(age=age, tf_data=tf_data, data=data, gene_type=gene_type)
        model.train(
            steps=args.steps,
            thresh=args.thresh,
            patience=args.patience,
            min_delta_ll=args.min_delta_ll,
            optimizer=optimizer,
            use_gpu_if_avail=not args.cpu_only,
            alpha=args.alpha,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.eps,
            verbose=not args.quiet,
            log_int=args.log_int,
            lambda_=lambda_value,
            eta=args.eta,
            autosave=False,
        )

        larr = model.Larr.detach().cpu() if torch.is_tensor(model.Larr) else torch.as_tensor(model.Larr)
        final_obj = float(larr[-1].item()) if larr.numel() else -math.inf
        if final_obj > best_obj:
            save_dtype = DTYPE_MAP[args.save_dtype]
            best_obj = final_obj
            best_restart = restart_idx
            best_payload = {
                "theta": model.theta.detach().cpu().to(save_dtype),
                "m": model.m.detach().cpu().to(save_dtype),
                "meta": {
                    "tissue": tissue,
                    "age_m": int(age),
                    "gene_type": gene_type.upper(),
                    "lambda": float(lambda_value),
                    "optimizer": optimizer,
                    "steps_requested": int(args.steps),
                    "steps_run": int(larr.numel()),
                    "n_restarts": int(args.n_restarts),
                    "best_restart_index": int(best_restart),
                    "seed_base": int(args.seed),
                    "save_dtype": args.save_dtype,
                    "source_data_file": str(data_path),
                    "theta_shape": list(model.theta.shape),
                },
                "final_penalized_obj": final_obj,
            }
            if args.save_loss:
                best_payload["Larr"] = larr.to(torch.float32)

        if not args.quiet:
            print(
                f"  restart {restart_idx + 1}/{args.n_restarts}: "
                f"final_penalized_obj={final_obj:.6f}"
            )

    if best_payload is None:
        raise RuntimeError("No successful restart was produced.")

    out_name = f"model_{tissue}_{age}m_{gene_type.upper()}_lam{lambda_tag(lambda_value)}.pt"
    out_path = args.out_dir / out_name
    torch.save(best_payload, out_path)
    return out_path, best_payload["meta"]["steps_run"], best_restart, best_obj


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    jobs = list(itertools.product(args.tissues, args.ages, args.gene_types, args.lambdas))
    failures = []
    saved = 0
    t0 = time.time()

    for idx, (tissue, age, gene_type, lambda_value) in enumerate(jobs, start=1):
        out_name = f"model_{tissue}_{age}m_{gene_type.upper()}_lam{lambda_tag(lambda_value)}.pt"
        out_path = args.out_dir / out_name
        if out_path.exists() and not args.overwrite:
            print(f"[{idx}/{len(jobs)}] SKIP existing {out_path}")
            continue

        print(f"[{idx}/{len(jobs)}] RUN tissue={tissue} age={age}m gene_type={gene_type} lambda={lambda_value}")
        try:
            saved_path, ran_steps, best_restart, best_obj = run_one_job(args, tissue, age, gene_type, lambda_value)
            print(
                f"[{idx}/{len(jobs)}] OK {saved_path} "
                f"(steps={ran_steps}, best_restart={best_restart}, final_penalized_obj={best_obj:.6f})"
            )
            saved += 1
        except Exception as exc:
            msg = f"[{idx}/{len(jobs)}] FAIL tissue={tissue} age={age}m gene_type={gene_type} lambda={lambda_value}: {exc}"
            print(msg)
            failures.append(msg)
            if args.fail_fast:
                break

    dt = time.time() - t0
    print(f"Finished in {dt:.2f}s, saved={saved}, failed={len(failures)}")
    if failures:
        print("Failures:")
        for msg in failures:
            print(msg)


if __name__ == "__main__":
    main()
