import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Create summary plots from model_stats.csv.")
    p.add_argument("--csv", type=Path, default=Path("outputs/model_stats.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/model_plots/summary"))
    p.add_argument("--dpi", type=int, default=160)
    return p.parse_args()


def save_lineplot(df, x, y, hue, title, out_path, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, g in df.groupby(hue):
        g2 = g.sort_values(x)
        ax.plot(g2[x], g2[y], marker="o", label=str(key))
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    required = {"tissue", "age_m", "gene_type", "lambda", "mean_pearson", "corr_pearson_upper", "mean_mae", "corr_mae"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "theta_threshold" not in df.columns:
        df["theta_threshold"] = 0.0
    df["theta_threshold"] = pd.to_numeric(df["theta_threshold"], errors="coerce").fillna(0.0)
    df["threshold_tag"] = df["theta_threshold"].map(lambda x: f"th={x:g}")
    df["group"] = df["tissue"].astype(str) + "_" + df["age_m"].astype(str) + "m_" + df["gene_type"].astype(str)
    df["group_with_threshold"] = df["group"] + "_" + df["threshold_tag"]

    save_lineplot(
        df=df,
        x="lambda",
        y="mean_pearson",
        hue="group_with_threshold",
        title="Mean PCC vs Lambda",
        out_path=args.out_dir / "mean_pcc_vs_lambda.png",
        ylabel="Mean PCC",
    )
    save_lineplot(
        df=df,
        x="lambda",
        y="corr_pearson_upper",
        hue="group_with_threshold",
        title="Correlation PCC vs Lambda",
        out_path=args.out_dir / "corr_pcc_vs_lambda.png",
        ylabel="Corr PCC",
    )
    save_lineplot(
        df=df,
        x="lambda",
        y="mean_mae",
        hue="group_with_threshold",
        title="Mean MAE vs Lambda",
        out_path=args.out_dir / "mean_mae_vs_lambda.png",
        ylabel="Mean MAE",
    )
    save_lineplot(
        df=df,
        x="lambda",
        y="corr_mae",
        hue="group_with_threshold",
        title="Correlation MAE vs Lambda",
        out_path=args.out_dir / "corr_mae_vs_lambda.png",
        ylabel="Corr MAE",
    )

    # Optional structure plots (present in newer compare_model_stats outputs)
    structure_cols = [
        ("theta_nonzero_frac", "Nonzero Fraction vs Lambda", "Nonzero Fraction", "theta_nonzero_frac_vs_lambda.png"),
        ("theta_top1pct_out_degree_share", "Top 1% Out-Degree Share vs Lambda", "Top1% Out-Degree Share", "theta_top1pct_share_vs_lambda.png"),
        ("theta_isolated_regulator_frac", "Isolated Regulator Fraction vs Lambda", "Isolated Regulator Fraction", "theta_isolated_regulator_frac_vs_lambda.png"),
        ("theta_avg_out_degree", "Average Out-Degree vs Lambda", "Average Out-Degree", "theta_avg_out_degree_vs_lambda.png"),
    ]
    for col, title, ylabel, fname in structure_cols:
        if col in df.columns:
            save_lineplot(
                df=df,
                x="lambda",
                y=col,
                hue="group_with_threshold",
                title=title,
                out_path=args.out_dir / fname,
                ylabel=ylabel,
            )

    # Quick best-lambda table by corr PCC (per tissue/age/gene_type/threshold)
    best = (
        df.sort_values("corr_pearson_upper", ascending=False)
        .groupby(["tissue", "age_m", "gene_type", "theta_threshold"], as_index=False)
        .first()[["tissue", "age_m", "gene_type", "theta_threshold", "lambda", "corr_pearson_upper", "mean_pearson"]]
    )
    best.to_csv(args.out_dir / "best_lambda_by_group.csv", index=False)
    print(f"Wrote summary plots and table to {args.out_dir}")


if __name__ == "__main__":
    main()
