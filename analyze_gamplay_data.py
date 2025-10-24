# analyze_gameplay_data.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_gameplay_data(csv_path: Path, show: bool = True):
    """Loads, cleans, and visualizes gameplay data from a CSV file."""
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return

    # ---------- Load & clean ----------
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df.replace("", pd.NA, inplace=True)

    # Coerce numerics (now includes ground_slope)
    for col in ["angle", "height_px", "ground_slope", "gas_pressed", "brake_pressed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Require slope if present; otherwise fall back gracefully
    subset_cols = ["angle", "height_px"]
    if "ground_slope" in df.columns:
        subset_cols.append("ground_slope")

    df.dropna(subset=subset_cols, inplace=True)

    if df.empty:
        print("No valid rows after cleaning — check your CSV.")
        return

    # Action label: 0=Coast, 1=Gas, 2=Brake (kept as strings for plotting)
    df["action"] = pd.Categorical(
        np.select(
            [df.get("gas_pressed", 0) == 1, df.get("brake_pressed", 0) == 1],
            ["Gas", "Brake"],
            default="Coast",
        )
    )

    # ---------- Stats ----------
    print("\n--- Data Overview ---")
    print(df.describe())
    print("\n--- Action Distribution ---")
    print(df["action"].value_counts(normalize=True).round(3))

    # ---------- Output directory ----------
    out_dir = csv_path.parent / "game_play_data_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    # ---------- Pairplot ----------
    pair_vars = ["angle", "height_px"]
    if "ground_slope" in df.columns:
        pair_vars.append("ground_slope")

    g = sns.pairplot(df, vars=pair_vars, hue="action", diag_kind="kde", corner=True)
    title = (
        "Gameplay Analysis: "
        + ", ".join(v.title().replace("_", " ") for v in pair_vars)
        + " by Action"
    )
    g.figure.suptitle(title, y=1.02)
    g.figure.savefig(
        out_dir / f"pairplot_{'_'.join(pair_vars)}_by_action.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(g.figure)

    # ---------- Correlation heatmap ----------
    fig_hm, ax_hm = plt.subplots(figsize=(6, 4))
    corr_cols = ["angle", "height_px"]
    if "ground_slope" in df.columns:
        corr_cols.append("ground_slope")
    # Controls if present
    for c in ["gas_pressed", "brake_pressed"]:
        if c in df.columns:
            corr_cols.append(c)

    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_hm)
    ax_hm.set_title("Correlation Heatmap: Features vs Controls")
    fig_hm.tight_layout()
    fig_hm.savefig(out_dir / "correlation_heatmap.png", dpi=300)
    plt.close(fig_hm)

    # ---------- Joint density (angle vs height) ----------
    j = sns.jointplot(
        data=df,
        x="angle",
        y="height_px",
        hue="action",
        kind="kde",
        fill=True,
        alpha=0.5,
    )
    j.figure.suptitle("Angle vs Height Density by Action", y=1.03)
    j.figure.savefig(
        out_dir / "joint_density_angle_vs_height.png", dpi=300, bbox_inches="tight"
    )
    plt.close(j.figure)

    # Optional: joint density (angle vs ground_slope), only if available
    if "ground_slope" in df.columns:
        j2 = sns.jointplot(
            data=df,
            x="angle",
            y="ground_slope",
            hue="action",
            kind="kde",
            fill=True,
            alpha=0.5,
        )
        j2.figure.suptitle("Angle vs Ground Slope Density by Action", y=1.03)
        j2.figure.savefig(
            out_dir / "joint_density_angle_vs_ground_slope.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(j2.figure)

    print(f"\n✅ Saved all analysis plots to: {out_dir}")

    if show:
        print("Set --no-show to suppress GUI windows while saving plots.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze HCR gameplay data and save plots."
    )
    parser.add_argument("csv_file", type=Path, help="Path to manual_play_data.csv")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display GUI windows; only save plots",
    )
    args = parser.parse_args()

    analyze_gameplay_data(args.csv_file, show=not args.no_show)
