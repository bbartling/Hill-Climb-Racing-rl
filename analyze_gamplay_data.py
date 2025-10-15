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
    df["angle"] = pd.to_numeric(df["angle"], errors="coerce")
    df["height_px"] = pd.to_numeric(df["height_px"], errors="coerce")
    df.dropna(subset=["angle", "height_px"], inplace=True)

    if df.empty:
        print("No valid rows after cleaning — check your CSV.")
        return

    # Action label: 0=Coast, 1=Gas, 2=Brake (kept as strings for plotting)
    df["action"] = pd.Categorical(
        np.select(
            [df["gas_pressed"] == 1, df["brake_pressed"] == 1],
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
    g = sns.pairplot(
        df, vars=["angle", "height_px"], hue="action", diag_kind="kde", corner=True
    )
    g.figure.suptitle("Gameplay Analysis: Angle & Height by Action", y=1.02)
    g.figure.savefig(
        out_dir / "pairplot_angle_height_by_action.png", dpi=300, bbox_inches="tight"
    )
    plt.close(g.figure)

    # ---------- Correlation heatmap ----------
    fig_hm, ax_hm = plt.subplots(figsize=(6, 4))
    corr = df[["angle", "height_px", "gas_pressed", "brake_pressed"]].corr()
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

    print(f"\n✅ Saved all analysis plots to: {out_dir}")

    if show:
        # If you still want to eyeball the last figure interactively:
        # (No-op here since we closed figures after saving.)
        print("Set --no-show to suppress GUI windows while saving plots.")
        # You can re-open any image file from out_dir if needed.


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
