"""
Heatmap analysis of WandB antmaze high-level DQN runs over
(reward-task-id) x (goal-radius) grid.

Group names are derived from submit_train_dqn_antmaze_grid.py::run_name_from_combo.
For each (task_id, goal_radius) combo we find the seed-42 run and average the
last 500k training steps of avg_episode_return and success_rate.

Usage:
    python visualization/analyze_wandb_antmaze_hl.py
    python visualization/analyze_wandb_antmaze_hl.py --out heatmap.png
    python visualization/analyze_wandb_antmaze_hl.py --entity my_wandb_user
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

VIZ_DIR = Path(__file__).resolve().parent
REPO_ROOT = VIZ_DIR.parent
GRID_CONFIG = REPO_ROOT / "hierarchical_training_scripts" / "grid_to_search.json"

PROJECT = "dqn_antmaze_hierarchical"
SEED = 42
TOTAL_TIMESTEPS = 2_000_000
LAST_N_STEPS = 500_000

METRICS = [
    "train/avg_episode_return",
    "train/success_rate",
]


# ---------------------------------------------------------------------------
# Grid loading + group name construction (mirrors submit_train_dqn_antmaze_grid.py)
# ---------------------------------------------------------------------------

def load_grid(path: Path = GRID_CONFIG) -> dict[str, list]:
    raw = json.loads(path.read_text())
    return {k.replace("_", "-").lstrip("-"): v for k, v in raw.items()}


def expand_grid_cartesian(grid: dict[str, list]) -> list[dict]:
    keys = sorted(grid.keys())
    return [dict(zip(keys, tup)) for tup in itertools.product(*[grid[k] for k in keys])]


def run_name_from_combo(combo: dict) -> str:
    parts = []
    for k in sorted(combo.keys()):
        key_slug = k.replace("-", "")
        val_slug = str(combo[k]).replace(".", "p")
        parts.append(f"{key_slug}{val_slug}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# WandB helpers
# ---------------------------------------------------------------------------

def get_api(entity: str | None) -> tuple[wandb.Api, str]:
    api = wandb.Api()
    entity = entity or api.default_entity
    if entity is None:
        sys.exit("Could not determine wandb entity. Pass --entity <username>.")
    return api, entity


def fetch_group_runs(api: wandb.Api, entity: str, group_name: str) -> list:
    """Return all runs whose group contains group_name as a substring.

    If the combo was submitted multiple times (different timestamps), only runs
    from the earliest matching group are returned.
    """
    path = f"{entity}/{PROJECT}"
    runs = list(api.runs(path, filters={"group": {"$regex": group_name}}))
    if not runs:
        print(f"  [warn] No runs found for group matching {group_name!r}")
        return []

    # Pick the latest group name in case of multiple submissions
    latest_group = max({r.group for r in runs if r.group}, key=lambda g: g)
    runs = [r for r in runs if r.group == latest_group]
    print(f"  Using group {latest_group!r} ({len(runs)} run(s))")
    return runs


def last_n_step_mean(run, metric: str) -> float:
    """Average `metric` over the last LAST_N_STEPS global steps for one run."""
    history = run.history(keys=["_step", metric], pandas=True)
    if history.empty or metric not in history.columns:
        return float("nan")
    threshold = TOTAL_TIMESTEPS - LAST_N_STEPS
    subset = history[history["_step"] >= threshold].dropna(subset=[metric])
    if subset.empty:
        return float("nan")
    return float(subset[metric].mean())


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def build_heatmap_data(
    api: wandb.Api, entity: str
) -> tuple[dict[str, dict[str, np.ndarray]], str, list, str, list]:
    """Returns (data, row_key, row_vals, col_key, col_vals).

    data[metric] = {"mean": 2D array, "std": 2D array} shaped (n_rows, n_cols).
    Axes come from the first two grid keys with >1 value (JSON file order).
    Keys with exactly 1 value are held fixed and folded into every group name.
    """
    grid = load_grid()

    varying = [(k, v) for k, v in grid.items() if len(v) > 1]
    fixed   = {k: v[0] for k, v in grid.items() if len(v) == 1}

    if len(varying) < 2:
        sys.exit("Need at least 2 keys with multiple values in grid for a 2D heatmap.")

    row_key, row_vals = varying[0]
    col_key, col_vals = varying[1]
    n_rows, n_cols = len(row_vals), len(col_vals)

    means = {m: np.full((n_rows, n_cols), np.nan) for m in METRICS}
    stds  = {m: np.full((n_rows, n_cols), np.nan) for m in METRICS}

    all_groups = [
        run_name_from_combo({**fixed, row_key: r, col_key: c})
        for r in row_vals for c in col_vals
    ]
    print(f"Target groups ({len(all_groups)}):")
    for g in all_groups:
        print(f"  {g}")
    print()

    for i, row_val in enumerate(row_vals):
        for j, col_val in enumerate(col_vals):
            combo = {**fixed, row_key: row_val, col_key: col_val}
            group = run_name_from_combo(combo)
            print(f"Fetching group={group!r} ...")
            runs = fetch_group_runs(api, entity, group)
            if not runs:
                continue
            for metric in METRICS:
                per_run_vals = [last_n_step_mean(r, metric) for r in runs]
                per_run_vals = [v for v in per_run_vals if not np.isnan(v)]
                if not per_run_vals:
                    continue
                m = float(np.mean(per_run_vals))
                s = float(np.std(per_run_vals)) if len(per_run_vals) > 1 else 0.0
                means[metric][i, j] = m
                stds[metric][i, j] = s
                print(f"  {metric}: {m:.4f} ± {s:.4f}  (n={len(per_run_vals)})")

    return (
        {m: {"mean": means[m], "std": stds[m]} for m in METRICS},
        row_key, row_vals,
        col_key, col_vals,
    )


# ---------------------------------------------------------------------------
# Heatmap plotting
# ---------------------------------------------------------------------------

def _cell_text_color(v: float, vmin: float, vmax: float) -> str:
    if np.isnan(v):
        return "black"
    mid = (vmin + vmax) / 2
    return "white" if v < mid else "black"


def plot_heatmaps(
    data: dict[str, dict[str, np.ndarray]],
    out_path: str | None,
    row_key: str,
    row_vals: list,
    col_key: str,
    col_vals: list,
) -> None:
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    col_labels = [str(v) for v in col_vals]
    row_labels = [str(v) for v in row_vals]

    for ax, metric in zip(axes, METRICS):
        mean_arr = data[metric]["mean"]
        std_arr  = data[metric]["std"]
        vmin = float(np.nanmin(mean_arr)) if not np.all(np.isnan(mean_arr)) else 0.0
        vmax = float(np.nanmax(mean_arr)) if not np.all(np.isnan(mean_arr)) else 1.0

        im = ax.imshow(mean_arr, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)

        ax.set_xticks(range(len(col_vals)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(row_vals)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel(col_key)
        ax.set_ylabel(row_key)
        ax.set_title(metric.replace("/", "\n"))

        for i in range(mean_arr.shape[0]):
            for j in range(mean_arr.shape[1]):
                m = mean_arr[i, j]
                s = std_arr[i, j]
                if np.isnan(m):
                    text = "N/A"
                elif np.isnan(s) or s == 0.0:
                    text = f"{m:.3f}"
                else:
                    text = f"{m:.3f}\n±{s:.3f}"
                color = _cell_text_color(m, vmin, vmax)
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    fig.suptitle(
        f"Mean ± std over last {LAST_N_STEPS:,} training steps (all seeds)",
        fontsize=12,
    )
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", default=None, help="WandB entity. Defaults to logged-in user.")
    p.add_argument("--out", default=str(VIZ_DIR / "heatmap_antmaze_hl.png"), help="Output path for the heatmap figure.")
    p.add_argument("--list", action="store_true", help="List all groups in the project and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api, entity = get_api(args.entity)

    if args.list:
        from collections import Counter
        path = f"{entity}/{PROJECT}"
        print(f"Fetching all runs in {path} ...")
        all_runs = list(api.runs(path))
        counts = Counter(r.group for r in all_runs if r.group)
        groups = sorted(counts)
        print(f"\n{'Group':<60}  {'# runs':>7}")
        print("-" * 70)
        for g in groups:
            print(f"{g:<60}  {counts[g]:>7}")
        ungrouped = [r.name for r in all_runs if not r.group]
        if ungrouped:
            print(f"\nUngrouped runs ({len(ungrouped)}): {ungrouped[:10]}")
        return

    data, row_key, row_vals, col_key, col_vals = build_heatmap_data(api, entity)
    plot_heatmaps(data, out_path=args.out, row_key=row_key, row_vals=row_vals, col_key=col_key, col_vals=col_vals)


if __name__ == "__main__":
    main()
