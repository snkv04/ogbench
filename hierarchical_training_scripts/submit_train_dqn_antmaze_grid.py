#!/usr/bin/env python3
"""Submit Slurm jobs for ``train_antmaze_highlevel_dqn`` over a hyperparameter grid.

The **grid** is a mapping from option name (tyro / CLI style) to a list of values; each point of
the Cartesian product becomes one job.

**Grid JSON** (``--grid-config``; defaults to ``hierarchical_training_scripts/grid_to_search.json``)::

    {
      "termination-time": [40, 50, 60],
      "max-episode-steps": [1200, 1600],
      "learning-rate": [1e-3, 3e-4]
    }

Keys may use hyphens or underscores and optional ``--`` prefix; boolean grid values use tyro-style
``--flag`` / ``--no-flag`` on the train command line.

Example::

    python hierarchical_training_scripts/submit_train_dqn_antmaze_grid.py --dry-run

    python hierarchical_training_scripts/submit_train_dqn_antmaze_grid.py \\
        --grid-config path/to/grid.json \\
        --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


# -----------------------------------------------------------------------------
# Defaults (keep in sync with hierarchical_training_scripts/train_dqn_antmaze.sh)
# -----------------------------------------------------------------------------

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_NODELISTS: list[str] = (
    [f"smblade24a{i}" for i in range(1, 7)] +
    [f"smblade16b{i}" for i in range(1, 9)] +
    [f"smblade16a{i}" for i in range(1, 15)]
)

DEFAULT_SLURM_ARGS: list[str] = [
    "#!/bin/bash",
    "#SBATCH --time=48:00:00",
    "#SBATCH --mem=100G",
    "#SBATCH --partition=compute",
    "#SBATCH --cpus-per-task=48",
]

# Train argv shared by every job except per-job grid tokens (see ``--fixed-args-file``).
DEFAULT_FIXED_TRAIN_ARGS: list[str] = [
    "--save-first-val-episodes-videos=4",
    "--no-validation-greedy",
    "--checkpoint-up=/home/svangaru/Desktop/ogbench/.ogbench/td3_runs/antmaze-arena__train_antmaze_lowlevel_td3__1__True__True__2026-04-07_02-05-17/checkpoints/checkpoint_step1950000.pt",
    "--checkpoint-down=/home/svangaru/Desktop/ogbench/.ogbench/td3_runs/antmaze-arena__train_antmaze_lowlevel_td3__1__True__True__2026-04-07_02-05-47/checkpoints/checkpoint_step1950000.pt",
    "--checkpoint-left=/home/svangaru/Desktop/ogbench/.ogbench/td3_runs/antmaze-arena__train_antmaze_lowlevel_td3__1__True__True__2026-04-07_02-06-36/checkpoints/checkpoint_step1100000.pt",
    "--checkpoint-right=/home/svangaru/Desktop/ogbench/.ogbench/td3_runs/antmaze-arena__train_antmaze_lowlevel_td3__1__True__True__2026-04-07_02-06-23/checkpoints/checkpoint_step1950000.pt",
    "--termination-time=50",
    "--max-episode-steps=2000",
    "--total-timesteps=2000000",
    "--track-with-wandb",
    "--reward-type=sparse",
    "--run-profiling",
]

# Short labels for Slurm job names (%x in log paths); unknown keys get a compact fallback.
_OPTION_ABBREV: dict[str, str] = {
    "termination-time": "tt",
    "max-episode-steps": "mes",
    "total-timesteps": "ts",
    "learning-rate": "lr",
    "reward-task-id": "rtid",
    "goal-radius": "gr",
}


def parse_fixed_args_file(path: Path) -> list[str]:
    """Load train CLI tokens from a text file (one per line; ``#`` starts a comment)."""
    tokens: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens.append(line)
    return tokens


def load_fixed_train_args(fixed_args_file: Path | None) -> list[str]:
    """Return the shared train arguments: either from file or built-in defaults."""
    if fixed_args_file is not None:
        return parse_fixed_args_file(fixed_args_file)
    return list(DEFAULT_FIXED_TRAIN_ARGS)


def resolve_repo_and_venv(repo_root: Path, venv_activate: Path | None) -> tuple[Path, Path]:
    """Absolute repo root and path to ``venv/bin/activate``."""
    root = repo_root.resolve()
    if venv_activate is not None:
        return root, venv_activate.resolve()
    return root, (root / "venv" / "bin" / "activate").resolve()


def normalize_option_key(key: str) -> str:
    """CLI option stem: strip ``--``, convert underscores to hyphens (tyro style)."""
    k = key.strip()
    if k.startswith("--"):
        k = k[2:]
    return k.replace("_", "-")


def parse_grid_mapping(raw: Mapping[str, Any]) -> dict[str, list[Any]]:
    """Validate and normalize user grid: ``option_name -> [v1, v2, ...]``."""
    if not isinstance(raw, Mapping):
        raise ValueError("Grid config must be a JSON object mapping strings to lists.")

    grid: dict[str, list[Any]] = {}
    for key, values in raw.items():
        opt = normalize_option_key(str(key))
        if not isinstance(values, list):
            raise ValueError(f"Grid entry {opt!r} must be a JSON array, got {type(values).__name__}.")
        if len(values) == 0:
            raise ValueError(f"Grid entry {opt!r} must be a non-empty list.")
        grid[opt] = values
    return grid


def load_grid_from_json_string(text: str) -> dict[str, list[Any]]:
    raw = json.loads(text)
    return parse_grid_mapping(raw)


def load_grid_from_json_file(path: Path) -> dict[str, list[Any]]:
    return load_grid_from_json_string(path.read_text())


def expand_grid_cartesian(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product: each item is one combo ``{option: single_value, ...}``.

    Key order in each combo follows sorted option names so generated bash scripts are stable.
    """
    if not grid:
        return [{}]

    keys = sorted(grid.keys())
    value_lists = [grid[k] for k in keys]
    combos: list[dict[str, Any]] = []
    for tup in itertools.product(*value_lists):
        combos.append(dict(zip(keys, tup)))
    return combos


def format_grid_cli_token(option_key: str, value: Any) -> str:
    """Single argv token for ``train_antmaze_highlevel_dqn`` (tyro conventions).

    Booleans become ``--flag`` or ``--no-flag``. Other scalars use ``--key=value`` with safe quoting
    for strings that would break bash word splitting.
    """
    k = normalize_option_key(option_key)

    if isinstance(value, bool):
        return f"--no-{k}" if not value else f"--{k}"

    if isinstance(value, str):
        rhs = value
        if any(ch in rhs for ch in " \t\n\"'$"):
            return f"--{k}={shlex.quote(rhs)}"
        return f"--{k}={rhs}"

    # int, float, and JSON numbers already decoded as int/float
    return f"--{k}={value}"


def combo_to_train_argv(combo: dict[str, Any]) -> list[str]:
    """Ordered argv fragments for one grid point."""
    return [format_grid_cli_token(k, combo[k]) for k in sorted(combo.keys())]


def _abbrev_option_key(key: str) -> str:
    if key in _OPTION_ABBREV:
        return _OPTION_ABBREV[key]
    parts = key.split("-")
    if parts:
        return "".join(p[:1] for p in parts if p) or key[:6]
    return key[:6]


def _abbrev_scalar_for_job_name(value: Any) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return str(value).replace(".", "p").replace("-", "m")
    s = str(value)
    return "".join(c if c.isalnum() else "_" for c in s)[:16]


def grid_job_name_from_combo(combo: dict[str, Any], *, prefix: str = "train_dqn_am", max_len: int = 64) -> str:
    """Slurm job name summarizing this grid point (truncated to ``max_len``)."""
    if not combo:
        name = prefix
        return name if len(name) <= max_len else name[:max_len]

    segments = [_abbrev_option_key(k) + _abbrev_scalar_for_job_name(combo[k]) for k in sorted(combo.keys())]
    body = "_".join(segments)
    name = f"{prefix}_{body}"
    return name if len(name) <= max_len else name[:max_len]


def run_name_from_combo(combo: dict[str, Any]) -> str:
    """Build a --run-name value encoding every searched hyperparameter."""
    parts = []
    for k in sorted(combo.keys()):
        key_slug = k.replace("-", "")
        val_slug = str(combo[k]).replace(".", "p")
        parts.append(f"{key_slug}{val_slug}")
    return "_".join(parts)


def build_train_argv(fixed_args: list[str], combo: dict[str, Any], extra_train_args: list[str]) -> list[str]:
    """Full argv for ``python -m ... train_antmaze_highlevel_dqn``."""
    run_name_arg = f"--run-name={run_name_from_combo(combo)}" if combo else ""
    return fixed_args + combo_to_train_argv(combo) + ([run_name_arg] if run_name_arg else []) + extra_train_args


def format_bash_python_invocation(train_args: list[str]) -> list[str]:
    """Lines for a bash script: ``python -m ... \\`` plus indented args with trailing ``\\`` except last."""
    header = ["python -m hierarchical_training_scripts.train_antmaze_highlevel_dqn \\"]
    body = []
    for i, arg in enumerate(train_args):
        continuation = " \\" if i < len(train_args) - 1 else ""
        body.append(f"    {arg}{continuation}")
    return header + body


def render_slurm_batch_script(
    *,
    job_name: str,
    output_dir: Path,
    nodelist: str,
    repo_root: Path,
    venv_activate: Path,
    train_args: list[str],
) -> str:
    """Full bash script body suitable for ``sbatch`` (includes shebang and #SBATCH directives)."""
    log_dir = output_dir.resolve()
    lines = [
        *DEFAULT_SLURM_ARGS,
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_dir}/%x_%j.out",
        f"#SBATCH --error={log_dir}/%x_%j.err",
        f"#SBATCH --nodelist={nodelist}",
        "",
        "export MUJOCO_GL=osmesa",
        f"cd {repo_root}",
        f"source {venv_activate}",
        *format_bash_python_invocation(train_args),
    ]
    return "\n".join(lines) + "\n"


def sbatch_from_stdin(script: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Submit ``script`` via sbatch stdin; avoids writing temporary .sh files on disk."""
    return subprocess.run(
        ["sbatch"],
        input=script,
        text=True,
        cwd=os.fspath(cwd),
        capture_output=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI definition kept in one place for readability."""
    p = argparse.ArgumentParser(
        description="Grid-submit Slurm jobs for train_antmaze_highlevel_dqn "
        "(Cartesian product of CLI options given as arg-name → list of values)."
    )
    p.add_argument(
        "--grid-config",
        type=Path,
        default=DEFAULT_REPO_ROOT / "hierarchical_training_scripts" / "grid_to_search.json",
        metavar="PATH",
        help="Path to JSON file: object mapping each train CLI option (string) to a JSON array of values.",
    )

    p.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repository root (working directory before launching training).",
    )
    p.add_argument(
        "--venv-activate",
        type=Path,
        default=None,
        help="Path to venv activate script; default: <repo-root>/venv/bin/activate",
    )
    p.add_argument(
        "--fixed-args-file",
        type=Path,
        default=None,
        help="Optional file: one train CLI token per line (# comments OK). Replaces built-in default fixed args.",
    )
    p.add_argument(
        "--extra-train-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Append extra args to every job (repeatable), e.g. --extra-train-arg --seed=2",
    )
    p.add_argument("--slurm-output-dir", type=Path, default=DEFAULT_REPO_ROOT / "slurm_logs")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print each generated batch script instead of calling sbatch.",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    repo_root, venv_activate = resolve_repo_and_venv(args.repo_root, args.venv_activate)
    fixed = load_fixed_train_args(args.fixed_args_file)

    try:
        grid = load_grid_from_json_file(args.grid_config.resolve())
    except (json.JSONDecodeError, ValueError, OSError) as e:
        print(f"Invalid grid configuration: {e}", file=sys.stderr)
        return 2

    combos = expand_grid_cartesian(grid)
    if not combos:
        print("No grid combinations produced.", file=sys.stderr)
        return 1

    nodelists = DEFAULT_NODELISTS
    for i, combo in enumerate(combos):
        train_args = build_train_argv(fixed, combo, args.extra_train_arg)
        job_name = grid_job_name_from_combo(combo)

        script = render_slurm_batch_script(
            job_name=job_name,
            output_dir=args.slurm_output_dir,
            nodelist=nodelists[i % len(nodelists)],
            repo_root=repo_root,
            venv_activate=venv_activate,
            train_args=train_args,
        )

        if args.dry_run:
            print("=" * 72)
            print(script, end="")
            continue

        proc = sbatch_from_stdin(script, cwd=repo_root)
        if proc.returncode != 0:
            print(proc.stderr or proc.stdout, file=sys.stderr)
            return proc.returncode
        print(proc.stdout.strip())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
