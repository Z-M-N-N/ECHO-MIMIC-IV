"""Seeding, config, run-directory, logging and checkpoint helpers."""
from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42, deterministic: bool = False) -> int:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    return seed


def worker_init_fn(worker_id: int) -> None:
    """Make DataLoader workers deterministic but decorrelated."""
    base = torch.initial_seed() % 2**32
    np.random.seed((base + worker_id) % 2**32)
    random.seed((base + worker_id) % 2**32)


# ---------------------------------------------------------------------------
# Config / JSON
# ---------------------------------------------------------------------------
def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def dump_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=indent, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def git_info(repo_root: str | Path) -> dict:
    repo_root = Path(repo_root)

    def run(*cmd: str) -> str:
        try:
            return subprocess.check_output(cmd, cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:  # noqa: BLE001
            return ""

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
    }


def environment_snapshot() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Run directories (timestamped, never overwrite previous results)
# ---------------------------------------------------------------------------
def _atomic_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    tmp = link.with_name(f".{link.name}.tmp{os.getpid()}")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(os.path.relpath(target, link.parent), tmp)
    os.replace(tmp, link)


def make_run_dir(root: str | Path, group: str, run_name: str, timestamp: str | None = None) -> Path:
    root = Path(root)
    ts = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = root / group / f"{ts}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic_symlink(run_dir, root / group / "latest")
    return run_dir


def link_latest_files(run_dir: Path, filenames: list[str]) -> None:
    """Expose ``<group>/<file>`` symlinks to the latest run for the spec layout."""
    group_dir = run_dir.parent
    for name in filenames:
        src = run_dir / name
        if src.exists():
            _atomic_symlink(src, group_dir / name)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else float("nan")


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-5, mode: str = "min") -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best: float | None = None
        self.best_epoch: int = -1
        self.num_bad: int = 0
        self.should_stop: bool = False

    def step(self, value: float, epoch: int) -> bool:
        """Return True if ``value`` is a new best."""
        improved = False
        if self.best is None:
            improved = True
        elif self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta
        if improved:
            self.best = value
            self.best_epoch = epoch
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.patience > 0 and self.num_bad >= self.patience:
                self.should_stop = True
        return improved


class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    def __str__(self) -> str:
        s = int(self.elapsed)
        return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def append_experiment_log(project_root: str | Path, title: str, fields: dict[str, str]) -> Path:
    """Append one section to ``EXPERIMENT_LOG.md`` using the AGENTS.md template."""
    path = Path(project_root) / "EXPERIMENT_LOG.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    labels = {
        "purpose": "目的",
        "command": "执行命令",
        "params": "关键参数",
        "input": "输入",
        "output": "输出",
        "result": "结果/指标",
        "conclusion": "结论与备注",
    }
    lines = [f"\n## [{ts}] {title}\n"]
    for key, label in labels.items():
        value = str(fields.get(key, "")).strip()
        if "\n" in value:
            lines.append(f"- {label}：\n```\n{value}\n```")
        else:
            lines.append(f"- {label}：{value}")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path
