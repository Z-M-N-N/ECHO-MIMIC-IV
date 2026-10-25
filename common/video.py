"""Video decoding / frame sampling shared by the feature extractor."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def sample_indices(num_frames: int, count: int) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("video has no decodable frames")
    if count <= 0:
        raise ValueError("num_frames must be positive")
    if num_frames >= count:
        idx = np.linspace(0, num_frames - 1, count)
    else:
        idx = np.concatenate([np.arange(num_frames), np.full(count - num_frames, num_frames - 1)])
    return np.clip(np.round(idx).astype(np.int64), 0, num_frames - 1)


def decode_video(path: str | Path, num_frames: int) -> np.ndarray:
    """Return uint8 frames ``[T, H, W, 3]`` uniformly sampled over the clip."""
    path = str(path)
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(path, ctx=cpu(0), num_threads=1)
        idx = sample_indices(len(vr), num_frames)
        frames = vr.get_batch(idx).asnumpy()
    except Exception:  # pragma: no cover - fallback
        from torchvision.io import read_video

        vframes, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
        idx = torch.from_numpy(sample_indices(int(vframes.shape[0]), num_frames)).long()
        frames = vframes[idx].numpy()
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"unexpected decoded shape {frames.shape} for {path}")
    return np.ascontiguousarray(frames)


def resize_uint8_frames(frames: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(frames.shape[-2:]) == tuple(size):
        return frames
    out = torch.nn.functional.interpolate(
        frames.float(), size=tuple(size), mode="bilinear", align_corners=False, antialias=True
    )
    return out.round().clamp_(0, 255).to(torch.uint8)


class RawViewDataset(Dataset):
    """One view's videos: returns uint8 frames ``[T,3,H,W]`` and metadata."""

    def __init__(self, df, num_frames: int = 16, image_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.num_frames = int(num_frames)
        self.image_size = (int(image_size), int(image_size))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        rec = self.df.iloc[i]
        frames = decode_video(rec["video_path"], self.num_frames)
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        tensor = resize_uint8_frames(tensor, self.image_size)
        return {
            "frames": tensor,
            "study_id": int(rec["study_id"]),
            "subject_id": int(rec["subject_id"]),
            "split": str(rec["split"]),
            "video_path": str(rec["video_path"]),
        }


def collate_raw(items: list[dict]) -> dict:
    return {
        "frames": torch.stack([it["frames"] for it in items], dim=0),
        "study_id": [it["study_id"] for it in items],
        "subject_id": [it["subject_id"] for it in items],
        "split": [it["split"] for it in items],
        "video_path": [it["video_path"] for it in items],
    }
