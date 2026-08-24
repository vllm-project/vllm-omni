from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torchvision

from vllm_omni.diffusion.models.internvla_a1.config import (  # noqa: E402
    OBS_IMAGES,
    OBS_STATE,
    OBS_TASK,
    InternVLAA1Config,
)


def tensor_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float32


def select_indices(dataset, num_samples: int) -> list[int]:
    indices: list[int] = []
    for _, episode_indices in dataset.episode_start_indices(max_episodes=num_samples):
        if episode_indices:
            indices.append(episode_indices[0])
    return indices[:num_samples]


def make_shared_noise(seed: int, sample_index: int, shape: tuple[int, ...], device: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + sample_index)
    noise = torch.randn(shape, generator=generator, dtype=torch.float32)
    return noise.to(device=device)


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _clamp_index(index: int, start: int, end: int) -> int:
    return max(start, min(end - 1, index))


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _stack_stats(stats: dict[str, Any], keys: list[str]) -> dict[str, torch.Tensor]:
    result = {}
    for stat_name in ("mean", "std"):
        values = []
        for key in keys:
            values.extend(stats[key][stat_name])
        result[stat_name] = torch.tensor(values, dtype=torch.float32)
    return result


def normalize_vector(values: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    denom = torch.where(stats["std"] == 0, torch.ones_like(stats["std"]), stats["std"])
    return (values - stats["mean"]) / denom


def unnormalize_vector(values: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    return values * stats["std"] + stats["mean"]


class TorchvisionVideoReaderCache:
    def __init__(self, backend: str = "pyav") -> None:
        self.backend = backend
        self._readers: dict[str, Any] = {}
        torchvision.set_video_backend(backend)

    def get(self, path: str) -> Any:
        reader = self._readers.get(path)
        if reader is None:
            reader = torchvision.io.VideoReader(path, "video")
            self._readers[path] = reader
        return reader

    def decode_frames(self, path: str, timestamps: list[float], tolerance_s: float = 1e-4) -> torch.Tensor:
        reader = self.get(path)
        first_ts = min(timestamps)
        last_ts = max(timestamps)
        reader.seek(first_ts, keyframes_only=self.backend == "pyav")

        loaded_frames: list[torch.Tensor] = []
        loaded_ts: list[float] = []
        for frame in reader:
            current_ts = float(frame["pts"])
            loaded_frames.append(frame["data"])
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break

        query_ts = torch.tensor(timestamps, dtype=torch.float32)
        loaded_ts_tensor = torch.tensor(loaded_ts, dtype=torch.float32)
        distances = torch.cdist(query_ts[:, None], loaded_ts_tensor[:, None], p=1)
        min_dist, argmin = distances.min(dim=1)
        if not torch.all(min_dist < tolerance_s):
            raise RuntimeError(
                f"Video timestamps are outside tolerance: query={query_ts.tolist()} "
                f"loaded={loaded_ts_tensor.tolist()} path={path}"
            )
        return torch.stack([loaded_frames[i] for i in argmin]).float() / 255.0


@dataclass
class A2DOpenLoopSample:
    index: int
    episode_index: int
    task: str
    state_raw: torch.Tensor
    action_raw: torch.Tensor
    inputs: dict[str, torch.Tensor]


class A2DOpenLoopDataset:
    image_keys = [
        "observation.images.head",
        "observation.images.hand_left",
        "observation.images.hand_right",
    ]
    state_keys = [
        "observation.states.joint.position",
        "observation.states.effector.position",
    ]
    action_keys = [
        "actions.joint.position",
        "actions.effector.position",
    ]

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        config: InternVLAA1Config,
        train_stats: dict[str, Any],
        image_offsets: tuple[int, int] = (-15, 0),
        tolerance_s: float = 1e-4,
    ) -> None:
        self.root = Path(dataset_root)
        self.config = config
        self.info = _load_json(self.root / "meta" / "info.json")
        self.dataset_stats = _load_json(self.root / "meta" / "stats.json")
        self.data_rows = _load_parquet_rows(self.root / "data" / "chunk-000" / "file-000.parquet")
        self.episode_rows = _load_parquet_rows(self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        self.task_rows = _load_parquet_rows(self.root / "meta" / "tasks.parquet")

        self.state_stats = _stack_stats(train_stats, self.state_keys)
        self.action_stats = _stack_stats(train_stats, self.action_keys)
        self.image_offsets = image_offsets
        self.tolerance_s = tolerance_s
        self.video_reader = TorchvisionVideoReaderCache(backend="pyav")

    @property
    def num_episodes(self) -> int:
        return len(self.episode_rows)

    @property
    def physical_action_dim(self) -> int:
        return 16

    def episode_start_indices(self, max_episodes: int | None = None) -> list[tuple[int, list[int]]]:
        rows = self.episode_rows if max_episodes is None else self.episode_rows[:max_episodes]
        result = []
        for ep in rows:
            start = int(ep["dataset_from_index"])
            end = int(ep["dataset_to_index"])
            result.append((int(ep["episode_index"]), list(range(start, end, self.config.chunk_size))))
        return result

    def _task_text(self, task_index: int) -> str:
        return self.task_rows[task_index]["__index_level_0__"]

    def _episode_for_index(self, row: dict[str, Any]) -> dict[str, Any]:
        return self.episode_rows[int(row["episode_index"])]

    def _state_vector(self, row: dict[str, Any]) -> torch.Tensor:
        return torch.tensor(row[self.state_keys[0]] + row[self.state_keys[1]], dtype=torch.float32)

    def _action_vector(self, row: dict[str, Any]) -> torch.Tensor:
        return torch.tensor(row[self.action_keys[0]] + row[self.action_keys[1]], dtype=torch.float32)

    def _query_rows(self, idx: int, deltas: list[int]) -> list[dict[str, Any]]:
        row = self.data_rows[idx]
        episode = self._episode_for_index(row)
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        return [self.data_rows[_clamp_index(idx + delta, start, end)] for delta in deltas]

    def _decode_camera_history(
        self, episode: dict[str, Any], camera_key: str, rows: list[dict[str, Any]]
    ) -> torch.Tensor:
        timestamps = [float(r["timestamp"]) for r in rows]
        shifted = [float(episode[f"videos/{camera_key}/from_timestamp"]) + ts for ts in timestamps]
        chunk_idx = int(episode[f"videos/{camera_key}/chunk_index"])
        file_idx = int(episode[f"videos/{camera_key}/file_index"])
        path = self.root / self.info["video_path"].format(
            video_key=camera_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        frames = self.video_reader.decode_frames(str(path), shifted, tolerance_s=self.tolerance_s)
        return frames

    def get_sample(self, idx: int) -> A2DOpenLoopSample:
        row = self.data_rows[idx]
        episode = self._episode_for_index(row)
        image_rows = self._query_rows(idx, list((-15, 0)))
        action_rows = self._query_rows(idx, list(range(self.config.chunk_size)))
        camera_images = [self._decode_camera_history(episode, camera_key, image_rows) for camera_key in self.image_keys]
        state_raw = self._state_vector(row)
        state_norm = normalize_vector(state_raw, self.state_stats)
        action_raw = torch.stack([self._action_vector(action_row) for action_row in action_rows], dim=0)
        task = self._task_text(int(row["task_index"]))
        inputs = {
            OBS_STATE: state_norm,
            OBS_TASK: task,
            f"{OBS_IMAGES}.image0": camera_images[0],
            f"{OBS_IMAGES}.image1": camera_images[1],
            f"{OBS_IMAGES}.image2": camera_images[2],
            f"{OBS_IMAGES}.image0_mask": torch.tensor(True),
            f"{OBS_IMAGES}.image1_mask": torch.tensor(True),
            f"{OBS_IMAGES}.image2_mask": torch.tensor(True),
        }
        return A2DOpenLoopSample(
            index=idx,
            episode_index=int(row["episode_index"]),
            task=task,
            state_raw=state_raw,
            action_raw=action_raw,
            inputs=inputs,
        )


def collate_open_loop_samples(
    samples: list[A2DOpenLoopSample],
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    first = samples[0]
    batch_inputs: dict[str, torch.Tensor] = {}
    for key in first.inputs:
        values = [sample.inputs[key] for sample in samples]
        if isinstance(values[0], torch.Tensor):
            tensor = torch.stack(values, dim=0)
            if tensor.dtype in (torch.int64, torch.bool):
                batch_inputs[key] = tensor.to(device=device)
            else:
                batch_inputs[key] = tensor.to(device=device, dtype=dtype)
        else:
            batch_inputs[key] = values

    metadata = {
        "indices": [sample.index for sample in samples],
        "episode_indices": [sample.episode_index for sample in samples],
        "tasks": [sample.task for sample in samples],
        "state_raw": torch.stack([sample.state_raw for sample in samples], dim=0),
        "action_raw": torch.stack([sample.action_raw for sample in samples], dim=0),
    }
    return batch_inputs, metadata


def plot_prediction_series(
    *,
    series: dict[str, np.ndarray],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    first_series = next(iter(series.values()))
    num_dims = int(first_series.shape[1])
    num_cols = 2
    num_rows = math.ceil(num_dims / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, max(6, 3 * num_rows)))
    axs = np.array(axs).reshape(-1)
    x_values = np.arange(first_series.shape[0])
    styles = [
        ("Ground Truth", "blue", "-", 1.5),
        ("VLLM", "green", "-.", 1.4),
        ("Predicted", "red", "--", 1.4),
    ]
    style_map = {name: (color, linestyle, linewidth) for name, color, linestyle, linewidth in styles}

    for dim in range(num_dims):
        ax = axs[dim]
        for name, values in series.items():
            color, linestyle, linewidth = style_map.get(name, (None, "-", 1.2))
            ax.plot(
                x_values,
                values[:, dim],
                label=name,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
        ax.set_title(f"Dimension {dim + 1}")
        ax.set_xlabel("Time Step / Sample Index")
        ax.set_ylabel(f"Value Dim {dim + 1}")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.7)

    for dim in range(num_dims, len(axs)):
        axs[dim].axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)


def summarize_prediction_metrics(
    *,
    gt_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    joint_dims: int = 14,
) -> dict[str, float]:
    return {
        "mse": float(F.mse_loss(gt_tensor, pred_tensor, reduction="mean").item()),
        "mae": float(F.l1_loss(gt_tensor, pred_tensor, reduction="mean").item()),
        "mse_joint": float(F.mse_loss(gt_tensor[:, :joint_dims], pred_tensor[:, :joint_dims], reduction="mean").item()),
        "mae_joint": float(F.l1_loss(gt_tensor[:, :joint_dims], pred_tensor[:, :joint_dims], reduction="mean").item()),
        "mse_gripper": float(
            F.mse_loss(gt_tensor[:, joint_dims:], pred_tensor[:, joint_dims:], reduction="mean").item()
        ),
        "mae_gripper": float(
            F.l1_loss(gt_tensor[:, joint_dims:], pred_tensor[:, joint_dims:], reduction="mean").item()
        ),
    }


def summarize_metric_list(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def run_open_loop_evaluation(
    *,
    mode: str,
    policy,
    config,
    dataset,
    train_meta,
    collate_samples: Callable[..., tuple[dict[str, torch.Tensor], dict[str, Any]]],
    run_sample_actions: Callable[[Any, dict[str, torch.Tensor], torch.Tensor], torch.Tensor],
    num_episodes: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
    output_dir: str | Path,
    skip_plots: bool,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    mse_values: list[float] = []
    mae_values: list[float] = []
    mse_joint_values: list[float] = []
    mae_joint_values: list[float] = []
    mse_gripper_values: list[float] = []
    mae_gripper_values: list[float] = []
    per_episode: list[dict[str, Any]] = []

    plotting_available = True
    if not skip_plots:
        try:
            import matplotlib.pyplot as _plt  # noqa: F401
        except ModuleNotFoundError:
            plotting_available = False
            print("[warning] matplotlib is not installed; plots will be skipped.")

    episode_specs = dataset.episode_start_indices(max_episodes=num_episodes)
    for episode_index, indices in episode_specs:
        print(f"[{mode}] episode: {episode_index}")
        pred_chunks = []
        gt_chunks = []
        visited_indices = []
        task = None

        for index in indices:
            sample = dataset.get_sample(index)
            task = sample.task
            batch_inputs, meta = collate_samples([sample], device=device, dtype=dtype)
            noise = make_shared_noise(
                seed,
                index,
                (batch_inputs[OBS_STATE].shape[0], config.chunk_size, config.max_action_dim),
                device,
            )
            with torch.no_grad():
                pred = run_sample_actions(policy, batch_inputs, noise)

            pred = pred[:, :, : dataset.physical_action_dim].to(torch.float32).cpu()
            pred_phys = unnormalize_vector(pred, dataset.action_stats)
            if train_meta.action_mode == "delta":
                pred_phys[:, :, :14] += meta["state_raw"][:, None, :14]

            gt_phys = meta["action_raw"][:, :, : dataset.physical_action_dim].to(torch.float32).cpu()
            pred_chunks.append(pred_phys[0])
            gt_chunks.append(gt_phys[0])
            visited_indices.append(index)

        pred_tensor = torch.cat(pred_chunks, dim=0)
        gt_tensor = torch.cat(gt_chunks, dim=0)
        metrics = summarize_prediction_metrics(gt_tensor=gt_tensor, pred_tensor=pred_tensor)

        mse_values.append(metrics["mse"])
        mae_values.append(metrics["mae"])
        mse_joint_values.append(metrics["mse_joint"])
        mae_joint_values.append(metrics["mae_joint"])
        mse_gripper_values.append(metrics["mse_gripper"])
        mae_gripper_values.append(metrics["mae_gripper"])

        if not skip_plots and plotting_available:
            plot_prediction_series(
                series={
                    "Ground Truth": gt_tensor.numpy(),
                    "VLLM": pred_tensor.numpy(),
                },
                output_path=plots_dir / f"{mode}_open_loop_ep{episode_index}.jpg",
                title=f"{mode} Ground Truth vs VLLM",
            )

        episode_log = {
            "episode_id": int(episode_index),
            "task": task,
            "visited_indices": visited_indices,
            "num_pred_steps": int(pred_tensor.shape[0]),
            **metrics,
        }
        per_episode.append(episode_log)
        print(json.dumps({"mode": mode, **episode_log}, ensure_ascii=False))

    summary = {
        "mode": mode,
        "num_episodes": len(per_episode),
        "mse": mse_values,
        "mae": mae_values,
        "average_mse": summarize_metric_list(mse_values),
        "average_mae": summarize_metric_list(mae_values),
        "mse_joint": mse_joint_values,
        "mae_joint": mae_joint_values,
        "average_mse_joint": summarize_metric_list(mse_joint_values),
        "average_mae_joint": summarize_metric_list(mae_joint_values),
        "mse_gripper": mse_gripper_values,
        "mae_gripper": mae_gripper_values,
        "average_mse_gripper": summarize_metric_list(mse_gripper_values),
        "average_mae_gripper": summarize_metric_list(mae_gripper_values),
        "episodes": per_episode,
    }
    with open(output_dir / "log.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
