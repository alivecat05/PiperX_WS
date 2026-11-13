"""Small LeRobot v3 Parquet/MP4 reader compatible with the upstream Python 3.9 envs."""
from collections import OrderedDict
from functools import lru_cache
import json
import os
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from .frame_cache import letterbox_rgb

CAMERAS = ("head", "left_wrist", "right_wrist")


def read_episode(root, episode, columns=None):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    path = root / info["data_path"].format(chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"])
    table = pq.read_table(path, columns=columns)
    # The converter uses one episode per file. Reject other layouts explicitly.
    if table.num_rows != episode["length"]:
        raise ValueError("This adapter requires one episode per Parquet file")
    return table


def load_arrays(root, episode):
    columns = ["action", "observation.state", "quality.sample_valid", "quality.action_valid", "quality.state_valid"]
    table = read_episode(root, episode, columns)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    valid = np.asarray(table["quality.sample_valid"].to_pylist(), dtype=bool)
    valid &= np.asarray(table["quality.action_valid"].to_pylist(), dtype=bool)
    valid &= np.asarray(table["quality.state_valid"].to_pylist(), dtype=bool)
    if not np.isfinite(state[valid]).all() or not np.isfinite(action[valid]).all():
        raise ValueError("Non-finite training vectors")
    return state, action, valid


def valid_runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def metadata(root):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    splits = json.loads((root / "meta/splits.json").read_text())
    episodes = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        episodes.extend(pq.read_table(path).to_pylist())
    episodes.sort(key=lambda e: e["episode_index"])
    if [e["episode_index"] for e in episodes] != list(range(info["total_episodes"])):
        raise ValueError("Episode metadata must be contiguous and zero based")
    if splits.get("val") or splits.get("validation") or splits.get("test"):
        raise ValueError("This corpus is training-only; holdout splits are not configured")
    train, excluded = (set(splits[k]) for k in ("train", "excluded"))
    if train & excluded or train | excluded != set(range(len(episodes))):
        raise ValueError("Invalid training/exclusion partition")
    exclusions_path = root / "meta/training_exclusions.json"
    if exclusions_path.exists():
        denied = set(json.loads(exclusions_path.read_text())["source_episodes"])
        leaked = [episodes[i]["source_episode"] for i in train if episodes[i]["source_episode"] in denied]
        if leaked:
            raise ValueError(f"Explicitly excluded source episodes in training: {leaked}")
    return info, splits, episodes


@lru_cache(maxsize=4)
def training_arrays(root):
    info, splits, episodes = metadata(root)
    states, actions = [], []
    for ep_index in splits["train"]:
        state, action, valid = load_arrays(root, episodes[ep_index])
        states.append(state[valid]); actions.append(action[valid])
    return np.concatenate(states), np.concatenate(actions)


class VideoFrames:
    """Per-process bounded decoder cache; select exact frames by PTS after keyframe seek."""
    def __init__(self, fps, image_size, capacity=9, return_uint8=False):
        self.fps, self.image_size, self.capacity = fps, tuple(image_size), capacity
        self.return_uint8=return_uint8
        self._pid = os.getpid()
        self._cache = OrderedDict()

    def close(self):
        for container, stream in self._cache.values():
            container.close()
        self._cache.clear()

    def __getstate__(self):
        state = self.__dict__.copy(); state["_cache"] = OrderedDict()
        return state

    def read(self, path, indices, timestamp_offset=0.):
        if os.getpid() != self._pid:
            self.close(); self._pid = os.getpid()
        path = str(path)
        if path not in self._cache:
            container = av.open(path)
            stream = container.streams.video[0]
            stream.thread_count = 1
            self._cache[path] = container, stream
            if len(self._cache) > self.capacity:
                old, _ = self._cache.popitem(last=False)[1]; old.close()
        self._cache.move_to_end(path)
        container, stream = self._cache[path]
        wanted = set(int(i) for i in indices)
        first, last = min(wanted), max(wanted)
        target_s = timestamp_offset + first / self.fps
        container.seek(int(target_s / float(stream.time_base)), stream=stream, backward=True, any_frame=False)
        frames = {}
        height, width = self.image_size
        for frame in container.decode(stream):
            actual_s = float(frame.pts * stream.time_base) - timestamp_offset
            frame_index = int(round(actual_s * self.fps))
            if frame_index in wanted:
                if abs(actual_s - frame_index / self.fps) > 1e-3:
                    raise ValueError(f"Unexpected video timestamp: {path}")
                canvas=letterbox_rgb(frame,self.image_size)
                frames[frame_index] = np.ascontiguousarray(canvas.transpose(2,0,1))
            if frame_index >= last:
                break
        if wanted - frames.keys():
            raise ValueError(f"Missing exact video frames in {path}: {wanted - frames.keys()}")
        result=np.stack([frames[int(i)] for i in indices])
        return result if self.return_uint8 else result.astype(np.float32)/255.


class LeRobotIndex:
    def __init__(self, root, split="train", image_size=(240,320), camera_names=CAMERAS, frame_cache=None, return_uint8=False):
        self.root = Path(root).resolve()
        self.info, self.splits, self.episodes = metadata(self.root)
        self.camera_names = tuple(camera_names)
        self.image_size = tuple(image_size)
        self.data, self.segments, counts = {}, [], []
        for ep_id in self.splits[split]:
            episode = self.episodes[ep_id]
            if not episode["training_eligible"]:
                raise ValueError(f"Excluded episode in {split}: {ep_id}")
            state, action, valid = load_arrays(self.root, episode)
            self.data[ep_id] = state, action, valid
            for start, end in valid_runs(valid):
                self.segments.append((ep_id,start,end)); counts.append(end-start)
        if not counts:
            raise ValueError(f"Empty {split} dataset")
        self.ends = np.cumsum(counts)
        if frame_cache:
            from .frame_cache import CachedFrames,prepare_frame_cache
            cache=prepare_frame_cache(self.root,frame_cache,image_size)
            self.decoder=CachedFrames(self.root,cache,image_size,return_uint8=return_uint8)
        else:
            self.decoder = VideoFrames(self.info["fps"], image_size, return_uint8=return_uint8)

    def __len__(self):
        return int(self.ends[-1])

    def locate(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        segment_index = int(np.searchsorted(self.ends, index, side="right"))
        episode, start, end = self.segments[segment_index]
        offset = index - (self.ends[segment_index-1] if segment_index else 0)
        return episode, start + int(offset), start, end

    def images(self, episode_id, frame_indices):
        episode = self.episodes[episode_id]
        images = {}
        for camera in self.camera_names:
            key = f"observation.images.{camera}"
            path = self.root / self.info["video_path"].format(video_key=key,
                chunk_index=episode[f"videos/{key}/chunk_index"], file_index=episode[f"videos/{key}/file_index"])
            images[camera] = self.decoder.read(path, frame_indices, episode[f"videos/{key}/from_timestamp"])
        return images
