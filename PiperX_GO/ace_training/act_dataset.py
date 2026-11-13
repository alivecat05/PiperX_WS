import numpy as np
import torch

from .data import LeRobotIndex, training_arrays, CAMERAS
from .preprocessing import normalize_state


class ACTLeRobotDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", chunk_size=100, image_size=(240,320), camera_names=CAMERAS, frame_cache=None, return_uint8=False):
        self.index = LeRobotIndex(root, split, image_size, camera_names,frame_cache=frame_cache,return_uint8=return_uint8)
        self.chunk_size = chunk_size
        state, action = training_arrays(str(self.index.root))
        self.norm_stats = {
            "qpos_mean":state.mean(0), "qpos_std":np.maximum(state.std(0,ddof=1),.01),
            "action_mean":action.mean(0), "action_std":np.maximum(action.std(0,ddof=1),.01),
        }

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        episode, anchor, start, end = self.index.locate(item)
        state, action, valid = self.index.data[episode]
        raw_indices = np.arange(anchor,anchor+self.chunk_size)
        indices = np.minimum(raw_indices, end-1)
        assert valid[indices].all()
        images = self.index.images(episode,[anchor])
        image = np.stack([images[c][0] for c in self.index.camera_names])
        qpos = normalize_state(state[anchor],self.norm_stats)
        target = (action[indices]-self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        return torch.from_numpy(image), torch.from_numpy(qpos), torch.from_numpy(target), torch.from_numpy(raw_indices>=end)
