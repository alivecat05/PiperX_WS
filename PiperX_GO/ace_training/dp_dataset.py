import copy
import numpy as np
import torch

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from .data import LeRobotIndex, training_arrays, CAMERAS


class DiffusionLeRobotDataset(BaseImageDataset):
    def __init__(self, root, horizon=16, n_obs_steps=2, image_size=(240,320), camera_names=CAMERAS, split="train", frame_cache=None):
        if horizon < n_obs_steps or n_obs_steps < 1:
            raise ValueError("Invalid observation/action horizon")
        self.root, self.horizon, self.n_obs_steps = root, horizon, n_obs_steps
        self.image_size, self.camera_names = tuple(image_size), tuple(camera_names)
        self.index = LeRobotIndex(root, split, image_size, camera_names,frame_cache=frame_cache)

    def get_validation_dataset(self):
        # Upstream interface compatibility; the training-only workspace never calls this.
        return BaseImageDataset()

    def get_normalizer(self, **kwargs):
        state, action = training_arrays(str(self.index.root))
        normalizer = LinearNormalizer()
        normalizer.fit({"qpos":state,"action":action}, mode="limits", **kwargs)
        # RGB stays [0,1]; the image encoder applies ImageNet normalization once.
        for camera in self.camera_names:
            normalizer[camera] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def get_all_actions(self):
        return torch.from_numpy(np.concatenate([action[valid] for state,action,valid in self.index.data.values()]))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        episode, anchor, start, end = self.index.locate(item)
        state, action, valid = self.index.data[episode]
        first = anchor-self.n_obs_steps+1
        obs_indices = np.clip(np.arange(first,anchor+1),start,end-1)
        action_indices = np.clip(np.arange(first,first+self.horizon),start,end-1)
        assert valid[obs_indices].all() and valid[action_indices].all()
        images = self.index.images(episode,obs_indices)
        obs = {c:torch.from_numpy(images[c]) for c in self.camera_names}
        obs["qpos"] = torch.from_numpy(state[obs_indices])
        return {"obs":obs,"action":torch.from_numpy(action[action_indices])}


from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class OfflineRunner(BaseImageRunner):
    def run(self, policy):
        return {}
