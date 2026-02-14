import logging
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional

import numpy as np

from policy_models.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class ShmDataset(BaseDataset):
    """
    Dataset that loads episodes from shared memory.
    """

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)
        self.episode_lookup_dict: Dict[str, List] = {}
        self.episode_lookup: Optional[np.ndarray] = None
        self.lang_lookup = None
        self.lang_ann = None
        self.shapes = None
        self.sizes = None
        self.dtypes = None
        self.dataset_type = None
        self.shared_memories = None

    def setup_shm_lookup(self, shm_lookup: Dict) -> None:
        """
        Initialize episode lookups.

        Args:
            shm_lookup: Dictionary containing precomputed lookups.
        """
        if self.with_lang:
            self.episode_lookup_dict = shm_lookup["episode_lookup_lang"]
            self.lang_lookup = shm_lookup["lang_lookup"]
            self.lang_ann = shm_lookup["lang_ann"]
        else:
            self.episode_lookup_dict = shm_lookup["episode_lookup_vision"]
        key = list(self.episode_lookup_dict.keys())[0]
        self.episode_lookup = np.array(self.episode_lookup_dict[key])[:, 1]
        self.shapes = shm_lookup["shapes"]
        self.sizes = shm_lookup["sizes"]
        self.dtypes = shm_lookup["dtypes"]
        self.dataset_type = "train" if "training" in self.abs_datasets_dir.as_posix() else "val"
        # attach to shared memories
        self.shared_memories = {
            key: SharedMemory(name=f"{self.dataset_type}_{key}") for key in self.episode_lookup_dict
        }

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames from shared memory and combine to episode dict.

        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.

        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        episode = {}
        for key, lookup in self.episode_lookup_dict.items():
            offset, j = lookup[idx]
            shape = (window_size + j,) + self.shapes[key]
            array = np.ndarray(shape, dtype=self.dtypes[key], buffer=self.shared_memories[key].buf, offset=offset)[j:]  # type: ignore
            episode[key] = array
        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]][0]  # TODO check  [0]
        return episode



class BesoSHmDataset(BaseDataset):
    
    def __init__(
        self, 
        obs_seq_len: int,
        action_seq_len: int,
        future_range: int,
        *args, 
        **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)
        self.episode_lookup_dict: Dict[str, List] = {}
        self.episode_lookup: Optional[np.ndarray] = None
        self.lang_lookup = None
        self.lang_ann = None
        self.shapes = None
        self.sizes = None
        self.dtypes = None
        self.dataset_type = None
        self.shared_memories = None
        
        # new stuff for our dataset
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len
        self.future_range = future_range

    def setup_shm_lookup(self, shm_lookup: Dict) -> None:
        """
        Initialize episode lookups.

        Args:
            shm_lookup: Dictionary containing precomputed lookups.
        """
        if self.with_lang:
            self.episode_lookup_dict = shm_lookup["episode_lookup_lang"]
            self.lang_lookup = shm_lookup["lang_lookup"]
            self.lang_ann = shm_lookup["lang_ann"]
        else:
            self.episode_lookup_dict = shm_lookup["episode_lookup_vision"]
        key = list(self.episode_lookup_dict.keys())[0]
        self.episode_lookup = np.array(self.episode_lookup_dict[key])[:, 1]
        self.shapes = shm_lookup["shapes"]
        self.sizes = shm_lookup["sizes"]
        self.dtypes = shm_lookup["dtypes"]
        self.dataset_type = "train" if "training" in self.abs_datasets_dir.as_posix() else "val"
        # attach to shared memories
        self.shared_memories = {
            key: SharedMemory(name=f"{self.dataset_type}_{key}") for key in self.episode_lookup_dict
        }

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        episode = {}
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        
        # Load main episode data from shared memory
        for key, lookup in self.episode_lookup_dict.items():
            offset, j = lookup[idx]
            shape = (window_size + j,) + self.shapes[key]
            array = np.ndarray(shape, dtype=self.dtypes[key], buffer=self.shared_memories[key].buf, offset=offset)[j:]
            
            # Slice the data to match action_seq_len and obs_seq_len
            if key == "rel_actions" or key == 'actions':
                episode[key] = array[:self.action_seq_len, :]
            else:
                episode[key] = array[:self.obs_seq_len, :]

        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]][0]  # TODO check  [0]
        
        # Logic for future goal
        delta = np.random.randint(self.future_range)
        goal_idx = self.episode_lookup[idx] + self.action_seq_len + delta
        eps_start_idx, eps_end_idx = self.find_sequence_boundaries(goal_idx)
        if eps_end_idx < goal_idx:
            goal_idx = eps_end_idx
        
        # Load future goal from shared memory
        offset, j = self.episode_lookup_dict['scene_obs'][goal_idx]  # Assuming 'scene_obs' is the key for goals
        shape = (1,) + self.shapes['scene_obs']
        goal_array = np.ndarray(shape, dtype=self.dtypes['scene_obs'], buffer=self.shared_memories['scene_obs'].buf, offset=offset)[j:]
        
        goal_episode = {'scene_obs': goal_array[:self.obs_seq_len, :]}
        
        # Merge episodes
        episode = self.merge_episodes(episode, goal_episode)
        
        return episode

    def merge_episodes(self, episode1: Dict[str, np.ndarray], episode2: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        merged_episode = {}
        all_keys = set(episode1.keys()).union(set(episode2.keys()))
        for key in all_keys:
            if key in episode1 and key in episode2:
                # Merge logic here, for example:
                merged_episode[key] = np.concatenate([episode1[key], episode2[key]], axis=0)
            elif key in episode1:
                merged_episode[key] = episode1[key]
            else:
                merged_episode[key] = episode2[key]
        return merged_episode