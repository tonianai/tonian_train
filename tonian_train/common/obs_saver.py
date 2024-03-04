from typing import Dict, List
import torch
import os, random, shutil

class ObservationSaver:
    
    def __init__(self, base_path: str, batch_size: int = 100, save_probability: float = 0.01):
        """
        Initializes the observation saver.

        Parameters:
        - base_path: The base directory path for saving the observations.
        - batch_size: Number of observations to accumulate before saving/appending to each file.
        - save_probability: Probability of saving a given observation.
        """
        self.base_path = base_path
        self.batch_size = batch_size
        self.save_probability = save_probability
        self.observations: Dict[str, List[torch.Tensor]] = {}
        # Ensure the base path exists
        os.makedirs(self.base_path, exist_ok=True)

    def maybe_save_obs(self, obs: Dict[str, torch.Tensor]) -> None:
        """
        Accumulates an observation for each key and saves batches of accumulated observations to separate files.

        Parameters:
        - obs: The observation to save, a dictionary mapping strings to torch.Tensors.
        """
        if random.random() >= self.save_probability:
            return  # Skip saving this observation

        for key, tensor in obs.items():
            if key not in self.observations:
                self.observations[key] = []
            self.observations[key].append(tensor)

            # Check if we have accumulated enough observations for this key
            if len(self.observations[key]) >= self.batch_size:
                self._save_and_clear_observations(key)

    def _save_and_clear_observations(self, key: str) -> None:
        """
        Saves the accumulated observations for a given key to a file without overriding existing files
        and clears the buffer.

        Parameters:
        - key: The key for which to save the observations.
        """
        existing_files = [f for f in os.listdir(self.base_path) if os.path.isfile(os.path.join(self.base_path, f)) and key in f]
        file_index = len(existing_files)
        file_path = os.path.join(self.base_path, f"{key}_{file_index}.pt")

        batch_tensor = torch.stack(self.observations[key])
        torch.save(batch_tensor, file_path)

        # Clear the accumulated observations for this key
        self.observations[key] = []

    def clear_base_path(self) -> None:
        """
        Clears the base path by deleting all files within it.
        """
        for filename in os.listdir(self.base_path):
            file_path = os.path.join(self.base_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    def flush(self) -> None:
        """
        Manually saves any remaining observations for all keys that haven't yet reached the batch size.
        """
        for key in list(self.observations.keys()):
            if self.observations[key]:
                self._save_and_clear_observations(key)
