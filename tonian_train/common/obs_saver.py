from typing import Dict, List
import torch
import os, random, shutil
from torch.utils.data import Dataset 

class ObservationSaver:
    
    def __init__(self, base_path: str, batch_size: int = 10, save_probability: float = 0.01):
        """
        Initializes the observation saver.

        Parameters:
        - base_path: The base directory path for saving the observations.
        - batch_size: Number of observations to accumulate before saving/appending to each file.
        - save_probability: Probability of saving a given observation.
        """
        self.base_path = base_path
        self.batch_size = batch_size
        self.current_batch = 0
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
            
        self.current_batch += 1
        
        if self.current_batch >= self.batch_size:
            self._save_and_clear_observations()

    def _save_and_clear_observations(self) -> None:
        """
        Saves the accumulated observations for a given key to a file without overriding existing files
        and clears the buffer.

        Parameters:
        - key: The key for which to save the observations.
        """ 
        if not self.observations:  # Check if there are any observations to save
            return
        
        # Determine the next files_index by finding the number of existing directories
        files_index = 0
        while os.path.exists(os.path.join(self.base_path, str(files_index))):
            files_index += 1

        # Create a new directory for this batch
        batch_dir = os.path.join(self.base_path, str(files_index))
        os.makedirs(batch_dir, exist_ok=True)

        # Loop through each key in observations and save the corresponding tensors
        for key, tensors in self.observations.items():
            # Concatenate tensors if you have more than one tensor per key, otherwise just save the single tensor
            if len(tensors) > 1:
                tensor_to_save = torch.cat(tensors, dim=0)
            else:
                tensor_to_save = tensors[0]

            # Define the save path
            save_path = os.path.join(batch_dir, f"{key}.pth")

            # Save the tensor
            torch.save(tensor_to_save, save_path)

        # Clear the observations now that they've been saved
        self.observations.clear()
        
        # Reset the current batch count
        self.current_batch = 0

        

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


class ObservationDataset(Dataset):
    def __init__(self, base_path: str, device: str = 'cuda:0'):
        """
        Initializes the dataset by listing all the observation folders in the base path.
        Each folder represents a batch of observations saved by the ObservationSaver.

        Parameters:
        - base_path: The directory containing saved observation folders.
        """
        self.base_path = base_path
        self.device = device
        # List all directories in base_path, each directory corresponds to a batch
        self.batch_folders = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        # Sort directories to ensure ordered access
        self.batch_folders.sort(key=lambda x: int(os.path.basename(x)))
        
    def get_keys(self):
        """
        Returns the keys of the first batch in the dataset.
        """
        # Path to the folder for the current index
        folder_path = self.batch_folders[0]
        # Initialize an empty dict to hold observations for this batch
        keys = []
        # Iterate over each file in the folder, assuming each file corresponds to a different observation key
        for file_name in os.listdir(folder_path):
            # Extract the key from the file name (remove the .pth extension)
            key = file_name.rsplit('.', 1)[0]
            keys.append(key)
        return keys

    def __len__(self):
        # Return the number of batches available
        return len(self.batch_folders)
    
    def __getitem__(self, idx):
        # Path to the folder for the current index
        folder_path = self.batch_folders[idx]
        # Initialize an empty dict to hold observations for this batch
        observations = {}
        # Iterate over each file in the folder, assuming each file corresponds to a different observation key
        for file_name in os.listdir(folder_path):
            # Construct the full file path
            file_path = os.path.join(folder_path, file_name)
            # Extract the key from the file name (remove the .pth extension)
            key = file_name.rsplit('.', 1)[0]
            # Load the tensor and add it to the observations dict
            batch_tensor = torch.load(file_path) 
            
            observations[key] = batch_tensor.to(self.device)
            
        return observations