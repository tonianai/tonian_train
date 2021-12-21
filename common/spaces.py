
from typing import Dict, Tuple
import numpy as np
from gym import spaces
import gym
 

class MultiSpace():
    """
    Defines the combination of multiple spaces to one multispace
    
    For expample an agent might require linear observations, visual observations and command input
    or an agent might output communication to other agents as well as actions
    These spaces can be summarized within multispace
    """
    def __init__(self, spaces: Dict[str, gym.Space] ) -> None:
        self.update(spaces) 
       
        
    def update(self, spaces: Dict[str, gym.Space]) -> None:
        self.spaces = spaces
        self.space_names = spaces.keys()
        self.num_spaces = len(spaces)
        self.shape = tuple([self.spaces[i].shape for i in self.spaces])
        
    def sample(self) -> Tuple[np.ndarray, ...]:
        return tuple([self.spaces[i].sample() for i in self.spaces])
        
    def __len__(self):
        return self.num_spaces         

    def __str__(self):
        return f"Multispace: \n Shape: {str(self.shape)} \n Contains: {str(self.spaces)}"
    
    
    def join_with(self, new_space):
        """Join together to create a new higher dimensional space
        
        Entries of the new_space dict will be appended to the current one, meaning that self.spaces will be in front
        Args:
            new_space ([MultiSpace]):
            
        Returns: 
            [MultiSpace] 
        """
        
        for key_self in self.spaces:
            for key_foreign in new_space.spaces:
                if key_self == key_foreign:
                    raise Exception(f"Two Multispaces containing each spaces with the same key, cannot be joined together. Key in Common: \"{key_self}\"")
        
        # Joining spaces
        self.update({**self.spaces , **new_space.spaces} )
        
        
        