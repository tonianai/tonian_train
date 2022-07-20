
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any


class VecTask(ABC):
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        
        
        