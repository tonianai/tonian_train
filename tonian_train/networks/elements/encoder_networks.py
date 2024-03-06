
from typing import Dict
import torch
import torch.nn as nn


from tonian_train.networks.elements.network_elements import * 

class ObsEncoder(nn.Module):
    
    
    def __init__(self, config: Dict,  
                       obs_space: MultiSpace,
                       sequence_length: Optional[int] = None) -> None:

        """
        The goal of this network is to order the Observations in a ordered latent space, ready for self attention 
        
        Example
        config: {
                encoder: 
                    network: [ <see conifg documentation of  MultispaceConfiguration> 
                        { 
                            name: linear_obs_net # this can be custom set
                            input: 
                                - obs: linear
                            mlp:
                                units: [256, 128]
                                activation: relu,
                                initializer: default
                        }
                    ]  
            >
            }

        Args:
            config (Dict): _description_
        """
        super().__init__() 
        assert 'network' in config, "The Input Embedding encoder needs a specified network. (List with mlp architecture)"
        
        self.network: MultispaceNet = MultiSpaceNetworkConfiguration(config['network']).build(obs_space)
        
        self.d_model = int(config['d_model'])
        self.sequence_length = sequence_length
        self.out_nn: nn.Linear = nn.Linear(self.network.out_size(), self.d_model)
        
    def forward(self, obs: Dict[str, torch.Tensor], sequential: bool = False):
        """_summary_

        Args:
            obs (Dict[str, torch.Tensor]): any tensor has the shape (batch_size, sequence_length, ) + obs.shape

        Returns:
            _type_: _description_
        """
        
        if not sequential:
            return self.out_nn(self.network(obs))
           
        unstructured_obs_dict = {} # the unstructuring has to happen, because the self.network only has one batch dimension, and here we essentially have two (batch, sequence_length) and would like to have one 
        for key, obs_tensor in obs.items():
                
            batch_size = obs_tensor.shape[0]
            
            # TODO: THe error could be here
            assert obs_tensor.shape[1] == self.sequence_length +1 , "The second dim of data sequence tensor must be equal to the sequence length +1"   
            unstructured_obs_dict[key] = obs_tensor.view((obs_tensor.shape[0] * obs_tensor.shape[1], ) + obs_tensor.shape[2::])
            
        unstructured_result = self.network(unstructured_obs_dict) # (batch_size * sequence_length, ) + output_dim
        
        # restructuring, so that the output is (batch_size, sequence_length, ) + output_dim
        return self.out_nn(unstructured_result).reshape((batch_size, self.sequence_length + 1, self.d_model) )
    
    
    def get_output_space(self) -> MultiSpace:
        return MultiSpace({'embedding':  gym.spaces.Box(low= -1, high= 1, shape = (self.d_model, ))})
    
    def load_pretrained_weights(self, path: str):
        """
        Loads pre-trained weights into the encoder from a given path.

        Args:
            path (str): The path to the saved model weights.
        """
        self.load_state_dict(torch.load(path))

    def freeze_parameters(self):
        """
        Freezes the encoder's parameters to prevent them from being updated during training.
        """
        for param in self.parameters():
            param.requires_grad = False
    

class ObsDecoder(nn.Module):
    
    def __init__(self, config: Dict, 
                       d_model: int, 
                       obs_space: MultiSpace) -> None:
        """
        The goal of this decoder is to decode the ordered latent space into the original observation space 
        we use the Residual Dynamics Net for this, after the net
        Example
        config: {
                decoder:  
                    network:  <see conifg documentation of  MlpConfiguration>
                        {
                            units: [256, 128, 128]
                            activation: relu,
                            initializer: default
                        }
            
                
            }

        Args:
            config (Dict): _description_
        """
        super().__init__()
        assert 'network' in config, "The Input Embedding encoder needs a specified network. (List with mlp architecture)"
        
        
        self.mlp_config =  MlpConfiguration(config['network'])
        self.network: nn.Sequential = self.mlp_config.build(d_model)
        self.residual_net: ResidualDynamicsNet = ResidualDynamicsNet(self.mlp_config.get_out_size(), obs_space)
        self.d_model = d_model
        
    def forward(self, encoded_obs: torch.Tensor, sequential: bool = False):
        """
        Forward pass for decoding the ordered latent space into the original observation space.

        Args:
            encoded_obs (torch.Tensor): The encoded observations with shape (batch_size, sequence_length, d_model).
            sequential (bool, optional): Flag to process observations sequentially. Defaults to False.

        Returns:
            torch.Tensor: The decoded observations, reshaped back to the original observation space dimensions.
        """

        
        # If processing sequentially, we may need to adjust the processing here
        if sequential:
            # Custom sequential processing can be added here
            # This is a placeholder for any specific logic needed when observations are to be processed sequentially
            pass

        # Apply the MLP network to the encoded observations
        mlp_output = self.network(encoded_obs)
        # Apply the residual dynamics network to the MLP output
        decoded_obs = self.residual_net(mlp_output)

        # Return the decoded observations
        return decoded_obs





