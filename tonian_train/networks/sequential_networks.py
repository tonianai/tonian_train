

from typing import Dict,  Optional
from abc import ABC

import torch, gym, os, math
import torch.nn as nn
import numpy as np

from tonian_train.common.spaces import MultiSpace
from tonian_train.common.aliases import ActivationFn, InitializerFn
from tonian_train.common.spaces import MultiSpace

from tonian_train.networks.network_elements import *
from tonian_train.networks.simple_networks import A2CSimpleNet



class InputEmbedding(nn.Module):
    
    def __init__(self, config: Dict, 
                       sequence_length: int,
                       d_model: int, 
                       obs_space: MultiSpace) -> None:

        """
        The goal of this network is to order the Observations in a ordered latent space, ready for self attention 
        
        Example
        config: {
                encoder: 
                    network: [ <see conifg documentation of  MlpConfiguration> 
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
        self.sequence_length = sequence_length
        assert 'encoder' in config, "Input Embeddings needs a encoder specified in the config"
        assert 'network' in config['encoder'], "The Input Embedding encoder needs a specified network. (List with mlp architecture)"
        
        self.network: MultispaceNet = MultiSpaceNetworkConfiguration(config['encoder']['network']).build(obs_space)
        
        self.d_model = d_model
        self.out_nn: nn.Linear = nn.Linear(self.network.out_size(), d_model )
        
    def forward(self, obs: Dict[str, torch.Tensor]):
        """_summary_

        Args:
            obs (Dict[str, torch.Tensor]): any tensor has the shape (batch_size, sequence_length, ) + obs.shape

        Returns:
            _type_: _description_
        """
        
        # TODO:  validate, that this is the right approach 
        # Note: Do we have a multiplication of gradients with this approach???
        # Please investigate @future schmijo
           
        unstructured_obs_dict = {} # the unstructuring has to happen, because the self.network only has one batch dimension, and here we essentially have two (batch, sequence_length) and would like to have one 
        for key, obs_tensor in obs.items():
                
            batch_size = obs_tensor.shape[0]
            
            # TODO: THe error could be here
            assert obs_tensor.shape[1] == self.sequence_length +1 , "The second dim of data sequence tensor must be equal to the sequence length +1"   
            unstructured_obs_dict[key] = obs_tensor.view((obs_tensor.shape[0] * obs_tensor.shape[1], ) + obs_tensor.shape[2::])
            
        unstructured_result = self.network(unstructured_obs_dict) # (batch_size * sequence_length, ) + output_dim
        
        # restructuring, so that the output is (batch_size, sequence_length, ) + output_dim
        return self.out_nn(unstructured_result).reshape((batch_size, self.sequence_length + 1, self.d_model) )
    
     
class OutputEmbedding(nn.Module):
    
    def __init__(self, config: Dict, 
                       sequence_length: int, 
                       d_model: int, 
                       action_space: gym.spaces.Space,
                       value_size: int = 1) -> None:
        """
        The goal of this network is to order the actions (mu & std), values in a ordered latent space

        Args:
            config (Dict):
            
                encoder:
                    mlp: 
                        units: [256, 128]
                        activation: relu,
                        initializer: default
        """
        super().__init__()
        
        print(config)
        self.sequence_length = sequence_length
        assert 'encoder' in config, "Output Embeddings needs a encoder specified in the config"
        assert 'mlp' in config['encoder'], "The Output Embedding encoder needs a specified mlp architecture (key = mlp)."
        
        assert isinstance(action_space, gym.spaces.Box), "Only continuous action spaces are implemented at the moment"
        
        
        self.action_size = action_space.shape[0]
        self.value_size = value_size
        self.mlp_input_size = self.action_size *2 + self.value_size
        
        mlp_config = MlpConfiguration(config['encoder']['mlp'])
        self.network: nn.Sequential = mlp_config.build(self.mlp_input_size)
        
        
        self.d_model = d_model
        self.out_nn: nn.Linear = nn.Linear(mlp_config.get_out_size(), d_model )
        
        
            
    def forward(self, action_mu: torch.Tensor, action_std: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Embedding for outputs 
        Transforms outputs in a fitting latent space for self attention

        Args:
            action_mu (torch.Tensor): shape (batch_size, seq_len, action_size)
            action_std (torch.Tensor):  shape (batch_size, seq_len, action_size)
            value (torch.Tensor): shape (batch_size, seq_len, value_size)

        Returns:
            torch.Tensor: _description_
        """
        
        all_outputs = torch.cat((action_mu, action_std, value), 2)
        
        assert all_outputs.shape[2] == self.mlp_input_size, "The mlp input size does not fit the concatinated mlp input in the output embeddings"
        
        
        # TODO:  validate, that this is the right approach 
        # Note: Do we have a multiplication of gradients with this approach???
        # Please investigate @future schmijo
           # the unstructuring has to happen, because the self.network only has one batch dimension, and here we essentially have two (batch, sequence_length) and would like to have one
        unstructured_all_outputs = all_outputs.view((
            all_outputs.shape[0] *  all_outputs.shape[1], all_outputs.shape[2]
        ))
        
        result: torch.Tensor = self.network(unstructured_all_outputs)
    
        return self.out_nn(result).reshape(all_outputs.shape[0], all_outputs.shape[1], self.d_model)
       

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int , dropout_p: float=  0.1 , max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        print("pos encoder init")
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(max_len, 1, d_model)
        pos_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class EncoderBlock(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
class Encoder(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
class DecoderBlock(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
             
class Decoder(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
class SequentialNet(ABC, nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask

class TransformerNetLogStd(SequentialNet):
    
    def __init__(self, 
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 n_heads: int,
                 input_embedding: InputEmbedding,
                 output_embedding: OutputEmbedding,
                 d_model: int,
                 target_seq_len: int, 
                 action_space: gym.spaces.Space, 
                 action_head: Optional[nn.Sequential] = None,
                 action_head_size: Optional[int] = None,
                 critic_head: Optional[nn.Sequential] = None,
                 critic_head_size: Optional[int] = None,
                 action_activation: ActivationFn = nn.Identity(),
                 is_std_fixed: bool = False, 
                 std_activation: ActivationFn = nn.Identity(),
                 value_activation: ActivationFn = nn.Identity(),
                 pos_encoder_dropout_p: float = 0.1,
                 dropout_p: float = 0.1,
                 value_size: int = 1) -> None:
        """_summary_
                                                        
                                                        
               Observations:                              Actions & Values:                                         
                                                        
               [...,obs(t-2),obs(t=1),obs(t=0)]           [...,action_mean(t-1), action_mean(t=0]
                                                          [...,action_std(t-1), action_std(t=0]
                                                          [...,value(t-1), value(t=0] 
                                                          
                                                          
                              
                               |                                       |   
                               |                                       |                           
                              \|/                                     \|/      
                              
                    |------------------|                      |------------------|     
<Multipsace Net> -> | Input Embeddings |                      | Output Embedding |  <- <MLP>            
                    |------------------|                      |------------------|            
                               |                                       |                           
                              \|/           Organized Latent Space    \|/            
             
|------------|            |--------|                              |--------|          |------------| 
|Pos Encoding|    ->      |    +   |                              |    +   |     <-   |Pos Encoding| 
|------------|            |--------|                              |--------|          |------------|      

                               |                                       |                           
                              \|/                                     \|/            


                      |---------------|                      |---------------|
                      |               |                      |               |
                      | Encoder Block |  x N           |---->| Decoder Block |  x N
                      |               |                |     |               |
                      |---------------|                |     |---------------|
                              |                        |                            
                              |                        |             |
                              |________________________|             |
                                                                     |
                                                        _____________|__________  
                                                        |                        |
                                                        |                        |
                                                        |                        |
                                                       \|/                      \|/
                                                
                                               |---------------|         |---------------|        
                                               |   Actor Head  |         |  Critic Head  |
                                               |---------------|         |---------------|
                                               
                                                        |                        |
                                                        |                        |
                                                       \|/                      \|/
                                                       
                                                     Actions (mu & std)        States
                                                       

        Args:
            action_space (gym.spaces.Space): _description_
            action_activation (ActivationFn, optional): _description_. Defaults to nn.Identity().
            is_std_fixed (bool, optional): _description_. Defaults to False.
            std_activation (ActivationFn, optional): _description_. Defaults to nn.Identity().
            value_activation (ActivationFn, optional): _description_. Defaults to nn.Identity().
            value_size (int, optional): _description_. Defaults to 1.
        """
        super().__init__()
        
        self.model_type = "Transformer"
        
        self.positional_encoder = PositionalEncoding(
            d_model=d_model , dropout_p=pos_encoder_dropout_p, max_len=5000
        )
        
        self.input_embedding = input_embedding
        self.output_embedding = output_embedding
        self.d_model = d_model
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead= n_heads, 
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first= True
        )
        
        
         
        self.action_head = action_head
        self.critic_head = critic_head
        self.num_actions = action_space.shape[0]
        
        if self.action_head is not None:    
            self.a_out = nn.Linear(action_head_size, self.num_actions)
        else:
            self.action_head = nn.Identity()
            self.a_out = nn.Linear(self.d_model, self.num_actions)
            
        if self.critic_head is not None:
            self.c_out = nn.Linear(critic_head_size, value_size)
        else:
            self.critic_head = nn.Identity()
            self.c_out = nn.Linear(self.d_model, critic_head_size)
        
        
        self.is_std_fixed = is_std_fixed
        if self.is_std_fixed:
            self.action_std = nn.Parameter(torch.zeros(self.num_actions, requires_grad=True))
        else:
            self.action_std = torch.nn.Linear(self.d_model * target_seq_len, self.num_actions)
        
        
        self.action_activation = action_activation
        self.std_activation = std_activation
        self.value_activation = value_activation
        
    def forward(self, src_obs: Dict[str, torch.Tensor], 
                      tgt_action_mu: torch.Tensor,  
                      tgt_action_std: torch.Tensor,  
                      tgt_value: torch.Tensor,  
                      tgt_mask=None, 
                      src_pad_mask=None,
                      tgt_pad_mask=None):
        """_summary_

        Args:
            src_obs (Dict[str, torch.Tensor]): tensor shape (batch_size, src_seq_length, ) + obs_shape
            tgt_action_mu (torch.Tensor): shape (batch_size, tgt_seq_length, action_length)
            tgt_action_std (torch.Tensor):  (batch_size, tgt_seq_length, action_length)
            tgt_value (torch.Tensor): (batch_size, sr)
            tgt_mask (_type_, optional): _description_. Defaults to None.
            src_pad_mask (_type_, optional): _description_. Defaults to None.
            tgt_pad_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
         
    
        
        
        src = self.input_embedding(src_obs) * math.sqrt(self.d_model)
        tgt = self.output_embedding(tgt_action_mu, tgt_action_std, tgt_value) * math.sqrt(self.d_model)
         
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
        transformer_out = self.transformer.forward(src=src, tgt=tgt, tgt_mask= tgt_mask)
        
        
        out_seq_length = transformer_out.shape[1]
        out_model_size = transformer_out.shape[2]
        
        result =  transformer_out.reshape(-1 , out_seq_length * out_model_size)
        
        value = self.value_activation(self.c_out(self.critic_head(result)))
        
        mu = self.action_activation(self.a_out(self.action_head(result)))
        
        if self.is_std_fixed:
            std = self.std_activation(self.action_std)
        else:
            std = self.std_activation(self.action_std(result))
        
        
        return mu, mu*0 + std, value 
          

    
def build_transformer_a2c_from_config(config: Dict,
                                      seq_len: int, 
                                      value_size: int, 
                                      obs_space: MultiSpace,
                                      action_space: gym.spaces.Space) -> TransformerNetLogStd:
    """Build a transformer model for Advantage Actor Crititc Policy

    Args:
        config (Dict): config
        obs_space (MultiSpace): Observations
        action_space (gym.spaces.Space): Actions Spaces

    Returns:
        TransformerNetLogStd: _description_
    """
    
    d_model = config['d_model']

    
    input_embedding = InputEmbedding(config['input_embedding'], 
                                     sequence_length= seq_len,
                                     d_model= d_model,
                                     obs_space= obs_space)
    
    output_embedding = OutputEmbedding(config=config['output_embedding'],
                                       sequence_length= seq_len,
                                       d_model= d_model,
                                       action_space= action_space,
                                       value_size= value_size )
    
    
    action_activation = ActivationConfiguration(config.get('action_activation', 'None')).build()
    std_activation = ActivationConfiguration(config.get('std_activation', 'None')).build()
    value_activation = ActivationConfiguration(config.get('value_activation', 'None')).build()
    
    action_head_factory = MlpConfiguration(config['action_head'])
    action_head = action_head_factory.build(d_model * seq_len)
    
    critic_head_factory =  MlpConfiguration(config['critic_head'])
    critic_head = critic_head_factory.build(d_model * seq_len)
    
    
    num_encoder_layers = config['num_encoder_layers']
    num_decoder_layers = config['num_decoder_layers']
    n_heads = config['n_heads']
    is_std_fixed = config['is_std_fixed']
    
    
    transformer = TransformerNetLogStd(
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            n_heads= n_heads,
            input_embedding=input_embedding,
            output_embedding=output_embedding,
            target_seq_len= seq_len,
            d_model=d_model, 
            action_space=action_space,
            action_head=action_head,
            action_head_size = action_head_factory.get_out_size(),
            critic_head=critic_head,
            critic_head_size = critic_head_factory.get_out_size(),
            action_activation=action_activation,
            is_std_fixed= is_std_fixed,
            std_activation=std_activation,
            value_activation=value_activation,
            pos_encoder_dropout_p= 0.1,
            dropout_p= 0.1,
            value_size=value_size
    )
    
    return transformer
    
    
class SimpleSequentialNet(SequentialNet):
    
    def __init__(self, 
                 main_body: nn.Sequential,
                 input_embedding: InputEmbedding,
                 output_embedding: OutputEmbedding, 
                 target_seq_len: int, 
                 d_model: int,
                 action_space: gym.spaces.Space, 
                 action_head: Optional[nn.Sequential] = None,
                 action_head_size: Optional[int] = None,
                 critic_head: Optional[nn.Sequential] = None,
                 critic_head_size: Optional[int] = None,
                 action_activation: ActivationFn = nn.Identity(),
                 is_std_fixed: bool = False, 
                 std_activation: ActivationFn = nn.Identity(),
                 value_activation: ActivationFn = nn.Identity(), 
                 dropout_p: float = 0.1,
                 value_size: int = 1) -> None:
        """_summary_
                                                        
                                                        
               Observations:                              Actions & Values:                                         
                                                        
               [...,obs(t-2),obs(t=1),obs(t=0)]           [...,action_mean(t-1), action_mean(t=0]
                                                          [...,action_std(t-1), action_std(t=0]
                                                          [...,value(t-1), value(t=0] 
                                                          
                                                          
                              
                               |                                       |   
                               |                                       |                           
                              \|/                                     \|/      
                              
                    |------------------|                      |------------------|     
                    | Input Embeddings |                      | Output Embedding |  <- <MLP>            
                    |------------------|                      |------------------|            
                               |                                       |                           
                              \|/           Organized Latent Space    \|/            
             
                                    |------------------------------|
                                    |     Simpe forward            |
                                    |------------------------------|

    
                                                 |
                                    _____________|__________  
                                    |                        |
                                    |                        |
                                    |                        |
                                    \|/                      \|/
                            
                            |---------------|         |---------------|        
                            |   Actor Head  |         |  Critic Head  |
                            |---------------|         |---------------|
                            
                                    |                        |
                                    |                        |
                                   \|/                      \|/
                                    
                                Actions (mu & std)        States
                                    

        Args:
            action_space (gym.spaces.Space): _description_
            action_activation (ActivationFn, optional): _description_. Defaults to nn.Identity().
            is_std_fixed (bool, optional): _description_. Defaults to False.
            std_activation (ActivationFn, optional): _description_. Defaults to nn.Identity().
            value_activation (ActivationFn, optional): _description_. Defaults to nn.Identity().
            value_size (int, optional): _description_. Defaults to 1.
        """
        super().__init__()
    
    
        self.input_embedding = input_embedding
        self.output_embedding = output_embedding
        self.main_body = main_body
        self.d_model = d_model
        

        self.action_head = action_head
        self.critic_head = critic_head
        self.num_actions = action_space.shape[0]
        
        if self.action_head is not None:    
            self.a_out = nn.Linear(action_head_size, self.num_actions)
        else:
            self.action_head = nn.Identity()
            self.a_out = nn.Linear(self.d_model, self.num_actions)
            
        if self.critic_head is not None:
            self.c_out = nn.Linear(critic_head_size, value_size)
        else:
            self.critic_head = nn.Identity()
            self.c_out = nn.Linear(self.d_model, critic_head_size)
        
        
        self.is_std_fixed = is_std_fixed
        if self.is_std_fixed:
            self.action_std = nn.Parameter(torch.zeros(self.num_actions, requires_grad=True))
        else:
            self.action_std = torch.nn.Linear(self.d_model, self.num_actions)
        
        
        self.action_activation = action_activation
        self.std_activation = std_activation
        self.value_activation = value_activation
    
     
    def forward(self, src_obs: Dict[str, torch.Tensor], 
                      tgt_action_mu: torch.Tensor,  
                      tgt_action_std: torch.Tensor,  
                      tgt_value: torch.Tensor, 
                      tgt_mask=None,
                      src_pad_mask=None,
                      tgt_pad_mask=None):
        """_summary_

        Args:
            src_obs (Dict[str, torch.Tensor]): tensor shape (batch_size, src_seq_length, ) + obs_shape
            tgt_action_mu (torch.Tensor): shape (batch_size, tgt_seq_length, action_length)
            tgt_action_std (torch.Tensor):  (batch_size, tgt_seq_length, action_length)
            tgt_value (torch.Tensor): (batch_size, sr)
            tgt_mask (_type_, optional): _description_. Defaults to None.
            src_pad_mask (_type_, optional): _description_. Defaults to None.
            tgt_pad_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
         
     
        src = self.input_embedding(src_obs) * math.sqrt(self.d_model)
        tgt = self.output_embedding(tgt_action_mu, tgt_action_std, tgt_value) * math.sqrt(self.d_model)
           
        main_input = torch.cat((src, tgt), dim=1)
        
        out_seq_length = main_input.shape[1]
        out_model_size = main_input.shape[2]
        result =  main_input.reshape(-1 , out_seq_length * out_model_size)
           
        result = self.main_body(result)

        
        value = self.value_activation(self.c_out(self.critic_head(result)))
        
        mu = self.action_activation(self.a_out(self.action_head(result)))
        
        if self.is_std_fixed:
            std = self.std_activation(self.action_std)
        else:
            std = self.std_activation(self.action_std(result))
        
        
        return mu, mu*0 + std, value      
    
         
    

def build_simple_sequential_nn_from_config(config: Dict,
                                      seq_len: int, 
                                      value_size: int, 
                                      obs_space: MultiSpace,
                                      action_space: gym.spaces.Space) -> TransformerNetLogStd:
    """Build a transformer model for Advantage Actor Crititc Policy

    Args:
        config (Dict): config
        obs_space (MultiSpace): Observations
        action_space (gym.spaces.Space): Actions Spaces

    Returns:
        TransformerNetLogStd: _description_
    """
    
    d_model = config['d_model']

    
    input_embedding = InputEmbedding(config['input_embedding'], 
                                     sequence_length= seq_len,
                                     d_model= d_model,
                                     obs_space= obs_space)
    
    output_embedding = OutputEmbedding(config=config['output_embedding'],
                                       sequence_length= seq_len,
                                       d_model= d_model,
                                       action_space= action_space,
                                       value_size= value_size )
    
    
    action_activation = ActivationConfiguration(config.get('action_activation', 'None')).build()
    std_activation = ActivationConfiguration(config.get('std_activation', 'None')).build()
    value_activation = ActivationConfiguration(config.get('value_activation', 'None')).build()
    
    action_head_factory = MlpConfiguration(config['action_head'])
    action_head = action_head_factory.build(d_model)
    
    critic_head_factory =  MlpConfiguration(config['critic_head'])
    critic_head = critic_head_factory.build(d_model)
    
    main_body_factory = MlpConfiguration(config['main_body'])
    main_body_factory.units.append(d_model)
    main_body = main_body_factory.build(d_model * (seq_len + seq_len + 1))
    
    is_std_fixed = config['is_std_fixed']
    
    
    sequental_nn = SimpleSequentialNet( 
            main_body= main_body,
            input_embedding=input_embedding,
            output_embedding=output_embedding,
            target_seq_len= seq_len,
            d_model=d_model, 
            action_space=action_space,
            action_head=action_head,
            action_head_size = action_head_factory.get_out_size(),
            critic_head=critic_head,
            critic_head_size = critic_head_factory.get_out_size(),
            action_activation=action_activation,
            is_std_fixed= is_std_fixed,
            std_activation=std_activation,
            value_activation=value_activation,
            dropout_p= 0.1,
            value_size=value_size
    )
    
    return sequental_nn

class SequentialNetWrapper(SequentialNet):
    
    def __init__(self) -> None:
        super().__init__()
        
        
    def __init__(self, 
                 simple_net: A2CSimpleNet) -> None:
        """_summary_
                                                        
                                                        
               Observations:                              Actions & Values:                                         
                                                        
               [...,obs(t-2),obs(t=1),obs(t=0)]           [...,action_mean(t-1), action_mean(t=0]
                                                          [...,action_std(t-1), action_std(t=0]
                                                          [...,value(t-1), value(t=0] 
                              |                            
                             \|/
                             
                         [obs(t=0)]                            
                              
                               |                              
                               |                                    
                              \|/             
                                              
                             |----------------------------------------|
                             |            A2C simple net              |
                             |----------------------------------------|
                              
                              
                            
                                    |                        |
                                    |                        |
                                   \|/                      \|/
                                    
                                Actions (mu & std)        States
                                    
 
        """
        super().__init__()
        self.simple_net = simple_net
     
    def forward(self, src_obs: Dict[str, torch.Tensor], 
                      tgt_action_mu: torch.Tensor,  
                      tgt_action_std: torch.Tensor,  
                      tgt_value: torch.Tensor, 
                      tgt_mask=None,
                      src_pad_mask=None,
                      tgt_pad_mask=None):
        """_summary_

        Args:
            src_obs (Dict[str, torch.Tensor]): tensor shape (batch_size, src_seq_length, ) + obs_shape  The current observations
            tgt_action_mu (torch.Tensor): shape (batch_size, tgt_seq_length, action_length) The actions made from the last observations
            tgt_action_std (torch.Tensor):  (batch_size, tgt_seq_length, action_length)
            tgt_value (torch.Tensor): (batch_size, sr)
            tgt_mask (_type_, optional): _description_. Defaults to None.
            src_pad_mask (_type_, optional): _description_. Defaults to None.
            tgt_pad_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        obs = {}
        # cut of the rest in favor of only the last obsevation
        for obs_key in src_obs.keys():
            obs[obs_key] = src_obs[obs_key][:, -1].squeeze() 

        return self.simple_net(obs)
        










