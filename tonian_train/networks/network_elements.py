
from typing import Callable, Dict, Union, List, Any, Tuple, Optional
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch, gym, os
import torch.nn as nn
import numpy as np

from tonian_train.common.spaces import MultiSpace
from tonian_train.common.aliases import ActivationFn, InitializerFn 


class DictConfigurationType(ABC, Callable):
    
    def __init__(self, config: Union[Dict, str, List]) -> None:
        """A Configuration Type is a Python object representation of values entered into the config dict
        
                
        The build function is capable of creating the object instance of the object articulated in the dict
        All Arguments in the config Dict are the arguments for the creation, that are set at configuration time and can be set in a config.yaml for examle
        All Arguments passed in the build function are the arguments that are only set at runtime
        
        Args:
            config (Union[Dict, str, List)
        """
        super().__init__()
        
    @abstractmethod
    def build(self, *args, **kwargs) -> Any:
        raise NotImplementedError()
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.build(*args, **kwargs)


class ActivationConfiguration(DictConfigurationType):
    
    def __init__(self, config: Union[Dict, str, None]) -> None:
        """Configuration type that describes an activation function

        Args:
            config (Dict): 
            Examples:
                activation: relu,
                
                activation: None,
                
                activation: 
                    name: elu
                    alpha: 0.5
            
        """
        super().__init__(config)
        
        if isinstance(config, str):
            self.name = config.lower()
            self.kwargs = {}
        elif isinstance(config, type(None)):
            self.name = 'none'
            self.kwargs = {}
        else:
            self.name = config['name'].lower()
            self.kwargs = config
            config.pop('name')
        
    def build(self) -> ActivationFn: 
        activation_fn_class_map = {
        'relu' : nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid' : nn.Sigmoid,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'softplus': nn.Softplus,
        'none': nn.Identity
        }
         
        assert self.name in activation_fn_class_map, f"The activation function {self.name} was not found. Please check your config."
    
        return activation_fn_class_map[self.name](**self.kwargs) 


class InitializerConfiguration(DictConfigurationType):
    
    def __init__(self, config: Dict) -> None:
        """Configuration type, that describes the Initializer

        Args:
            config (Dict): Examples: 
                initializer: None
                initializer: const_initializer
        """
        
        super().__init__(config)
        
        if isinstance(config, str):
            self.name = config.lower()
            self.kwargs = {}
        elif isinstance(config, type(None)):
            self.name = 'default'
            self.kwargs = {}
        else:
            self.name = config['name'].lower()
            self.kwargs = config
            self.kwargs.pop('name')
            
    def build(self) -> InitializerFn:
        
        def _create_initializer(func, **kwargs):
            return lambda v : func(v, **kwargs)   
    
        intializer_fn_class_map = {
            'const_initializer': _create_initializer(nn.init.constant_, **self.kwargs),
            'orthogonal_initializer': _create_initializer(nn.init.orthogonal_, **self.kwargs),
            'glorot_normal_initializer': _create_initializer(nn.init.xavier_normal_, **self.kwargs),
            'glorot_uniform_initializer': _create_initializer(nn.init.xavier_uniform_, **self.kwargs), 
            'random_uniform_initializer': _create_initializer(nn.init.uniform_, **self.kwargs),
            'kaiming_normal': _create_initializer(nn.init.kaiming_normal_, **self.kwargs),
            'orthogonal': _create_initializer(nn.init.orthogonal_, **self.kwargs),
            'default' : nn.Identity()
        }
        
        assert self.name in intializer_fn_class_map, f"The initializer function {self.name} was not found. Please check your config."
    
        return intializer_fn_class_map[self.name]


class CnnConfiguration(DictConfigurationType):
    
    def __init__(self, config: Dict) -> None:
        """Configuration for a convolutional neural net

        Args:
            config (Dict): Example:
             type: conv2d
             activation: elu
             initializer:
               name: glorot_normal_initializer
               gain: 1
             convs:    
               - filters: 32
                 kernel_size: 8
                 strides: 4
                 padding: 0
               - filters: 64
                 kernel_size: 4
                 strides: 2
                 padding: 0
               - filters: 64
                 kernel_size: 3
                 strides: 1
                 padding: 0
      
        """
        super().__init__(config)
        
        self.type = config['type']
        self.activation = ActivationConfiguration(config['activation'])
        self.initializer = InitializerConfiguration(config['initializer']) 
        self.convs = config['convs']
        
        if self.type == 'conv2d':
            self.conv_func = torch.nn.Conv2d
        else:
            raise Exception('Only Conv2d is supported at this time')
        
    
        
    def build(self, in_channels: int) -> nn.Sequential: 
        layers = []
        for conv in self.convs:
            layers.append(
                self.conv_func(in_channels=in_channels,
                out_channels = conv['filters'],
                kernel_size = conv['kernel_size'],
                stride = conv['strides'],
                padding = conv['padding']
                ))
            act = self.activation.build()
            layers.append(act)
            in_channels = conv['filters']
            layers.append(torch.nn.BatchNorm2d(in_channels))
            
        return nn.Sequential(*layers, nn.Flatten())    
          
            
class MultispaceNetElement(nn.Module):
    
    def __init__(self, name: str, inputs_names: List[str], net: nn.Sequential, out_size: int) -> None:
        """Element Network for the multispace network
        The multispace network is made up of multiple of MultispaceNetElements
        Examples would be a cnn, mlp 
        Args:
            name (str): name of the multispace net element, cannot collide with any obersvation name the multispace net is going to use
            inputs_names (List[str]): names of inputs for the network, that will be used and concativated together 
            net (nn.Sequential): Network, that takes the cumulative size of all inputs names as input
            out_size (int): The size of the flattened output of the network 
        """
        super().__init__()
        self.name = name
        assert self.name != 'residual', "A Multispace element cannot be called residual, since this is a protected name for residual nets"
        
        self.input_names = inputs_names
        self.net = net
        self.out_size = out_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultispaceNet(nn.Module):
    
    
    def __init__(self, network_layers: List[List[MultispaceNetElement]]) -> None:
        """Takes an multispace input dict as input for the foward function 

        Args:
            network_layers (List[List[MultispaceNetElement]]
                 consecutive layers the network is made of.
                    layer 0 only takes observations
                    layer 1 takes outputs from layer 0 and obsersvations
                    layer 2 takes outputs from layer 1 and layer 0 and observations
                    .... and so on...
        """
        super().__init__()
        self.network_layers = network_layers
        
        all_nets =  []
        for layer in self.network_layers:
            for network in layer:
                all_nets.append(network)
        # add all networks to the module list, so that they appear in the parameters
        self.network_modules = nn.ModuleList(all_nets)
        
    def out_size(self):
        """retreives the size of the flattened output
        """
        return self.network_layers[-1][-1].out_size
        
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Todo: make more performant by creating a single network and storing it as such

        Args:
            x (Dict[str, torch.Tensor]): _description_

        Returns:
            _type_: torch.Tensor
        """
        
        for i_layer in range(len(self.network_layers)):
                
            for element in self.network_layers[i_layer]:
                # concat inputs
                input = None
                for input_name in element.input_names:
                    if input is None:
                        input = x[input_name]
                    else: 
                        input = torch.cat((input, x[input_name]), dim=1)
        
                if i_layer != len(self.network_layers) - 1:
                    x[element.name] = element(input) 
                else: 
                    
                    return element(input)
                
    def save(self, path: str, nets_to_save: Optional[List[str]] = None):
        """Save all state dicts of the network children as separate files in the given folder

        Args:
            path (str): base folder for the models
            nets_to_save (Optional[List[str]], optional): All the nets, that must be saved. If it is none all nets will be saved. Defaults to None.
        """
        
        for layer in self.network_layers:
            for network in layer:
                if not nets_to_save or network.name in nets_to_save:
                    torch.save(network.state_dict(), os.path.join(path, network.name + '.pth'))
        
    def load(self, path: str, nets_to_load: Optional[List[str]] = None):
        
        for layer in self.network_layers:
            for network in layer:
                if not nets_to_load or network.name in nets_to_load:
                    # check if network exists
                    file_name = os.path.join(path, network.name + '.pth')
                    if os.path.exists(file_name):
                        network.load_state_dict(torch.load(file_name))
                    else:
                        print("Warning, loading of mulispacenet requested sumnets, that do not exist")
    
    
class MlpConfiguration(DictConfigurationType):
    
    def __init__(self, config: Dict) -> None:
        """The MlpConfiguration describes how a multi level perceptron network is constructed

        Args:
            config (Dict): Example:
                {
                    units: [256, 128, 128]
                    activation: relu,
                    initializer: default
                }
            
        """
        super().__init__(config)
        
        self.units : List[int] =  config['units']
        self.activation = ActivationConfiguration(config.get('activation', 'relu'))
        self.initializer = InitializerConfiguration(config.get('intitializer', 'None'))
        self.dropout_prob = config.get('dropout', [])
        
    def build(self, input_size: int) -> nn.Sequential:
        """Buld the Sequential linear mlp neural network

        Args:
            input_size (int): Size of the initial input

        Returns:
            nn.Sequential: Neural Network Output
        """
        
        layers = []
        for unit in self.units:
            layers.append(nn.Linear(input_size, unit))
            layers.append(self.activation.build())
            input_size = unit
            
        for i in reversed(range(len(self.dropout_prob))):
            prob_net = nn.Dropout(p = self.dropout_prob[i]['prob'])
            
            layers.insert(self.dropout_prob[i]['after_layer_index'] +1 , prob_net)
                
            
        return nn.Sequential(*layers)

    def get_out_size(self) -> int:
        """Get the size of the flat tensor, that the mlp returns
        Returns:
            int: last unit size
        """
        return self.units[-1]
    
        
class MultiSpaceNetworkConfiguration(DictConfigurationType):
    
    def __init__(self, config: List) -> None:
        """
        The multispace network configuration describes which parts of a multispace are used as inputs for which network 
        It is imparative, that there is a final network, that takes all residual networks as input 
        Args:
            config (List): Examples:
                - [
                    {
                        name: linear_obs_net # this can be custom set
                        input: 
                            - obs: linear
                        mlp:
                            units: [256, 128]
                            activation: relu,
                            initializer: default

                    },
                    {
                        name: visual_obs_output
                        input_obs_name: visual
                        cnn:
                            type: conv2d
                            activation: elu
                            initializer:
                              name: glorot_normal_initializer
                              gain: 1
                            convs:    
                              - filters: 32
                                kernel_size: 8
                                strides: 4
                                padding: 0
                              - filters: 64
                                kernel_size: 4
                                strides: 2
                                padding: 0
                              - filters: 64
                                kernel_size: 3
                                strides: 1
                                padding: 0
                    },
                    {
                        
                        name: final_output
                        input: 
                            - obs: command
                            - net: linear_obs_net
                            - net: visual_obs_net
                        mlp:
                            units: [128, 128]
                            activation: relu,
                            initializer: default

                    },
                    
                ]
        """
        super().__init__(config)
         
        self.config = config
        
        self.mlp_networks = []
        self.cnn_networks = []
        for network in config:
            
            type = ''
            if 'cnn' in network:
                type = 'cnn'
            elif 'mlp' in network:
                type = 'mlp'    
                
            else:
                raise Exception(f"The network must either contain a cnn or a mlp. Input {str(network)}, name: {network['name']}")
            
            if type == 'cnn':
                net_config = CnnConfiguration(network['cnn'])

                self.cnn_networks.append({
                    'input_obs_space': network['input_obs_space'], # cnn can only directly take on space as input and not a network, nor multiple spaces
                    'name' : network['name'],
                    'net_config': net_config
                    })
            elif type == 'mlp':
                net_config = MlpConfiguration(network['mlp'])
            
                self.mlp_networks.append({
                    'input': network['input'],
                    'name' : network['name'],
                    'net_config': net_config
                    })
                
    def build(self, multi_space: MultiSpace) -> MultispaceNet:
        """ build the multispace network, that takes a multispace as an input and outputs a flattened tensor

        Args:
            multi_space (MultiSpace): Multispace with all possible observations, that will be used on the net

        Returns:
            MultispaceNet: Network as described int the config
        """ 
                
        for space_key in multi_space.keys():
            assert space_key not in [mlp_net['name'] for mlp_net in  self.mlp_networks], "The name of a net cannot be the same as any multispace key name"
            assert space_key not in [cnn_net['name'] for cnn_net in  self.cnn_networks], "The name of a net cannot be the same as any multispace key name"
        
        cnn_name_to_built = {}
        
        
        # build the cnns and determine their output shape 
        # cnns are bult first, since they cannot be preceided by any mlp
        for cnn_network in self.cnn_networks:
            # determine input and output size of the cnns
            for space_name, space in multi_space:
                if space_name == cnn_network['input_obs_space']:
                    
                    # determine input width and channels from dict 
                    space_shape = space.shape
                    
                    in_channels = 1
                    
                    # TODO: generalize when cnns with more dimensions are possible
                    if cnn_network['net_config'].type == 'conv2d': 
                        if len(space_shape) == 2:
                            in_channels = 1
                            mock_input = torch.from_numpy(space.sample()).unsqueeze(dim= 0) # add a dim for mock channel
                        elif len(space_shape) == 3:
                            in_channels = space_shape[0]
                            mock_input = torch.from_numpy(space.sample())
                        else:
                            raise Exception('The cnn input gym space shape is not correct, only 2d visual spaces with 1 dimension for channels are supported at the moment')
                    else: raise Exception(f"Only conv2d is supported at this time. Given was {cnn_network['net_config'].type} ")
                    
                    
                    built_cnn = cnn_network['net_config'].build(in_channels)
                    
                    mock_input = mock_input.unsqueeze(dim= 0) # add a dim for moch batch size
                    
                    out_size = built_cnn(mock_input).shape[1] # the cnn flattens the values, so, that they can be used in an mlp
                    cnn_name_to_built[cnn_network['name']] = { 'net': built_cnn, 'out_size': out_size, 'input': space_name }
                    break
        
                    
        
        
        layered_mlps_to_build = [] 
        #two dimensional list configuring the mlps into layers
        # the first layer only has obs inputs
        # the second layer only has first layer or lower inputs
        # the third layer only has second layer or lower inputs
        # and so on....
        
        def are_mlp_inputs_in_layers_above(mlp_input: List[Dict], layer: int) -> bool:
            """Deterrmines wheter all the inputs of the network are acconted in the layer above

            Args:
                mlp_inputs (List[Dict]): _description_
                layer (int): _description_

            Raises:
                Exception: _description_
            """ 
            for input in mlp_input:
                
                if 'obs' in input: # obs never have a dependencie issue
                    continue
                elif 'net' in input:
                    
                    if layer == 0: # layer 0 nets can only have obs as input
                        return False
                    # -> layer >= 0
                    # if the requested input is an cnn it is definately okay, since cnns always reside in the zeroth layer
                    
                    input_name = input['net']
                    # check if the net is an cnn
                    if input_name in cnn_name_to_built.keys():
                        continue
                    
                    is_in_any_layer_above = False
                    # => the requested input is neither an cnn, nor a obs and the layer >= 0
                    for i_layer in range(layer):
                        # check whether it is in i_layer layer
                        i_layer_keys = [mlps_in_layer['name'] for  mlps_in_layer in layered_mlps_to_build[i_layer] ] 
                        
                        is_in_layer =  input_name in i_layer_keys
                        
                        if is_in_layer:
                            is_in_any_layer_above = True
                            break
                    
                    if not is_in_any_layer_above:
                       return False
            return True            
        
        
        unaccounted_mlps = self.mlp_networks.copy()
        
        mlp_configure_iter = 0
        i_layer = 0
        while len(unaccounted_mlps) > 0:
            
            layered_mlps_to_build.append([])
            
            mlps_accounted_for_in_this_layer = []
            
            for mlp in unaccounted_mlps:
                
                if are_mlp_inputs_in_layers_above(mlp['input'], i_layer):
                    # the mlp can be accounted for -> add it to the current layer and remove from the unaccounted
                    layered_mlps_to_build[i_layer].append(mlp)
                    mlps_accounted_for_in_this_layer.append(mlp)
                
                mlp_configure_iter += 1
                        
            i_layer += 1
            for mlp in mlps_accounted_for_in_this_layer:
                unaccounted_mlps.remove(mlp)
            
            if len(layered_mlps_to_build[i_layer-1]) == 0:
                raise Exception("A layer specified in the config for the multispace net could not be found. Check your config file")
            
            if mlp_configure_iter > 1000:
                raise Exception("Found circular dependency in the config for multispace net. Check your multispace configs")
            
            
        # --- The mlps are now listed in an layered array, with paralell networks nexto each other, and cosecutive networks in layers below
    
        
        def find_input_size(mlp_input: List[Dict], layer: int) -> int:
            """Find the accumulated output length of the given inputs

            Args:
                mlp_input (List[Dict]): inputs for a given mlp
                layer (int): current layer of the input

            Returns:
                int: accumulated input length
            """
            cumulated_input_size = 0
            
            for input in mlp_input:
                
                if 'obs' in input: # obs never have a dependencie issue
                    cumulated_input_size += sum(multi_space.spaces[input['obs']].shape)
                elif 'net' in input:
                
                    input_name = input['net']
                    # check if the net is an cnn
                
                    if input_name in cnn_name_to_built.keys():
                        cumulated_input_size += cnn_name_to_built[input_name]['out_size']
                        continue
                        
                    for i_layer in range(layer):
                        was_mlp_found = False                        
                        for mlp in layered_mlps_to_build[i_layer]:
                            
                            if mlp['name'] == input_name:
                                cumulated_input_size +=  mlp['net_config'].get_out_size()
                                was_mlp_found = True
                                break
                        if was_mlp_found:
                            break
                        
            return cumulated_input_size
         
        def input_dict_list_to_str_list(mlp_input: List[Dict]) -> List[str]:
            """Turn the dict list of the mlp input, which differtiates between obs and net into a string list, 
            which does not

            Args:
                mlp_input (List[Dict]): inputs for a given mlp as dict list

            Returns:
                List[str]: inputs for a given mlp as a str list
            """
            resulting_list = []
            for input in mlp_input:
                if 'obs' in input: # obs never have a dependencie issue
                    resulting_list.append(input['obs'])
                elif 'net' in input:
                    resulting_list.append(input['net'])
        
            return resulting_list
        
        # This gets used to in the end initialize the MultispaceNet
        network_layers = []
        
        
        # build the networks
        for i_layer in range(max(len(layered_mlps_to_build), 1)):
            
            network_layers.append([])
            if i_layer == 0:
                # the zeroth index is special, because it contains all cnns ->  add all cnns
                for net_name, cnn in cnn_name_to_built.items():
                    
                    network_layers[i_layer].append(MultispaceNetElement(name=net_name, inputs_names=[cnn['input']], net=cnn['net'], out_size=cnn['out_size']))
        
            if len(layered_mlps_to_build) > 0:
                # add the mlp multispace elements
                for mlp in layered_mlps_to_build[i_layer]:
                    
                    # build the net
                    in_size = find_input_size(mlp['input'], layer=i_layer)
                    
                    out_size = mlp['net_config'].get_out_size()
                    
                    built_net = mlp['net_config'].build(in_size)
                    
                    input_names = input_dict_list_to_str_list(mlp_input=mlp['input'])
                    
                    network_layers[i_layer].append(MultispaceNetElement(name= mlp['name'], inputs_names= input_names, net= built_net, out_size= out_size))
            
        assert len(network_layers[-1]) == 1, "There can only be one output layer for the multispace. Check the your multispace configs" 
            
        return MultispaceNet(network_layers)
        
        