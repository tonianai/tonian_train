from typing import Callable, Dict, Union, List, Any, Tuple
import torch.nn as nn
from abc import ABC, abstractmethod

from tonian.common.spaces import MultiSpace, MultiSpaceIterator
from tonian.training2.common.aliases import ActivationFn, InitializerFn
from tonian.training2.networks import MultispaceNet, MultispaceNetElement

import torch



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
            self.name = config
            self.kwargs = {}
        elif isinstance(config, type(None)):
            self.name = 'None'
            self.kwargs = {}
        else:
            self.name = config['name']
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
        'None': nn.Identity
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
            self.name = config
            self.kwargs = {}
        elif isinstance(config, type(None)):
            self.name = 'default'
            self.kwargs = {}
        else:
            self.name = config['name']
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
        
        