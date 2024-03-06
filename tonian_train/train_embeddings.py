import torch
from torch.utils.data import DataLoader
from tonian_train.common.obs_saver import ObservationDataset
from tonian_train.networks.elements.encoder_networks import ObsEncoder, ObsDecoder

import os

def train_embeddings(config, obs_space, obs_path, num_epochs=10):
    """_summary_

    Args:
        config (_type_): Example: 
                 
                encoder:
                    model_path: 'encoder_models/08_mk1_target/encoder.pth'
                    network:
                        - name: encoder_net
                        input: 
                            - obs: linear 
                            - obs: command
                        mlp:
                            units: [512, 128]
                            activation: elu
                            initializer: default 
                            
                decoder:
                    model_path: 'encoder_models/08_mk1_target/decoder.pth'
                    network: 
                        units: [256, 128, 128]
                        activation: relu,
                        initializer: default
                        
        obs_space (_type_): _description_
        obs_path (_type_): _description_
        num_epochs (int, optional): _description_. Defaults to 10.
    """
    device = 'cuda:0'

    # Assuming obs_space, config for encoder and decoder are defined
    encoder = ObsEncoder(config=config['encoder'], d_model=128, obs_space=obs_space).to(device)
    decoder = ObsDecoder(config=config['decoder'], d_model=128, obs_space=obs_space).to(device)

    dataset = ObservationDataset(base_path=obs_path, device=device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    
    obs_keys = list(obs_space.spaces.keys())

    for epoch in range(num_epochs):
        for obs_batch in dataset:
            optimizer.zero_grad()
            
            
            
            encoded = encoder(obs_batch)
            decoded = decoder(encoded)
            
            # add the loss for each key in the observation space
            loss = 0
            for key in obs_keys:
                loss += loss_fn(decoded[key], obs_batch[key])
                
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item()}')
        
    encoder_path = config['encoder']['model_path']    
    decoder_path = config['decoder']['model_path']


    # Extract the directory path from the full file paths
    encoder_dir = os.path.dirname(encoder_path)
    decoder_dir = os.path.dirname(decoder_path)

    # Ensure directories exist for both encoder and decoder
    os.makedirs(encoder_dir, exist_ok=True)
    os.makedirs(decoder_dir, exist_ok=True)

    # Save models
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
