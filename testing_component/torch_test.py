import torch

dones = torch.randint(0, 2, size=(10,), dtype=torch.uint8)

print(dones)

dones_inv = -(dones -1)

print(dones_inv)
