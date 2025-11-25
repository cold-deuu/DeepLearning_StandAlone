import torch
import torchvision

import numpy as np

# Download MNIST
dataset = torchvision.datasets.MNIST('./', download=True, transform=torchvision.transforms.ToTensor())

# Check Dataset
# Type Check 
print(type(dataset)) # torchvision.datasets.mnist.MNIST
print(len(dataset))

n = int(np.random.choice(len(dataset), 1)[0])
data, target = dataset[n] # data : image, target : label

print(type(data)) #Tensor
print(type(target))


data_loader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle=True)



for data, target in data_loader:
    print(f"Data Type : {type(data)}")
    print(f"Target Type : {type(target)}")

    print(f"Data Shape : {data.shape}")
    print(f"Target Shape : {target.shape}")

    # Data Reshape
    data = data.squeeze(1)
    data = data.reshape(data.shape[0],-1)
    print(f"Reshaped Data Shape : {data.shape}")
