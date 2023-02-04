import torchvision
import torch

# pull the fashion MNIST datset from pythorch repos
def load_fashion_MNIST(batch_size):

  train_data = torchvision.datasets.FashionMNIST(
    root='./fashionMNIST/',
    train=True,
    transform= torchvision.transforms.Compose([ torchvision.transforms.ToTensor() ])
  )
  train_loader = torch.utils.data.DataLoader( train_data, batch_size=batch_size, shuffle=True )
  
  test_data = torchvision.datasets.FashionMNIST(
    root='./fashionMNIST/',
    train=False,
    transform= torchvision.transforms.Compose([ torchvision.transforms.ToTensor() ])
  )
  test_loader = torch.utils.data.DataLoader( test_data, batch_size=batch_size, shuffle=True )
  
  return train_loader, test_loader
