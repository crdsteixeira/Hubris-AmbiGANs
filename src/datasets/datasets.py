import torchvision
from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import io


def get_mnist(dataroot, train=True):
    dataset = torchvision.datasets.MNIST(root=dataroot, download=True, train=train,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                 (0.5,), (0.5,)),
                                         ]))

    return dataset


def get_fashion_mnist(dataroot, train=True):
    dataset = torchvision.datasets.FashionMNIST(root=dataroot, download=True,
                                                train=train,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        (0.5,), (0.5,)),
                                                ]))

    return dataset


def get_cifar10(dataroot, train=True):
    dataset = torchvision.datasets.CIFAR10(root=dataroot, download=True, train=train,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))

    return dataset

def get_chest_xray(dataroot, train=True):
    split = "train" if train else "test"
    ds = load_dataset("keremberke/chest-xray-classification", name="full")[split]
    transform = torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                (0.5,), (0.5,)),
                                                ])
    class ChestXrayDataset(Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.hf_dataset = hf_dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.hf_dataset)
        
        def __getitem__(self, idx):
            sample = self.hf_dataset[idx]
            image = Image.open(io.BytesIO(sample['image']))
        
            if self.transform:
                image = self.transform(image)
            
            label = sample['label']
            return image, label
    
    return ChestXrayDataset(ds, transform=transform)


