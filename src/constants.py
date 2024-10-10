from src.datasets.datasets import (get_chest_xray, get_cifar10,
                                   get_fashion_mnist, get_mnist)


class DatasetManager:
    datasets = {
        "mnist": get_mnist,
        "fashion-mnist": get_fashion_mnist,
        "cifar10": get_cifar10,
        "chest-xray": get_chest_xray,
    }

    @classmethod
    def get_dataset(cls, name, dataroot, train=True):
        if name in cls.datasets:
            return cls.datasets[name](dataroot, train)
        else:
            raise ValueError(f"Dataset {name} is not supported.")

    @classmethod
    def valid_dataset(cls, name):
        return name in cls.datasets
