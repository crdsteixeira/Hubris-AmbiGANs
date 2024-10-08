from src.datasets.datasets import get_chest_xray, get_cifar10, get_fashion_mnist, get_mnist


dataset_getters = {
    'mnist': get_mnist,
    'fashion-mnist': get_fashion_mnist,
    'cifar10': get_cifar10,
    'chest-xray': get_chest_xray
}