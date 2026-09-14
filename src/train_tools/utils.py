from .models import *

__all__ = ["create_models"]

MODELS = {
    "fedavg_mnist": fedavgnet.FedAvgNetMNIST,
    "fedavg_cifar": fedavgnet.FedAvgNetCIFAR,
    "fedavg_tiny": fedavgnet.FedAvgNetTiny,
    "vgg11": vgg.vgg11,
    "res10": resnet.resnet10,
    "res18": resnet.resnet18,
}

NUM_CLASSES = {
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "cinic10": 10,
    "tinyimagenet": 200,
}


def create_models(model_name, dataset_name, **params):
    """Create a network model"""

    num_classes = NUM_CLASSES[dataset_name]
    if model_name == "vgg11" and dataset_name == "tinyimagenet":
        params.setdefault("img_size", 3 * 64 * 64)
    model = MODELS[model_name](num_classes=num_classes, **params)

    return model
