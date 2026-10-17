import inspect
import random
from contextlib import contextmanager

import numpy as np
import torch


def classifier_module(model):
    """Use the project's existing classifier-input feature space unchanged."""
    if "get_features" not in inspect.signature(model.forward).parameters:
        raise NotImplementedError(
            "FedHPro requires forward(data, get_features=True); use "
            "fedavg_cifar, fedavg_tiny, or vgg11."
        )
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, torch.nn.Linear) or classifier.bias is None:
        raise ValueError("FedHPro requires a linear classifier with weight and bias")
    return classifier


def forward_with_features(model, data):
    # Project return order is (logits, features), opposite to upstream.
    return model(data, get_features=True)


def replace_classifier(weights, classifier_weights):
    weights["classifier.weight"] = classifier_weights["weight"].detach().clone()
    weights["classifier.bias"] = classifier_weights["bias"].detach().clone()
    return weights


@contextmanager
def preserve_evaluation_rng(loaders):
    """Project-only local statistics must not advance the training RNG streams."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    generators = {}
    for loader in loaders:
        generator = getattr(loader, "generator", None)
        if generator is not None and generator not in generators:
            generators[generator] = generator.get_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        for generator, state in generators.items():
            generator.set_state(state)
