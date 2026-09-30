import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn


def create_optimizer(model, name="sgd", **params):
    """Retain both upstream optimizers, with experiment settings from JSON."""
    if name == "sgd":
        options = dict(momentum=0.5)
        options.update(params)
        return torch.optim.SGD(model.parameters(), **options)
    if name == "adam":
        options = dict(weight_decay=1e-4)
        options.update(params)
        return torch.optim.Adam(model.parameters(), **options)
    raise ValueError("Unknown FedProto optimizer: {}".format(name))


def feature_module(model):
    """ResNet uses pre-pooling layer4 maps; project CNNs use hidden features."""
    if hasattr(model, "layer4") and isinstance(getattr(model, "linear", None), nn.Linear):
        return model.layer4, True
    for name in ("classifier", "linear_2"):
        module = getattr(model, name, None)
        if isinstance(module, nn.Linear):
            return module, False
    raise ValueError("FedProto requires a project CNN, VGG, or ResNet model")


def forward_with_features(model, data):
    """Observe one unmodified forward without adding layers or normalization."""
    module, use_output = feature_module(model)
    features = []

    def collect_input(module, inputs):
        features.append(inputs[0])

    def collect_output(module, inputs, output):
        features.append(output)

    handle = (module.register_forward_hook(collect_output) if use_output
              else module.register_forward_pre_hook(collect_input))
    try:
        logits = model(data)
    finally:
        handle.remove()
    if len(features) != 1:
        raise ValueError("FedProto expects exactly one feature extraction per forward")
    return logits, features[0]


def average_prototypes(prototypes):
    """Upstream agg_func: retain sequential sums and the singleton branch."""
    result = {}
    for label, values in prototypes.items():
        if not values:
            raise ValueError("Cannot average an empty prototype list")
        if len(values) > 1:
            prototype = 0 * values[0].detach()
            for value in values:
                prototype += value.detach()
            result[label] = prototype / len(values)
        else:
            result[label] = values[0].detach()
    return result


def aggregate_prototypes(local_prototypes):
    """Equal mean over contributing clients; rebuild from this round only."""
    by_class = {}
    for prototypes in local_prototypes:
        for label, prototype in prototypes.items():
            by_class.setdefault(label, []).append(prototype)
    return average_prototypes(by_class)


def cpu_snapshot(weights):
    return {key: value.detach().cpu().clone() for key, value in weights.items()}


@contextmanager
def preserve_evaluation_rng(loaders):
    """Extra reporting must not consume the next training round's RNG streams."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    generators = {}
    for loader in loaders:
        generator = getattr(loader, "generator", None)
        if generator is not None and id(generator) not in generators:
            generators[id(generator)] = (generator, generator.get_state())
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        for generator, state in generators.values():
            generator.set_state(state)
