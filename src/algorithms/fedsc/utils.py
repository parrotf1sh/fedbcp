import math

import numpy as np
import torch


def average_prototypes(prototypes):
    """Source average_kns: average raw last-epoch features, including singletons."""
    for label, prototype_list in prototypes.items():
        if len(prototype_list) > 1:
            prototype = 0 * prototype_list[0].data
            for feature in prototype_list:
                prototype += feature.data
            average = prototype / len(prototype_list)
            prototypes[label] = average.detach()
        else:
            prototypes[label] = prototype_list[0].detach()
    return prototypes


def get_relational_prototypes(local_prototypes):
    """Preserve get_re_p's insertion order and its uncommitted temporary update."""
    relational_prototypes = {}
    for prototypes in local_prototypes:
        for label, prototype in prototypes.items():
            if label in relational_prototypes:
                relational_prototypes[label].append(prototype)
            else:
                relational_prototypes[label] = [prototype]
    for label, values in relational_prototypes.items():
        temporary = torch.stack(values)
        average = temporary.mean(dim=0)
        centers = average.unsqueeze(0).expand(temporary.size(0), -1)
        # Source compares with repeated class centers, not pairwise prototypes.
        distances = torch.cdist(temporary, centers, p=2)
        distances.fill_diagonal_(float("inf"))
        _, min_indices = torch.min(distances, dim=1)
        for prototype_idx, neighbor_idx in enumerate(min_indices):
            temporary[prototype_idx] = (
                values[prototype_idx] + values[neighbor_idx]
            ) * 0.5
        # Deliberately no write-back: torch.stack made a separate tensor.
    return relational_prototypes


def get_distribution_distances(local_class_counts, num_classes):
    """Measure each participating client's distance from the uniform prior."""
    distances = []
    uniform = [1 / num_classes for _ in range(num_classes)]
    for counts in local_class_counts:
        if sum(counts) == 0:
            raise ValueError("FedSC source distribution distance requires a nonempty client")
        probabilities = [count / sum(counts) for count in counts]
        squared_distance = 0.0
        for class_idx in range(len(probabilities)):
            squared_distance += (probabilities[class_idx] - uniform[class_idx]) ** 2
        distances.append(math.sqrt(squared_distance * 0.5))
    return distances


def get_consistent_prototypes(relational_prototypes, distances, client_sizes, device):
    """Source get_co_p: positional weights and in-place relational reweighting."""
    if sum(client_sizes) == 0 or sum(distances) == 0:
        raise ValueError("FedSC source prototype weights require positive size and distance sums")
    weights = []
    consistent_prototypes = {}
    for distance, size in zip(distances, client_sizes):
        value = size / sum(client_sizes) - distance / sum(distances)
        value = 1 / (1 + np.exp(-value))
        weights.append(value)
    # Keep NumPy's source scalar arithmetic and dtype, without a float32 cast.
    weights = weights / sum(weights)
    weights = np.array(weights)
    weights = torch.from_numpy(weights).to(device)
    for label, values in relational_prototypes.items():
        for prototype_idx in range(len(values)):
            # Index is a class-list position, not the contributing client's ID.
            relational_prototypes[label][prototype_idx] = (
                relational_prototypes[label][prototype_idx] * weights[prototype_idx]
            )
    for label, values in relational_prototypes.items():
        consistent_prototypes[label] = torch.sum(torch.stack(values), dim=0)
    return consistent_prototypes
