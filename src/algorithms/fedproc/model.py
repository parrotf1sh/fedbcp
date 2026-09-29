import torch.nn as nn
import torch.nn.functional as F

from .utils import classifier_module, forward_with_features

__all__ = ["ModelWithFeatures", "ModelWithProjection"]


class ModelWithFeatures(nn.Module):
    """Keep the project model intact and expose FedProc's classifier-input features."""

    def __init__(self, model):
        super().__init__()
        classifier_module(model)
        self.model = model

    def forward(self, data, get_features=False):
        if get_features:
            return forward_with_features(self.model, data)
        return self.model(data)


class ModelWithProjection(nn.Module):
    """Optional original FedProc head on the project's existing feature extractor."""

    def __init__(self, model, out_dim=256):
        super().__init__()
        name, classifier = classifier_module(model)
        feature_dim = classifier.in_features
        num_classes = classifier.out_features
        setattr(model, name, nn.Identity())
        self.features = model
        self.linear_1 = nn.Linear(feature_dim, feature_dim)
        self.linear_2 = nn.Linear(feature_dim, out_dim)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, data, get_features=False):
        features = self.features(data)
        features = self.linear_1(features)
        features = F.relu(features)
        features = self.linear_2(features)
        features = F.normalize(features, p=2, dim=1)
        logits = self.classifier(features)
        if get_features:
            return logits, features
        return logits
