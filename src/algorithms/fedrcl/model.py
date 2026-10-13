import torch.nn as nn

from train_tools.models.fedavgnet import FedAvgNetCIFAR, FedAvgNetTiny
from train_tools.models.vgg import VGG

__all__ = ["ModelWithLayerFeatures"]


class ModelWithLayerFeatures(nn.Module):
    """Observe the project model's activations without replacing its forward."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.feature_layers = self._feature_layers(model)
        # As in the source, the last layer is the classifier output.
        self.num_layers = len(self.feature_layers) + 1

    def _feature_layers(self, model):
        if isinstance(model, (FedAvgNetCIFAR, FedAvgNetTiny)):
            # One shared ReLU is called after conv1, conv2, and linear_1.
            return [("relu", 0), ("relu", 1), ("relu", 2)]
        if isinstance(model, VGG):
            layers = []
            last_relu = None
            for name, module in model.features.named_children():
                if isinstance(module, nn.ReLU):
                    last_relu = "features." + name
                elif isinstance(module, nn.MaxPool2d):
                    if last_relu is None:
                        raise ValueError("FedRCL requires a ReLU in each VGG stage")
                    layers.append((last_relu, 0))
                    last_relu = None
            if last_relu is not None:
                layers.append((last_relu, 0))
            layers.extend(
                ("fc." + name, 0)
                for name, module in model.fc.named_children()
                if isinstance(module, nn.ReLU)
            )
            if len(layers) != 7:
                raise ValueError("FedRCL requires five VGG stages and two FC ReLUs")
            return layers
        raise ValueError("FedRCL supports fedavg_cifar, fedavg_tiny, and vgg11")

    def forward(self, data, get_features=False, get_layer_features=False, no_relu=True):
        if not get_layer_features:
            if get_features:
                return self.model(data, get_features=True)
            return self.model(data)

        results = {}
        selections = {}
        for layer_idx, (name, occurrence) in enumerate(self.feature_layers):
            selections.setdefault(name, {})[occurrence] = layer_idx
        calls = {name: 0 for name in selections}
        handles = []
        modules = dict(self.model.named_modules())

        def collect(name, feature):
            occurrence = calls[name]
            calls[name] += 1
            if occurrence in selections[name]:
                layer_idx = selections[name][occurrence]
                # VGG uses inplace ReLUs. Clone without detaching the loss branch.
                results["layer{}".format(layer_idx)] = feature.clone()

        def pre_hook(name):
            def hook(module, inputs):
                collect(name, inputs[0])
            return hook

        def post_hook(name):
            def hook(module, inputs, output):
                collect(name, output)
            return hook

        def collect_classifier_input(module, inputs):
            results["feature"] = inputs[0]

        try:
            for name in selections:
                if no_relu:
                    handle = modules[name].register_forward_pre_hook(pre_hook(name))
                else:
                    handle = modules[name].register_forward_hook(post_hook(name))
                handles.append(handle)
            handles.append(
                self.model.classifier.register_forward_pre_hook(collect_classifier_input)
            )
            logits = self.model(data)
        finally:
            for handle in handles:
                handle.remove()

        if any("layer{}".format(i) not in results for i in range(self.num_layers - 1)):
            raise RuntimeError("FedRCL did not observe every configured feature layer")
        results["logit"] = logits
        results["layer{}".format(self.num_layers - 1)] = logits
        return results
