import copy

from ..BaseClientTrainer import BaseClientTrainer
from .criterion import PrototypeLoss
from .utils import average_prototypes, cpu_snapshot, forward_with_features

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, lambda_proto, optimizer_class, **kwargs):
        super().__init__(**kwargs)
        self.criterion = PrototypeLoss(lambda_proto)
        self.optimizer_class = optimizer_class
        self.global_prototypes = {}
        self.local_prototypes = {}

    def download_global(self, local_weights, server_optimizer, global_prototypes):
        """Restore this client's own model and download shared prototypes only."""
        self.model.load_state_dict(local_weights, strict=True)
        self.model.to(self.device)
        # The server optimizer is an untrained schedule template with empty state.
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=0)
        self.optimizer.load_state_dict(copy.deepcopy(server_optimizer))
        self.global_prototypes = {
            label: prototype.detach().to(self.device)
            for label, prototype in global_prototypes.items()
        }

    def train(self):
        """Follow update_weights_het, including last-epoch online prototypes."""
        self.model.train()
        self.model.to(self.device)
        if self.trainloader is None or len(self.trainloader) == 0:
            raise ValueError("FedProto requires a nonempty client training loader")
        epoch_losses = {key: [] for key in ("total_loss", "ce_loss", "proto_loss")}

        for _ in range(self.local_epochs):
            batch_losses = {key: [] for key in epoch_losses}
            prototypes = {}
            for data, targets in self.trainloader:
                data, targets = data.to(self.device), targets.to(self.device)
                self.model.zero_grad()
                logits, features = forward_with_features(self.model, data)
                loss, loss_results = self.criterion(
                    logits, targets, features, self.global_prototypes
                )
                loss.backward()
                self.optimizer.step()

                # These are the pre-step forward features, not a second pass.
                for index, label in enumerate(targets):
                    prototypes.setdefault(label.item(), []).append(features[index].detach())
                predictions = logits[:, :self.num_classes].argmax(dim=1)
                last_batch_acc = predictions.eq(targets).float().mean().item()
                for key, value in loss_results.items():
                    batch_losses[key].append(value.item())
            if not batch_losses["total_loss"]:
                raise ValueError("FedProto client training produced no batches")
            for key, values in batch_losses.items():
                epoch_losses[key].append(sum(values) / len(values))

        self.local_prototypes = average_prototypes(prototypes)
        local_results = {key: sum(values) / len(values)
                         for key, values in epoch_losses.items()}
        local_results["last_batch_acc"] = last_batch_acc
        return local_results, self.datasize

    def upload_local(self):
        """Save local state for the next participation, never for model averaging."""
        return cpu_snapshot(self.model.state_dict())

    def upload_prototypes(self):
        return {label: value.detach().clone()
                for label, value in self.local_prototypes.items()}

    def reset(self):
        self.datasize = None
        self.trainloader = None
        self.testloader = None
        self.optimizer = None
        self.global_prototypes = {}
        self.local_prototypes = {}
