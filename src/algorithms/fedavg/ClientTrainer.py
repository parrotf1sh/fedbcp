import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """

    def train(self):
        if not self.algo_params.get("metrics_only", False):
            return super().train()
        self.model.train()
        self.model.to(self.device)
        loss_sum, correct, seen = 0.0, 0, 0
        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                data, targets = data.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(data)
                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()
                count = targets.numel()
                loss_sum += loss.detach().item() * count
                correct += (logits.detach().argmax(1) == targets).sum().item()
                seen += count
        if not seen:
            raise ValueError("Selected client has no training samples")
        return {"loss_sum": loss_sum, "correct": correct, "seen": seen}, self.datasize
