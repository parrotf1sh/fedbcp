import torch

from ..BaseClientTrainer import BaseClientTrainer
from .criterion import RetentionLoss
from .utils import classifier_module, cpu_state

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, global_counts, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.criterion = RetentionLoss(global_counts, cfg).to(self.device)

    def train_client(self, state, optimizer_state, data, teacher, round_idx, probe=None, batch_trace=None):
        self.model.load_state_dict(state)
        self.model.to(self.device)
        # Fresh local optimizer per selected client; never share momentum between clients.
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        self.optimizer.load_state_dict(optimizer_state)
        if self.cfg["student_init"] == "teacher":
            classifier_module(self.model)[1].load_state_dict(classifier_module(teacher)[1].state_dict())
        self.model.train()
        if teacher is not None:
            teacher.eval()
        counts = torch.as_tensor(data["class_counts"], dtype=torch.float32, device=self.device)
        sums = dict(classification=0.0, retention=0.0, band_active=0.0, teacher_correct=0.0)
        seen = correct = 0
        strength = float(round_idx >= self.cfg["warmup_rounds"])
        use_teacher = (teacher is not None and self.cfg["retention_mode"] != "none"
                       and self.cfg["retention_weight"] > 0 and strength > 0)
        for _ in range(self.local_epochs):
            for images, targets in data["train"]:
                if batch_trace is not None:
                    batch_trace.observe(images, targets)
                images, targets = images.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                teacher_logits = None
                if use_teacher:
                    with torch.no_grad():
                        teacher_logits = teacher(images)
                loss, metrics = self.criterion(logits, targets, teacher_logits, counts, strength)
                if probe is not None and teacher_logits is not None:
                    probe.observe(logits.detach(), teacher_logits.detach(), targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite FedBTR training loss")
                loss.backward()
                self.optimizer.step()
                batch = targets.numel()
                seen += batch
                correct += logits.detach().argmax(dim=1).eq(targets).sum().item()
                for key, value in metrics.items():
                    sums[key] += value.item() * batch
        if seen == 0:
            raise ValueError("Empty local training run")
        result = {key: value / seen for key, value in sums.items()}
        result.update(train_accuracy=correct / seen, training_examples=seen,
                      teacher_forward_examples=seen if use_teacher else 0)
        state = cpu_state(self.model)
        if any(not torch.isfinite(value).all() for value in state.values() if value.is_floating_point()):
            raise FloatingPointError("Non-finite client model update")
        return state, result
