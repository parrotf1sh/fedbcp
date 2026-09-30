"""CPU-only client history. Snapshots are read-only inputs to load_state_dict.

No torch import is needed here; tensor operations are provided by the model.
Unvisited clients share one immutable initialization; updates get separate copies.
"""


def cpu_snapshot(weights):
    return {key: tensor.detach().cpu().clone() for key, tensor in weights.items()}


def state_bytes(weights):
    return sum(tensor.numel() * tensor.element_size() for tensor in weights.values())


class ClientHistory:
    def __init__(self, weights, n_clients):
        self.n_clients = n_clients
        self.initial = cpu_snapshot(weights)
        self.states = {}
        self.last_updated = {}

    def _validate(self, client):
        if not 0 <= client < self.n_clients:
            raise IndexError("Client index out of range")

    def get(self, client):
        self._validate(client)
        return self.states.get(client, self.initial)

    def age(self, client, round_idx):
        self._validate(client)
        previous = self.last_updated.get(client)
        return None if previous is None else round_idx - previous

    def update(self, client, weights, round_idx):
        self._validate(client)
        self.states[client] = cpu_snapshot(weights)
        self.last_updated[client] = round_idx

    def nbytes(self):
        return state_bytes(self.initial) + sum(state_bytes(state) for state in self.states.values())
