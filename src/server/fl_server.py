import torch
import copy

class FLServer:
    def __init__(self, model_class, config):
        self.config       = config
        self.global_model = model_class(
            config.INPUT_SIZE,
            config.HIDDEN_SIZE,
            config.NUM_CLASSES
        )
        self.round_num  = 0
        self.weight_log = []

    def get_weights(self):
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, client_updates):
        total_weight = sum(w for _, w in client_updates)
        new_state    = copy.deepcopy(client_updates[0][0])
        for key in new_state:
            new_state[key] = torch.zeros_like(new_state[key], dtype=torch.float32)
            for state_dict, w in client_updates:
                new_state[key] += (w / total_weight) * state_dict[key].float()
        self.global_model.load_state_dict(new_state)
        self.round_num += 1
        self.weight_log.append([w for _, w in client_updates])

    def evaluate(self, test_loader):
        self.global_model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in test_loader:
                preds    = self.global_model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
        return correct / total if total > 0 else 0.0