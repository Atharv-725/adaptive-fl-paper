import torch
import torch.nn as nn
import torch.optim as optim
import copy
from river.drift import ADWIN

class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class FLClient:
    def __init__(self, client_id, dataloader, config):
        self.client_id    = client_id
        self.dataloader   = dataloader
        self.config       = config
        self.model        = SimpleMLP(config.INPUT_SIZE, config.HIDDEN_SIZE, config.NUM_CLASSES)
        self.criterion    = nn.CrossEntropyLoss()
        self.loss_history = []
        self.drift_score  = 0.0
        self.adwin        = ADWIN()

    def set_weights(self, global_weights):
        self.model.load_state_dict(copy.deepcopy(global_weights))

    def train(self):
        self.model.train()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            momentum=0.9
        )
        total_loss = 0.0
        batches    = 0
        for _ in range(self.config.LOCAL_EPOCHS):
            for X, y in self.dataloader:
                optimizer.zero_grad()
                loss = self.criterion(self.model(X), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches    += 1
        avg_loss = total_loss / max(batches, 1)
        self.loss_history.append(avg_loss)
        self.adwin.update(avg_loss)
        self._update_drift_score()
        return self.model.state_dict(), avg_loss

    def _update_drift_score(self):
        if len(self.loss_history) < 2:
            self.drift_score = 0.0
            return
        recent   = self.loss_history[-1]
        baseline = sum(self.loss_history[:-1]) / len(self.loss_history[:-1])
        raw      = (recent - baseline) / (baseline + 1e-8)
        self.drift_score = float(min(max(raw, 0.0), 1.0))

    def get_weight(self):
        return max(1.0 - self.drift_score, 0.05)

    def evaluate(self, test_loader):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in test_loader:
                preds    = self.model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
        return correct / total if total > 0 else 0.0