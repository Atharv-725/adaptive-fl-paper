import sys, os
sys.path.append('.')

from config import Config
from src.data.data_loader import get_mnist, iid_split, apply_drift, get_dataloader
from src.client.fl_client import FLClient, SimpleMLP
from src.server.fl_server import FLServer
from src.utils.metrics import save_csv, plot_accuracy
from tqdm import tqdm
from collections import defaultdict
from river.drift import ADWIN, PageHinkley
import copy


class FLClientPH(FLClient):
    """Client using Page-Hinkley detector instead of ADWIN."""
    def __init__(self, client_id, dataloader, config):
        super().__init__(client_id, dataloader, config)
        self.ph = PageHinkley()

    def train(self):
        weights, avg_loss = super().train()
        self.ph.update(avg_loss)
        return weights, avg_loss


class FLClientRolling(FLClient):
    """Client using simple rolling mean drift detection."""
    def _update_drift_score(self):
        if len(self.loss_history) < 5:
            self.drift_score = 0.0
            return
        recent   = sum(self.loss_history[-3:]) / 3
        baseline = sum(self.loss_history[:-3]) / max(len(self.loss_history[:-3]), 1)
        raw      = (recent - baseline) / (baseline + 1e-8)
        self.drift_score = float(min(max(raw, 0.0), 1.0))


def run_detector_experiment(client_class, label):
    config = Config()
    config.NUM_ROUNDS = 30

    train_data, test_data = get_mnist()
    client_splits = iid_split(train_data, config.NUM_CLIENTS)
    test_loader   = get_dataloader(test_data, batch_size=256, shuffle=False)

    server  = FLServer(SimpleMLP, config)
    clients = [
        client_class(i, get_dataloader(client_splits[i], config.BATCH_SIZE), config)
        for i in range(config.NUM_CLIENTS)
    ]

    drift_applied = [False] * config.NUM_CLIENTS
    results       = []
    drift_detected_round = None

    for r in tqdm(range(config.NUM_ROUNDS), desc=label):
        global_weights = server.get_weights()
        updates        = []

        for client in clients:
            if r >= config.DRIFT_START_ROUND and client.client_id < 3 and not drift_applied[client.client_id]:
                drifted_ds = apply_drift(client_splits[client.client_id], intensity=0.6)
                client.dataloader = get_dataloader(drifted_ds, config.BATCH_SIZE)
                drift_applied[client.client_id] = True

            client.set_weights(global_weights)
            weights, loss = client.train()
            w = client.get_weight()
            updates.append((weights, w))

            if drift_detected_round is None and client.drift_score > 0.1:
                drift_detected_round = r + 1

        server.aggregate(updates)
        acc = server.evaluate(test_loader)
        results.append({
            'round':    r + 1,
            'accuracy': round(acc * 100, 4),
            'method':   label,
            'drift_detected_round': drift_detected_round or -1
        })

    return results


if __name__ == '__main__':
    os.makedirs('./results/csv',   exist_ok=True)
    os.makedirs('./results/plots', exist_ok=True)

    all_results = []
    for cls, label in [
        (FLClient,       'ADWIN'),
        (FLClientPH,     'PageHinkley'),
        (FLClientRolling,'RollingMean')
    ]:
        all_results.extend(run_detector_experiment(cls, label))

    save_csv(all_results, './results/csv/experiment2_results.csv',
             fieldnames=['round', 'accuracy', 'method', 'drift_detected_round'])

    grouped = defaultdict(list)
    for row in all_results:
        grouped[row['method']].append(row['accuracy'])

    rounds = list(range(1, Config().NUM_ROUNDS + 1))
    plot_accuracy(rounds, dict(grouped),
                title='Detector comparison: ADWIN vs Page-Hinkley vs Rolling mean',
                save_path='./results/plots/experiment2_detectors.png')

    for method in ['ADWIN', 'PageHinkley', 'RollingMean']:
        rows = [r for r in all_results if r['method'] == method]
        d    = rows[0]['drift_detected_round']
        print(f"{method}: drift detected at round {d}")

    print("\nExperiment 2 complete.")