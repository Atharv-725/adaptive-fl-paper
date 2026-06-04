import sys, os
sys.path.append('.')

from config import Config
from src.data.data_loader import get_mnist, iid_split, apply_drift, get_dataloader
from src.client.fl_client import FLClient, SimpleMLP
from src.server.fl_server import FLServer
from src.utils.metrics import save_csv, plot_accuracy
from tqdm import tqdm
from collections import defaultdict
import math


def make_weight_fn(formula):
    if formula == 'linear':
        return lambda d: max(1.0 - d, 0.05)
    elif formula == 'exponential':
        return lambda d: math.exp(-d)
    elif formula == 'binary':
        return lambda d: 0.1 if d > 0.3 else 1.0


def run_weight_experiment(formula, label):
    config = Config()
    weight_fn = make_weight_fn(formula)

    train_data, test_data = get_mnist()
    client_splits = iid_split(train_data, config.NUM_CLIENTS)
    test_loader   = get_dataloader(test_data, batch_size=256, shuffle=False)

    server  = FLServer(SimpleMLP, config)
    clients = [
        FLClient(i, get_dataloader(client_splits[i], config.BATCH_SIZE), config)
        for i in range(config.NUM_CLIENTS)
    ]

    drift_applied = [False] * config.NUM_CLIENTS
    results       = []

    for r in tqdm(range(config.NUM_ROUNDS), desc=label):
        global_weights = server.get_weights()
        updates        = []

        for client in clients:
            if r >= config.DRIFT_START_ROUND and client.client_id < 3 and not drift_applied[client.client_id]:
                drifted_ds = apply_drift(client_splits[client.client_id], intensity=0.6)
                client.dataloader = get_dataloader(drifted_ds, config.BATCH_SIZE)
                drift_applied[client.client_id] = True

            client.set_weights(global_weights)
            weights, _ = client.train()
            w = weight_fn(client.drift_score)
            updates.append((weights, w))

        server.aggregate(updates)
        acc = server.evaluate(test_loader)
        results.append({
            'round':    r + 1,
            'accuracy': round(acc * 100, 4),
            'method':   label
        })

    return results


if __name__ == '__main__':
    os.makedirs('./results/csv',   exist_ok=True)
    os.makedirs('./results/plots', exist_ok=True)

    all_results = []
    for formula, label in [
        ('linear',      'w = 1 - d'),
        ('exponential', 'w = exp(-d)'),
        ('binary',      'w = binary threshold')
    ]:
        all_results.extend(run_weight_experiment(formula, label))

    save_csv(all_results, './results/csv/experiment3_results.csv',
            fieldnames=['round', 'accuracy', 'method'])

    grouped = defaultdict(list)
    for row in all_results:
        grouped[row['method']].append(row['accuracy'])

    rounds = list(range(1, Config().NUM_ROUNDS + 1))
    plot_accuracy(rounds, dict(grouped),
                title='Weight formula sensitivity analysis',
                save_path='./results/plots/experiment3_weights.png')

    print("\nExperiment 3 complete.")