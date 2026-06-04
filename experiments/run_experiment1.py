import sys
import os
sys.path.append('.')

from config import Config
from src.data.data_loader import get_mnist, iid_split, apply_drift, get_dataloader
from src.client.fl_client import FLClient, SimpleMLP
from src.server.fl_server import FLServer
from src.utils.metrics import save_csv, plot_accuracy
from tqdm import tqdm
from collections import defaultdict


def run_experiment(drift_fraction=0.3, use_adaptive=True, label='AdaptiveFL'):
    config = Config()
    print(f"\n{'='*55}")
    print(f"  Running: {label} | Drift fraction: {drift_fraction}")
    print(f"{'='*55}")

    train_data, test_data = get_mnist()
    client_splits = iid_split(train_data, config.NUM_CLIENTS)
    test_loader   = get_dataloader(test_data, batch_size=256, shuffle=False)

    server  = FLServer(SimpleMLP, config)
    clients = [
        FLClient(i, get_dataloader(client_splits[i], config.BATCH_SIZE), config)
        for i in range(config.NUM_CLIENTS)
    ]

    num_drift_clients = int(config.NUM_CLIENTS * drift_fraction)
    drift_applied     = [False] * config.NUM_CLIENTS
    results           = []

    for r in tqdm(range(config.NUM_ROUNDS), desc=label):
        global_weights = server.get_weights()
        updates        = []

        for client in clients:
            if (r >= config.DRIFT_START_ROUND
                    and client.client_id < num_drift_clients
                    and not drift_applied[client.client_id]):
                drifted_ds = apply_drift(
                    client_splits[client.client_id], intensity=0.6
                )
                client.dataloader = get_dataloader(drifted_ds, config.BATCH_SIZE)
                drift_applied[client.client_id] = True

            client.set_weights(global_weights)
            weights, loss = client.train()
            w = client.get_weight() if use_adaptive else 1.0
            updates.append((weights, w))

        server.aggregate(updates)
        acc = server.evaluate(test_loader)

        client_weights = [client.get_weight() for client in clients]
        avg_weight     = sum(client_weights) / len(client_weights)

        results.append({
            'round':      r + 1,
            'accuracy':   round(acc * 100, 4),
            'avg_weight': round(avg_weight, 4),
            'method':     label,
            'drift_frac': drift_fraction
        })

        if (r + 1) % 5 == 0:
            tqdm.write(
                f"  Round {r+1:02d} | Acc: {acc*100:.2f}% | Avg weight: {avg_weight:.3f}"
            )

    return results


if __name__ == '__main__':
    os.makedirs('./results/csv',   exist_ok=True)
    os.makedirs('./results/plots', exist_ok=True)

    all_results = []

    for drift_frac in [0.0, 0.3, 0.6]:
        r1 = run_experiment(
            drift_fraction=drift_frac,
            use_adaptive=False,
            label=f'FedAvg_drift{int(drift_frac*100)}'
        )
        r2 = run_experiment(
            drift_fraction=drift_frac,
            use_adaptive=True,
            label=f'AdaptiveFL_drift{int(drift_frac*100)}'
        )
        all_results.extend(r1)
        all_results.extend(r2)

    save_csv(
        all_results,
        './results/csv/experiment1_results.csv',
        fieldnames=['round', 'accuracy', 'avg_weight', 'method', 'drift_frac']
    )

    grouped = defaultdict(list)
    for row in all_results:
        grouped[row['method']].append(row['accuracy'])

    rounds = list(range(1, Config().NUM_ROUNDS + 1))
    plot_accuracy(
        rounds,
        dict(grouped),
        title='FedAvg vs AdaptiveFL under concept drift',
        save_path='./results/plots/experiment1_accuracy.png'
    )

    print("\nExperiment 1 complete. Results saved to ./results/")