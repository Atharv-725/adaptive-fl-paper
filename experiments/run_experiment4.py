import sys, os
sys.path.append('.')

from config import Config
from src.data.data_loader import get_mnist, iid_split, apply_drift, get_dataloader
from src.client.fl_client import FLClient, SimpleMLP
from src.server.fl_server import FLServer
from src.utils.metrics import save_csv, plot_accuracy
from tqdm import tqdm
from collections import defaultdict
import time


def run_scale_experiment(num_clients, label):
    config = Config()
    config.NUM_CLIENTS = num_clients
    config.NUM_ROUNDS  = 20

    train_data, test_data = get_mnist()
    client_splits = iid_split(train_data, config.NUM_CLIENTS)
    test_loader   = get_dataloader(test_data, batch_size=256, shuffle=False)

    server  = FLServer(SimpleMLP, config)
    clients = [
        FLClient(i, get_dataloader(client_splits[i], config.BATCH_SIZE), config)
        for i in range(config.NUM_CLIENTS)
    ]

    num_drift = max(1, int(num_clients * 0.3))
    drift_applied = [False] * config.NUM_CLIENTS
    results       = []
    round_times   = []

    for r in tqdm(range(config.NUM_ROUNDS), desc=label):
        t0             = time.time()
        global_weights = server.get_weights()
        updates        = []

        for client in clients:
            if r >= 10 and client.client_id < num_drift and not drift_applied[client.client_id]:
                drifted_ds = apply_drift(client_splits[client.client_id], intensity=0.6)
                client.dataloader = get_dataloader(drifted_ds, config.BATCH_SIZE)
                drift_applied[client.client_id] = True

            client.set_weights(global_weights)
            weights, _ = client.train()
            w = client.get_weight()
            updates.append((weights, w))

        server.aggregate(updates)
        acc = server.evaluate(test_loader)
        elapsed = time.time() - t0
        round_times.append(elapsed)

        results.append({
            'round':       r + 1,
            'accuracy':    round(acc * 100, 4),
            'round_time':  round(elapsed, 3),
            'num_clients': num_clients,
            'method':      label
        })

    avg_time = sum(round_times) / len(round_times)
    print(f"\n  {label}: avg round time = {avg_time:.2f}s")
    return results


if __name__ == '__main__':
    os.makedirs('./results/csv',   exist_ok=True)
    os.makedirs('./results/plots', exist_ok=True)

    all_results = []
    for n, label in [(5, '5 clients'), (20, '20 clients'), (50, '50 clients')]:
        all_results.extend(run_scale_experiment(n, label))

    save_csv(all_results, './results/csv/experiment4_results.csv',
            fieldnames=['round', 'accuracy', 'round_time', 'num_clients', 'method'])

    grouped = defaultdict(list)
    for row in all_results:
        grouped[row['method']].append(row['accuracy'])

    rounds = list(range(1, 21))
    plot_accuracy(rounds, dict(grouped),
                title='Scalability: accuracy vs number of clients',
                save_path='./results/plots/experiment4_scalability.png')

    print("\nExperiment 4 complete.")