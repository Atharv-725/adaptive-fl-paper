import sys
sys.path.append('.')

from config import Config
from src.data.data_loader import get_mnist, iid_split, get_dataloader
from src.client.fl_client import FLClient, SimpleMLP
from src.server.fl_server import FLServer

config = Config()
config.NUM_CLIENTS = 3
config.NUM_ROUNDS  = 2
config.LOCAL_EPOCHS = 1

print("Loading MNIST...")
train_data, test_data = get_mnist()
client_splits = iid_split(train_data, config.NUM_CLIENTS)
test_loader   = get_dataloader(test_data, batch_size=256, shuffle=False)

print("Setting up server and clients...")
server  = FLServer(SimpleMLP, config)
clients = [FLClient(i, get_dataloader(client_splits[i], config.BATCH_SIZE), config)
           for i in range(config.NUM_CLIENTS)]

print("Running 2 test rounds...")
for r in range(config.NUM_ROUNDS):
    global_weights = server.get_weights()
    updates = []
    for client in clients:
        client.set_weights(global_weights)
        weights, loss = client.train()
        updates.append((weights, 1.0))
        print(f"  Round {r+1} | Client {client.client_id} | Loss: {loss:.4f}")
    server.aggregate(updates)
    acc = server.evaluate(test_loader)
    print(f"  Round {r+1} | Global accuracy: {acc*100:.2f}%")

print("\nSetup verified successfully!")
