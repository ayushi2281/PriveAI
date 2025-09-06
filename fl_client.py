import argparse, os, json, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import flwr as fl
from opacus import PrivacyEngine
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x)

def load_site(site="A"):
    df = pd.read_csv(f"hospital_{site}.csv")
    y = df['target'].values.astype(np.float32)
    X = df.drop(columns=['target']).values.astype(np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_t = torch.tensor(Xs, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return X_t, y_t, Xs.shape[1]

def make_loader(X_t, y_t, batch_size=64, shuffle=True):
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def train_epoch(model, loader, optimizer, dp=False):
    model.train()
    total_loss = 0.0
    loss_fn = nn.BCEWithLogitsLoss()
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def get_weights(model):
    return [val.cpu().detach().numpy() for _, val in model.state_dict().items()]

def set_weights(model, weights):
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, site, epochs, batch_size, lr, dp, clip, noise, delta):
        X_t, y_t, in_dim = load_site(site=site)
        self.model = MLP(in_dim)
        self.loader = make_loader(X_t, y_t, batch_size=batch_size, shuffle=True)
        self.epochs = epochs
        self.dp = dp == 1
        self.clip = clip
        self.noise = noise
        self.delta = delta
        self.sample_rate = batch_size / len(self.loader.dataset)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.privacy_engine = None
        if self.dp:
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.loader,
                noise_multiplier=self.noise,
                max_grad_norm=self.clip,
            )
    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        for _ in range(self.epochs):
            train_epoch(self.model, self.loader, self.optimizer, dp=self.dp)
        metrics = {}
        if self.dp and self.privacy_engine is not None:
            eps = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
            metrics["epsilon"] = float(eps)
        return get_weights(self.model), len(self.loader.dataset), metrics

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        # Simple training loss on local data (proxy); real test is on server
        self.model.eval()
        loss_fn = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in self.loader:
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                total_loss += loss.item() * xb.size(0)
        return total_loss / len(self.loader.dataset), len(self.loader.dataset), {}

def main(site="A", epochs=2, batch_size=64, lr=1e-3, dp=0, clip=1.0, noise=1.1, delta=1e-5):
    client = FlowerClient(site=site, epochs=epochs, batch_size=batch_size, lr=lr, dp=dp, clip=clip, noise=noise, delta=delta)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=str, choices=["A","B"], default="A")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dp", type=int, choices=[0,1], default=0)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1.1)
    parser.add_argument("--delta", type=float, default=1e-5)
    args = parser.parse_args()
    main(site=args.site, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, dp=args.dp, clip=args.clip, noise=args.noise, delta=args.delta)
