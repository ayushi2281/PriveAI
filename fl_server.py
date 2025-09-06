import argparse, os, json
import flwr as fl
from typing import Dict, OrderedDict, List, Tuple
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch

SEED = 42

def get_test_data(test_size=0.2):
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)
    scaler = StandardScaler()
    X_test_s = scaler.fit_transform(X)  # use full dataset to fit scaler for convenience
    X_test_s = scaler.transform(X_test)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    return X_test_t, y_test_t

def get_evaluate_fn(test_size=0.2):
    X_test_t, y_test_t = get_test_data(test_size=test_size)
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config):
        # parameters correspond to the MLP weights
        # The client defines the model architecture; here we'll just compute AUC via a helper client eval
        # For simplicity, return 0 loss and 0 metrics; real evaluation happens on a dedicated evaluation client or by saving weights.
        # In practice, you'd reconstruct the model here.
        return 0.0, {"server_round": server_round}
    return evaluate

def main(rounds=10, eval_every=1, test_split=0.2):
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(test_size=test_split),
        evaluate_metrics_aggregation_fn=None,
    )
    fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=rounds), strategy=strategy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--test-split", type=float, default=0.2)
    args = parser.parse_args()
    main(rounds=args.rounds, eval_every=args.eval_every, test_split=args.test_split)
