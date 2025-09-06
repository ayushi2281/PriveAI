# PriveAI
PriveAI is a simulation of privacy-preserving federated learning on the Breast Cancer dataset. It compares centralized models, federated learning, and federated learning with differential privacy, showing how institutions can collaborate without sharing raw data while balancing accuracy and privacy.
# Privacy-Preserving Federated Learning (Breast Cancer)

This repo simulates **federated learning across two hospitals** on the Breast Cancer Wisconsin (Diagnostic) dataset, adds **differential privacy (DP)** on clients, and compares against a **centralized baseline**.

> **Highlights**
> - Clean, end-to-end **Jupyter Notebook** (`notebooks/assignment4_federated_dp.ipynb`)
> - **Flower (flwr)** for federated orchestration
> - **Opacus** for DP-SGD on clients
> - Clear comparisons: Centralized vs FL (no-DP) vs FL (DP)

---

## Quickstart

### 1) Setup environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> If torch fails to install via `pip`, install a wheel from https://pytorch.org/get-started/locally/ that matches your CUDA/CPU.

### 2) Run the notebook
Open and run this: `notebooks/assignment4_federated_dp.ipynb`  
It will: load data → create non-IID hospital splits → train & evaluate centralized baselines → produce plots/tables.

### 3) Federated Learning (no DP)
Open **two terminals** (A and B) for the two hospitals, and **one terminal** for the server.

**Terminal S (server):**
```bash
python -m src.fl_server --rounds 10 --eval-every 1 --test-split 0.2
```

**Terminal A (hospital A client):**
```bash
python -m src.fl_client --site A --epochs 2 --batch-size 64 --lr 0.001 --dp 0
```

**Terminal B (hospital B client):**
```bash
python -m src.fl_client --site B --epochs 2 --batch-size 64 --lr 0.001 --dp 0
```

After the rounds finish, the global model will be saved to `models/fed_nodp_final.pt` and metrics will be printed on the server side.

### 4) Federated Learning (with DP)
Use the **same** server command. For each client, enable DP and set hyperparams:

```bash
python -m src.fl_client --site A --epochs 2 --batch-size 64 --lr 0.001   --dp 1 --clip 1.0 --noise 1.1 --delta 1e-5
python -m src.fl_client --site B --epochs 2 --batch-size 64 --lr 0.001   --dp 1 --clip 1.0 --noise 1.1 --delta 1e-5
```

The DP run saves `models/fed_dp_final.pt`, logs ε at the end, and prints metrics (server side).

---

## Repo Structure

```
federated-learning-dp/
├── notebooks/
│   └── assignment4_federated_dp.ipynb
├── src/
│   ├── data_split.py
│   ├── centralized_training.py
│   ├── fl_server.py
│   ├── fl_client.py
│   ├── dp_utils.py
│   └── utils.py
├── models/
├── results/
├── requirements.txt
├── README.md
└── report.pdf
```

---

## Expected Outputs

- **Class distribution per hospital** (table/plot)
- **Centralized baselines** (LR & RF): Accuracy, Precision, Recall, F1, ROC-AUC; plus ROC curves
- **FL curves**: global test ROC-AUC vs rounds (no-DP vs DP)
- **Comparison table**: Centralized vs FL no-DP vs FL DP
- **Saved models**:
  - `models/centralized_best.joblib`
  - `models/fed_nodp_final.pt`
  - `models/fed_dp_final.pt`

---

## Reproducibility

- Fixed random seeds (numpy, torch, sklearn)
- Version-pinned `requirements.txt`
- Notebook runs **top-to-bottom** on clean env
- Clear commands to run server/clients

---

## Notes

- Target mapping in scikit-learn breast cancer dataset is typically `0 = malignant`, `1 = benign`. The notebook confirms this.
- DP accounting uses `Opacus`'s privacy engine to report `ε` for a given `δ ≈ 1/N²`.
- Keep the MLP small to avoid overfitting local shards.
- Non-IID split uses a feature threshold to induce heterogeneity by default (configurable).

