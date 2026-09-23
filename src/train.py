import argparse, csv, random
import numpy as np
import torch
from torch_geometric.utils import to_dense_adj

from src.data import load_injected_cora
from src.models.dominant import Dominant
from src.evaluate import evaluate


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(seed, args, device):
    set_seed(seed)
    data = load_injected_cora(seed=seed).to(device)
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
    model = Dominant(data.num_features, args.hidden, args.dropout, args.alpha).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = model.node_scores(data.x, data.edge_index, adj).mean()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        scores = model.node_scores(data.x, data.edge_index, adj).cpu().numpy()
    return evaluate(data.y.cpu().numpy(), scores)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in range(args.seeds):
        auroc, auprc = run(seed, args, device)
        rows.append((seed, auroc, auprc))
        print(f"seed {seed}: AUROC={auroc:.4f} AUPRC={auprc:.4f}")

    a = np.array([[r[1], r[2]] for r in rows])
    print(f"AUROC {a[:,0].mean():.4f} ± {a[:,0].std():.4f} | "
          f"AUPRC {a[:,1].mean():.4f} ± {a[:,1].std():.4f}")

    with open("results/dominant_cora.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "auroc", "auprc"])
        w.writerows(rows)