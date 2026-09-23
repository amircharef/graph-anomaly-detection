from torch_geometric.datasets import Planetoid
from pygod.generator import gen_contextual_outlier, gen_structural_outlier


def load_injected_cora(root="data", seed=0):
    """Cora with injected anomalies (~5%): contextual + structural."""
    data = Planetoid(root, "Cora")[0]
    data, y_ctx = gen_contextual_outlier(data, n=75, k=50, seed=seed)
    data, y_str = gen_structural_outlier(data, m=15, n=5, seed=seed)
    data.y = (y_ctx.bool() | y_str.bool()).long()
    return data