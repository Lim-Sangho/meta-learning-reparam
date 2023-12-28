import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from pyro.ops.stats import effective_sample_size

def select_columns(df, param):
    return df[[column for column in df if f"{param}" in column]]

def get_ess(df, param, num_chains, num_samples):
    df = select_columns(df, param)
    samples = torch.Tensor(df.to_numpy()).reshape(num_samples, num_chains, -1)
    ess = effective_sample_size(samples, chain_dim=1, sample_dim=0).numpy()
    return pd.DataFrame(ess[None], columns=df.columns)

def plot_ESS(df, params, title, num_chains, num_samples):
    assert type(params) in (tuple, list, np.ndarray)
    if type(params) is np.ndarray: params = params.tolist()
    if not type(params[0]) in (tuple, list):
        params = [params]

    nrows = len(params)
    ncols = max(map(len, params))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if nrows == 1: axes = [axes]
    if ncols == 1: axes = [[axis] for axis in axes]

    for i, j in product(range(nrows), range(ncols)):
        if j < len(params[i]):
            param = params[i][j]
            axes[i][j].set_title(f"ESS for {param}")
            ess = get_ess(df, param, num_chains, num_samples)
            if len(ess.columns) <= 5:
                axes[i][j].axis("off")
                axes[i][j].table(cellText=np.stack([ess.columns, ess.values.round().flatten()]).T,
                                    colLabels=["parameter", "ESS"], colLoc="right", loc="center")
            else:
                axes[i][j].hist(ess.values.flatten())
        else:
            axes[i][j].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
