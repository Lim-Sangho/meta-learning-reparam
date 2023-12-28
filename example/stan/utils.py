# %%
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from numpyro.diagnostics import effective_sample_size, gelman_rubin

def select_columns(df, param):
    return df.iloc[:,map(lambda column: bool(re.search(rf"\b{param}\b", column)), df.columns)]

def samples_to_df(samples):
    samples_1d = {}
    for var, sample in samples.items():
        sample = np.array(sample)
        event_dim = sample.ndim - 1
        for index in product(*[range(sample.shape[dim + 1]) for dim in range(event_dim)]):
            samples_1d[(var + "[%d]" * event_dim) % index] = sample[(...,) + index]
    df = pd.DataFrame(samples_1d)
    df.index.name = "draws"
    return df

def df_to_samples(df, params, shapes):
    samples = {}
    for param, shape in zip(params, shapes):
        df_param = select_columns(df, param)
        samples[param] = df_param.to_numpy().reshape((-1,) + shape)
    return samples

def get_ess(df, num_chains, num_samples):
    samples = df.to_numpy().reshape(num_chains, num_samples, -1)
    ess = effective_sample_size(samples)
    ess = pd.DataFrame(ess[None], columns=df.columns)
    ess[ess.isnull()] = 0
    return ess

def get_gr(df, num_chains, num_samples):
    samples = df.to_numpy().reshape(num_chains, num_samples, -1)
    gr = gelman_rubin(samples)
    gr = pd.DataFrame(gr[None], columns=df.columns)
    gr[gr.isnull()] = 0
    return gr

def plot_ESS(dfs, labels, params, num_chains, num_samples, title=None):
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
            esses = [get_ess(select_columns(df, param), num_chains, num_samples).values.flatten() for df in dfs]
            bins = np.histogram(np.hstack(esses))[1]
            for ess, label in zip(esses, labels):
                axes[i][j].hist(ess, bins=bins, label=label, alpha=0.5)
            axes[i][j].legend()
        else:
            axes[i][j].axis("off")

    fig.suptitle("Effective Sample Size" if title is None else title)
    plt.tight_layout()
    plt.show()

def plot_GR(dfs, labels, params, num_chains, num_samples, title=None):
    plt.figure(figsize=(len(params), len(params)))
    for df, label in zip(dfs, labels):
        xs, ys = np.array([]), np.array([])
        for param in params:
            gr_param = get_gr(select_columns(df, param), num_chains, num_samples).to_numpy().flatten()
            xs = np.concatenate([xs, [param] * len(gr_param)])
            ys = np.concatenate([ys, gr_param])
        plt.scatter(xs, ys, label=label, alpha=0.5)

    for tick in plt.gca().get_xticklabels():
        tick.set_rotation(45)

    plt.title("Gelman Rubin" if title is None else title)
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_posterior(dfs, labels, params, num_subsamples, title=None):
    sns.set(style="ticks", color_codes=True)
    df_pair = pd.concat([df.sample(num_subsamples).assign(label=label) for df, label in zip(dfs, labels)], ignore_index=True)
    fig = sns.pairplot(df_pair, hue="label", vars=params, height=2, kind="kde")
    fig.fig.suptitle("Pairwise Joint Posterior Density" if title is None else title, y=1.01)
    

if __name__ == "__main__":
    num_chains = 5
    num_samples = 2000

    df_snp500 = pd.read_csv("result/stochastic_volatility_snp500.csv")
    df_snp500_dct = pd.read_csv("result/stochastic_volatility_snp500_dct.csv")

    dfs = [df_snp500, df_snp500_dct]
    labels = ["none", "dct"]

    plot_ESS(dfs, labels,
            params=["v", "sigma", "nu", "r_loc"],
            num_chains=num_chains, num_samples=num_samples)

    # plot_GR(dfs, labels,
    #         params=["v", "sigma", "nu", "r_loc"],
    #         num_chains=num_chains, num_samples=num_samples)

    # plot_posterior(dfs, labels,
    #             params=["v[0]", "v[1]", "v[10]", "sigma", "nu", "r_loc"],
    #             num_subsamples=1000)

# %%