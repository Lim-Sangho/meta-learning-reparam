import torch
import random
import numpy as np
import torch.optim as to
import pyro.optim  as po
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pyro.infer import Trace_ELBO
from pyro.poutine import reparam

from VIPReparam import VIPReparam
from AutoReparam import GNN, AutoReparam
from model import neals_funnel, eight_schools
from util import run_nuts, samples_to_df, plot_scatter3d


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_default_device("cuda:0")
    
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = neals_funnel
    batch_depth = 1
    n_hidden = 5
    n_layer = 5
    n_model_svi = 1000
    n_model_hmc = 25

    gnn = GNN(n_hidden, n_layer)
    loss = Trace_ELBO(num_particles=100, vectorize_particles=True)

    y_svi = torch.rand(n_model_svi) * 40 - 20
    sigma_svi = torch.rand(n_model_svi) * 20
    args_svi = (y_svi, sigma_svi)
    kwargs_svi = {}

    n_sample = 10000
    n_warmup = 1000
    n_chain = 20

    bound = 20
    step = 5
    y_hmc = torch.linspace(-bound, bound, step)
    sigma_hmc = torch.linspace(0.1, bound, step)
    y_hmc, sigma_hmc = map(torch.flatten, torch.meshgrid(y_hmc, sigma_hmc, indexing="ij"))
    
    # y_hmc = torch.rand(8, n_model_hmc) * 40 - 20
    # sigma_hmc = torch.rand(8, n_model_hmc) * 20
    args_hmc = (y_hmc.cpu(), sigma_hmc.cpu())
    kwargs_hmc = {}

    scatter_idx = slice(None)
    scatter_name = "x"

    # Warmup
    print("==> Warmup")
    optims_warmup = {"gnn": to.Adam(gnn.parameters(), lr=0), "svi": po.ClippedAdam({"lr": 1e-2})}
    train = AutoReparam(gnn, optims_warmup, loss, model, batch_depth, *args_svi, **kwargs_svi)
    for _ in (epoch := tqdm(range(1000))):
        _loss = train.step()
        epoch.set_description("Loss: {:.4f}".format(_loss))

    # Train
    print("==> Train")
    optims_train = {"gnn": to.Adam(gnn.parameters(), lr=1e-2), "svi": po.ClippedAdam({"lr": 1e-2})}
    train = AutoReparam(gnn, optims_train, loss, model, batch_depth, *args_svi, **kwargs_svi)
    for _ in (epoch := tqdm(range(2000))):
        _loss = train.step()
        epoch.set_description("Loss: {:.4f}".format(_loss))
    plot_scatter3d(args_svi[0][scatter_idx], args_svi[1][scatter_idx], train.get_context(scatter_name)[scatter_idx], alpha=0.1, save="result/neals_funnel/train.pdf")
    torch.save(torch.stack([args_svi[0], args_svi[1], train.get_context(scatter_name)]).detach().cpu(), "result/neals_funnel/train.pt")

    # HMC
    torch.set_default_device("cpu")
    for y, sigma in zip(*args_hmc):
        test = AutoReparam(gnn.cpu(), None, None, model, batch_depth, y.unsqueeze(-1), sigma.unsqueeze(-1), **kwargs_hmc)

        models = {"no": model,
                  "full": reparam(model, config={scatter_name: VIPReparam(0.0)}),
                  "half": reparam(model, config={scatter_name: VIPReparam(0.5)}),
                  "auto": test.get_reparam_model()}

        for label in models:
            sample = run_nuts(models[label], n_sample, n_warmup, n_chain, y.unsqueeze(-1), sigma.unsqueeze(-1), **kwargs_hmc)
            samples_to_df(sample).to_csv(f"result/neals_funnel/{label}_y_{y:.2f}_sigma_{sigma:.2f}.csv")