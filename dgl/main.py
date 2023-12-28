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


def main(gnn, optims, loss, model, batch_depth, args_svi, kwargs_svi,
         n_sample, n_warmup, n_chain, args_hmc, kwargs_hmc,
         scatter_name, scatter_idx):

    # SVI
    print("==> SVI")
    train = AutoReparam(gnn, optims, loss, model, batch_depth, *args_svi, **kwargs_svi)
    for _ in (epoch := tqdm(range(1000))):
        loss = train.step()
        epoch.set_description("Loss: {:.4f}".format(loss))
    plot_scatter3d(args_svi[0][scatter_idx], args_svi[1][scatter_idx], train.get_context(scatter_name)[scatter_idx], alpha=0.05, save="result/eight_schools/train.pdf")
    torch.save(torch.stack([args_svi[0], args_svi[1], train.get_context(scatter_name)]).detach().cpu(), "result/eight_schools/train.pt")

    # HMC
    torch.set_default_tensor_type(torch.FloatTensor)
    test = AutoReparam(gnn.cpu(), None, None, model, batch_depth, *args_hmc, **kwargs_hmc)
    plot_scatter3d(args_hmc[0][scatter_idx], args_hmc[1][scatter_idx], test.get_context(scatter_name)[scatter_idx], alpha=1, save="result/eight_schools/test.pdf")
    torch.save(torch.stack([args_hmc[0], args_hmc[1], test.get_context(scatter_name)]).detach().cpu(), "result/eight_schools/test.pt")

    reparam_models = [model, reparam(model, config={scatter_name: VIPReparam(0.0)}), reparam(model, config={scatter_name: VIPReparam(0.5)}), test.get_reparam_model()]
    labels = ["no", "full", "half", "auto"]

    for reparam_model, label in zip(reparam_models, labels):
        print(f"==> HMC ({label})")
        samples = run_nuts(reparam_model, n_sample, n_warmup, n_chain, *args_hmc, **kwargs_hmc)
        samples_to_df(samples).to_csv(f"result/{label}.csv")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    
    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    model = eight_schools
    batch_depth = 2
    n_hidden = 5
    n_layer = 5
    n_model_svi = 1000
    n_model_hmc = 25

    gnn = GNN(n_hidden, n_layer)
    optim_gnn = to.Adam(gnn.parameters(), lr=1e-3)
    optim_svi = po.ClippedAdam({"lr": 1e-6})
    optims = {"gnn": optim_gnn, "svi": optim_svi}
    loss = Trace_ELBO(num_particles=100, vectorize_particles=True)

    y_svi = torch.cat([torch.rand(1, n_model_svi) * 40 - 20, torch.zeros(7, n_model_svi)])
    sigma_svi = torch.cat([torch.rand(1, n_model_svi) * 20, torch.ones(7, n_model_svi)])
    args_svi = (y_svi, sigma_svi)
    kwargs_svi = {}

    n_sample = 10000
    n_warmup = 1000
    n_chain = 10

    # bound = 20
    # step = 5
    # y_hmc = torch.linspace(-bound, bound, step)
    # sigma_hmc = torch.linspace(0.1, bound, step)
    # y_hmc, sigma_hmc = map(torch.flatten, torch.meshgrid(y_hmc, sigma_hmc, indexing="ij"))
    
    y_hmc = torch.rand(8, n_model_hmc) * 40 - 20
    sigma_hmc = torch.rand(8, n_model_hmc) * 20
    args_hmc = (y_hmc.cpu(), sigma_hmc.cpu())
    kwargs_hmc = {}

    scatter_idx = [0]
    scatter_name = "theta"

    main(gnn, optims, loss, model, batch_depth, args_svi, kwargs_svi,
         n_sample, n_warmup, n_chain, args_hmc, kwargs_hmc,
         scatter_name, scatter_idx)