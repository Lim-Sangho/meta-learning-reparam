import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import NUTS, MCMC, HMC
from pyro.infer.reparam import AutoReparam
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam

import matplotlib.pyplot as plt
import torch

from multiESS import multiESS

def eight_school(treatment_effects, treatment_stddevs):
        assert(len(treatment_effects) == len(treatment_stddevs))

        mu = pyro.sample("mu",dist.Normal(0,10))
        log_tau = pyro.sample("log_tau",dist.Normal(5,1))

        with pyro.plate("effect_obs",len(treatment_effects)):
            school_effects = pyro.sample("school_effect",dist.Normal(mu,torch.exp(log_tau)))
            return pyro.sample("obs", dist.Normal(school_effects, treatment_stddevs), obs=treatment_effects)

def print_ess(diag):
    for key in diag:
        if key != "divergences" and key != "acceptance rate":
            print(key, " : ", diag[key])

if __name__ == "__main__":

    treatment_effects = torch.tensor(
        [28, 8, -3, 7, -1, 1, 18, 12], dtype=torch.float32)  # treatment effects
    treatment_stddevs = torch.tensor(
        [15, 10, 16, 11, 9, 11, 10, 18], dtype=torch.float32)

    num_samples = 5000
    warmup_steps = 1000
    num_chains = 20

    step_num = 2000

    #################### naive (centered) eight school model #####################
    pyro.clear_param_store()

    guide = AutoNormal(eight_school)

    svi = SVI(eight_school, guide, Adam({"lr": 0.02}),Trace_ELBO(num_particles=20))

    losses = []
    for step in range(step_num):
        loss = svi.step(treatment_effects,treatment_stddevs)
        losses.append(loss)
        if step % 100 == 0:
            print("loss : ", loss)
    
    plt.plot(losses)
    plt.show()

    final_loss = losses[-1]

    nuts_kernel = NUTS(eight_school)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps,num_chains=num_chains,mp_context="spawn")
    mcmc.run(treatment_effects,treatment_stddevs)

    mcmc.summary()

    
    #################### decentered eight school model #####################
    pyro.clear_param_store()
    decentered_school = AutoReparam(centered=0.0)(eight_school)

    guide = AutoNormal(decentered_school)

    svi = SVI(decentered_school, guide, Adam({"lr": 0.02}),Trace_ELBO(num_particles=20))

    losses = []
    for step in range(step_num):
        loss = svi.step(treatment_effects,treatment_stddevs)
        losses.append(loss)
        if step % 100 == 0:
            print("loss : ", loss)
    
    plt.plot(losses)
    plt.show()

    final_loss_dc = losses[-1]

    nuts_kernel_dc = NUTS(decentered_school)
    mcmc_dc = MCMC(nuts_kernel_dc, num_samples=num_samples, warmup_steps=warmup_steps,num_chains=num_chains,mp_context="spawn")
    mcmc_dc.run(treatment_effects,treatment_stddevs)

    mcmc_dc.summary()




    #################### VIP eight school model #####################
    pyro.clear_param_store()

    learnable_school = AutoReparam()(eight_school)

    guide = AutoNormal(learnable_school)

    svi = SVI(learnable_school, guide, Adam({"lr": 0.02}),Trace_ELBO(num_particles=20))

    losses = []
    for step in range(step_num):
        loss = svi.step(treatment_effects,treatment_stddevs)
        losses.append(loss)
        if step % 100 == 0:
            print("loss : ", loss)
    
    plt.plot(losses)
    plt.show()

    final_loss_l = losses[-1]

    nuts_kernel_l = NUTS(learnable_school)
    mcmc_l = MCMC(nuts_kernel_l, num_samples=num_samples, warmup_steps=warmup_steps,num_chains=num_chains,mp_context="spawn")
    mcmc_l.run(treatment_effects,treatment_stddevs)
    mcmc_l.summary()

    sample=mcmc.get_samples(group_by_chain=True)
    sample_dc=mcmc_dc.get_samples(group_by_chain=True)
    sample_l=mcmc_l.get_samples(group_by_chain=True)

    multi_sample = torch.cat([torch.stack([sample['log_tau'],sample['mu']],dim=2), sample['school_effect']],dim=2)
    multi_sample_dc = torch.cat([torch.stack([sample_dc['log_tau_decentered'],sample_dc['mu_decentered']],dim=2), sample_dc['school_effect_decentered']],dim=2)
    multi_sample_l = torch.cat([torch.stack([sample_l['log_tau_decentered'],sample_l['mu_decentered']],dim=2), sample_l['school_effect_decentered']],dim=2)


    totalESS = 0.0
    totalESS_dc = 0.0
    totalESS_l = 0.0
    for i in range(num_chains):
        print(i)
        totalESS += multiESS(multi_sample[i].numpy())
        totalESS_dc += multiESS(multi_sample_dc[i].numpy())
        totalESS_l += multiESS(multi_sample_l[i].numpy())

    print("           |  multiESS  |    ELBO    |")
    print("centered   | {:>10.2f} | {:>10.2f} |".format(totalESS, final_loss) )
    print("decentered | {:>10.2f} | {:>10.2f} |".format(totalESS_dc, final_loss_dc) )
    print("VIP        | {:>10.2f} | {:>10.2f} |".format(totalESS_l, final_loss_l) )