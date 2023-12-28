#%%
import pyro
from FlowDistributions import Normal
from FlowGraph import FlowGraph
import torch

def model():
    x = pyro.sample("x", Normal(0,1))
    y = pyro.sample("y", Normal(x,torch.FloatTensor([1,2,3])), obs = torch.tensor([1.5, 1.5,13]))
    #z = pyro.sample("z", Normal(x,y))

def model_obs():
    x = pyro.sample("x", Normal(0,1))
    y = pyro.sample("y", Normal(x,torch.FloatTensor([1,2,3])), obs = torch.tensor([1.5, 1.5]))

#%%
fg = FlowGraph(model)


import networkx as nx, dgl
import matplotlib.pyplot as plt

g = fg.get_graph()
#print(g)
ng = dgl.to_networkx(dgl.to_homogeneous(g))

plt.figure(figsize=[7,7])
nx.draw(ng)

#%%
import pyro

tr = pyro.poutine.trace(model).get_trace()