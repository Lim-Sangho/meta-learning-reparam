import dgl
import pyro
import torch
from torch import nn
from pyro.infer import SVI
from pyro.poutine import reparam
from pyro.infer.autoguide import AutoNormal, AutoDiagonalNormal, init_to_feasible
from pyro.infer.autoguide.utils import deep_getattr, deep_setattr

from FlowGraph import FlowGraph, edge_type, edge_id
from VIPReparam import VIPReparam
from util import run_nuts


class MLP(nn.Module):
    
    def __init__(self, dims, bias=True):
        super().__init__()
        self.layer = [nn.Linear(dims[0], dims[1], bias=bias)]
        for i in range(1, len(dims) - 1):
            self.layer.append(nn.Tanh())
            self.layer.append(nn.Linear(dims[i], dims[i+1], bias=bias))
        self.layer = nn.Sequential(*self.layer)

    def forward(self, x):
        return self.layer(x)


class WeightedConv(nn.Module):

    def __init__(self, n_hidden):
        super(WeightedConv, self).__init__()
        self.mlp = MLP([n_hidden*2, n_hidden * 2, n_hidden])

    def forward(self, G, feature):
        """
        G.ndata['feature']: (n_nodes, n_hidden)
        G.edata['weight']: (n_edges, n_hidden, n_hidden)
        -> G.nodes.mailbox['message']: (n_edges, n_hidden)
        -> G.ndata['aggregate']: (n_nodes, n_hidden)
        -> output: (n_nodes, n_hidden)
        """
        with G.local_scope():
            G.ndata['feature'] = feature
            G.update_all(message_func=self.udf_u_mm_e, reduce_func=dgl.function.sum('message', 'aggregate'))
            output = self.mlp(torch.cat([feature, G.ndata['aggregate']], dim=-1))
            output = (feature + G.ndata['aggregate']) / 2
            return output

    @staticmethod
    def udf_u_mm_e(edges):
        """
        edges.src['feature']: (n_edges, n_hidden)
        edges.data['weight']: (n_edges, n_hidden, n_hidden)
        -> edges.dst['message']: (n_edges, n_hidden)
        """
        return {'message': torch.einsum('ei,eij->ej', edges.src['feature'], edges.data['weight'])}


class GNN(nn.Module):

    def __init__(self, n_hidden, n_layer):
        super(GNN, self).__init__()
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.param = nn.Parameter(torch.Tensor(len(edge_type), n_hidden, n_hidden).normal_())
        self.conv = WeightedConv(n_hidden)
        self.mlp = MLP([n_hidden, n_hidden * 2, n_hidden * 2, 1])
    
    def to_homogeneous(self, G):
        """
        Convert G into a homogeneous graph G_homo only equipped with G_homo.ndata['feature'].
        Cache the edge ids of G to allocate G_homo.edata['weight'] in each forward pass.
        This method is called only once before a training starts.
        """
        const = G.ndata['const']['const']
        self.param_id = []
        for etype in G.canonical_etypes:
            self.param_id += [edge_id[etype]] * G.num_edges(etype)
        self.param_id = torch.tensor(self.param_id)
        G.ndata['feature'] = {'const': const.unsqueeze(-1) ** torch.ones(self.n_hidden),
                              'sample': torch.zeros(G.num_nodes('sample'), self.n_hidden)}
        G_homo = dgl.to_homogeneous(G, ndata=['feature'])
        return G_homo

    def forward(self, G_homo):
        """
        G_homo.ndata['feature']: (n_nodes, n_hidden)
        G_homo.edata['weight']: (n_edges, n_hidden, n_hidden)
        -> output: (n_nodes,)
        """
        G_homo.edata['weight'] = self.param[self.param_id]
        output = G_homo.ndata['feature']
        for _ in range(self.n_layer):
            output = self.conv(G_homo, output)
        output = self.mlp(nn.functional.normalize(output, dim=-1)).squeeze(-1).sigmoid()
        return output


class AutoReparam():
    
    def __init__(self, gnn, optims, loss, model, batch_depth, *args, init_loc_fn=init_to_feasible, **kwargs):
        self.gnn = gnn
        self.optims = optims
        self.loss = loss
        self.model = model
        self.batch_depth = batch_depth
        self.args = args
        self.kwargs = kwargs

        self.G, index = FlowGraph(self.model).get_graph(*args, **kwargs)
        reG = dgl.to_heterogeneous(dgl.to_homogeneous(self.G), self.G.ntypes, self.G.etypes)
        self.index = {name: torch.take(reG.ndata[dgl.NID]["sample"], heteroid) for name, heteroid in index.items()}
        reparam_model = reparam(model, config={name: VIPReparam(batch_depth=batch_depth) for name in index})
        self.guide = AutoNormal(reparam_model, init_loc_fn=init_loc_fn)
        self.G_homo = gnn.to_homogeneous(self.G)
    
    def step(self):
        context = self.gnn(self.G_homo)
        config = {name: VIPReparam(torch.take(context, homoID), batch_depth=self.batch_depth) for name, homoID in self.index.items()}
        reparam_model = reparam(self.model, config)

        self.optims["gnn"].zero_grad()
        svi = SVI(reparam_model, self.guide, self.optims["svi"], self.loss)
        loss = svi.step(*self.args, **self.kwargs)
        self.optims["gnn"].step()

        return loss

    def get_context(self, name):
        context = self.gnn(self.G_homo)
        return torch.take(context, self.index[name])

    def get_reparam_model(self):
        context = self.gnn(self.G_homo).detach()
        config = {name: VIPReparam(torch.take(context, homoID.cpu()), batch_depth=self.batch_depth) for name, homoID in self.index.items()} 
        return reparam(self.model, config)