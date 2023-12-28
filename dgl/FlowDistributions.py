import pyro.distributions as dist
import torch

from torch.distributions.utils import broadcast_all
from dgl import DGLGraph
from FlowGraph import NodeTensor,node_type,tensor_to_node

class Normal(dist.Normal):

    def __init__(self, loc, scale, validate_args=None):
        loc, scale = broadcast_all(loc, scale)

        if isinstance(loc,NodeTensor):
            assert not hasattr(loc, "detached") or not loc.detached
            loc.set_node(detached=True)
            loc.realise_node()

        if isinstance(scale,NodeTensor):
            assert not hasattr(scale, "detached") or not scale.detached
            scale.set_node(detached=True)
            scale.realise_node()

        super().__init__(loc, scale, validate_args)
        
    def expand(self, batch_shape):
        return super().expand(batch_shape, self)

    def draw_graph(self, result_node:NodeTensor):
        graph = NodeTensor.graph

        loc_node = self.loc
        if not isinstance(loc_node, NodeTensor):
            loc_node = tensor_to_node(self.loc, "const")

        scale_node = self.scale
        if not isinstance(scale_node, NodeTensor):
            scale_node = tensor_to_node(self.scale, "const")

        for i, name in enumerate(node_type):
            indices = torch.nonzero(loc_node.ntype == i, as_tuple=True)
            if len(indices[0]) == 0:
                continue

            loc_nid = loc_node.nid[indices] if len(loc_node.shape) != 0 else loc_node.nid

            result_nid = result_node.nid[indices] if len(result_node.shape) != 0 else result_node.nid
            
            graph.add_edges(loc_nid, result_nid, etype=(name, "loc", "sample"))

            if name != "const":
                graph.add_edges(result_nid, loc_nid, etype=("sample", "loc.inv", name))

        for i, name in enumerate(node_type):
            indices = torch.nonzero(scale_node.ntype == i, as_tuple=True)

            if len(indices[0]) == 0:
                continue

            scale_nid = scale_node.nid[indices] if len(scale_node.shape) != 0 else scale_node.nid

            result_nid = result_node.nid[indices] if len(result_node.shape) != 0 else result_node.nid
            
            graph.add_edges(scale_nid, result_nid, etype=(name, "scale", "sample"))

            if name != "const":
                graph.add_edges(result_nid, scale_nid, etype=("sample", "scale.inv", name))