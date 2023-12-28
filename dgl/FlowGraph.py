from warnings import warn
import torch
from torch import Tensor
from enum import Enum
from pathlib import Path
from typing import (
    Any, BinaryIO, Callable, ContextManager, Dict, Iterable, Iterator, List,
    NamedTuple, Optional, overload, Sequence, Tuple, TypeVar, Type, Union,
    Generic, Set, AnyStr)
# from typing_extensions import Literal, Self
if torch.__version__ >= "2.0.0":
    from torch import inf
else:
    from torch._six import inf

from torch.types import _int, Number

from pyro.poutine.messenger import Messenger
import dgl
from dgl import DGLGraph

import itertools

BROADCAST = 1
OUT_OF_PLACE = 0
INPLACE = -1

shapeop = {"__getitem__":OUT_OF_PLACE, "expand":OUT_OF_PLACE, "broadcast_tensors":BROADCAST, "__setitem__":INPLACE}

op = {"add":2, "mul":2, "exp":1} #might use tuples to represent commutativity
sample = ["obs","scale", "loc"]
node_type = ["null", "const", "sample"]  #null, const must be the first and the second elements.

NULL = 0
CONST = 1

edge_type = [(ut, et, vt) for ut, et, vt in itertools.product(node_type, list(op.keys())+ sample, node_type)]
edge_type.extend([(ut, et + ".inv", vt) for ut, et, vt in edge_type])
edge_id = {et: i for i, et in enumerate(edge_type)}

node_type_dict = {tp : 0 for tp in node_type}
edge_type_dict = {(ut, et, vt) : ([],[]) for ut, et, vt in edge_type}

tensor_dtype = torch.float


class FlowGraph(object):

    def __init__(self, fn):
        self.FGMessenger = FlowGraphMessenger()
        self.fn = fn
        super().__init__()

    def __call__(self, *args, **kwargs):
        with self.FGMessenger:
            self.fn(*args, **kwargs)
        return self

    def get_graph(self, *args, **kwargs)->Tuple[DGLGraph,Dict[str,Tensor]]:
        with self.FGMessenger:
            self.fn(*args, **kwargs)
        return self.FGMessenger.graph, self.FGMessenger.index


class FlowGraphMessenger(Messenger):
    def __init__(self) -> None:
        super().__init__()
        self.graph:DGLGraph = dgl.heterograph(edge_type_dict, node_type_dict)
        self.index = {}
    
    def __enter__(self):
        NodeTensor.set_graph(self.graph)
        return super().__enter__()

    def _pyro_post_sample(self, msg:dict):
        if type(msg["fn"]).__name__ == "_Subsample":
            return
        
        sample_node = tensor_to_node(msg["value"], "sample")
        
        msg["fn"].draw_graph(sample_node)
        if msg["is_observed"]:
            obs_node = tensor_to_node(msg["value"], "const")
            self.graph.add_edges(obs_node.nid.flatten(), sample_node.nid.flatten(), etype=("const","obs", "sample"))
        else:
            self.index[msg["name"]]=sample_node.nid

        msg["value"] = sample_node

    def __exit__(self, exc_type, exc_value, traceback):
        NodeTensor.set_graph(None)
        return super().__exit__(exc_type, exc_value, traceback)

        
def tensor_to_node(tensor:Tensor, ntype:Union[Tensor,str], detached = False):
    node:NodeTensor = tensor.as_subclass(NodeTensor)
    node.init_node(ntype, detached)
    return node

def null_node(tensor:Tensor):
    if NodeTensor.graph is None:
        return tensor
    else:
        return tensor_to_node(tensor, "null")


class NodeTensor(Tensor):
    graph = None
    def init_node(self, ntype:Union[Tensor,str], detached = False) -> None:
        graph = NodeTensor.graph

        assert graph is not None

        raw = self.as_subclass(Tensor)
        raw = raw.type(tensor_dtype)
        if isinstance(ntype, str):
            ntype = torch.ones_like(raw, dtype=torch.int64) * node_type.index(ntype)

        assert ntype.shape == raw.shape

        self.nid = torch.zeros_like(raw, dtype=torch.int64)

        for i, name in enumerate(node_type[CONST:], start=CONST):
            indices = torch.nonzero(ntype == i, as_tuple=True)
            num_element = len(indices[0])
            if num_element == 0:
                continue
            cur_nid = graph.num_nodes(name)

            if len(raw.shape) == 0:
                self.nid = torch.tensor(cur_nid).type_as(self.nid)
                graph.add_nodes(num_element, {"const":raw.clone().unsqueeze(-1)} if i==CONST else None, name)
                break
            self.nid[indices] = torch.arange(cur_nid, cur_nid+num_element).type_as(self.nid)
            graph.add_nodes(num_element, {"const":raw[indices].clone()} if i==CONST else None, name)

        self.ntype = ntype
        self.detached = detached
    
    def realise_node(self) -> None:
        graph = NodeTensor.graph
        
        raw = self.as_subclass(Tensor)
        indices = torch.nonzero(self.ntype == NULL, as_tuple=True)
        num_element = len(indices[0])
        if num_element == 0:
            return
        
        cur_nid = graph.num_nodes("const")
        if len(raw.shape) == 0:
            self.nid = torch.tensor(cur_nid).type_as(self.nid)
            graph.add_nodes(num_element, {"const":raw.clone().unsqueeze(-1)}, "const")
        else:
            self.nid[indices] = torch.arange(cur_nid, cur_nid+num_element).type_as(self.nid)
            graph.add_nodes(num_element, {"const":raw[indices].clone()}, "const")

        self.ntype[indices] = CONST

    def set_node(self, ntype:Tensor = None, nid:Tensor = None, detached:bool = None) -> None:
        if ntype != None:
            self.ntype = ntype

        if nid != None:
            self.nid = nid

        if detached != None:
            self.detached = detached
    
    @classmethod        
    def set_graph(cls, graph:DGLGraph):
        cls.graph = graph

    def __repr__(self, *, tensor_contents=None):
        return "NodeTensor({}, type:{}, id:{}, detached:{})".format(self.as_subclass(Tensor), self.ntype if hasattr(self, "ntype") else None, self.nid if hasattr(self, "ntype") else None, self.detached if hasattr(self, "detached") else None)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # print(func)
        # print(cls)
        # print("name: {}".format(func.__name__))
        # print("types: ", types)
        # print("args: ", args)
        # print("kwargs: ", kwargs)
        # print("////////////////////////////////")

        fname = func.__name__

        if fname in shapeop.keys():
            
            if shapeop[fname] == INPLACE and not isinstance(args[0], NodeTensor):
                warn("Inplace operator on non-Node type: the information of input NodeTensor might be lost.")

            args_type = [t.ntype if isinstance(t,NodeTensor) else t for t in args]
            ntypes = func(*args_type, **kwargs)

            args_id = [t.nid if isinstance(t,NodeTensor) else t for t in args]
            nids = func(*args_id, **kwargs)

            args_val = [t.as_subclass(Tensor) if isinstance(t,NodeTensor) else t for t in args]
            vals = func(*args_val, **kwargs)

            if shapeop[fname] == BROADCAST:
                def wrap(val, ntype, nid, arg):
                    if not isinstance(arg,NodeTensor):
                        return val
                    node_tensor = val.as_subclass(NodeTensor)
                    node_tensor.set_node(ntype,nid,arg.detached)
                    return node_tensor

                return tuple(map(wrap, vals, ntypes, nids, args))
            
            
            elif shapeop[fname] == OUT_OF_PLACE:
                check_detached = lambda arg: isinstance(arg, NodeTensor) and arg.detached

                result_node = vals.as_subclass(NodeTensor)
                result_node.set_node(ntypes,nids,any(map(check_detached, args)))
                return result_node
            
            elif shapeop[fname] == INPLACE:
                return
            
            else:
                raise ValueError("Invalid shapeop types: ", shapeop[fname])

        
        else:
            args = tuple(t.as_subclass(Tensor).clone() if isinstance(t, NodeTensor) and t.detached else t for t in args)

            if all(map(lambda t: not isinstance(t, NodeTensor), args)):
                return func(*args, **kwargs)

            if fname in op.keys():                
                arg_num = op[fname]

                graph = NodeTensor.graph
                assert graph is not None

                for t in args:
                    if isinstance(t, Tensor):
                        dtype = t.dtype
                        device = t.device
                        break

                assert graph != None

                def arg_to_node(value):
                    if not isinstance(value, Tensor):
                        value = torch.tensor(value, dtype=dtype, device=device)
                    if not isinstance(value, NodeTensor):
                        value = tensor_to_node(value, "const")
                    value.realise_node()
                    return value
                
                nodes = torch.broadcast_tensors(*map(arg_to_node, args[0:arg_num]))
                
                new_args = [t.as_subclass(Tensor) if isinstance(t,NodeTensor) else t for t in args]
                
                
                result_node = func(*new_args, **kwargs)
                
                
                op_ntype_name = "sample"  #need modification if a new node type for op is implemented
                op_ntype = node_type.index(op_ntype_name)  
                
                result_ntype = torch.any(torch.stack(list(map(lambda nt: nt.ntype > CONST, nodes)), dim=0), dim=0) * (op_ntype-1) + CONST
                result_node = tensor_to_node(result_node, result_ntype)

                result_indices = torch.nonzero(result_node.ntype == op_ntype, as_tuple=True)
                if len(result_indices[0]) != 0:
                    if len(result_node.shape) == 0:
                        for node in nodes:
                            graph.add_edges(node.nid, result_node.nid, etype=(node_type[node.ntype], fname, op_ntype_name))
                            if node.ntype != CONST:
                                graph.add_edges(result_node.nid, node.nid, etype=(op_ntype_name, fname, node_type[node.ntype]))
                    
                    else:
                        compact_result_node = result_node[result_indices]
                        for node in nodes:
                            compact_node = node[result_indices]
                            for i, name in enumerate(node_type):
                                indices = torch.nonzero(compact_node.ntype == i, as_tuple=True)
                                if len(indices[0]) == 0:
                                    continue

                                result_nid = compact_result_node.nid[indices]
                                nid = compact_node.nid[indices]

                                graph.add_edges(nid, result_nid, etype=(name, fname, op_ntype_name))
                                if i != CONST:
                                    graph.add_edges(result_nid, nid, etype=(op_ntype_name, "{}.inv".format(fname), name))
                            
                            
                
                return result_node

        args = tuple(t.as_subclass(Tensor) if isinstance(t, NodeTensor) else t for t in args)
        
        return func(*args, **kwargs)