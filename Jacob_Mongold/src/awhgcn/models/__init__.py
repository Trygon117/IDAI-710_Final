from .attention import HyperedgeAttentionMLP
from .hypergraph_conv import AWHGConv

def _lazy(name):
    def _load():
        import importlib
        return getattr(importlib.import_module(f".{name}", package=__name__), {
            "cnn_backbone": "CNNBackbone",
            "awhgcn": "AWHGCN",
        }[name])
    return _load

# CNNBackbone and AWHGCN require MONAI — import on demand
def CNNBackbone(*a, **kw):
    from .cnn_backbone import CNNBackbone as _C; return _C(*a, **kw)

def AWHGCN(*a, **kw):
    from .awhgcn import AWHGCN as _A; return _A(*a, **kw)
