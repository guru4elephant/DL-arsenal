import collections
from itertools import repeat
from paddle import compat as cpt

def volumn(var_desc, batch):
    val = reduce(lambda x, y: x * y, var_desc.shape(), 1)
    if val < 0:
        return abs(val * batch)
    else:
        return val

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return list(repeat(x, n))
    return parse

def find_var_by_name(block, name):
    return block.desc.var(cpt.to_bytes(name))

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)
