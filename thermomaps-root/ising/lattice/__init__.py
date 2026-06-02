# thermomaps-root/ising/lattice/__init__.py
from .arch_lattices import gen_lattice

_CACHE = {}

def adjacency_list(size: int, lattice_type: str = "square", pbc: bool = True):
    key = (size, lattice_type, pbc)
    if key not in _CACHE:
        graph = gen_lattice(lattice_type, [size, size], pbc)
        _CACHE[key] = graph.adjacency_list()
    return _CACHE[key]