import aiger
from aiger.lazy import lazy


def test_lazy_call_smoke():
    x = aiger.atom('x')

    circ = x.aig
    lcirc = lazy(circ)

    assert lcirc({'x': True}) == circ({'x': True})
