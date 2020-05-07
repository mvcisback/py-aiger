import aiger
from aiger.lazy import lazy


def test_lazy_call_smoke():
    x = aiger.atom('x')

    circ = x.aig
    lcirc = lazy(circ)

    assert lcirc({'x': True}) == circ({'x': True})


def test_lazy_flatten_smoke():
    x = aiger.atom('x')

    circ = x.aig
    circ2 = lazy(circ).aig

    assert circ2.inputs == circ.inputs
    assert circ2.outputs == circ.outputs


def test_lazy_seq_compose_smoke():
    x, y, z = aiger.atoms('x', 'y', 'z')

    circ = (x & y).with_output('z').aig
    lcirc1 = lazy(circ)
    lcirc2 = lazy(z.aig)
    lcirc3 = lcirc1 >> lcirc2

    assert lcirc3.inputs == {'x', 'y'}
    assert lcirc3.outputs == {z.output}

    expr = aiger.BoolExpr(lcirc3)
    assert expr({'x': True, 'y': True})
    assert not expr({'x': False, 'y': True})
    assert not expr({'x': True, 'y': False})
    assert not expr({'x': False, 'y': False})


def test_lazy_par_compose_smoke():
    x, y = aiger.atoms('x', 'y')

    lcirc1 = lazy(x.with_output('x').aig)
    lcirc2 = lazy(y.with_output('y').aig)
    lcirc12 = lcirc1 | lcirc2

    assert lcirc12.inputs == {'x', 'y'}
    assert lcirc12.outputs == {'x', 'y'}

    assert lcirc12({'x': True, 'y': True})[0] == {'x': True, 'y': True}
