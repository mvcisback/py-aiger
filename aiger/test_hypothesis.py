from aiger.hypothesis import make_circuit


def test_smoke():
    make_circuit(['(~(a & b) & a)'])
    make_circuit(['Z(~(a & b) & a)'])
