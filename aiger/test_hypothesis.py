from aiger.hypothesis import make_circuit


def test_smoke():
    make_circuit(['(~(a & b) & a)'])
