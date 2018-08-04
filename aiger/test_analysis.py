from aiger import atom
from aiger.analysis import satisfiable

x, y = atom('x'), atom('y')
expr_sat = x | y
expr_unsat = expr_sat & ~ expr_sat


def test_satisfiable():
    assert satisfiable(expr_sat)


def test_unsatisfiable():
    assert not satisfiable(expr_unsat)
