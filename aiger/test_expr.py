import hypothesis.strategies as st
from hypothesis import given

from aiger.expr import atom


@given(st.data())
def test_expr_and(data):
    x, y = atom('x'), atom('y')
    expr = x & y

    vals = {f'{i}': data.draw(st.booleans()) for i in expr.inputs}
    assert (x(vals) and y(vals)) == expr(vals)


@given(st.data())
def test_expr_or(data):
    x, y = atom('x'), atom('y')
    expr = x | y

    vals = {f'{i}': data.draw(st.booleans()) for i in expr.inputs}
    assert (x(vals) or y(vals)) == expr(vals)


@given(st.data())
def test_expr_xor(data):
    x, y = atom('x'), atom('y')
    expr = x ^ y

    vals = {f'{i}': data.draw(st.booleans()) for i in expr.inputs}
    assert (x(vals) ^ y(vals)) == expr(vals)


@given(st.data())
def test_expr_eq(data):
    x, y = atom('x'), atom('y')
    expr = x == y

    vals = {f'{i}': data.draw(st.booleans()) for i in expr.inputs}
    assert (x(vals) == y(vals)) == expr(vals)


@given(st.data())
def test_expr_implies(data):
    x, y = atom('x'), atom('y')
    expr = x.implies(y)

    vals = {f'{i}': data.draw(st.booleans()) for i in expr.inputs}
    assert ((not x(vals)) or y(vals)) == expr(vals)


@given(st.data())
def test_expr_invert(data):
    x = atom('x')
    expr = ~x

    vals = {f'{i}': data.draw(st.booleans()) for i in expr.inputs}
    assert x(vals) == (not expr(vals))
