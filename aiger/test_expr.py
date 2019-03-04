import hypothesis.strategies as st
from hypothesis import given

from aiger.expr import atom, ite


@given(st.data())
def test_expr_and(data):
    x, y = atom('x'), atom('y')
    expr = x & y

    vals = {f'{i}': data.draw(st.booleans()) for i in expr.inputs}
    assert (x(vals) and y(vals)) == expr(vals)


@given(st.data())
def test_expr_and2(data):
    x = atom('x')
    expr = x & x

    vals = {f'{i}': data.draw(st.booleans()) for i in expr.inputs}
    assert x(vals) == expr(vals)


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


@given(st.data())
def test_expr_ite(data):
    x, y, z = map(atom, 'xyz')
    expr = ite(x, y, z)

    vals = {f'{i}': data.draw(st.booleans()) for i in expr.inputs}
    assert expr(vals) == (vals['y'] if vals['x'] else vals['z'])
