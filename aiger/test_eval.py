import hypothesis.strategies as st
from hypothesis import given

from aiger import common


@given(st.integers(2, 10), st.data())
def test_and(n_inputs, data):
    aag = common.and_gate([f'x{i}' for i in range(n_inputs)], "out")
    test_input = {f'x{i}': data.draw(st.booleans()) for i in range(n_inputs)}
    out, _ = aag(test_input)
    assert out['out'] == all(test_input.values())


@given(st.integers(2, 10), st.data())
def test_or(n_inputs, data):
    aag = common.or_gate([f'x{i}' for i in range(n_inputs)], "out")
    test_input = {f'x{i}': data.draw(st.booleans()) for i in range(n_inputs)}
    out, _ = aag(test_input)
    assert out['out'] == any(test_input.values())
