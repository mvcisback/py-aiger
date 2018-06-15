import hypothesis.strategies as st
from hypothesis import given

from aiger import hypothesis as aigh
from aiger.bv import BV


@given(aigh.Circuits, st.data())
def test_aig_to_aag(circ, data):
    circ2 = circ._to_aag()._to_aig()
    assert circ.inputs == circ2.inputs
    assert circ.outputs == circ2.outputs
    test_input = {f'{i}': data.draw(st.booleans()) for i in circ.inputs}
    assert circ(test_input) == circ2(test_input)


@given(st.data())
def test_bv_aig_to_aag(data):
    # TODO: generate random circuit generator.
    x = BV(4, 'x')
    y = BV(4, 'y')
    z = BV(4, 'z')
    circ = ((x + y) & BV(4, 'z') < BV(4, 3)).aig
    circ2 = circ._to_aag()._to_aig()
    assert circ.inputs == circ2.inputs
    assert circ.outputs == circ2.outputs
    test_input = {f'{i}': data.draw(st.booleans()) for i in circ.inputs}
    assert circ(test_input) == circ2(test_input)
