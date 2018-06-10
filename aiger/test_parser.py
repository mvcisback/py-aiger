from tempfile import NamedTemporaryFile

import hypothesis.strategies as st
from hypothesis import given

from aiger import hypothesis as aigh
from aiger import parser as aigp


@given(aigh.Circuits, st.data())
def test_load(circ, data):
    with NamedTemporaryFile() as f:
        circ.write(f.name)
        circ2 = aigp.load(f.name)

    assert circ.inputs == circ2.inputs
    assert circ.outputs == circ2.outputs
    test_input = {f'{i}': data.draw(st.booleans()) for i in circ.inputs}
    assert circ(test_input) == circ2(test_input)


@given(aigh.Circuits, st.data())
def test_parse(circ, data):
    circ2 = aigp.parse(repr(circ))

    assert circ.inputs == circ2.inputs
    assert circ.outputs == circ2.outputs
    test_input = {f'{i}': data.draw(st.booleans()) for i in circ.inputs}
    assert circ(test_input) == circ2(test_input)
