from tempfile import NamedTemporaryFile

import hypothesis.strategies as st
from hypothesis import given, settings

from aiger import hypothesis as aigh
from aiger import parser as aigp


@settings(deadline=500)
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


TEST1 = """aag 5 1 2 1 1
2
6 9 0
4 2 1
9
8 7 5
i0 ap1
o0 x
l0 y
l1 z
c
PZap1
"""


@given(st.data())
def test_smoke1(data):
    circ1 = aigp.parse(TEST1)
    circ2 = aigp.parse(str(circ1))
    test_input = {f'{i}': data.draw(st.booleans()) for i in circ1.inputs}
    assert circ1(test_input) == circ2(test_input)


TEST2 = """aag 2 1 0 1 0
2
3
i0 ap1
o0 x
c
~ap1
"""


@given(st.data())
def test_smoke2(data):
    circ1 = aigp.parse(TEST2)
    circ2 = aigp.parse(str(circ1))
    test_input = {f'{i}': data.draw(st.booleans()) for i in circ1.inputs}
    assert circ1(test_input) == circ2(test_input)


def test_mutex_example_smoke():
    aigp.load('tests/mutex_converted.aag')
