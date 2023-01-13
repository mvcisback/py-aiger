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


def test_degenerate_smoke():
    import aiger as A

    expr = A.BoolExpr(A.parse("""aag 0 0 0 1 0
0
"""))
    assert expr({}) is False
    expr = A.BoolExpr(A.parse("""aag 0 0 0 1 0
1
"""))
    assert expr({}) is True
    circ = A.parse("""aag 0 0 0 0 0
""")
    assert len(circ.node_map) == 0
    assert circ.inputs == circ.outputs == circ.latches == set()

    circ = A.parse("""aag 0 0 0 2 0
0
0
""")
    assert not any(circ({})[0].values())


def test_io_order():
    import aiger as A

    circ1 = A.parse("""aag 2 2 0 2 0
2  
4  
4  
2  
i0 a
i1 b
o0 ob
o1 oa
""")
    circ2 = A.parse("""aag 2 2 0 2 0
2  
4  
2  
4  
i0 a
i1 b
o0 oa
o1 ob
""")
    circ3 = A.parse("""aag 2 2 0 2 0
4  
2  
2  
4  
i0 b
i1 a
o1 ob
o0 oa
""")
    circ4 = A.parse("""aag 2 2 0 2 0
4  
2  
4  
2  
i0 b
i1 a
o0 ob
o1 oa
""")

    data = {'a': False, 'b': True}
    assert circ1(data) \
           == circ2(data) \
           == circ3(data) \
           == circ4(data)


TEST_NO_SYMBOL_TABLE = """aag 4 2 1 1 1
2
4
6 2
8
8 7 5
"""


@given(st.data())
def test_no_symbol_table(data):
    circ1 = aigp.parse(TEST_NO_SYMBOL_TABLE)
    circ2 = aigp.parse(str(circ1))
    test_input = {f'{i}': data.draw(st.booleans()) for i in circ1.inputs}
    assert circ1(test_input) == circ2(test_input)


# ----------- BINARY FILE PARSER TESTS ----------------

@given(st.data())
def test_smoke1_aig(data):
    circ1 = aigp.parse(TEST1)
    circ2 = aigp.load("tests/aig/test1.aig")
    test_input = {f'{i}': data.draw(st.booleans()) for i in circ1.inputs}
    assert circ1(test_input) == circ2(test_input)


@given(st.data())
def test_smoke2_aig(data):
    circ1 = aigp.parse(TEST2)
    circ2 = aigp.load("tests/aig/test2.aig")
    test_input = {f'{i}': data.draw(st.booleans()) for i in circ1.inputs}
    assert circ1(test_input) == circ2(test_input)


def test_mutex_example_smoke_aig():
    aigp.load('tests/aig/mutex_converted.aig')


def test_degenerate_smoke_aig():
    import aiger as A

    expr = A.BoolExpr(A.load("tests/aig/test_degenerate1.aig"))
    assert expr({}) is False
    expr = A.BoolExpr(A.load("tests/aig/test_degenerate2.aig"))
    assert expr({}) is True
    circ = A.load("tests/aig/test_degenerate3.aig")
    assert len(circ.node_map) == 0
    assert circ.inputs == circ.outputs == circ.latches == set()

    circ = A.load("tests/aig/test_degenerate4.aig")
    assert not any(circ({})[0].values())
