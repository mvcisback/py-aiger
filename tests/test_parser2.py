import pytest

from aiger import parser


TEST1 = """aag 14 2 2 2 10
4
14
2 26 0
12 28 0
28
26
6 2 4
8 3 5
10 2 4
16 12 14
18 13 15
20 17 19
22 10 20
24 11 21
26 7 9
28 23 25
i0 y[0]
i1 y[1]
o0 z[1]
o1 z[0]
l0 x[0]
l1 x[1]
c
Test 1
Test 2
"""


def test_parser2_test1():
    aag = parser.parse(TEST1, to_aig=False)
    aag2 = parser.parse(str(aag), to_aig=False)
    assert TEST1 == str(aag) == str(aag2)


TEST2 = """aag 1 1 0 1 0
2
2
i0 ap1
o0 3aedbf54-22f7-11eb-97f0-f1c009e72b66
c
ap1
"""


def test_parser2_test2():
    aag = parser.parse(TEST2, to_aig=False)
    aag2 = parser.parse(str(aag), to_aig=False)
    assert TEST2 == str(aag) == str(aag2)



TEST3 = """aag 1 1 0 1 0
2
2
i0 ap1
c
ap1
"""

TEST3_2 = """aag 1 1 0 1 0
2
2
i0 ap1
o0 {}
c
ap1
"""


def test_parser2_test3():
    aag = parser.parse(TEST3, to_aig=False)
    assert len(aag.outputs) == 1
    out, *_ = aag.outputs
    expected = TEST3_2.format(out)
    assert str(aag) == expected
