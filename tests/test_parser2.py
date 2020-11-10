import pytest

from aiger import parser2


TEST = """aag 14 2 2 2 10
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


def test_parser2_smoke():
    aag = parser2.parse(TEST)
    aag2 = parser2.parse(str(aag))
    assert TEST == str(aag) == str(aag2)

