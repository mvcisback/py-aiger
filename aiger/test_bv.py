import os

from aiger.bv import BV
# from aiger.bv import *
from aiger.bv_utils import unsigned_value, value


def test_bv_class():
    # TODO: make these tests work on Travis
    if "TRAVIS" in os.environ and os.environ["TRAVIS"] == "true":
        return True

    # Repeat
    assert unsigned_value(BV(1, 1).repeat(3)) == 7
    assert unsigned_value(BV(2, 1).repeat(3)) == 21

    # Items and Slicing
    assert unsigned_value(BV(4, 6)[2]) == 1
    assert unsigned_value(BV(4, 6)[0]) == 0
    assert unsigned_value(BV(4, 6)[1:3]) == unsigned_value(BV(2, 3))
    assert unsigned_value(BV(4, 6)[::-1]) == unsigned_value(BV(4, 6))

    # Concatenation
    assert unsigned_value((BV(4, 1).concat(BV(3, 0)))) == 1
    assert unsigned_value((BV(4, 0).concat(BV(3, 1)))) == 16

    # Values
    assert value(BV(4, 6)) == 6
    assert unsigned_value(BV(4, -6)) == 10
    assert value(BV(4, -6)) == -6

    # Addition
    assert value(BV(16, 6) + BV(16, 3)) == value(BV(16, 9))
    assert value(-BV(16, -127)) == value(BV(16, 127))
    assert value(BV(16, -127)) == -value(BV(16, 127))
    assert value(BV(16, 0) - BV(16, 42)) == -value(BV(16, 42))

    # Assignment
    assert value(BV(5, 'x').assign({'x': 12})) == 12
    assert value((BV(8, 'x') - BV(8, 'y')).assign({'x': 12, 'y': 2})) == 10

    # Bitwise operators
    assert value(BV(8, 67) and BV(8, 66)) == 66
    assert value(BV(8, 67) or BV(8, 66)) == 67
    assert value(BV(8, 42) ^ BV(8, 42)) == 0
    assert value(BV(8, 127) ^ BV(8, 126)) == 1

    # Abs
    assert value(abs(BV(8, -17))) == 17
    assert value(abs(BV(8, 42))) == 42

    # Equality
    assert unsigned_value(BV(4, 2) == BV(4, 2)) == 1
    assert unsigned_value(BV(4, 2) != BV(4, 2)) == 0
    assert unsigned_value(BV(4, 2) == BV(4, 3)) == 0
    assert unsigned_value(BV(4, 2) != BV(4, 3)) == 1

    # Comparison
    assert unsigned_value(BV(4, 2) < BV(4, 3)) == 1
    assert unsigned_value(BV(4, 3) < BV(4, 2)) == 0
    assert unsigned_value(BV(4, 2) > BV(4, 3)) == 0
    assert unsigned_value(BV(4, 3) > BV(4, 2)) == 1
    assert unsigned_value(BV(4, 2) <= BV(4, 3)) == 1
    assert unsigned_value(BV(4, 3) <= BV(4, 3)) == 1
    assert unsigned_value(BV(4, 4) <= BV(4, 3)) == 0
