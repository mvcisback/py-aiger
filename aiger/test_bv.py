from aiger.bv import BV

# additional imports for testing frammework
import hypothesis.strategies as st
from hypothesis import given


@given(st.integers(-128, 127))
def test_signed_value(int_value):
    var = BV(8, 'x')
    assert int_value == var({'x': int_value})


@given(st.text(), st.integers(-128, 127))
def test_assign(var_name, int_value):
    var = BV(8, var_name)
    assigned = var.assign({var_name: int_value})
    assert int_value == assigned()


@given(st.integers(-128, 127), st.integers(-128, 127))
def test_assign2(a, b):
    var = BV(16, 'a')
    assigned = var.assign({'a': a})
    num = BV(16, b)
    assert a + b == (assigned + num)()


def test_repeat():
    assert BV(1, 1).repeat(3)() == -1
    assert BV(2, 1).repeat(3)() == 21


def test_slicing():
    assert BV(4, 6)[2]() == -1
    assert BV(4, 6)[0]() == 0
    assert BV(4, 6)[1:3]() == BV(2, 3)()
    assert BV(4, 6)[::-1]() == BV(4, 6)()


def test_concat():
    assert (BV(4, 1).concat(BV(3, 0)))() == 1
    assert (BV(4, 0).concat(BV(3, 1)))() == 16


@given(st.integers(-128,127), st.integers(-128,127))
def test_addition(a, b):
    assert (BV(9, 'a') + BV(9, 'b'))({'a': a, 'b': b}) == a + b


@given(st.integers(-128,127), st.integers(-128,127))
def test_and(a, b):
    e = BV(8, a) & BV(8, b)
    assert e() == a & b


@given(st.integers(-128,127), st.integers(-128,127))
def test_or(a, b):
    e = BV(8, a) | BV(8, b)
    assert e() == a | b


@given(st.integers(-128,127), st.integers(-128,127))
def test_xor(a, b):
    e = BV(8, a) ^ BV(8, b)
    assert e() == a ^ b


@given(st.integers(-128,127), st.integers(-128,127))
def test_eq(a, b):
    e = (BV(8, a) == BV(8, b))
    assert e() == (a == b)


@given(st.integers(-128, 127))
def test_abs(int_value):
    bv = abs(BV(8, int_value))
    assert bv() == abs(int_value)



# def test_bv_class():
#     # Abs
#     assert value(abs(BV(8, -17))) == 17
#     assert value(abs(BV(8, 42))) == 42

#     # Equality
#     assert unsigned_value(BV(4, 2) == BV(4, 2)) == 1
#     assert unsigned_value(BV(4, 2) != BV(4, 2)) == 0
#     assert unsigned_value(BV(4, 2) == BV(4, 3)) == 0
#     assert unsigned_value(BV(4, 2) != BV(4, 3)) == 1

#     # Comparison
#     assert unsigned_value(BV(4, 2) < BV(4, 3)) == 1
#     assert unsigned_value(BV(4, 3) < BV(4, 2)) == 0
#     assert unsigned_value(BV(4, 2) > BV(4, 3)) == 0
#     assert unsigned_value(BV(4, 3) > BV(4, 2)) == 1
#     assert unsigned_value(BV(4, 2) <= BV(4, 3)) == 1
#     assert unsigned_value(BV(4, 3) <= BV(4, 3)) == 1
#     assert unsigned_value(BV(4, 4) <= BV(4, 3)) == 0
