from aiger.bv import BV, VAR_NAME_ALPHABET

# additional imports for testing frammework
import hypothesis.strategies as st
from hypothesis import given, settings, unlimited

var_name_generator = st.text(alphabet=VAR_NAME_ALPHABET)


@given(st.integers(-128, 127))
def test_signed_value(int_value):
    var = BV(8, 'x')
    assert int_value == var({'x': int_value})


@given(var_name_generator, st.integers(-128, 127))
def test_assign(var_name, int_value):
    var = BV(8, var_name)
    assigned = var.assign({var_name: int_value})
    assert int_value == assigned()


@given(st.integers(-128, 127), st.integers(-128, 127))
def test_assign2(a, b):
    var = BV(10, 'a')
    assigned = var.assign({'a': a})
    num = BV(10, b)
    assert a + b == (assigned + num)()


def test_repeat():
    assert BV(1, 1).repeat(3)() == -1
    assert BV(2, 1).repeat(3)() == 21


def test_slicing():
    assert BV(4, 6)[2]() == 1
    assert BV(4, 6)[0]() == 0
    assert BV(4, 6)[1:3]() == BV(2, 3)()
    assert BV(4, 6)[::-1]() == BV(4, 6)()


def test_concat():
    assert (BV(4, 1).concat(BV(3, 0)))() == 1
    assert (BV(4, 0).concat(BV(3, 1)))() == 16


@given(st.integers(-64, 63), st.integers(-64, 63))
def test_addition(a, b):
    assert (BV(8, 'a') + BV(8, 'b'))({'a': a, 'b': b}) == a + b


@given(st.integers(-128, 127), st.integers(-128, 127))
def test_and(a, b):
    e = BV(8, a) & BV(8, b)
    assert e() == a & b


@given(st.integers(-128, 127), st.integers(-128, 127))
def test_or(a, b):
    e = BV(8, a) | BV(8, b)
    assert e() == a | b


@given(st.integers(-128, 127), st.integers(-128, 127))
def test_xor(a, b):
    e = BV(8, a) ^ BV(8, b)
    assert e() == a ^ b


@given(st.integers(-128, 127), st.integers(-128, 127))
def test_eq(a, b):
    e = (BV(8, a) == BV(8, b))
    assert e() == (a == b)


@given(st.integers(-7, 7), st.integers(-7, 7))
def test_neq(a, b):
    e = (BV(4, a) != BV(4, b))
    assert e() == (a != b)


@given(st.integers(-7, 7))  # note that abs(-8) == -8
def test_abs(int_value):
    bv = abs(BV(4, int_value))
    assert bv() == abs(int_value)


@settings(max_examples=20, timeout=unlimited)
@given(st.integers(-8, 7), st.integers(-8, 7))
def test_lt(a, b):
    e = (BV(4, a) < BV(4, b))
    assert e() == (a < b)


@settings(max_examples=20, timeout=unlimited)
@given(st.integers(-8, 7), st.integers(-8, 7))
def test_gt(a, b):
    e = (BV(4, a) > BV(4, b))
    assert e() == (a > b)


@settings(max_examples=20, timeout=unlimited)
@given(st.integers(-8, 7), st.integers(-8, 7))
def test_le(a, b):
    e = (BV(4, a) <= BV(4, b))
    assert e() == (a <= b)


@settings(max_examples=20, timeout=unlimited)
@given(st.integers(-8, 7), st.integers(-8, 7))
def test_ge(a, b):
    e = (BV(4, a) >= BV(4, b))
    assert e() == (a >= b)


@settings(max_examples=20, timeout=unlimited)
@given(st.integers(-8, 7))
def test_reverse(a):
    e = BV(4, a)
    assert e() == (e.reverse().reverse())()
