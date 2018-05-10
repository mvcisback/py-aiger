
from bv import *
from calc import *

# Items and Slicing
assert unsigned_value(BV(4, 6)[2]) == 1
assert unsigned_value(BV(4, 6)[0]) == 0
assert unsigned_value(BV(4, 6)[1:3]) == unsigned_value(BV(2, 3))
assert unsigned_value(BV(4, 6)[::-1]) == unsigned_value(BV(4, 6))

# Concatenation
assert unsigned_value((BV(4, 1).concat(BV(3,0)))) == 1
assert unsigned_value((BV(4, 0).concat(BV(3,1)))) == 16


# Values
assert value(BV(4, 6)) == 6
assert unsigned_value(BV(4, -6)) == 10
assert value(BV(4, -6)) == -6

# Addition
assert value(BV(16, 6) + BV(16, 3)) == value(BV(16, 9))
assert value(- BV(16, -127)) == value(BV(16, 127))
assert value(BV(16, -127)) == - value(BV(16, 127))
assert value(BV(16, 0) - BV(16, 42)) == - value(BV(16, 42))

# Assignment
assert value(BV(5, 'x').assign({'x': 12})) == 12
assert value((BV(8, 'x') - BV(8, 'y')).assign({'x': 12, 'y': 2})) == 10


