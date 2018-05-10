
import common
import parser


def signal(name):
    return common.and_gate([name], output=name)

def rename_output(wordlen, aig, cur_name='x', new_name='y'):
    return aig['o', {cur_name + '[{}]'.format(i): new_name + '[{}]'.format(i) for i in range(wordlen)}]

# print(rename_output(2, word(2)))

def rename_input(wordlen, aig, cur_name='x', new_name='y'):
    return aig['i', {cur_name + '[{}]'.format(i): new_name + '[{}]'.format(i) for i in range(wordlen)}]

# print(rename_input(2, word(2)))

def bv_and(wordlen, output='x&y', left='x', right='y'):
    aig = common.empty()
    for i in range(wordlen):
        aig = aig | common.and_gate([left + '[{}]'.format(i), right + '[{}]'.format(i)], output=output + '[{}]'.format(i))
    return aig

def bv_or(wordlen, output='x&y', left='x', right='y'):
    aig = common.empty()
    for i in range(wordlen):
        aig = aig | common.or_gate([left + '[{}]'.format(i), right + '[{}]'.format(i)], output=output + '[{}]'.format(i))
    return aig

# print(bv_or(2))

def const(wordlen, value, output='x'):
    assert 2**wordlen > value
    aig = common.empty()
    for i in range(wordlen):
        aig = aig | common.source({output + '[{}]'.format(i): value % 2 == 1})
        value = value // 2
    return aig

# print(const(3, 2))
# print(const(3, 8))


def _full_adder(x, y, carry_in, result, carry_out):
    return parser.parse("aag 10 3 0 2 7\n2\n4\n6\n18\n21\n8 4 2\n10 5 3\n12 11 9\n14 12 6\n16 13 7\n18 17 15\n20 15 9\ni0 {}\ni1 {}\ni2 {}\no0 {}\no1 {}\n".format(x, y, carry_in, result, carry_out))
# print(full_adder('x', 'y', 'carry_in', 'x+y', 'carry_out'))



def adder(wordlen, output='x+y', left='x', right='y'):
    carry_name = output + '_carry'
    aig = common.source({carry_name: False})
    assert left != carry_name and right != carry_name
    for i in range(wordlen):
        fa = _full_adder(left + "[{}]".format(i), right + "[{}]".format(i), carry_name, output+'[{}]'.format(i), carry_name)
        aig = aig >> fa
    return aig

# print(adder(4))

def inc(wordlen, output='x+1', input='x'):
    return const(wordlen, 1, output='y') >> adder(wordlen, output=output, left=input, right='y')

# print(inc(3))

def negation(wordlen, output='not x', input='x'):
    # word(word) >> 
    return common.bit_flipper(inputs=[input + '[{}]'.format(i) for i in range(wordlen)],
                              outputs=[output + '[{}]'.format(i) for i in range(wordlen)])

# print(negation(2))

def negative(wordlen, output='-x', input='x'):
    """Returns the circuit computing x*(-1) in Two's complement"""
    return negation(wordlen, output='tmp', input=input) >> inc(wordlen, output=output, input='tmp')

# print(negative(4))

def subtraction(wordlen, output='x-y', left='x', right='y'):
    return negation(wordlen, input=right, output='-'+right) >> adder(wordlen, output=output, left=left, right='-'+right)



class BV:
    def __init__(self, size, kind):

        self.size = size
        self.variables = []

        out_name = 'bv{}'.format(size)  # all BVs use this name for outputs

        if isinstance(kind, int):  # Constant
            self.aig = const(size, kind, output=out_name)
        if isinstance(kind, str):  # Variable
            self.variables.append(kind)
            self.aig = common.empty()
            for i in range(self.size):
                self.aig = self.aig | signal(kind + '[{}]'.format(i))
        if isinstance(kind, tuple):  # for internal use only
            assert isinstance(kind[0], list)  # variables
            self.variables = kind[0]
            assert isinstance(kind[1], common.AAG)
            self.aig = kind[1]            
        
        assert len(self.aig.outputs) == self.size
        assert len(self.aig.inputs) == self.size*len(self.variables)

    def name(self):
        return "bv{}".format(self.size)

    def __repr__(self):
        return self.name() + ":\n" + str(self.aig)

    def rename(self, name):
        """Renames the output of the expression; mostly used internally"""
        new_aig = self.aig['o', {self.name() + '[{}]'.format(i): name + '[{}]'.format(i) for i in range(self.size)}]
        return BV(self.size, (self.variables.copy(), new_aig))

    def subtitute(self, subst):  # TODO: test this function; does the simultaneous thing work?
        """Simultaniously substitutes one set of input words by another"""
        d = dict()
        for old, new in subst:
            d.update({old + '[{}]'.format(i): new + '[{}]'.format(i) for i in range(self.size)}) 
        return BV(self.size, (self.variables.copy(), self.aig['i', d]))

    def __add__(self, other):
        assert self.size == other.size
        other = other.rename("other")
        return BV(self.size, 
                    (self.variables + other.variables, 
                     (self.aig | other.aig)
                        >> (adder(self.size, output=self.name(), left=self.name(), right='other') >> common.sink([self.name() + '_carry']))
                    ))

    def __len__(self):
        return self.size

    # TODO
    # Arithmetic operations
    # def __neg__(self):
    # def __pos__(self):
        # return self
    # def __sub__(self, other):
    # def __abs__(self):

    # Difficult arithmetic operations
    # def __mul__(self, other):
    # def __mod__(self, other):
    # def __div__(self, other):
    # def __pow__(self, other):

    # Word-level comparisons
    # def __lt__(self, other):
    # def __le__(self, other):
    # def __eq__(self, other):
    # def __ne__(self, other):
    # def __gt__(self, other):
    # def __ge__(self, other):

    # def __hash__(self):  # for use in strash; remember hash with every expression to avoid recomputation; remember global map from hashes to subexpressions
    
    # Bitwise opeators
    # def __getitem__(self, k):  # element access, and slice exctraction
    # def __invert__(self):  # ~x
    # def __rshift__(self, other):
    # def __lshift__(self, other):
    # def __or__(self, other):
    # def __and__(self, other):
    # def __xor__(self, other):


print(BV(4, 6) + BV(4, 3))


# self + other    __add__(self, other)
# self - other    __sub__(self, other)
# self * other    __mul__(self, other)
# self / other    __div__(self, other) or __truediv__(self,other) if __future__.division is active.
# self // other   __floordiv__(self, other)
# self % other    __mod__(self, other)
# divmod(self,other)  __divmod__(self, other)
# self ** other   __pow__(self, other)
# self & other    __and__(self, other)
# self ^ other    __xor__(self, other)
# self | other    __or__(self, other)
# self << other   __lshift__(self, other)
# self >> other   __rshift__(self, other)
# bool(self)  __nonzero__(self) (used in boolean testing)
# -self   __neg__(self)
# +self   __pos__(self)
# abs(self)   __abs__(self)
# ~self   __invert__(self) (bitwise)
# self += other   __iadd__(self, other)
# self -= other   __isub__(self, other)
# self *= other   __imul__(self, other)
# self /= other   __idiv__(self, other) or __itruediv__(self,other) if __future__.division is in effect.
# self //= other  __ifloordiv__(self, other)
# self %= other   __imod__(self, other)
# self **= other  __ipow__(self, other)
# self &= other   __iand__(self, other)
# self ^= other   __ixor__(self, other)
# self |= other   __ior__(self, other)
# self <<= other  __ilshift__(self, other)
# self >>= other  __irshift__(self, other)



# def exactly_one(wordlen, output='sum(x[i])==1', input='x'):


