
import common
from common import AAG
import parser
import aiger  # needed only for the isinstance(kind[1], aiger.common.AAG) check

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

def _full_adder(x, y, carry_in, result, carry_out):
    return parser.parse("aag 10 3 0 2 7\n2\n4\n6\n18\n21\n8 4 2\n10 5 3\n12 11 9\n14 12 6\n16 13 7\n18 17 15\n20 15 9\ni0 {}\ni1 {}\ni2 {}\no0 {}\no1 {}\n".format(x, y, carry_in, result, carry_out))
# print(full_adder('x', 'y', 'carry_in', 'x+y', 'carry_out'))


def _adder_circuit(wordlen, output='x+y', left='x', right='y'):
    carry_name = output + '_carry'
    aig = common.source({carry_name: False})
    assert left != carry_name and right != carry_name
    for i in range(wordlen):
        fa = _full_adder(left + "[{}]".format(i), right + "[{}]".format(i), carry_name, output+'[{}]'.format(i), carry_name)
        aig = aig >> fa
    return aig
# print(_adder_circuit(4))

def _incrementer_circuit(wordlen, output='x+1', input='x'):
    return const(wordlen, 1, output='y') >> _adder_circuit(wordlen, output=output, left=input, right='y')
# print(_incrementer_circuit(3))

def _negation_circuit(wordlen, output='not x', input='x'):
    # word(word) >> 
    return common.bit_flipper(inputs=[input + '[{}]'.format(i) for i in range(wordlen)],
                              outputs=[output + '[{}]'.format(i) for i in range(wordlen)])
# print(_negation_circuit(2))

def _negative_circuit(wordlen, output='-x', input='x'):
    """Returns the circuit computing x*(-1) in Two's complement"""
    return _negation_circuit(wordlen, output='tmp', input=input) >> _incrementer_circuit(wordlen, output=output, input='tmp')
# print(_negative_circuit(4))



class BV:
    def __init__(self, size, kind):

        self.size = size
        self.variables = []
        
        if self.size == 0:
            self.aig = common.empty()
            return

        if isinstance(kind, int):  # Constant
            self.aig = const(size, abs(kind), output=self.name())
            if kind < 0:
                neg = - self
                self.aig = neg.aig
        if isinstance(kind, str):  # Variable
            self.variables.append(kind)
            self.aig = common.empty()
            for i in range(self.size):
                self.aig = self.aig | common.and_gate([kind + '[{}]'.format(i)], output=self.name(i))
        if isinstance(kind, tuple):  # for internal use only
            assert isinstance(kind[0], list)  # variables
            self.variables.extend(kind[0])
            assert isinstance(kind[1], common.AAG) or isinstance(kind[1], aiger.common.AAG)
            self.aig = kind[1]
        
        assert len(self.aig.outputs) == self.size
        # assert len(self.aig.inputs) == self.size*len(self.variables)

    def __len__(self):
        return self.size

    def __repr__(self):
        return str(self.aig)

    def name(self, idx=None):
        if idx is None:
            return "bv{}".format(self.size)
        else:
            return "bv{}[{}]".format(self.size, idx)

    def rename(self, name):
        """Renames the output of the expression; mostly used internally"""
        new_aig = self.aig['o', {self.name(i): name + '[{}]'.format(i) for i in range(self.size)}]
        return BV(self.size, (self.variables, new_aig))

    def subtitute(self, subst):  # TODO: test this function; does the simultaneous thing work?
        """Simultaniously substitutes one set of input words by another"""
        d = dict()
        for old, new in subst:
            d.update({old + '[{}]'.format(i): new + '[{}]'.format(i) for i in range(self.size)}) 
        return BV(self.size, (self.variables, self.aig['i', d]))

    def assign(self, assignment):
        """assignment must be map from names to integer values"""
        aig = self.aig
        names = assignment.keys()
        for name, value in assignment.items():
            c = const(self.size, value, output=name)
            aig = c >> aig
        variables = list(filter(lambda n: n not in names, self.variables))
        return BV(self.size, (variables, aig))

    def __add__(self, other):
        assert self.size == other.size
        other = other.rename('other')
        adder = _adder_circuit(self.size, output=self.name(), left=self.name(), right='other')  >>  common.sink([self.name() + '_carry'])
        return BV(self.size, (self.variables + other.variables, self.aig >> (other.aig >> adder)))

    def __invert__(self):  # ~x
        neg = _negation_circuit(self.size, output=self.name(), input=self.name())
        return BV(self.size, (self.variables, self.aig >> neg))

    def __neg__(self):  #-x
        # aig = self.aig >> _negation_circuit(wordlen, input=self.name(), output=self.name())
        # return BV(self.size, (self.variables, aig))
        return ~self + BV(self.size, 1)

    def __sub__(self, other):
        return self + (-other)

    def __getitem__(self, k):
        if isinstance(k, int):
            out_idxs = [k]
        else:
            out_idxs = range(self.size).__getitem__(k)
        out_map = dict(zip(out_idxs, range(len(out_idxs))))  # maps indices to their new position
        if len(out_idxs) == 0:
            return BV(0, 0)
        outputs_to_remove = []
        rename = dict()
        for i in range(self.size):
            if i in out_map:
                new_idx = out_map[i]
                rename.update({self.name(i): "bv{}[{}]".format(len(out_idxs), new_idx)})
            else: 
                outputs_to_remove.append(self.name(i))
        aig = self.aig
        if len(outputs_to_remove) > 0:
            aig = aig >> common.sink(outputs_to_remove)
        aig = aig['o', rename]
        return BV(len(out_idxs), (self.variables, aig))

    def reverse(self):
        return self[::-1]

    def concat(self, other):
        self_rename = dict()
        for i in range(self.size):
            self_rename.update({self.name(i): "bv{}[{}]".format(self.size+other.size, i)})

        other_rename = dict()
        for i in range(other.size):
            other_rename.update({other.name(i): "bv{}[{}]".format(self.size+other.size, self.size + i)})
        return BV(self.size + other.size, (self.variables + other.variables, self.aig['o', self_rename] | other.aig['o', other_rename])) 

    # Bitwise opeators
    # def __rshift__(self, other):
    # def __lshift__(self, other):
    # def __or__(self, other):
    # def __and__(self, other):
    # def __xor__(self, other):

    # TODO
    # Arithmetic operations
    # def __neg__(self):
    # def __pos__(self):
        # return self
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


