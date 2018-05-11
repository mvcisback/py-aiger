
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
    def __init__(self, size, kind, name="bv"):
        """
        Creates a bitvector expression.

        'kind' can be either an integer, a variable name, or a tuple of variable names and an aiger.
        'name' allows us to label the outputs of the circuit
        """
        self.size = size
        self.variables = []

        assert isinstance(name, str)
        self._name = name  # name of all circuit outputs

        if self.size == 0:
            self.aig = common.empty()
            return

        if isinstance(kind, int):  # Constant
            assert kind < 2**size and kind > - 2**size
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
        
        assert len(self.aig.outputs) == self.size
        # assert len(self.aig.inputs) == self.size*len(self.variables)

    def __len__(self):
        return self.size

    def __repr__(self):
        return str(self.aig)

    def name(self, idx=None):
        assert idx == None or idx >= 0
        if idx is None:
            return self._name
        else:
            return self._name + "[{}]".format(idx)

    def rename(self, name):
        """Renames the output of the expression; mostly used internally"""
        new_aig = self.aig['o', {self.name(i): name + '[{}]'.format(i) for i in range(self.size)}]
        return BV(self.size, (self.variables, new_aig), name=name)

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

    # Arithmetic operations
    def __add__(self, other):
        assert self.size == other.size
        other = other.rename('other')
        adder = _adder_circuit(self.size, output=self.name(), left=self.name(), right='other')  >>  common.sink([self.name() + '_carry'])
        return BV(self.size, (self.variables + other.variables, self.aig >> (other.aig >> adder)))

    def __invert__(self):  # ~x
        neg = _negation_circuit(self.size, output=self.name(), input=self.name())
        return BV(self.size, (self.variables, self.aig >> neg))

    def __neg__(self):  #-x
        return ~self + BV(self.size, 1)

    def __pos__(self):  # +x
        return self

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
                rename.update({self.name(i): self.name(new_idx)})
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
        other_rename = dict()
        for i in range(other.size):
            other_rename.update({other.name(i): other.name(self.size + i)})
        return BV(self.size + other.size, (self.variables + other.variables, self.aig | other.aig['o', other_rename])) 

    # Bitwise opeators
    def unsigned_rightshift(self, k):
        """Unsigned rightshift by a fixed integer; big endian encoding"""
        return self[:-k].concat(BV(k, 0))
    
    def repeat(self, k):
        """Repeats the bitvector k times; resulting size is self.size*k"""
        assert k > 0
        copies = dict()
        for i in range(self.size):
            copies[self.name(i)] = [self.name(i+j) for j in range(0, k*self.size, self.size)]
        return BV(self.size*k, (self.variables, self.aig >> common.tee(copies)))

    def __rshift__(self, k):
        """Signed rightshift by a fixed integer; big endian encoding; index 0 of bitvector is rightmost"""
        right_side = self[-1:].repeat(k)
        assert right_side.size == k
        return self[k:].concat(right_side)

    def __lshift__(self, other):
        """Leftshift by a fixed integer; big endian encoding; index 0 of bitvector is rightmost"""
        return BV(k, 0).concat(self[:-k])

    def __or__(self, other):
        assert self.size == other.size
        if self.name() == other.name():
            other = other.rename(self.name() + '_other')
        bitwise_or = common.empty()
        for i in range(self.size):
            bitwise_or = bitwise_or | common.or_gate([self.name(i), other.name(i)], output=self.name(i))
        aig = (self.aig | other.aig) >> bitwise_or
        return BV(self.size, (self.variables + other.variables, aig))

    def __and__(self, other):
        assert self.size == other.size
        if self.name() == other.name():
            other = other.rename(self.name() + '_other')
        bitwise_and = common.empty()
        for i in range(self.size):
            bitwise_and = bitwise_and | common.and_gate([self.name(i), other.name(i)], output=self.name(i))
        aig = (self.aig | other.aig) >> bitwise_and
        return BV(self.size, (self.variables + other.variables, aig))

    def __xor__(self, other):
        assert self.size == other.size
        if self.name() == other.name():
            other = other.rename(self.name() + '_other')

        def xor(i):
            tee = common.tee({self.name(i):[self.name(i), self.name(i) + '_alt'], other.name(i): [other.name(i), other.name(i) + '_alt']})
            negated_inputs = aiger.bit_flipper([self.name(i) + '_alt', other.name(i) + '_alt'], 
                                                outputs=[self.name(i) + '_neg', other.name(i) + '_neg'])

            or_gate_pos = common.or_gate([self.name(i), other.name(i)], output=self.name(i) + '_pos')
            or_gate_neg = common.or_gate([self.name(i) + '_neg', other.name(i) + '_neg'], output=self.name(i) + '_neg')

            and_gate = common.and_gate([self.name(i) + '_pos', self.name(i) + '_neg'], output=self.name(i))
            aig = (or_gate_pos | or_gate_neg)
            return tee >> negated_inputs >> (or_gate_pos | or_gate_neg) >> and_gate

        bitwise_xor = common.empty()
        for i in range(self.size):
            bitwise_xor = bitwise_xor | xor(i)

        aig = (self.aig | other.aig) >> bitwise_xor
        return BV(self.size, (self.variables + other.variables, aig))


    def __abs__(self):
        mask = self >> self.size - 1
        assert mask.size == self.size
        return (self + mask) ^ mask

    def is_nonzero(self, output='bool'):
        return BV(1, (self.variables, self.aig >> common.or_gate(self.aig.outputs, output=output + '[0]')), name=output)

    def is_zero(self, output='bool'):
        check_zero = common.bit_flipper(self.aig.outputs) >> common.and_gate(self.aig.outputs, output=output + '[0]')
        return BV(1, (self.variables, self.aig >> check_zero), name=output)

    def __eq__(self, other):
        return (self ^ other).is_zero()

    def __ne__(self, other):
        return (self ^ other).is_nonzero()
    
    def __lt__(self, other):
        """signed comparison"""
        return (self - other)[-1:]

    def __gt__(self, other):
        """signed comparison"""
        return (other - self)[-1:]

    def __le__(self, other):
        """signed comparison"""
        return ~(self > other)

    def __ge__(self, other):
        """signed comparison"""
        return ~(self < other)

        # assert self.size == other.size
        # if self.name() == other.name():
        #     other = other.rename(self.name() + '_other')

        # local_lt =   BV(self.size, self.name())  & ~BV(other.size, other.name())
        # local_gt =  ~BV(self.size, self.name())  &  BV(other.size, other.name())

        # print(local_cmp)
        # quit()
        # lt = BV(1,1)  # proof obligation show that it is less than
        # for i in range(self.size -1, -1, -1):
        #     lt = lt 

        # aig = self.aig >> (other.aig >> lt.aig)
        # print(aig)
        # return BV(1, (self.variables + other.variables, aig))


    # Difficult arithmetic operations
    # def __mul__(self, other):
    # def __mod__(self, other):
    # def __div__(self, other):
    # def __pow__(self, other):

    # def __hash__(self):  # for use in strash; remember hash with every expression to avoid recomputation; remember global map from hashes to subexpressions

