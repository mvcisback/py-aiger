from aiger import common
from aiger.common import AAG
import parser


def _bv_andor(wordlen, gate, output='x&y', left='x', right='y'):
    aig = common.empty()
    for i in range(wordlen):
        aig |= common.and_gate([f'{left}[{i}]', f'{right}[{i}]'], 
                              output=f'{output}[{i}]')
    return aig


def bv_and(wordlen, output='x&y', left='x', right='y'):
    return _bv_andor(wordlen, common.and_gate, output, left, right)


def bv_or(wordlen, output='x&y', left='x', right='y'):
    return _bv_andor(wordlen, common.or_gate, output, left, right)


def const(wordlen, value, output='x'):
    assert 2**wordlen > value
    aig = common.empty()
    for i in range(wordlen):
        aig = aig | common.source({output + '[{}]'.format(i): value % 2 == 1})
        value = value // 2
    return aig


def _full_adder(x, y, carry_in, result, carry_out):
    return parser.parse(
        "aag 10 3 0 2 7\n2\n4\n6\n18\n21\n8 4 2\n10 5 3\n"
        "12 11 9\n14 12 6\n16 13 7\n18 17 15\n20 15 9\n"
        f"i0 {x}\ni1 {y}\ni2 {carry_in}\no0 {result}\no1 {carry_out}\n")


def _adder_circuit(wordlen, output='x+y', left='x', right='y'):
    carry_name = f'{output}_carry'
    assert left != carry_name and right != carry_name

    aig = common.source({carry_name: False})
    for i in range(wordlen):
        aig >>= _full_adder(x=f"{left}[{i}]", 
                         y=f"{right}[{i}]",
                         carry_in=carry_name,
                         result=f'{output}[{i}]',
                         carry_out=carry_name)
    return aig


def _incrementer_circuit(wordlen, output='x+1', input='x'):
    const_1 = const(wordlen, 1, output='y')
    adder = _adder_circuit(wordlen, output=output, left=input, right='y')
    return const_1 >> adder


def _negation_circuit(wordlen, output='not x', input='x'):
    return common.bit_flipper(
        inputs=[f'{input}[{i}]' for i in range(wordlen)],
        outputs=[f'{output}[{i}]' for i in range(wordlen)]
    )


def _negative_circuit(wordlen, output='-x', input='x'):
    """Returns the circuit computing x*(-1) in Two's complement"""
    neg = _negation_circuit(wordlen, output='tmp', input=input)
    inc = _incrementer_circuit(wordlen, output=output, input='tmp')
    return neg >> inc


class BV(object):
    def __init__(self, size, kind):
        self.size = size
        self.variables = []
        
        if self.size == 0:
            self.aig = common.empty()
            return

        elif isinstance(kind, int):  # Constant
            self.aig = const(size, abs(kind), output=self.name())
            if kind < 0:
                self.aig = (-self).aig

        elif isinstance(kind, str):  # Variable
            self.variables.append(kind)
            self.aig = common.empty()
            for i in range(self.size):
                self.aig |= common.and_gate([f'{kind}[{i}]'], 
                                            output=self.name(i))

        elif isinstance(kind, tuple):  # for internal use only
            assert isinstance(kind[0], list)  # variables
            self.variables.extend(kind[0])
            assert isinstance(kind[1], common.AAG) \
                or isinstance(kind[1], aiger.common.AAG)
            self.aig = kind[1]
        
        assert len(self.aig.outputs) == self.size

    def __len__(self):
        return self.size

    def __repr__(self):
        return str(self.aig)

    def name(self, idx=None):
        return f"bv{self.size}" if idx is None else f"bv{self.size}[{idx}]"

    def rename(self, name):
        """Renames the output of the expression; mostly used internally."""
        rename_map = {self.name(i): f'{name}[{i}]' for i in range(self.size)}
        return BV(self.size, (self.variables, self.aig['o',  rename_map]))

    def subtitute(self, subst):
        """Simultaniously substitutes one set of input words by another."""
        # TODO: test this function; does the simultaneous thing work?
        d = dict()
        for old, new in subst:
            d.update({f'{old}[{i}]': f'{new}[{i}]' for i in range(self.size)}) 

        return BV(self.size, (self.variables, self.aig['i', d]))

    def assign(self, assignment):
        """Assignment must be map from names to integer values."""
        aig = self.aig
        for name, value in assignment.items():
            aig = const(self.size, value, output=name) >> aig

        names = assignment.keys()
        variables = list(filter(lambda n: n not in names, self.variables))
        return BV(self.size, (variables, aig))

    def __add__(self, other):
        assert self.size == other.size
        other = other.rename('other')
        adder = _adder_circuit(self.size, output=self.name(),
                               left=self.name(), right='other')
        adder >>=  common.sink([self.name() + '_carry'])
        return BV(self.size, (self.variables + other.variables,
                              self.aig >> (other.aig >> adder)))

    def __invert__(self):  # ~x
        neg = _negation_circuit(self.size, output=self.name(), input=self.name())
        return BV(self.size, (self.variables, self.aig >> neg))

    def __neg__(self):  #-x
        return ~self + BV(self.size, 1)

    def __sub__(self, other):
        return self + (-other)

    def __getitem__(self, k):
        out_idxs = [k] if isinstance(k, int) else range(self.size)[k]

        if len(out_idxs) == 0:
            return BV(0, 0)

        # maps indices to their new position
        out_map = dict(zip(out_idxs, range(len(out_idxs))))

        outputs_to_remove, rename = [], dict()
        for i in range(self.size):
            if i in out_map:
                new_idx = out_map[i]
                rename.update({self.name(i): f"bv{len(out_idxs)}[{new_idx}]"})
            else: 
                outputs_to_remove.append(self.name(i))

        aig = self.aig
        if len(outputs_to_remove) > 0:
            aig >>= common.sink(outputs_to_remove)

        return BV(len(out_idxs), (self.variables, aig['o', rename]))

    def reverse(self):
        return self[::-1]

    def concat(self, other):
        new_size = self.size+other.size
        bits = range(self.size)

        self_rename = {self.name(i): f"bv{new_size}[{i}]" for i in bits}
        other_rename = {other.name(i): f"bv{new_size}[{self.size + i}]"
                        for i in bits}

        new_aig = self.aig['o', self_rename] | other.aig['o', other_rename]
        new_variables = self.variables + other.variables
        return BV(self.size + other.size, (new_variables, new_aig)) 

# TODO:
# Make iterable

# Bitwise opeators
# def __rshift__(self, other):
# def __lshift__(self, other):

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


