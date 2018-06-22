from lenses import bind

import aiger

import string
import re

_FORBIDDEN = (' ', '[', ']', '\n', '\t', '\r', '\x0b', '\x0c')
VAR_NAME_ALPHABET = frozenset(
    [c for c in string.printable if c not in _FORBIDDEN])


def _const(wordlen, value, output='x'):
    assert 2**wordlen > value
    aig = aiger.empty()
    for i in range(wordlen):
        aig = aig | aiger.source({output + '[{}]'.format(i): value % 2 == 1})
        value = value // 2
    return aig


def _full_adder(x, y, carry_in, result, carry_out):
    return aiger.parse(
        "aag 10 3 0 2 7\n2\n4\n6\n18\n21\n8 4 2\n10 5 3\n"
        "12 11 9\n14 12 6\n16 13 7\n18 17 15\n20 15 9\n"
        f"i0 {x}\ni1 {y}\ni2 {carry_in}\no0 {result}\no1 {carry_out}\n")


def _adder_circuit(wordlen, output='x+y', left='x', right='y'):
    carry_name = f'{output}_carry'
    assert left != carry_name and right != carry_name

    aig = aiger.source({carry_name: False})
    for i in range(wordlen):
        aig >>= _full_adder(
            x=f"{left}[{i}]",
            y=f"{right}[{i}]",
            carry_in=carry_name,
            result=f'{output}[{i}]',
            carry_out=carry_name)
    return aig


def _negation_circuit(wordlen, output='not x', input='x'):
    return aiger.bit_flipper(
        inputs=[f'{input}[{i}]' for i in range(wordlen)],
        outputs=[f'{output}[{i}]' for i in range(wordlen)])


def _indent(strings):
    return list(map(lambda s: '  ' + s, strings))


def _testBit(value, bit_idx):
    assert isinstance(value, int)
    assert isinstance(bit_idx, int)
    mask = 1 << bit_idx
    return value & mask


def _split_output_name(name):
    return re.match('^(.*)\[(\d*)\]$', name).groups()


class BV(object):
    def __init__(self, size, kind, name="bv"):
        """
        Creates a bitvector expression.

        'kind' can be either an integer, a variable name, or a tuple
        of variable names and an aiger.
        'name' allows us to label the outputs of the circuit
        """
        self.size = size
        self.variables = []

        assert isinstance(name, str)
        for c in name:
            assert c in VAR_NAME_ALPHABET
        self._name = name  # name of all circuit outputs

        if self.size == 0:
            self.aig = aiger.empty()
            return

        elif isinstance(kind, int):  # Constant
            assert kind < 2**size and kind > -2**size
            self.aig = _const(size, abs(kind), output=self.name())

            if kind < 0:
                negative = -self
                self.aig = negative.rename(self.name()).aig

            # nice comments
            self._replace_comments([f'{kind}'])

        elif isinstance(kind, str):  # Variable
            self.variables.append(kind)
            inputs = [kind + f'[{i}]' for i in range(self.size)]
            outputs = [self.name(i) for i in range(self.size)]
            self.aig = aiger.identity(inputs, outputs)

            # nice comments
            self._replace_comments([f'{kind}'])

        elif isinstance(kind, tuple):  # for internal use only
            assert isinstance(kind[0], list)  # variables
            self.variables.extend(kind[0])
            assert isinstance(kind[1], aiger.AIG)
            self.aig = kind[1]
            assert len(self.aig.outputs) == self.size

            # get actual output name of circuit
            if self.size > 0:
                actual_name, _ = _split_output_name(list(self.aig.outputs)[0])
                if actual_name != self.name():
                    self._name = actual_name
                    self.aig = self.rename(name).aig
                    self._name = name

        # final sanity check
        assert len(self.aig.outputs) == self.size
        assert list(self.aig.outputs)[0].startswith(self.name())

    def _replace_comments(self, comments):
        self.aig = bind(self.aig).comments.set(tuple(comments))

    def __len__(self):
        return self.size

    def __repr__(self):
        return str(self.aig)

    def name(self, idx=None):
        assert idx is None or idx >= 0
        return self._name if idx is None else f'{self._name}[{idx}]'

    def subtitute(self, subst):
        """Simultaniously substitutes one set of input words by another."""
        # TODO: test this function; does the simultaneous thing work?
        d = dict()
        for old, new in subst:
            d.update({f'{old}[{i}]': f'{new}[{i}]' for i in range(self.size)})

        return BV(self.size, (self.variables, self.aig['i', d]))

    def assign(self, assignment):
        """Assignment must be map from names to integer values."""
        value_aig = aiger.empty()
        for name, value in assignment.items():
            value_aig |= BV(self.size, value, name=name).aig
        composed_aig = value_aig >> self.aig
        names = assignment.keys()
        variables = list(filter(lambda n: n not in names, self.variables))
        return BV(self.size, (variables, composed_aig))

    # Arithmetic operations
    def __add__(self, other):
        assert self.size == other.size
        if self.name() == other.name():
            other = other.rename('other')
        outname = 'bv'
        if self.name() == outname or other.name() == outname:
            outname = f'{self.name()}_+_{other.name()}'

        adder = _adder_circuit(
            self.size, output=outname, left=self.name(), right=other.name())
        adder >>= aiger.sink([outname + '_carry'])
        inputs = self.aig | other.aig
        result = inputs >> adder
        all_vars = self.variables + other.variables
        res = BV(self.size, (all_vars, result), name=outname)

        # nice comments
        res._replace_comments([f'add'] + _indent(self.aig.comments) + _indent(
            other.aig.comments))
        return res

    def __invert__(self):
        """Implements -x."""
        neg = _negation_circuit(
            self.size, output=self.name(), input=self.name())
        aig = self.aig >> neg
        res = BV(self.size, (self.variables, aig), name=self.name())

        # nice comments
        res._replace_comments([f'invert'] + _indent(self.aig.comments))
        return res

    def __neg__(self):
        """Implements -x."""
        res = ~self + BV(self.size, 1)

        # nice comments
        res._replace_comments([f'unary minus'] + _indent(self.aig.comments))
        return res

    def __pos__(self):
        return self

    def __sub__(self, other):
        res = self + (-other)

        # nice comments
        res._replace_comments([f'subtract'] + _indent(self.aig.comments) +
                              _indent(other.aig.comments))
        return res

    def __getitem__(self, k):
        comments = self.aig.comments

        out_idxs = [k] if isinstance(k, int) else range(self.size)[k]

        if len(out_idxs) == 0:
            return BV(0, 0)

        # maps indices to their new position
        out_map = dict(zip(out_idxs, range(len(out_idxs))))

        outputs_to_remove, rename = [], dict()
        for i in range(self.size):
            if i in out_map:
                new_idx = out_map[i]
                rename.update({self.name(i): self.name(new_idx)})
            else:
                outputs_to_remove.append(self.name(i))

        aig = self.aig
        if len(outputs_to_remove) > 0:
            aig >>= aiger.sink(outputs_to_remove)

        res = BV(len(out_idxs), (self.variables, aig['o', rename]))

        # nice comments
        res._replace_comments([f'get({k})'] + _indent(comments))
        return res

    def reverse(self):
        comments = self.aig.comments
        res = self[::-1]

        # nice comments
        res._replace_comments([f'reverse'] + _indent(comments))
        return res

    def concat(self, other):
        other_rename = dict()
        for i in range(other.size):
            other_rename.update({other.name(i): other.name(self.size + i)})

        res = BV(self.size + other.size,
                 (self.variables + other.variables,
                  self.aig | other.aig['o', other_rename]))

        # nice comments
        res._replace_comments([f'concat'] + _indent(self.aig.comments) +
                              _indent(other.aig.comments))
        return res

    # Bitwise opeators
    def unsigned_rightshift(self, k):
        """Unsigned rightshift by a fixed integer; big endian encoding"""
        res = self[:-k].concat(BV(k, 0))

        # nice comments
        res._replace_comments([f'>> {k}  (unsigned)'] + _indent(
            self.aig.comments))

        return res

    def repeat(self, k):
        """Repeats the bitvector k times; resulting size is self.size*k"""
        assert k > 0
        copies = dict()
        for i in range(self.size):
            copies[self.name(i)] = [
                self.name(i + j) for j in range(0, k * self.size, self.size)
            ]
        res = BV(self.size * k,
                 (self.variables, self.aig >> aiger.tee(copies)))

        # nice comments
        res._replace_comments([f'repeat({k})'] + _indent(self.aig.comments))
        return res

    def __rshift__(self, k):
        """Signed rightshift by a fixed integer; big endian encoding;
        index 0 of bitvector is rightmost"""
        right_side = self[-1:].repeat(k)
        assert right_side.size == k
        res = self[k:].concat(right_side)

        # nice comments
        res._replace_comments([f'>> {k}'] + _indent(self.aig.comments))

        return res

    def __lshift__(self, k):
        """Leftshift by a fixed integer; big endian encoding; index 0
        of bitvector is rightmost"""
        res = BV(k, 0).concat(self[:-k])

        # nice comments
        res._replace_comments([f'<< {k}'] + _indent(self.aig.comments))

        return res

    def __or__(self, other):
        assert self.size == other.size
        if self.name() == other.name():
            other = other.rename(self.name() + '_other')
        bitwise_or = aiger.empty()
        for i in range(self.size):
            bitwise_or = bitwise_or | aiger.or_gate(
                [self.name(i), other.name(i)], output=self.name(i))
        aig = (self.aig | other.aig) >> bitwise_or

        # nice comments
        res = BV(self.size, (self.variables + other.variables, aig))
        res._replace_comments(['or'] + _indent(self.aig.comments) + _indent(
            other.aig.comments))
        return res

    def __and__(self, other):
        assert self.size == other.size
        if self.name() == other.name():
            other = other.rename(self.name() + '_other')
        bitwise_and = aiger.empty()
        for i in range(self.size):
            bitwise_and = bitwise_and | aiger.and_gate(
                [self.name(i), other.name(i)], output=self.name(i))
        aig = (self.aig | other.aig) >> bitwise_and

        res = BV(self.size, (self.variables + other.variables, aig))
        # nice comments
        res._replace_comments(['and'] + _indent(self.aig.comments) + _indent(
            other.aig.comments))
        return res

    def __xor__(self, other):
        assert self.size == other.size
        if self.name() == other.name():
            other = other.rename(self.name() + '_other')

        def xor(i):
            tee = aiger.tee({
                self.name(i): [self.name(i),
                               self.name(i) + '_alt'],
                other.name(i): [other.name(i),
                                other.name(i) + '_alt']
            })
            negated_inputs = aiger.bit_flipper(
                [self.name(i) + '_alt',
                 other.name(i) + '_alt'],
                outputs=[self.name(i) + '_neg',
                         other.name(i) + '_neg'])

            or_gate_pos = aiger.or_gate(
                [self.name(i), other.name(i)], output=self.name(i) + '_pos')
            or_gate_neg = aiger.or_gate(
                [self.name(i) + '_neg',
                 other.name(i) + '_neg'],
                output=self.name(i) + '_neg')

            and_gate = aiger.and_gate(
                [self.name(i) + '_pos',
                 self.name(i) + '_neg'],
                output=self.name(i))
            aig = (or_gate_pos | or_gate_neg)
            return tee >> negated_inputs >> aig >> and_gate

        bitwise_xor = aiger.empty()
        for i in range(self.size):
            bitwise_xor = bitwise_xor | xor(i)

        aig = (self.aig | other.aig) >> bitwise_xor

        res = BV(self.size, (self.variables + other.variables, aig))
        res._replace_comments(['xor'] + _indent(self.aig.comments) + _indent(
            other.aig.comments))
        return res

    def __abs__(self):
        mask = self >> self.size - 1
        assert mask.size == self.size
        res = (self + mask) ^ mask

        # nice comments
        res._replace_comments(['abs'] + _indent(self.aig.comments))

        return res

    def is_nonzero(self, output='bool'):
        return BV(
            1, (self.variables, self.aig >> aiger.or_gate(
                self.aig.outputs, output=output + '[0]')),
            name=output)

    def is_zero(self, output='bool'):
        check_zero = aiger.bit_flipper(self.aig.outputs) >> aiger.and_gate(
            self.aig.outputs, output=output + '[0]')
        return BV(1, (self.variables, self.aig >> check_zero), name=output)

    def __eq__(self, other):
        res = (self ^ other).is_zero()

        # nice comments
        res._replace_comments(['=='] + _indent(self.aig.comments) + _indent(
            other.aig.comments))
        return res

    def __ne__(self, other):
        res = (self ^ other).is_nonzero()

        # nice comments
        res._replace_comments(['!='] + _indent(self.aig.comments) + _indent(
            other.aig.comments))

        return res

    def __lt__(self, other):
        """signed comparison"""
        assert self.size == other.size

        left = self.concat(self[-1:])
        assert left.size == self.size + 1
        right = other.concat(other[-1:])

        res = (left - right)[-1:]

        # nice comments
        res._replace_comments(['<'] + _indent(self.aig.comments) + _indent(
            other.aig.comments))

        return res

    def __gt__(self, other):
        """signed comparison"""
        res = other < self

        # nice comments
        res._replace_comments(['>'] + _indent(self.aig.comments) + _indent(
            other.aig.comments))

        return res

    def __le__(self, other):
        """signed comparison"""
        res = ~(self > other)

        # nice comments
        res._replace_comments(['<='] + _indent(self.aig.comments) + _indent(
            other.aig.comments))

        return res

    def __ge__(self, other):
        """signed comparison"""
        res = ~(self < other)

        # nice comments
        res._replace_comments(['>='] + _indent(self.aig.comments) + _indent(
            other.aig.comments))

        return res

    def rename(self, name):
        """Renames the output of the expression; mostly used internally"""
        m = {self.name(i): f'{name}[{i}]' for i in range(self.size)}
        return BV(self.size, (self.variables, self.aig['o', m]), name=name)

    def __call__(self, args=None, signed=True, interpreted=True):
        '''
        Eval for unsigned integers:
        - inputs must be unsigned
        - outputs are unsigned
        - args is a dict mapping variable names to non-negative integers
          smaller than 2**bitwidth
        - signed controls if the output is interpreted signed or unsigned
        - interpreted=False returns the bitvector instead of an int
        '''
        if args is None:
            args = {}

        # Check completeness of inputs; check ranges
        for key, value in args.items():
            assert value >= -2**(self.size - 1)
            assert value < 2**(self.size)
            assert key in self.variables
        # Check if the correct number of inputs is given
        assert len(self.aig.inputs) // self.size == len(args)

        # Tanslate integers values to bit values; Challenge here is that we
        # don't know the bit widths of the different variables
        inputs = {}
        for input_name in self.aig.inputs:
            # split name into variable name and index
            var_name, idx = _split_output_name(input_name)

            # populate input map
            assert var_name in args
            inputs[input_name] = _testBit(args[var_name], int(idx))

        outputs, gates = self.aig(inputs=inputs)

        if not interpreted:
            return outputs

        # Interpret result
        out_value = 0
        if signed and self.size > 1 and outputs[f'{self.name(self.size - 1)}']:
            for idx in range(self.size):
                if not outputs[f'{self.name()}[{idx}]']:
                    out_value -= 2**idx
            out_value -= 1
        else:
            for idx in range(self.size):
                if outputs[f'{self.name()}[{idx}]']:
                    out_value += 2**idx

        return out_value

    # Difficult arithmetic operations
    # def __mul__(self, other):
    # def __mod__(self, other):
    # def __div__(self, other):
    # def __pow__(self, other):


# TODO:
# Make iterable

# def __hash__(self): # for use in strash; remember hash with every
# expression to avoid recomputation; remember global map from hashes
# to subexpressions
