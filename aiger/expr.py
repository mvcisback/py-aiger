from typing import Union

import attr

from aiger import aig
from aiger import common as cmn


@attr.s(frozen=True, slots=True, eq=False, auto_attribs=True, hash=True)
class BoolExpr:
    aig: aig.AIG

    def __call__(self, inputs):
        return self.aig(inputs)[0][self.output]

    def __and__(self, other):
        return _binary_gate(cmn.and_gate, self, other)

    def __or__(self, other):
        return _binary_gate(cmn.or_gate, self, other)

    def __xor__(self, other):
        return _binary_gate(cmn.parity_gate, self, other)

    def __invert__(self):
        return type(self)(
            aig=self.aig >> cmn.bit_flipper(self.aig.outputs, [cmn._fresh()])
        )

    def __eq__(self, other):
        return ~(self ^ other)

    def implies(self, other):
        return ~self | other

    @property
    def output(self):
        return list(self.aig.outputs)[0]

    @property
    def inputs(self):
        return self.aig.inputs

    def _fresh_output(self, name=None):
        if name is None:
            name = cmn._fresh()
        return type(self)(self.aig['o', {self.output: name}])


def _binary_gate(gate, expr1, expr2):
    aig = expr1._fresh_output().aig | expr2._fresh_output().aig
    aig >>= gate(inputs=aig.outputs, output=cmn._fresh())
    return type(expr1)(aig=aig)


def ite(test, expr_true, expr_false):
    return test.implies(expr_true) & (~test).implies(expr_false)


def atom(val: Union[str, bool, None]) -> BoolExpr:
    output = cmn._fresh()
    if val is None:
        val = cmn._fresh()

    if isinstance(val, str):
        assert val not in ('True', 'False')
        aig = cmn.identity([val], [output])
    else:
        aig = cmn.source({output: val})

    return BoolExpr(aig)


def atoms(*vals):
    return tuple(map(atom, vals))
