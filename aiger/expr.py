from typing import Union

import attr
import funcy as fn

from aiger import aig
from aiger import common as cmn


@attr.s(frozen=True, slots=True, cmp=False, auto_attribs=True)
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
        return BoolExpr(
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


def _binary_gate(gate, expr1, expr2):
    circ1, circ2 = expr1.aig, expr2.aig
    aig = _parcompose(circ1, circ2)
    aig >>= gate(inputs=aig.outputs, output=cmn._fresh())
    return BoolExpr(aig=aig)


def _fresh_relabel(keys):
    return {k: cmn._fresh() for k in keys}


def _parcompose(circ1, circ2):
    inputs_collide = circ1.inputs & circ2.inputs
    outputs_collide = circ1.outputs & circ2.outputs

    if outputs_collide:
        circ1 = circ1['o', _fresh_relabel(circ1.outputs)]
        circ2 = circ2['o', _fresh_relabel(circ2.outputs)]

    if not inputs_collide:
        return circ1 | circ2
    else:
        subs1 = _fresh_relabel(circ1.inputs)
        subs2 = _fresh_relabel(circ2.inputs)
        tee = cmn.tee(fn.merge_with(tuple, subs1, subs2))
        return tee >> (circ2['i', subs1] | circ1['i', subs2])


def atom(val: Union[str, bool]) -> BoolExpr:
    output = cmn._fresh()
    if isinstance(val, str):
        assert val not in ('True', 'False')
        aig = cmn.identity([val], [output])
    else:
        aig = cmn.source({output: val})

    return BoolExpr(aig)
