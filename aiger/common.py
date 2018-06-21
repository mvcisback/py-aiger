# TODO: factor out common parts of seq_compose and par_compose
from itertools import starmap

import funcy as fn

from aiger import aig


def _map_tree(inputs, f):
    queue = fn.lmap(aig.Input, inputs)
    while len(queue) > 1:
        queue = fn.lmap(f, fn.chunks(2, queue))
    return queue[0]


def _and(left_right):
    if len(left_right) == 1:
        return left_right[0]

    return aig.AndGate(*left_right)


def and_gate(inputs, output=None):
    output = f'#and_output#{hash(tuple(inputs))}' if output is None else output

    return aig.AIG(
        inputs=frozenset(inputs),
        latches=frozenset(),
        node_map=frozenset(((output, _map_tree(inputs, f=_and)), )),
        comments=(' ∧ '.join(inputs),))


def identity(inputs, outputs=None):
    if outputs is None:
        outputs = inputs

    return aig.AIG(
        inputs=frozenset(inputs),
        latches=frozenset(),
        node_map=frozenset(zip(outputs, map(aig.Input, inputs))),
        comments=('identity',))


def empty():
    return identity([])


def _inverted_input(name):
    return aig.Inverter(aig.Input(name))


def bit_flipper(inputs, outputs=None):
    if outputs is None:
        outputs = inputs
    else:
        assert len(outputs) == len(inputs)

    return aig.AIG(
        inputs=frozenset(inputs),
        latches=frozenset(),
        node_map=frozenset(zip(outputs, map(_inverted_input, inputs))),
        comments=('~',))


def _const(val):
    return aig.Inverter(aig.ConstFalse()) if val else aig.ConstFalse()


def source(outputs):
    return aig.AIG(
        inputs=frozenset(),
        latches=frozenset(),
        node_map=frozenset((k, _const(v)) for k, v in outputs.items()),
        comments=('source',))


def sink(inputs):
    return aig.AIG(
        inputs=frozenset(inputs),
        latches=frozenset(),
        node_map=frozenset(),
        comments=('sink',))


def tee(outputs):
    def tee_output(name, renames):
        return frozenset((r, aig.Input(name)) for r in renames)

    return aig.AIG(
        inputs=frozenset(outputs),
        latches=frozenset(),
        node_map=frozenset.union(*starmap(tee_output, outputs.items())),
        comments=('T',))


def or_gate(inputs, output=None):
    output = f'#or_output#{hash(tuple(inputs))}' if output is None else output
    circ = and_gate(inputs, output)

    return bit_flipper(inputs) >> circ >> bit_flipper([output])
