# TODO: factor out common parts of seq_compose and par_compose
from collections import namedtuple
from itertools import starmap

import funcy as fn

from aiger import aig


def _map_tree(inputs, f):
    queue = fn.lmap(aig.Input, inputs)
    while len(queue) > 1:
        queue = list(starmap(f, zip(queue, queue[1:])))
    return queue[0]


def and_gate(inputs, output=None):
    if len(inputs) <= 1:
        return identity(inputs)

    output = f'#and_output#{hash(tuple(inputs))}' if output is None else output

    return aig.AIG(
        inputs=frozenset(inputs),
        top_level=frozenset(((output, _map_tree(inputs, f=aig.AndGate)), )),
        comments=())


def identity(inputs):
    return aig.AIG(
        inputs=frozenset(inputs),
        top_level=frozenset(zip(inputs, map(aig.Input, inputs))),
        comments=())


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
        top_level=frozenset(zip(outputs, map(_inverted_input, inputs))),
        comments=())


def _const(val):
    return aig.Inverter(aig.ConstFalse()) if val else aig.ConstFalse()


def source(outputs):
    return aig.AIG(
        inputs=frozenset(),
        top_level=frozenset((k, _const(v)) for k, v in outputs.items()),
        comments=())


def sink(inputs):
    return aig.AIG(
        inputs=frozenset(inputs), 
        top_level=frozenset(), 
        comments=())


def tee(outputs):
    def tee_output(name, renames):
        return frozenset((r, aig.Input(name)) for r in renames)

    return aig.AIG(
        inputs=frozenset(outputs),
        top_level=frozenset.union(*starmap(tee_output, outputs.items())),
        comments=())


def or_gate(inputs, output=None):
    outputs = [f'#or_output#{hash(tuple(inputs))}' if output is None else output]
    circ = and_gate(inputs, output)
    return bit_flipper(inputs) >> circ >> bit_flipper(outputs)
