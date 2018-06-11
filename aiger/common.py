# TODO: factor out common parts of seq_compose and par_compose
from itertools import starmap
from uuid import uuid1

import funcy as fn
from pyrsistent import pmap

from aiger import aig


def _binary_and(left, right):
    output_name, and_ref = uuid1(), uuid1()
    left_ref, right_ref = uuid1(), uuid1()
    return aig.AIG(
        input_map=pmap({left: left_ref, right: right_ref}),
        output_map=pmap({output_name: and_ref}),
        latch_map=pmap(),
        node_map=pmap({
            and_ref: aig.AndGate(left_ref, right_ref),
            left_ref: aig.Input(),
            right_ref: aig.Input()
        }),
        comments=())


def _and_compose(left, right):
    circ = left | right
    assert len(circ.outputs) == 2
    return circ >> _binary_and(*circ.outputs)


def _reduce_and(inputs):
    queue = fn.lmap(lambda x: identity([x]), inputs)
    while len(queue) > 1:
        queue = list(starmap(_and_compose, zip(queue, queue[1:])))
    return queue[0]


def and_gate(inputs, output=None):
    output = f'#and_output#{hash(tuple(inputs))}' if output is None else output

    if len(inputs) == 0:
        return source({output: False})
    
    input_map = pmap({uuid1(): name for name in inputs})
    return _reduce_and(inputs)


def identity(inputs, outputs=None):
    if outputs is None:
        outputs = inputs
    else:
        assert len(outputs) == len(inputs)

    names = [(i, o, uuid1()) for i, o in zip(inputs, outputs)]
    return aig.AIG(
        input_map=pmap({i: ref for i, _, ref in names}),
        output_map=pmap({o: ref for _, o, ref in names}),
        latch_map=pmap(),
        node_map=pmap({ref: aig.Input() for i, _, ref in names}),
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

    names = [(i, o, uuid1(), uuid1()) for i, o in zip(inputs, outputs)]
    return aig.AIG(
        input_map=pmap({i: ref1 for i, _, ref1, _ in names}),
        output_map=pmap({o: ref2 for _, o, _, ref2 in names}),
        latch_map=pmap(),
        node_map=pmap(fn.merge(
            {ref1: aig.Input() for _, _, ref1, _ in names},
            {ref2: aig.Inverter(ref1) for _, _, ref1, ref2 in names}
        )),
        comments=())


def _const(val):
    return aig.Inverter(aig.ConstFalse()) if val else aig.ConstFalse()


def _false_source(outputs):
    names = [(o, uuid1()) for o in outputs]
    return aig.AIG(
        input_map=pmap(),
        latch_map=pmap(),
        output_map=pmap(names),
        node_map=pmap({ref: aig.ConstFalse() for _, ref in names}),
        comments=(),
    )


def source(outputs):
    flipped_outputs = {k for k, v in outputs.items() if v}
    return _false_source(outputs) >> bit_flipper(flipped_outputs)


def sink(inputs):
    names = [(i, uuid1()) for i in inputs]
    return aig.AIG(
        input_map=pmap({i: ref for i, ref in names}),
        latch_map=pmap(),
        output_map=pmap(),
        node_map={ref: Input() for _, ref in names},
        comments=()
    )


def tee(outputs):
    def tee_output(name, renames):
        return frozenset((r, aig.Input(name)) for r in renames)

    return aig.AIG(
        inputs=frozenset(outputs),
        top_level=frozenset.union(*starmap(tee_output, outputs.items())),
        comments=())


def or_gate(inputs, output=None):
    output = f'#or_output#{hash(tuple(inputs))}' if output is None else output
    circ = and_gate(inputs, output)

    return bit_flipper(inputs) >> circ >> bit_flipper([output])
