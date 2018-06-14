# TODO: factor out common parts of seq_compose and par_compose
from uuid import uuid1

import funcy as fn
from pyrsistent import pmap

from aiger import aig


def _binary_and(left, right, output_name=None):
    if output_name is None:
        output_name = str(uuid1())

    and_ref, left_ref, right_ref = str(uuid1()), str(uuid1()), str(uuid1())
    return aig.AIG(
        input_map=pmap({
            left: left_ref,
            right: right_ref
        }),
        output_map=pmap({
            output_name: and_ref
        }),
        latch_map=pmap(),
        node_map=pmap({
            and_ref: aig.AndGate(left_ref, right_ref),
            left_ref: aig.Input(),
            right_ref: aig.Input()
        }),
        comments=())


def _and_compose(left_right, output_name=None):
    if len(left_right) == 1:
        return left_right[0]

    left, right = left_right
    circ = left | right
    assert len(circ.outputs) == 2
    return circ >> _binary_and(*circ.outputs, output_name=output_name)


def _reduce_and(inputs, output_name):
    queue = fn.lmap(lambda x: identity([x]), inputs)
    while len(queue) > 1:
        if len(queue) == 2:
            return _and_compose(queue, output_name=output_name)

        queue = list(map(_and_compose, fn.chunks(2, queue)))

    return queue[0]


def and_gate(inputs, output=None):
    output = f'#and#{hash(tuple(inputs))}' if output is None else output

    if len(inputs) == 0:
        return source({output: False})

    return _reduce_and(inputs, output)


def identity(inputs, outputs=None):
    if outputs is None:
        outputs = inputs
    else:
        assert len(outputs) == len(inputs)

    names = [(i, o, str(uuid1())) for i, o in zip(inputs, outputs)]
    return aig.AIG(
        input_map=pmap({i: ref
                        for i, _, ref in names}),
        output_map=pmap({o: ref
                         for _, o, ref in names}),
        latch_map=pmap(),
        node_map=pmap({ref: aig.Input()
                       for i, _, ref in names}),
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

    names = [(i, o, str(uuid1()), str(uuid1()))
             for i, o in zip(inputs, outputs)]
    return aig.AIG(
        input_map=pmap({i: ref1
                        for i, _, ref1, _ in names}),
        output_map=pmap({o: ref2
                         for _, o, _, ref2 in names}),
        latch_map=pmap(),
        node_map=pmap(
            fn.merge({ref1: aig.Input()
                      for _, _, ref1, _ in names},
                     {ref2: aig.Inverter(ref1)
                      for _, _, ref1, ref2 in names})),
        comments=())


def _const(val):
    return aig.Inverter(aig.ConstFalse()) if val else aig.ConstFalse()


def _false_source(outputs):
    names = [(o, str(uuid1())) for o in outputs]
    return aig.AIG(
        input_map=pmap(),
        latch_map=pmap(),
        output_map=pmap(names),
        node_map=pmap({ref: aig.ConstFalse()
                       for _, ref in names}),
        comments=(),
    )


def source(outputs):
    flipped_outputs = {k for k, v in outputs.items() if v}
    return _false_source(outputs) >> bit_flipper(flipped_outputs)


def sink(inputs):
    names = [(i, str(uuid1())) for i in inputs]
    return aig.AIG(
        input_map=pmap({i: ref
                        for i, ref in names}),
        latch_map=pmap(),
        output_map=pmap(),
        node_map={ref: aig.Input()
                  for _, ref in names},
        comments=())


def tee(outputs):
    input_map = {k: str(uuid1()) for k in outputs.keys()}
    output_map = {}
    for name, renames in outputs.items():
        output_map.update({name2: input_map[name] for name2 in renames})

    return aig.AIG(
        input_map=pmap(input_map),
        output_map=pmap(output_map),
        latch_map=pmap(),
        node_map=pmap({ref: aig.Input()
                       for ref in input_map.values()}),
        comments=())


def nand_gate(inputs, output=None):
    output = f'#nand#{hash(tuple(inputs))}' if output is None else output
    circ = and_gate(inputs, output)
    return circ >> bit_flipper([output])


def or_gate(inputs, output=None):
    output = f'#or#{hash(tuple(inputs))}' if output is None else output
    return bit_flipper(inputs) >> nand_gate(inputs, output)


def nor_gate(inputs, output=None):
    output = f'#nor#{hash(tuple(inputs))}' if output is None else output
    return bit_flipper(inputs) >> and_gate(inputs, output)


def delay(inputs):
    names = [(i, str(uuid1()), str(uuid1())) for i in inputs]
    return aig.AIG(
        input_map=pmap({i: ref
                        for i, ref, _ in names}),
        output_map=pmap({str(uuid1()): ref
                         for _, _, ref in names}),
        latch_map=pmap({str(uuid1()): ref
                        for _, _, ref in names}),
        node_map=pmap(
            fn.merge(
                {refi: aig.Input()
                 for _, refi, _ in names},
                {refl: aig.Latch(refi, inputs[i])
                 for i, refi, refl in names},
            )),
        comments=())
