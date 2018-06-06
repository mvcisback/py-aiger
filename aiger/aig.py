from collections import defaultdict
from itertools import starmap
from typing import Tuple, FrozenSet, Mapping, NamedTuple, Union

import funcy as fn
import lenses.hooks  # TODO: remove on next lenses version release.
from lenses import bind
from toposort import toposort


# TODO: Remove on next lenses lenses version release.
# Needed because 0.4 does not know about frozensets.
@lenses.hooks.from_iter.register(frozenset)
def _frozenset_from_iter(self, iterable):
    return frozenset(iterable)


class AndGate(NamedTuple):
    left: 'Gate' # TODO: replace with Gate once 3.7 lands.
    right: 'Gate'

    @property
    def children(self):
        return tuple((self.left, self.right))

    
class Latch(NamedTuple):
    name: str
    input: 'Gate'
    initial: bool

    @property
    def children(self):
        return tuple((self.input,))


class Inverter(NamedTuple):
    input: 'Gate'

    @property
    def children(self):
        return tuple((self.input,))

    
# Enables filtering for InputSignal via lens library.
class InputSignal(NamedTuple):
    name: str  

    @property
    def children(self):
        return tuple()


class Ground(NamedTuple):
    @property
    def children(self):
        return tuple()


Gate = Union[AndGate, Latch, Ground, Inverter, InputSignal]

class AIG(NamedTuple):
    inputs: FrozenSet[str]
    top_level: FrozenSet[Tuple[str, Gate]]
    comments: Tuple[str]

    # TODO:
    # __repr__(self):
    
    @property
    def outputs(self):
        return frozenset(fn.pluck(0, self.top_level))

    @property
    def latches(self):
        return frozenset(bind(self).Recur(Latch).collect())

    @property
    def gates(self):
        return frozenset(fn.pluck(1, self.top_level))

    def __rshift__(self, other):
        return seq_compose(self, other)

    def __or__(self, other):
        return par_compose(self, other)

    @property
    def _eval_order(self):
        return list(toposort(_dependency_graph(self.gates)))
    
    def __call__(self, inputs, latches=None):
        # TODO: Implement partial evaluation.
        if latches is not None:
            latches = dict()
            
        latches = {l: latches.get(l.name, l.initial) for l in self.latches}
        lookup = fn.merge(inputs, latches)
        for gate in fn.cat(self._eval_order[1:]):
            if isinstance(gate, AndGate):
                lookup[gate] = lookup[gate.left] and lookup[gate.right]
            elif isinstance(gate, Inverter):
                lookup[gate] = not lookup[gate.input]
            elif isinstance(gate, Latch):
                lookup[gate] = lookup[gate.input]
            elif isinstance(gate, InputSignal):
                lookup[gate] = lookup[gate.name]
            elif isinstance(gate, Ground):
                lookup[gate] = False
            else:
                raise NotImplementedError
            
        outputs = {name: lookup[gate] for name, gate in self.top_level}
        latches = {l.name: lookup[l] for l in latches}
        return outputs, latches

    def simulator(self, latches=None):
        inputs = yield
        while True:
            outputs, latches = self(inputs, latches)
            inputs = yield outputs, latches

    def simulate(self, input_seq, latches=None):
        sim = self.simulator()
        next(sim)
        return [sim.send(inputs) for inputs in input_seq]
    

    def cutlatches(self, latches=None):
        raise NotImplementedError
    
    def unroll(self, horizon, *, init=True, omit_latches=True):
        # TODO: Port cutlatches
        raise NotImplementedError



def _dependency_graph(gates):
    queue, deps = list(gates), defaultdict(set)
    while queue:
        gate = queue.pop()
        children = gate.children
        queue.extend(children)
        deps[gate].update(children)

    return deps


def _map_tree(inputs, f):
    queue = fn.lmap(InputSignal, inputs)
    while len(queue) > 1:
        queue = list(starmap(f, zip(queue, queue[1:])))
    return queue[0]


def and_gate(inputs, output=None):
    if len(inputs) <= 1:
        # TODO: return identity or empty circuits.
        raise NotImplementedError

    output = f'#and_output#{hash(tuple(inputs))}' if output is None else output

    return AIG(
        inputs=frozenset(inputs),
        top_level=frozenset(((output, _map_tree(inputs, f=AndGate)),)),
        comments=()
    )


def identity(inputs):
    return AIG(
        inputs=frozenset(inputs),
        top_level=frozenset(zip(inputs, map(InputSignal, inputs))),
        comments=()
    )


def empty():
    return identity([])


def _inverted_input(name):
    return Inverter(InputSignal(name))


def inverter(inputs, outputs=None):
    if outputs is None:
        outputs = inputs
    else:
        assert len(outputs) == len(inputs)

    return AIG(
        inputs=tuple(inputs),
        top_level=tuple(zip(outputs, map(_inverted_input, inputs))),
        comments=()
    )


def _const(val):
    return Inverter(Ground()) if val else Ground()


def source(outputs):
    return AIG(
        inputs=tuple(),
        top_level=tuple((k, _const(v)) for k, v in outputs.items()),
        comments=()
    )


def sink(inputs):
    return AIG(
        inputs=tuple(inputs),
        top_level=tuple(),
        comments=()
    )


def par_compose(aig1, aig2, check_precondition=True):
    if check_precondition:
        assert not (aig1.latches & aig2.latches)
        assert not (aig1.outputs & aig2.outputs)

    return AIG(
        inputs=aig1.inputs | aig2.inputs,
        top_level=aig1.top_level | aig2.top_level,
        comments=()
    )


def seq_compose(aig1, aig2, check_precondition=True):
    # TODO: apply simple optimizations such as unit propogation and excluded middle.

    interface = aig1.outputs & aig2.inputs
    if check_precondition:
        assert not (aig1.outputs - interface) & aig2.outputs
        assert not aig1.latches & aig2.latches
    
    lookup = dict(aig1.top_level)
    def sub(input_sig):
        return lookup.get(input_sig.name, input_sig)

    composed = bind(aig2.top_level).Recur(InputSignal).modify(sub)
    passthrough = frozenset((k, v) for k, v in aig1.top_level if k not in interface)

    return AIG(
        inputs=aig1.inputs | (aig2.inputs - interface),
        top_level=composed | passthrough,
        comments=()
    )
