from functools import reduce
from typing import Tuple, FrozenSet, NamedTuple, Union

import attr
import funcy as fn
from pyrsistent import pmap

from aiger import common as cmn
from aiger import parser


def timed_name(name, time):
    return f"{name}##time_{time}"


class AndGate(NamedTuple):
    left: 'Node'  # TODO: replace with Node once 3.7 lands.
    right: 'Node'

    @property
    def children(self):
        return (self.left, self.right)

    def __hash__(self):
        return id(self)


class Inverter(NamedTuple):
    input: 'Node'

    @property
    def children(self):
        return (self.input, )

    def __hash__(self):
        return id(self)


# Enables filtering for Input via lens library.
@attr.s(frozen=True, slots=True, auto_attribs=True)
class Input:
    name: str

    @property
    def children(self):
        return ()


@attr.s(frozen=True, slots=True, auto_attribs=True)
class LatchIn:
    name: str

    @property
    def children(self):
        return ()


class ConstFalse(NamedTuple):
    @property
    def children(self):
        return ()


def _is_const_true(node):
    return isinstance(node, Inverter) and isinstance(node.input, ConstFalse)


Node = Union[AndGate, ConstFalse, Inverter, Input, LatchIn]


@attr.s(frozen=True, slots=True, auto_attribs=True, repr=False)
class AIG:
    inputs: FrozenSet[str] = frozenset()
    node_map: FrozenSet[Tuple[str, Node]] = attr.ib(
        default=pmap(), converter=pmap
    )
    latch_map: FrozenSet[Tuple[str, Node]] = frozenset()
    latch2init: FrozenSet[Tuple[str, Node]] = frozenset()
    comments: Tuple[str] = ()

    _to_aag = parser.aig2aag

    def __repr__(self):
        return repr(self._to_aag())

    def __getitem__(self, others):
        assert isinstance(others, tuple) and len(others) == 2
        kind, relabels = others
        assert kind in {'i', 'o', 'l'}

        basis = {
            'i': self.inputs, 'o': self.outputs, 'l': self.latches
        }.get(kind)
        relabels = fn.project(relabels, basis)

        if kind == 'o':
            relabels = {k: [v] for k, v in relabels.items()}
            return self >> cmn.tee(relabels)
        elif kind == 'i':
            relabels_ = {v: [k] for k, v in relabels.items()}
            return cmn.tee(relabels_) >> self

        # Latches act like inputs and outputs...
        def fix_keys(mapping):
            return fn.walk_keys(lambda x: relabels.get(x, x), mapping)

        circ = self.evolve(
            latch_map=fix_keys(self.latch_map),
            latch2init=fix_keys(self.latch2init)
        )

        def sub(node):
            if isinstance(node, LatchIn):
                return LatchIn(relabels.get(node.name, node.name))
            return node

        return circ._modify_leafs(sub)

    def evolve(self, **kwargs):
        return attr.evolve(self, **kwargs)

    @property
    def aig(self):
        return self

    @property
    def outputs(self):
        return frozenset(self.node_map.keys())

    @property
    def latches(self):
        return frozenset(fn.pluck(0, self.latch2init))

    @property
    def cones(self):
        return frozenset(self.node_map.values())

    @property
    def latch_cones(self):
        return frozenset(fn.pluck(1, self.latch_map))

    def __rshift__(self, other):
        return seq_compose(self, other)

    def __lshift__(self, other):
        return seq_compose(other, self)

    def __or__(self, other):
        return par_compose(self, other)

    def __call__(self, inputs, latches=None):
        if latches is None:
            latches = dict()
        latchins = fn.merge(dict(self.latch2init), latches)
        # Remove latch inputs not used by self.
        latchins = fn.project(latchins, self.latches)

        # Turn into a combinatorial circuit
        circ, lmap = self.cutlatches(self.latches)
        latchins = fn.walk_keys(lambda n: lmap[n][0], latchins)
        inputs = fn.merge(inputs, latchins)

        circ = cmn.source(inputs) >> circ

        all_outputs = {
            n: _is_const_true(node) for n, node in circ.node_map.items()
        }
        outputs = fn.project(all_outputs, self.outputs)
        latch_outputs = fn.omit(all_outputs, self.outputs)

        # Fix up latch names
        lmap_inv = {k: n for n, (k, _) in lmap.items()}
        latch_outputs = fn.walk_keys(lmap_inv.get, latch_outputs)

        return outputs, latch_outputs

    def simulator(self, latches=None):
        inputs = yield
        while True:
            outputs, latches = self(inputs, latches)
            inputs = yield outputs, latches

    def simulate(self, input_seq, latches=None):
        sim = self.simulator()
        next(sim)
        return [sim.send(inputs) for inputs in input_seq]

    def cutlatches(self, latches, check_postcondition=True):
        l_map = {n: (cmn._fresh(), init) for (n, init) in self.latch2init}

        assert len(
            set(fn.pluck(0, l_map.values())) & (self.inputs | self.outputs)
        ) == 0

        def sub(node):
            if isinstance(node, LatchIn):
                return Input(l_map[node.name][0])
            return node

        circ = self._modify_leafs(sub)
        _cones = {l_map[k][0]: v for k, v in circ.latch_map if k in latches}
        aig = self.evolve(
            node_map=circ.node_map + _cones,
            inputs=self.inputs | {n for n, _ in l_map.values()},
            latch_map={(k, v) for k, v in circ.latch_map if k not in latches},
            latch2init={(k, v) for k, v in self.latch2init if k not in latches}
        )
        return aig, l_map

    def feedback(
        self, inputs, outputs, initials=None, latches=None, keep_outputs=False
    ):
        if latches is None:
            latches = inputs

        if initials is None:
            initials = [False for _ in inputs]

        assert len(inputs) == len(initials) == len(outputs) == len(latches)
        assert len(set(inputs) & self.inputs) != 0
        assert len(set(outputs) & self.outputs) != 0

        in2latch = {iname: lname for iname, lname in zip(inputs, latches)}

        def sub(node):
            if isinstance(node, Input) and node.name in inputs:
                return LatchIn(in2latch[node.name])
            return node

        aig = self._modify_leafs(sub)

        _latch_map, node_map = fn.lsplit(
            lambda x: x[0] in outputs, aig.node_map.items()
        )
        out2latch = {oname: lname for oname, lname in zip(outputs, latches)}
        _latch_map = {(out2latch[k], v) for k, v in _latch_map}
        l2init = frozenset((n, val) for n, val in zip(latches, initials))
        return aig.evolve(
            inputs=aig.inputs - set(inputs),
            node_map=aig.node_map if keep_outputs else pmap(node_map),
            latch_map=aig.latch_map | _latch_map,
            latch2init=aig.latch2init | l2init
        )

    def unroll(self, horizon, *, init=True, omit_latches=True,
               only_last_outputs=False):
        # TODO:
        # - Check for name collisions.
        latches = self.latches
        aag0, l_map = self.cutlatches({l for l in latches})

        def _unroll():
            prev = aag0
            for t in range(1, horizon + 1):
                tmp = prev['i', {k: timed_name(k, t - 1) for k in aag0.inputs}]
                yield tmp['o', {k: timed_name(k, t) for k in aag0.outputs}]

        unrolled = reduce(seq_compose, _unroll())
        if init:
            source = {timed_name(n, 0): init for n, init in l_map.values()}
            unrolled = cmn.source(source) >> unrolled

        if omit_latches:
            latch_names = [timed_name(n, horizon) for n, _ in l_map.values()]
            unrolled = unrolled >> cmn.sink(latch_names)

        if only_last_outputs:
            odrop = fn.lfilter(
                lambda o: int(o.split('##time_')[1]) < horizon,
                unrolled.outputs
            )
            unrolled = unrolled >> cmn.sink(odrop)

        return unrolled

    def write(self, path):
        with open(path, 'w') as f:
            f.write(repr(self))

    def _modify_leafs(self, func):
        @fn.memoize
        def _mod(node):
            if isinstance(node, AndGate):
                left = _mod(node.left)
                right = _mod(node.right)
                if ConstFalse() in (left, right):
                    return ConstFalse()
                elif _is_const_true(left):
                    return right
                elif _is_const_true(right):
                    return left
                else:
                    return node._replace(left=left, right=right)

            elif isinstance(node, Inverter):
                child = _mod(node.input)
                if isinstance(child, Inverter):
                    return child.input
                else:
                    return node._replace(input=child)

            return func(node)

        node_map = ((name, _mod(cone)) for name, cone in self.node_map.items())
        latch_map = ((name, _mod(cone)) for name, cone in self.latch_map)
        return self.evolve(
            node_map=pmap(node_map),
            latch_map=frozenset(latch_map)
        )


def par_compose(aig1, aig2, check_precondition=True):
    assert not aig1.latches & aig2.latches
    assert not aig1.outputs & aig2.outputs

    shared_inputs = aig1.inputs & aig2.inputs
    if shared_inputs:
        relabels1 = {n: cmn._fresh() for n in shared_inputs}
        relabels2 = {n: cmn._fresh() for n in shared_inputs}
        aig1, aig2 = aig1['i', relabels1], aig2['i', relabels2]

    circ = AIG(
        inputs=aig1.inputs | aig2.inputs,
        latch_map=aig1.latch_map | aig2.latch_map,
        latch2init=aig1.latch2init | aig2.latch2init,
        node_map=aig1.node_map + aig2.node_map,
        comments=aig1.comments + aig2.comments
    )

    if shared_inputs:
        for orig in shared_inputs:
            new1, new2 = relabels1[orig], relabels2[orig]
            circ <<= cmn.tee({orig: [new1, new2]})

    return circ


def seq_compose(circ1, circ2, *, input_kind=Input):
    interface = circ1.outputs & circ2.inputs
    assert not (circ1.outputs - interface) & circ2.outputs
    assert not circ1.latches & circ2.latches

    passthrough = {
        k: v for k, v in circ1.node_map.items() if k not in interface
    }

    circ3 = circ2
    for mapping in [circ1.node_map, circ1.latch_map]:
        lookup = dict(mapping)

        def sub(node):
            if isinstance(node, input_kind):
                return lookup.get(node.name, node)
            return node

        circ3 = circ3._modify_leafs(sub)

    return AIG(
        inputs=circ1.inputs | (circ2.inputs - interface),
        latch_map=circ1.latch_map | circ3.latch_map,
        latch2init=circ1.latch2init | circ2.latch2init,
        node_map=circ3.node_map + passthrough,
        comments=circ1.comments + circ2.comments
    )
