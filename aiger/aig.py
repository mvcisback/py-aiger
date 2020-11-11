"""
Abstractions for compositions/manipulations of And Inverter Graphs.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
import operator as op
import pathlib
from typing import Tuple, FrozenSet

import attr
import funcy as fn
from pyrsistent import pmap
from pyrsistent.typing import PMap

import aiger as A
from aiger import common as cmn
from aiger import parser
from aiger import writer


@attr.s(frozen=True, auto_attribs=True, eq=False)
class Node(metaclass=ABCMeta):
    def __and__(self, other: Node) -> Node:
        if self.is_false or other.is_false:
            return ConstFalse()
        elif self.is_true:
            return other
        elif other.is_true:
            return self
        return AndGate(self, other)

    def __invert__(self) -> Node:
        if isinstance(self, Inverter):
            return self.input
        return Inverter(self)

    @property
    def is_false(self):
        return isinstance(self, ConstFalse)

    @property
    def is_true(self):
        return (~self).is_false

    @property
    @abstractmethod
    def children(self):
        pass


@attr.s(frozen=True, auto_attribs=True, eq=False)
class AndGate(Node):
    left: Node
    right: Node

    @property
    def children(self):
        return (self.left, self.right)


@attr.s(frozen=True, auto_attribs=True)  # Allow Hashing.
class Inverter(Node):
    input: Node

    @property
    def children(self):
        return (self.input, )


@attr.s(frozen=True, auto_attribs=True)
class Input(Node):
    name: str

    @property
    def children(self):
        return ()


@attr.s(frozen=True, auto_attribs=True)
class LatchIn(Node):
    name: str

    @property
    def children(self):
        return ()


@attr.s(frozen=True, auto_attribs=True)
class ConstFalse(Node):
    @property
    def children(self):
        return ()

    def __hash__(self):
        return hash(False)


@attr.s(frozen=True, auto_attribs=True, repr=False)
class AIG:
    inputs: FrozenSet[str] = frozenset()
    node_map: PMap[str, Node] = attr.ib(default=pmap(), converter=pmap)
    latch_map: PMap[str, Node] = attr.ib(default=pmap(), converter=pmap)
    latch2init: PMap[str, bool] = attr.ib(default=pmap(), converter=pmap)
    comments: Tuple[str] = ()

    def __repr__(self):
        return writer.dump(self)

    def __getitem__(self, others):
        return self.lazy_aig[others].aig

    def __iter_nodes__(self):
        """Returns an iterator over iterators of nodes in an AIG.

        If completely flattened this iterator would give topological
        a order on the nodes, starting on the inputs.

        The reason for the iterator over iterators is to mark
        dependencies. Namely, to compute the value of any node
        requires just the value of the nodes in the previous iterator.
        """
        return [cmn.dfs(self)]

    def evolve(self, **kwargs):
        return attr.evolve(self, **kwargs)

    def relabel(self, key, mapping):
        assert key in {'input', 'output', 'latch'}
        return self[key[0], mapping]

    @property
    def aig(self):
        return self

    @property
    def lazy_aig(self):
        return A.lazy(self)

    @property
    def outputs(self):
        return frozenset(self.node_map.keys())

    @property
    def latches(self):
        return frozenset(self.latch2init.keys())

    @property
    def cones(self):
        return frozenset(self.node_map.values())

    @property
    def latch_cones(self):
        return frozenset(self.latch_map.values())

    def __rshift__(self, other):
        return seq_compose(self, other)

    def __lshift__(self, other):
        return seq_compose(other, self)

    def __or__(self, other):
        return par_compose(self, other)

    def __call__(self, inputs, latches=None, *, lift=None):
        """Evaluate AIG on inputs (and latches).
        If `latches` is `None` initial latch value is used.

        `lift` is an optional argument used to interpret constants
        (False, True) in some other Boolean algebra over (&, ~).

        - See py-aiger-bdd and py-aiger-cnf for examples.
        """
        if latches is None:
            latches = dict()

        if lift is None:
            lift = fn.identity
            and_, neg = op.and_, op.not_
        else:
            and_, neg = op.__and__, op.__invert__

        latchins = fn.merge(dict(self.latch2init), latches)
        # Remove latch inputs not used by self.
        latchins = fn.project(latchins, self.latches)

        latch_map = dict(self.latch_map)
        boundary = set(self.node_map.values()) | set(latch_map.values())

        store, prev, mem = {}, set(), {}
        for node_batch in self.__iter_nodes__():
            prev = set(mem.keys()) - prev
            mem = fn.project(mem, prev)  # Forget about unnecessary gates.

            for gate in node_batch:
                if isinstance(gate, Inverter):
                    mem[gate] = neg(mem[gate.input])
                elif isinstance(gate, AndGate):
                    mem[gate] = and_(mem[gate.left], mem[gate.right])
                elif isinstance(gate, Input):
                    mem[gate] = lift(inputs[gate.name])
                elif isinstance(gate, LatchIn):
                    mem[gate] = lift(latchins[gate.name])
                elif isinstance(gate, ConstFalse):
                    mem[gate] = lift(False)

                if gate in boundary:
                    store[gate] = mem[gate]  # Store for eventual output.

        outs = {out: store[gate] for out, gate in self.node_map.items()}
        louts = {out: store[gate] for out, gate in latch_map.items()}
        return outs, louts

    def simulator(self, latches=None):
        inputs = yield
        while True:
            outputs, latches = self(inputs, latches)
            inputs = yield outputs, latches

    def simulate(self, input_seq, latches=None):
        sim = self.simulator()
        next(sim)
        return [sim.send(inputs) for inputs in input_seq]

    def cutlatches(self, latches=None, check_postcondition=True, renamer=None):
        lcirc, lmap = self.lazy_aig.cutlatches(latches, renamer=renamer)
        return lcirc.aig, lmap

    def loopback(self, *wirings):
        return self.lazy_aig.loopback(*wirings).aig

    def feedback(
        self, inputs, outputs, initials=None, latches=None, keep_outputs=False
    ):
        import warnings
        warnings.warn("deprecated", DeprecationWarning)

        def create_wire(val):
            iname, oname, lname, init = val
            return {
                'input': iname, 'output': oname, 'latch': lname, 'init': init,
                'keep_output': keep_outputs
            }

        if initials is None:
            initials = fn.repeat(False)

        if latches is None:
            assert (set(inputs) & self.latches) == set()
            latches = inputs

        vals = zip(inputs, outputs, latches, initials)
        return self.loopback(*map(create_wire, vals))

    def unroll(self, horizon, *, init=True, omit_latches=True,
               only_last_outputs=False):
        return self.lazy_aig.unroll(
            horizon, init=init, omit_latches=omit_latches,
            only_last_outputs=only_last_outputs
        ).aig

    def write(self, path):
        with open(path, 'w') as f:
            f.write(repr(self))

    def reinit(self, latch2init):
        """Update late initial values based on mapping provided."""
        return self.lazy_aig.reinit(latch2init).aig


def par_compose(aig1, aig2, check_precondition=True):
    return A.Parallel(aig1, aig2).aig


def seq_compose(circ1, circ2, *, input_kind=Input):
    return A.Cascading(circ1, circ2).aig


def to_aig(circ, *, allow_lazy=False) -> AIG:
    if isinstance(circ, pathlib.Path) and circ.is_file():
        circ = parser.load(circ)
    elif isinstance(circ, str):
        if circ.startswith('aag '):
            circ = parser.parse(circ)  # Assume it is an AIGER string.
        else:
            circ = parser.load(circ)  # Assume it is a file path.

    if allow_lazy and hasattr(circ, '.lazy_aig'):
        return circ.lazy_aig

    return circ.aig  # Extract AIG from potential wrapper.
