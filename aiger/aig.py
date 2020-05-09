import operator as op
import pathlib
from typing import Tuple, FrozenSet, NamedTuple, Union

import attr
import funcy as fn
from pyrsistent import pmap

import aiger as A
from aiger import common as cmn
from aiger import parser


def timed_name(name, time):
    return f"{name}##time_{time}"


@attr.s(frozen=True, slots=True, auto_attribs=True, cache_hash=True)
class AndGate:
    left: 'Node'                # TODO: replace with Node once 3.7 lands.
    right: 'Node'

    _replace = attr.evolve      # Backwards compat with NamedTuple code.

    @property
    def children(self):
        return (self.left, self.right)


@attr.s(frozen=True, slots=True, auto_attribs=True, cache_hash=True)
class Inverter:
    input: 'Node'

    _replace = attr.evolve      # Backwards compat with NamedTuple code.

    @property
    def children(self):
        return (self.input, )


# Enables filtering for Input via lens library.
@attr.s(frozen=True, slots=True, auto_attribs=True)
class Input:
    name: str

    @property
    def children(self):
        return ()


@attr.s(frozen=True, slots=True, auto_attribs=True, cache_hash=True)
class Shim:
    old: 'Node'
    new: 'Node'


@attr.s(frozen=True, slots=True, auto_attribs=True)
class LatchIn:
    name: str

    @property
    def children(self):
        return ()


class ConstFalse(NamedTuple):
    @property
    def children(se0lf):
        return ()

    def __hash__(self):
        return hash(False)


def _is_const_true(node):
    return isinstance(node, Inverter) and isinstance(node.input, ConstFalse)


Node = Union[AndGate, ConstFalse, Inverter, Input, LatchIn]


@attr.s(frozen=True, slots=True, auto_attribs=True, repr=False)
class AIG:
    inputs: FrozenSet[str] = frozenset()
    node_map: FrozenSet[Tuple[str, Node]] = attr.ib(
        default=pmap(), converter=pmap
    )
    # TODO: change to pmap
    latch_map: FrozenSet[Tuple[str, Node]] = frozenset()
    latch2init: FrozenSet[Tuple[str, bool]] = frozenset()
    comments: Tuple[str] = ()

    _to_aag = parser.aig2aag

    def __repr__(self):
        return repr(self._to_aag())

    def __getitem__(self, others):
        return A.lazy(self)[others].aig

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
        return (A.lazy(self) >> A.lazy(other)).aig

    def __lshift__(self, other):
        return other >> self

    def __or__(self, other):
        return (A.lazy(self) | A.lazy(other)).aig

    def __call__(self, inputs, latches=None, *, false=False):
        """Evaluate AIG on inputs (and latches).

        If `latches` is `None` initial latch value is used.

        `false` is an optional argument used to interpet the AIG as
        an object in some other Boolean algebra over (&, ~).
          - See py-aiger-bdd and py-aiger-cnf for examples.
        """
        if latches is None:
            latches = dict()

        if false is False:
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
                elif isinstance(gate, Shim):
                    mem[gate.new] = mem[gate.old]
                    gate = gate.new         # gate.new could be the output.
                elif isinstance(gate, Input):
                    mem[gate] = inputs[gate.name]
                elif isinstance(gate, LatchIn):
                    mem[gate] = latchins[gate.name]
                elif isinstance(gate, ConstFalse):
                    mem[gate] = false

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

    def cutlatches(self, latches=None, renamer=None):
        lcirc = A.lazy(self)
        lcirc2, l_map = lcirc.cutlatches(latches=latches, renamer=renamer)
        return lcirc2.aig, l_map

    def loopback(self, *wirings):
        return A.lazy(self).loopback(*wirings)

    def unroll(self, horizon, *, init=True, omit_latches=True,
               only_last_outputs=False):
        return A.lazy(self).unroll(
            horizon, init=init,
            omit_latches=omit_latches,
            only_last_outputs=only_last_outputs
        ).aig

    def write(self, path):
        with open(path, 'w') as f:
            f.write(repr(self))


def to_aig(circ) -> AIG:
    if isinstance(circ, pathlib.Path) and circ.is_file():
        circ = parser.load(circ)
    elif isinstance(circ, str):
        if circ.startswith('aag '):
            circ = parser.parse(circ)  # Assume it is an AIGER string.
        else:
            circ = parser.load(circ)  # Assume it is a file path.

    return circ.aig  # Extract AIG from potential wrapper.
