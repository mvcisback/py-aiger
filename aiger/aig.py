from collections import defaultdict
from itertools import starmap
from typing import Tuple, FrozenSet, NamedTuple, Union, Mapping, List

import funcy as fn
import lenses.hooks  # TODO: remove on next lenses version release.
from lenses import bind, lens
from toposort import toposort


# TODO: Remove on next lenses lenses version release.
# Needed because 0.4 does not know about frozensets.
@lenses.hooks.from_iter.register(frozenset)
def _frozenset_from_iter(self, iterable):
    return frozenset(iterable)


class AndGate(NamedTuple):
    left: 'Node'  # TODO: replace with Node once 3.7 lands.
    right: 'Node'

    @property
    def children(self):
        return tuple((self.left, self.right))


class Latch(NamedTuple):
    name: str
    input: 'Node'
    initial: bool

    @property
    def children(self):
        return tuple((self.input, ))


class Inverter(NamedTuple):
    input: 'Node'

    @property
    def children(self):
        return tuple((self.input, ))


# Enables filtering for Input via lens library.
class Input(NamedTuple):
    name: str

    @property
    def children(self):
        return tuple()


class ConstFalse(NamedTuple):
    @property
    def children(self):
        return tuple()


Node = Union[AndGate, Latch, ConstFalse, Inverter, Input]


class AIG(NamedTuple):
    inputs: FrozenSet[str]
    top_level: FrozenSet[Tuple[str, Node]]
    comments: Tuple[str]

    # TODO:
    # __repr__(self):

    def __getitem__(self, others):
        if not isinstance(others, tuple):
            return super().__getitem__(others)

        kind, relabels = others
        if kind not in {'i', 'o', 'l'}:
            raise NotImplementedError

        def _relabel(n):
            return relabels.get(n, n)

        return {
            'i': lens.Fork(lens.Recur(Input).name, lens.inputs.Each()),
            'o': lens.top_level.Each()[0],
            'l': lens.Recur(Latch).name
        }.get(kind).modify(_relabel)(self)

    @property
    def outputs(self):
        return frozenset(fn.pluck(0, self.top_level))

    @property
    def latches(self):
        return frozenset(bind(self).Recur(Latch).collect())

    @property
    def cones(self):
        return frozenset(fn.pluck(1, self.top_level))

    def __rshift__(self, other):
        return seq_compose(self, other)

    def __or__(self, other):
        return par_compose(self, other)

    @property
    def _eval_order(self):
        return list(toposort(_dependency_graph(self.nodes)))

    def __call__(self, inputs, latches=None):
        # TODO: Implement partial evaluation.
        if latches is None:
            latches = dict()

        latches = {l: latches.get(l.name, l.initial) for l in self.latches}
        lookup = fn.merge(inputs, latches)
        for node in fn.cat(self._eval_order[1:]):
            if isinstance(node, AndGate):
                lookup[node] = lookup[node.left] and lookup[node.right]
            elif isinstance(node, Inverter):
                lookup[node] = not lookup[node.input]
            elif isinstance(node, Latch):
                lookup[node] = lookup[node.input]
            elif isinstance(node, Input):
                lookup[node] = lookup[node.name]
            elif isinstance(node, ConstFalse):
                lookup[node] = False
            else:
                raise NotImplementedError

        outputs = {name: lookup[node] for name, node in self.top_level}
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

    def cutlatches(self, latches):
        # TODO: assert relabels won't collide with existing labels.
        latch_top_level = {(l.name, l.input) for l in self.latches}
        return AIG(
            inputs=fn.merge(self.inputs, frozenset(name_to_latch)),
            top_level=self.top_level | latch_top_level,
            comments=())

    def unroll(self, horizon, *, init=True, omit_latches=True):
        # TODO:
        # - Check for name collisions.
        aag0 = cutlatches(self, self.latches.keys())

        def _unroll():
            prev = aag0
            for t in range(1, horizon + 1):
                tmp = prev['i', {k: f"{k}##time_{t-1}" for k in aag0.inputs}]
                yield tmp['o', {k: f"{k}##time_{t}" for k in aag0.outputs}]

        unrolled = reduce(seq_compose, _unroll())
        if init:
            latch_source = {
                f"{k}##time_0": val
                for k, (_, _, val) in self.latches.items()
            }
            unrolled = source(latch_source) >> unrolled

        if omit_latches:
            latch_names = [f"{k}##time_{horizon}" for k in self.latches.keys()]

            unrolled = unrolled >> sink(latch_names)

        return unrolled

    def to_aag(self):
        # TODO: toposort.
        # TODO: convert
        pass


class AAG(NamedTuple):
    lit_map: Mapping[Node, int]
    inputs: Mapping[str, int]
    latches: Mapping[str, Tuple[int]]
    outputs: Mapping[str, int]
    gates: List[Tuple[int]]
    comments: Tuple[str]

    @property
    def header(self):
        return (max((v >> 1
                     for v in self.lit_map.values())), len(self.inputs), len(
                         self.latches), len(self.outputs), len(self.gates))

    def __repr__(self):
        if self.inputs:
            input_names, input_lits = zip(*list(self.inputs.items()))
        if self.outputs:
            output_names, output_lits = zip(*list(self.outputs.items()))
        if self.latches:
            latch_names, latch_lits = zip(*list(self.latches.items()))

        out = f"aag " + " ".join(map(str, self.header)) + '\n'
        if self.inputs:
            out += '\n'.join(map(str, input_lits)) + '\n'
        if self.latches:
            out += '\n'.join([' '.join(map(str, xs))
                              for xs in latch_lits]) + '\n'
        if self.outputs:
            out += '\n'.join(map(str, output_lits)) + '\n'
        if self.gates:
            out += '\n'.join([' '.join(map(str, xs))
                              for xs in self.gates]) + '\n'
        if self.inputs:
            out += '\n'.join(f"i{idx} {name}"
                             for idx, name in enumerate(input_names)) + '\n'
        if self.outputs:
            out += '\n'.join(f"o{idx} {name}"
                             for idx, name in enumerate(output_names)) + '\n'
        if self.latches:
            out += '\n'.join(f"l{idx} {name}"
                             for idx, name in enumerate(latch_names)) + '\n'
        if self.comments:
            out += 'c\n' + '\n'.join(self.comments) + '\n'
        return out


def _to_aag(aig):
    aag = __to_aag(aig.cones, AAG({}, {}, {}, {}, [], aig.comments))
    aag.outputs.update({k: aag.lit_map[cone] for k, cone in aig.top_level})
    return aag


def __to_aag(gates, aag: AAG = None, *, max_idx=1):
    if not gates:
        return aag

    # Recurse to update get aag for subtrees.
    children = fn.cat(g.children for g in gates)
    children = [c for c in children if c not in aag.lit_map]
    aag = __to_aag(children, aag, max_idx=max_idx)

    # Update aag with current level.
    lit_map = aag.lit_map
    for gate in gates:
        if gate in lit_map:
            continue

        if isinstance(gate, Inverter):
            lit_map[gate] = lit_map[gate.input] + 1
            continue
        elif isinstance(gate, ConstFalse):
            lit_map[gate] = 0
            continue

        # Must be And, Latch, or Input
        lit_map[gate] = 2 * max_idx
        max_idx += 1
        if isinstance(gate, AndGate):
            encoded = tuple(map(lit_map.get, (gate, gate.left, gate.right)))
            aag.gates.append(encoded)

        elif isinstance(gate, Latch):
            encoded = (lit_map[gate], lit_map[gate.input], int(gate.initial))
            aag.latches[gate.name] = encoded

        elif isinstance(gate, Input):
            aag.inputs[gate.name] = lit_map[gate]

    return aag


def _dependency_graph(nodes):
    queue, deps = list(nodes), defaultdict(set)
    while queue:
        node = queue.pop()
        children = node.children
        queue.extend(children)
        deps[node].update(children)

    return deps


def _map_tree(inputs, f):
    queue = fn.lmap(Input, inputs)
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
        top_level=frozenset(((output, _map_tree(inputs, f=AndGate)), )),
        comments=())


def identity(inputs):
    return AIG(
        inputs=frozenset(inputs),
        top_level=frozenset(zip(inputs, map(Input, inputs))),
        comments=())


def empty():
    return identity([])


def _inverted_input(name):
    return Inverter(Input(name))


def inverter(inputs, outputs=None):
    if outputs is None:
        outputs = inputs
    else:
        assert len(outputs) == len(inputs)

    return AIG(
        inputs=frozenset(inputs),
        top_level=frozenset(zip(outputs, map(_inverted_input, inputs))),
        comments=())


def _const(val):
    return Inverter(ConstFalse()) if val else ConstFalse()


def source(outputs):
    return AIG(
        inputs=frozenset(),
        top_level=frozenset((k, _const(v)) for k, v in outputs.items()),
        comments=())


def sink(inputs):
    return AIG(inputs=frozenset(inputs), top_level=frozenset(), comments=())


def tee(outputs):
    def tee_output(name, renames):
        return frozenset((r, Input(name)) for r in renames)

    return AIG(
        inputs=frozenset(outputs),
        top_level=frozenset.union(*starmap(tee_output, outputs.items())),
        comments=[])


def par_compose(aig1, aig2, check_precondition=True):
    if check_precondition:
        assert not (aig1.latches & aig2.latches)
        assert not (aig1.outputs & aig2.outputs)

    return AIG(
        inputs=aig1.inputs | aig2.inputs,
        top_level=aig1.top_level | aig2.top_level,
        comments=())


def seq_compose(aig1, aig2, check_precondition=True):
    # TODO: apply simple optimizations such as unit propogation and
    # excluded middle.

    interface = aig1.outputs & aig2.inputs
    if check_precondition:
        assert not (aig1.outputs - interface) & aig2.outputs
        assert not aig1.latches & aig2.latches

    lookup = dict(aig1.top_level)

    def sub(input_sig):
        return lookup.get(input_sig.name, input_sig)

    composed = bind(aig2.top_level).Recur(Input).modify(sub)
    passthrough = frozenset(
        (k, v) for k, v in aig1.top_level if k not in interface)

    return AIG(
        inputs=aig1.inputs | (aig2.inputs - interface),
        top_level=composed | passthrough,
        comments=())
