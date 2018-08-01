from collections import defaultdict
from itertools import chain
from functools import reduce
from typing import Tuple, FrozenSet, NamedTuple, Union, Mapping, List

import funcy as fn
import lenses.hooks  # TODO: remove on next lenses version release.
from lenses import bind, lens
from toposort import toposort

from aiger import common as cmn


# TODO: Remove on next lenses lenses version release.
# Needed because 0.4 does not know about frozensets.
@lenses.hooks.from_iter.register(frozenset)
def _frozenset_from_iter(self, iterable):
    return frozenset(iterable)


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
class Input(NamedTuple):
    name: str

    @property
    def children(self):
        return ()


class LatchIn(NamedTuple):
    name: str

    @property
    def children(self):
        return ()


class ConstFalse(NamedTuple):
    @property
    def children(self):
        return ()


Node = Union[AndGate, ConstFalse, Inverter, Input, LatchIn]


class AIG(NamedTuple):
    inputs: FrozenSet[str]  # TODO: use internal names to make relabels fast.
    node_map: FrozenSet[Tuple[str, Node]]
    latch_map: FrozenSet[Tuple[str, Node]]
    latch2init: FrozenSet[Tuple[str, Node]]
    comments: Tuple[str]

    def __repr__(self):
        return repr(self._to_aag())

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
            'o': lens.node_map.Each()[0],
            # TODO: Test this lens.
            'l': lens.Fork(lens.Recur(LatchIn).name, lens.latch_map.Each()[0]),
        }.get(kind).modify(_relabel)(self)

    @property
    def outputs(self):
        return frozenset(fn.pluck(0, self.node_map))

    @property
    def latches(self):
        return frozenset(fn.pluck(0, self.latch2init))

    @property
    def cones(self):
        return frozenset(fn.pluck(1, self.node_map))

    @property
    def latch_cones(self):
        return frozenset(fn.pluck(1, self.latch_map))

    def __rshift__(self, other):
        return seq_compose(self, other)

    def __or__(self, other):
        return par_compose(self, other)

    @property
    def _eval_order(self):
        return list(toposort(_dependency_graph(self.cones | self.latch_cones)))

    def __call__(self, inputs, latches=None):
        # TODO: Implement partial evaluation.
        # TODO: Implement via DFS. In practice this was faster for _to_aag
        if latches is None:
            latches = dict()

        lookup = dict(inputs)  # Copy inputs as initial lookup table.
        latch2init = dict(self.latch2init)
        for node in fn.cat(self._eval_order):
            if isinstance(node, AndGate):
                lookup[node] = lookup[node.left] and lookup[node.right]
            elif isinstance(node, Inverter):
                lookup[node] = not lookup[node.input]
            elif isinstance(node, LatchIn):
                lookup[node] = latches.get(node.name, latch2init[node.name])
            elif isinstance(node, Input):
                lookup[node] = lookup[node.name]
            elif isinstance(node, ConstFalse):
                lookup[node] = False
            else:
                raise NotImplementedError

        outputs = {name: lookup[node] for name, node in self.node_map}
        latch_outputs = {name: lookup[node] for name, node in self.latch_map}
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

        aig = bind(self).Recur(LatchIn) \
                        .Filter(lambda x: x.name in latches) \
                        .modify(lambda x: Input(l_map[x.name][0]))

        _cones = {(l_map[k][0], v) for k, v in aig.latch_map if k in latches}
        aig = aig._replace(
            node_map=aig.node_map | _cones,
            inputs=aig.inputs | {n for n, _ in l_map.values()},
            latch_map={(k, v) for k, v in aig.latch_map if k not in latches},
            latch2init={(k, v) for k, v in aig.latch2init if k not in latches}
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
        aig = bind(self).Recur(Input). \
            Filter(lambda x: x.name in inputs). \
            modify(lambda x: LatchIn(in2latch[x.name]))

        _latch_map, node_map = fn.lsplit(
            lambda x: x[0] in outputs, aig.node_map
        )
        out2latch = {oname: lname for oname, lname in zip(outputs, latches)}
        _latch_map = {(out2latch[k], v) for k, v in _latch_map}
        l2init = frozenset((n, val) for n, val in zip(latches, initials))
        return aig._replace(
            inputs=aig.inputs - set(inputs),
            node_map=aig.node_map if keep_outputs else frozenset(node_map),
            latch_map=aig.latch_map | _latch_map,
            latch2init=aig.latch2init | l2init
        )

    def unroll(self, horizon, *, init=True, omit_latches=True):
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

        return unrolled

    def _to_aag(self):
        aag, max_idx, lit_map = _to_aag(
            self.cones | self.latch_cones,
            AAG({}, {}, {}, [], self.comments),
        )

        # Check that all inputs have a lit.
        for name in filter(lambda x: x not in aag.inputs, self.inputs):
            aag.inputs[name] = lit_map[name] = 2 * max_idx
            max_idx += 1

        # Update cone maps.
        aag.outputs.update({k: lit_map[cone] for k, cone in self.node_map})
        latch2init = dict(self.latch2init)
        for name, cone in self.latch_map:
            latch = LatchIn(name)
            if latch not in lit_map:
                lit = lit_map[latch] = 2 * max_idx
                max_idx += 1
            else:
                lit = lit_map[latch]

            init = int(latch2init[name])
            ilit = lit_map[cone]
            aag.latches[name] = lit, ilit, init

        return aag

    def write(self, path):
        with open(path, 'w') as f:
            f.write(repr(self))


def _to_idx(lit):
    """AAG format uses least significant bit to encode an inverter.
    The index is thus the interal literal shifted by one bit."""
    return lit >> 1


def _polarity(i):
    return Inverter if i & 1 == 1 else lambda x: x


class Header(NamedTuple):
    max_var_index: int
    num_inputs: int
    num_latches: int
    num_outputs: int
    num_ands: int


class AAG(NamedTuple):
    inputs: Mapping[str, int]
    latches: Mapping[str, Tuple[int]]
    outputs: Mapping[str, int]
    gates: List[Tuple[int]]
    comments: Tuple[str]

    @property
    def header(self):
        literals = chain(
            self.inputs.values(),
            self.outputs.values(),
            fn.pluck(0, self.gates), fn.pluck(0, self.latches.values())
        )
        max_idx = max(map(_to_idx, literals), default=0)
        return Header(max_idx, *map(len, self[:-1]))

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
            out += '\n'.join(
                f"i{idx} {name}" for idx, name in enumerate(input_names)
            ) + '\n'
        if self.outputs:
            out += '\n'.join(
                f"o{idx} {name}" for idx, name in enumerate(output_names)
            ) + '\n'

        if self.latches:
            out += '\n'.join(
                f"l{idx} {name}" for idx, name in enumerate(latch_names)
            ) + '\n'

        if self.comments:
            out += 'c\n' + '\n'.join(self.comments)
            if out[-1] != '\n':
                out += '\n'
        return out

    def _to_aig(self):
        gate_order, latch_order = self.eval_order_and_gate_lookup

        lookup = fn.merge(
            {0: ConstFalse()},
            {_to_idx(l): Input(n) for n, l in self.inputs.items()},
            {
                _to_idx(l): LatchIn(n)
                for n, (l, _, init) in self.latches.items()
            },
        )
        latches = set()
        and_dependencies = {i: (l, r) for i, l, r in self.gates}
        for gate in fn.cat(gate_order):
            if _to_idx(gate) in lookup:
                continue

            inputs = and_dependencies[gate]
            sources = [_polarity(i)(lookup[_to_idx(i)]) for i in inputs]
            lookup[_to_idx(gate)] = AndGate(*sources)

        latch_dependencies = {
            i: (n, dep) for n, (i, dep, _) in self.latches.items()
        }
        for gate in fn.cat(latch_order):
            assert _to_idx(gate) in lookup
            if not isinstance(lookup[_to_idx(gate)], LatchIn):
                continue

            name, dep = latch_dependencies[gate]
            source = _polarity(dep)(lookup[_to_idx(dep)])
            latches.add((name, source))

        def get_output(v):
            idx = _to_idx(v)
            return _polarity(v)(lookup[idx])

        top_level = ((k, get_output(v)) for k, v in self.outputs.items())
        return AIG(
            inputs=frozenset(self.inputs),
            node_map=frozenset(top_level),
            latch_map=frozenset(latches),
            latch2init=frozenset(
                (n, bool(init)) for n, (_, _, init) in self.latches.items()
            ),
            comments=self.comments
        )

    @property
    def eval_order_and_gate_lookup(self):
        deps = fn.merge(
            {a & -2: {b & -2, c & -2} for a, b, c in self.gates},
            {a & -2: set() for a in self.inputs.values()},
            {a & -2: set() for a, _, _ in self.latches.values()},  # LatchIn
        )
        latch_deps = {a & -2: {b & -2} for a, b, _ in self.latches.values()}
        return list(toposort(deps)), list(toposort(latch_deps))


def _to_aag(gates, aag: AAG = None, *, max_idx=1, lit_map=None):
    if lit_map is None:
        lit_map = {}

    if not gates:
        return aag, max_idx, lit_map

    # Recurse to update get aag for subtrees.
    for c in fn.mapcat(lambda g: g.children, gates):
        if c in lit_map:
            continue
        aag, max_idx, lit_map = _to_aag(
            [c], aag, max_idx=max_idx, lit_map=lit_map
        )

    # Update aag with current level.
    for gate in gates:
        if gate in lit_map:
            continue

        if isinstance(gate, Inverter):
            input_lit = lit_map[gate.input]
            lit_map[gate] = (input_lit & -2) | (1 ^ (input_lit & 1))
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

        elif isinstance(gate, Input):
            aag.inputs[gate.name] = lit_map[gate]

    return aag, max_idx, lit_map


def _dependency_graph(nodes):
    queue, deps = list(nodes), defaultdict(set)
    while queue:
        node = queue.pop()
        children = node.children
        queue.extend(children)
        deps[node].update(children)

    return deps


def par_compose(aig1, aig2, check_precondition=True):
    if check_precondition:
        assert not (aig1.latches & aig2.latches)
        assert not (aig1.outputs & aig2.outputs)

    return AIG(
        inputs=aig1.inputs | aig2.inputs,
        latch_map=aig1.latch_map | aig2.latch_map,
        latch2init=aig1.latch2init | aig2.latch2init,
        node_map=aig1.node_map | aig2.node_map,
        comments=aig1.comments + ('|', ) + aig2.comments
    )


def _is_const_true(node):
    return isinstance(node, Inverter) and isinstance(node.input, ConstFalse)


def sub_inputs(node, sub):
    if isinstance(node, AndGate):
        left = sub_inputs(node.left, sub)
        right = sub_inputs(node.right, sub)
        if ConstFalse() in (left, right):
            return ConstFalse()
        elif _is_const_true(left):
            return right
        elif _is_const_true(right):
            return left
        else:
            return node._replace(left=left, right=right)

    elif isinstance(node, Inverter):
        child = sub_inputs(node.input, sub)
        if isinstance(child, Inverter):
            return child.input
        else:
            return node._replace(input=child)

    elif isinstance(node, LatchIn):
        return node

    elif isinstance(node, Input):

        return sub.get(node.name, node)

    return ConstFalse()


def seq_compose(aig1, aig2, check_precondition=True):
    # TODO: apply simple optimizations such as unit propogation and
    # excluded middle.

    interface = aig1.outputs & aig2.inputs
    if check_precondition:
        assert not (aig1.outputs - interface) & aig2.outputs
        assert not aig1.latches & aig2.latches

    lookup = dict(aig1.node_map)
    composed = frozenset(
        (name, sub_inputs(cone, lookup)) for name, cone in aig2.node_map
    )

    composed_lmap = frozenset(
        (name, sub_inputs(cone, lookup)) for name, cone in aig2.latch_map
    )

    passthrough = frozenset(
        (k, v) for k, v in aig1.node_map if k not in interface
    )

    return AIG(
        inputs=aig1.inputs | (aig2.inputs - interface),
        latch_map=aig1.latch_map | composed_lmap,
        latch2init=aig1.latch2init | aig2.latch2init,
        node_map=composed | passthrough,
        comments=aig1.comments + ('>>', ) + aig2.comments
    )
