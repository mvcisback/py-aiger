from itertools import chain
from functools import reduce
from typing import Tuple, FrozenSet, NamedTuple, Union, Mapping, List
from uuid import uuid1

import funcy as fn
import lenses.hooks  # TODO: remove on next lenses version release.
from pyrsistent import pmap
from toposort import toposort_flatten as toposort

from aiger import common


# TODO: Remove on next lenses lenses version release.
# Needed because 0.4 does not know about frozensets.
@lenses.hooks.from_iter.register(frozenset)
def _frozenset_from_iter(self, iterable):
    return frozenset(iterable)


Reference = Union[str, int]


class AndGate(NamedTuple):
    left: Reference
    right: Reference

    @property
    def children(self):
        return (self.left, self.right)


class Latch(NamedTuple):
    input: Reference
    initial: bool

    @property
    def children(self):
        return (self.input, )


class Inverter(NamedTuple):
    input: Reference

    @property
    def children(self):
        return (self.input, )


# Enables filtering for Input via lens library.
class Input(NamedTuple):
    @property
    def children(self):
        return ()


class ConstFalse(NamedTuple):
    @property
    def children(self):
        return ()


Node = Union[AndGate, Latch, ConstFalse, Inverter, Input]


def _invert_lit(input_lit):
    return (input_lit & -2) | (1 ^ (input_lit & 1))


def _walk_values(f, mapping):
    return {k: f(v) for k, v in mapping.items()}


def _omit(mapping, keys):
    return {k: v for k, v in mapping.items() if k not in keys}


class AIG(NamedTuple):
    input_map: Mapping[str, Reference]
    output_map: Mapping[str, Reference]
    latch_map: Mapping[str, Reference]
    node_map: Mapping[Reference, Node]
    comments: Tuple[str]

    def __repr__(self):
        return repr(self._to_aag())

    def __getitem__(self, others):
        if not isinstance(others, tuple):
            return super().__getitem__(others)

        kind, relabels = others

        def _relabel(n):
            return relabels.get(n, n)

        if kind not in {'i', 'o', 'l'}:
            raise NotImplementedError

        key = {'i': 'input_map', 'o': 'output_map', 'l': 'latch_map'}.get(kind)
        return self._replace(**{
            key:
            pmap({_relabel(k): v
                  for k, v in getattr(self, key).items()})
        })

    @property
    def outputs(self) -> FrozenSet[str]:
        return frozenset(self.output_map.keys())

    @property
    def latches(self) -> FrozenSet[str]:
        return frozenset(self.latch_map.keys())

    @property
    def inputs(self) -> FrozenSet[str]:
        return frozenset(self.input_map.keys())

    def __rshift__(self, other):
        return seq_compose(self, other)

    def __or__(self, other):
        return par_compose(self, other)

    def __call__(self, inputs, latches=None):
        # TODO: Implement partial evaluation.
        if latches is None:
            latches = dict()

        lookup = fn.merge({
            ref: latches.get(name, self.node_map[ref].initial)
            for name, ref in self.latch_map.items()
        }, {ref: inputs[name]
            for name, ref in self.input_map.items()})

        for ref in self.output_map.values():
            self._eval(ref, lookup)

        outputs = {name: lookup[ref] for name, ref in self.output_map.items()}
        latches = {
            name: lookup[self.node_map[ref].input]
            for name, ref in self.latch_map.items()
        }
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
        # 1. Each latch becomes a new input.
        new_inputs = {name: self.latch_map[name] for name in latches}

        # 2. Each latch becomes a new output.
        new_outputs = _walk_values(lambda ref: self.node_map[ref].input,
                                   self.latch_map)

        return AIG(
            input_map=self.input_map.update(new_inputs),
            output_map=self.output_map.update(new_outputs),
            latch_map=pmap(_omit(self.latch_map, latches)),
            node_map=pmap(_omit(self.node_map, new_inputs.values())),
            comments=())

    def unroll(self, horizon, *, init=True, omit_latches=True):
        # TODO:
        # - Check for name collisions.
        aag0 = self.cutlatches(self.latches)
        latches = _walk_values(self.node_map.get, self.latch_map).items()

        def _unroll():
            prev = aag0
            for t in range(1, horizon + 1):
                tmp = prev['i', {k: f"{k}##time_{t-1}" for k in aag0.inputs}]
                yield tmp['o', {k: f"{k}##time_{t}" for k in aag0.outputs}]

        unrolled = reduce(seq_compose, _unroll())
        if init:
            latch_source = {f"{n}##time_0": l.initial for n, l in latches}
            unrolled = common.source(latch_source) >> unrolled

        if omit_latches:
            latch_names = [f"{n}##time_{horizon}" for n, _ in latches]
            unrolled = unrolled >> common.sink(latch_names)

        return unrolled

    def _to_aag(self):
        # Compute ref -> lit map.
        refs = set(
            chain(
                self.input_map.values(),
                self.latch_map.values(),
                (k for k, v in self.node_map.items()
                 if not isinstance(v, Inverter)),
            ))
        ref_to_lit = {ref: (idx + 1) << 1 for idx, ref in enumerate(refs)}

        # Propogate lits to inverters.
        # Toposort inverters.
        inverter_map = {
            ref: {node.input}
            for ref, node in self.node_map.items()
            if isinstance(node, Inverter)
        }
        eval_order = list(toposort(inverter_map))
        for ref in eval_order:
            if ref in ref_to_lit:
                continue

            node = self.node_map[ref]
            ref_to_lit[ref] = _invert_lit(ref_to_lit[node.input])

        # Convert AIG to ASCII encoding.
        gates = [(ref_to_lit[k], ref_to_lit[v.left], ref_to_lit[v.right])
                 for k, v in self.node_map.items() if isinstance(v, AndGate)]
        latches = {
            k: (ref_to_lit[v], ref_to_lit[self.node_map[v].input], int(
                self.node_map[v].initial))
            for k, v in self.latch_map.items()
        }
        return AAG(
            inputs=_walk_values(ref_to_lit.get, self.input_map),
            outputs=_walk_values(ref_to_lit.get, self.output_map),
            latches=latches,
            gates=gates,
            comments=self.comments)

    def write(self, path):
        with open(path, 'w') as f:
            f.write(repr(self))

    def _eval(self, node_ref, lookup):
        node = self.node_map[node_ref]
        for child in node.children:
            self._eval(child, lookup)

        if isinstance(node, AndGate):
            lookup[node_ref] = lookup[node.left] and lookup[node.right]
        elif isinstance(node, Inverter):
            lookup[node_ref] = not lookup[node.input]
        elif isinstance(node, (Latch, Input)):
            return
        elif isinstance(node, ConstFalse):
            lookup[node_ref] = False
        else:
            raise NotImplementedError


def _to_idx(lit):
    """AAG format uses least significant bit to encode an inverter.
    The index is thus the interal literal shifted by one bit."""
    return lit >> 1


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
        literals = chain(self.inputs.values(),
                         self.outputs.values(),
                         fn.pluck(0, self.gates),
                         fn.pluck(0, self.latches.values()))
        max_idx = max(map(_to_idx, literals))
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
            out += '\n'.join(f"i{idx} {name}"
                             for idx, name in enumerate(input_names)) + '\n'
        if self.outputs:
            out += '\n'.join(f"o{idx} {name}"
                             for idx, name in enumerate(output_names)) + '\n'

        if self.latches:
            out += '\n'.join(f"l{idx} {name}"
                             for idx, name in enumerate(latch_names)) + '\n'

        if self.comments:
            out += 'c\n' + '\n'.join(self.comments)
            if out[-1] != '\n':
                out += '\n'
        return out

    def _to_aig(self):
        lits = set(
            chain(
                self.inputs.values(),
                self.outputs.values(),
                chain(*self.gates),
                fn.pluck(0, self.latches.values()),
                fn.pluck(1, self.latches.values()),
            ))

        lookup = {lit: str(uuid1()) for lit in lits}
        input_map = fn.walk_values(lookup.get, self.inputs)
        output_map = fn.walk_values(lookup.get, self.outputs)
        latch_map = fn.walk_values(lambda x: lookup[x[0]], self.latches)
        node_map = fn.merge(
            {ref: Input()
             for ref in input_map.values()}, {
                 lookup[out]: AndGate(lookup[left], lookup[right])
                 for out, left, right in self.gates
             }, {
                 lookup[lit]: Latch(lookup[lit2], bool(init))
                 for lit, lit2, init in self.latches.values()
             }, {
                 lookup[lit]: Inverter(lookup[lit & -2])
                 for lit in lits if lit & 1
             })
        return AIG(
            input_map=input_map,
            output_map=output_map,
            latch_map=latch_map,
            node_map=node_map,
            comments=self.comments)


def par_compose(aig1, aig2, check_precondition=True):
    if check_precondition:
        assert not (aig1.latches & aig2.latches)
        assert not (aig1.outputs & aig2.outputs)

    interface = aig1.inputs & aig2.inputs
    if interface:  # Need to split wire.
        renames1 = {i: str(uuid1()) for i in interface}
        renames2 = {i: str(uuid1()) for i in interface}
        aig1 = aig1['i', renames1]
        aig2 = aig2['i', renames2]

    res = AIG(
        input_map=aig1.input_map.update(aig2.input_map),
        output_map=aig1.output_map.update(aig2.output_map),
        latch_map=aig1.latch_map.update(aig2.latch_map),
        node_map=aig1.node_map.update(aig2.node_map),
        comments=aig1.comments + aig2.comments)

    if interface:
        split = common.tee({i: (renames1[i], renames2[i]) for i in interface})
        res = split >> res

    return res


def seq_compose(aig1, aig2, check_precondition=True):
    # TODO: apply simple optimizations such as unit propogation and
    # excluded middle.

    interface = aig1.outputs & aig2.inputs
    if check_precondition:
        assert not (aig1.outputs - interface) & aig2.outputs
        assert not aig1.latches & aig2.latches

    # Update aig2's references to link up with aig1.
    new_refs = {aig2.input_map[i]: aig1.output_map[i] for i in interface}
    node_map2 = _omit(aig2.node_map, set(new_refs.keys()))

    def lookup(ref):
        return new_refs.get(ref, ref)

    def _update_node_ref(node):
        if isinstance(node, AndGate):
            return AndGate(lookup(node.left), lookup(node.right))
        elif isinstance(node, (Latch, Inverter)):
            return node._replace(input=lookup(node.input))
        return node

    node_map2 = _walk_values(_update_node_ref, node_map2)
    latch_map2 = _walk_values(lookup, aig2.latch_map)
    input_map2 = _walk_values(lookup, _omit(aig2.input_map, interface))
    output_map2 = pmap(_walk_values(lookup, aig2.output_map))
    return AIG(
        input_map=aig1.input_map.update(input_map2),
        latch_map=aig1.latch_map.update(latch_map2),
        node_map=aig1.node_map.update(node_map2),
        output_map=output_map2.update(_omit(aig1.output_map, interface)),
        comments=aig1.comments + aig2.comments)
