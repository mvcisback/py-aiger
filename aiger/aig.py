from collections import defaultdict
from itertools import chain
from functools import reduce
from typing import Tuple, FrozenSet, NamedTuple, Union, Mapping, List
from uuid import uuid1

import funcy as fn
import lenses.hooks  # TODO: remove on next lenses version release.
from lenses import bind, lens
from toposort import toposort

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


class AIG(NamedTuple):
    input_map: Mapping[str, Reference]
    output_map: Mapping[str, Reference]
    latch_map: Mapping[str, Reference]
    node_map: Mapping[Reference, Node]
    comments: Tuple[str]

    def __repr__(self):
        return repr(self._to_aag())


    """
    def __getitem__(self, others):
        if not isinstance(others, tuple):
            return super().__getitem__(others)

        kind, relabels = others
        if kind not in {'i', 'o'}:
            raise NotImplementedError

        def _relabel(n):
            return relabels.get(n, n)

        return {
            'i': lens.Fork(lens.Recur(Input).name, lens.inputs.Each()),
            'o': lens.top_level.Each()[0],
        }.get(kind).modify(_relabel)(self)
    """

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

    @property
    def _eval_order(self):
        return list(toposort(_dependency_graph(self.cones)))

    def __call__(self, inputs, latches=None):
        # TODO: Implement partial evaluation.
        if latches is None:
            latches = dict()

        latches = {l: latches.get(l.name, l.initial) for l in self.latches}
        lookup = fn.merge(inputs, latches)
        for node in fn.cat(self._eval_order):
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
            inputs=fn.merge(self.inputs, frozenset(latch_top_level)),
            top_level=self.top_level | latch_top_level,
            comments=())

    def unroll(self, horizon, *, init=True, omit_latches=True):
        # TODO:
        # - Check for name collisions.
        latches = self.latches
        aag0 = self.cutlatches({l.name for l in latches})

        def _unroll():
            prev = aag0
            for t in range(1, horizon + 1):
                tmp = prev['i', {k: f"{k}##time_{t-1}" for k in aag0.inputs}]
                yield tmp['o', {k: f"{k}##time_{t}" for k in aag0.outputs}]

        unrolled = reduce(seq_compose, _unroll())
        if init:
            latch_source = {f"{l.name}##time_0": l.initial for l in latches}
            unrolled = common.source(latch_source) >> unrolled

        if omit_latches:
            latch_names = [f"{l.name}##time_{horizon}" for l in latches]
            unrolled = unrolled >> common.sink(latch_names)

        return unrolled

    def _to_aag(self):
        # Compute ref -> lit map.
        refs = set(chain(
            self.input_map.values(),
            self.output_map.values(),
            self.latch_map.values(),
            (k for k, v in self.node_map.items() 
             if not isinstance(v, Inverter)),
        ))
        ref_to_lit = {ref: (idx + 1) << 1 for idx, ref in enumerate(refs)}

        inverters = ((k, v) for k, v in self.node_map.items()
                     if isinstance(v, Inverter))
        ref_to_lit.update(
            {ref: _invert_lit(ref_to_lit[v.input]) for ref, v in inverters}
        )

        # Convert AIG to ASCII encoding.
        gates = [
            (ref_to_lit[k], ref_to_lit[v.left], ref_to_lit[v.right]) 
            for k, v in self.node_map.items() if isinstance(v, AndGate)
        ]
        latches = {
            k: (ref_to_lit[k], ref_to_lit[v.input], int(v.initial))
            for k, v in self.latch_map.items()
        }
        return AAG(
            inputs=_walk_values(ref_to_lit.get, self.input_map),
            outputs=_walk_values(ref_to_lit.get, self.output_map),
            latches=latches,
            gates=gates,
            comments=self.comments
        )


    def write(self, path):
        with open(path, 'w') as f:
            f.write(repr(self))


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
        literals = chain(
            self.inputs.values(),
            self.outputs.values(),
            fn.pluck(0, self.gates),
            fn.pluck(0, self.latches.values())
        )
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
        lits = set(chain(
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
            {ref: Input() for ref in input_map.values()},
            {lookup[out]: AndGate(lookup[left], lookup[right]) 
             for out, left, right in self.gates},
            {lookup[out]: AndGate(lookup[left], lookup[right]) 
             for out, left, right in self.latches},
            {lookup[lit]: Inverter(lookup[lit & -2]) 
             for lit in lits if lit & 1}
        )
        return AIG(
            input_map=input_map,
            output_map=output_map,
            latch_map=latch_map,
            node_map=node_map,
            comments=self.comments
        )


    @property
    def eval_order_and_gate_lookup(self):
        deps = {a & -2: {b & -2, c & -2} for a, b, c in self.gates}
        deps.update(
            {a & -2: {b & -2}
             for _, (a, b, _) in self.latches.items()})

        lookup = {v[0] & -2: ('AND', v) for v in self.gates}
        lookup.update(
            {v[0] & -2: ('LATCH', (v, k))
             for k, v in self.latches.items()})
        return list(toposort(deps)), lookup


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

    interface = aig1.inputs & aig2.inputs
    if interface:  # Need to split wire.
        import pdb; pdb.set_trace()
        renames1 = {i: str(uuid1()) for i in interface}
        renames2 = {i: str(uuid1()) for i in interface}
        aig1 = aig1['i', renames1]
        aig2 = aig2['i', renames2]

    res = AIG(
        input_map=aig1.input_map.update(aig2.input_map),
        output_map=aig1.output_map.update(aig2.output_map),
        latch_map=aig1.latch_map.update(aig2.latch_map),
        node_map=aig1.node_map.update(aig2.node_map),
        comments=aig1.comments + aig2.comments
    )

    if interface:
        res >>= tee({i: (renames1[i], renames2[i]) for i in interface})
    return res


def _omit(mapping, keys):
    return {k: v for k, v in mapping.items() if k not in keys}


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

    def _update_node_ref(node):
        if isinstance(node, AndGate):
            return AndGate(
                new_refs.get(node.left, node.left),
                new_refs.get(node.right, node.right)
            )
        elif isinstance(node, (Latch, Inverter)):
            return node._replace(
                input=new_refs.get(node.input, node.input)
            )
        return node
    
    node_map2 = fn.walk_values(_update_node_ref, node_map2)

    return AIG(
        input_map=aig1.input_map.update(
            _omit(aig2.input_map, interface)),
        output_map=aig2.output_map.update(
            _omit(aig1.output_map, interface)),
        latch_map=aig1.latch_map.update(aig2.latch_map),
        node_map=aig1.node_map.update(node_map2),
        comments=aig1.comments + aig2.comments
    )
