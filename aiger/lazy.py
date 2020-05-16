"""
Abstractions for lazy compositions/manipulations of And Inverter
Graphs.
"""

from __future__ import annotations

from typing import (Union, FrozenSet, Callable, Iterator, Tuple,
                    Mapping, Sequence, Optional)

import attr
import funcy as fn
from bidict import bidict
from pyrsistent import pmap
from pyrsistent.typing import PMap

import aiger as A
from aiger.aig import AIG, Node, Shim, Input, AndGate, LatchIn
from aiger.aig import ConstFalse, Inverter, _is_const_true


@attr.s(auto_attribs=True, frozen=True)
class NodeAlgebra:
    node: Node

    def __and__(self, other: NodeAlgebra) -> NodeAlgebra:
        if isinstance(self.node, ConstFalse):
            return self
        elif isinstance(other.node, ConstFalse):
            return other
        elif _is_const_true(self.node):
            return other
        elif _is_const_true(other.node):
            return self

        return NodeAlgebra(AndGate(self.node, other.node))

    def __invert__(self) -> NodeAlgebra:
        if isinstance(self.node, Inverter):
            return NodeAlgebra(self.node.input)
        return NodeAlgebra(Inverter(self.node))


@attr.s(frozen=True, auto_attribs=True)
class LazyAIG:
    iter_nodes: Callable[[], Iterator[Iterator[Node]]]

    inputs: FrozenSet[str] = attr.ib(default=frozenset(), converter=frozenset)
    latch2init: PMap[str, bool] = attr.ib(default=pmap(), converter=pmap)

    # Note: Unlike in aig.AIG, here Nodes **only** serve as keys.
    node_map: PMap[str, Node] = attr.ib(default=pmap(), converter=pmap)
    latch_map: PMap[str, Node] = attr.ib(default=pmap(), converter=pmap)
    comments: Sequence[str] = attr.ib(default=(), converter=tuple)

    __call__ = AIG.__call__
    relabel = AIG.relabel

    @property
    def __iter_nodes__(self) -> Callable[[], Sequence[Sequence[Node]]]:
        return self.iter_nodes

    @property
    def outputs(self) -> FrozenSet[str]:
        return frozenset(self.node_map.keys())

    @property
    def latches(self) -> FrozenSet[str]:
        return frozenset(self.latch_map.keys())

    @property
    def lazy_aig(self) -> LazyAIG:
        return self

    @property
    def aig(self) -> AIG:
        """Return's flattened AIG represented by this LazyAIG."""
        false = NodeAlgebra(ConstFalse())
        inputs = {i: Input(i) for i in self.inputs}
        latches = {i: LatchIn(i) for i in self.latches}

        def lift(obj):
            if isinstance(obj, NodeAlgebra):
                return obj
            elif isinstance(obj, bool):
                return ~false if obj else false
            assert isinstance(obj, (Input, LatchIn))
            return NodeAlgebra(obj)

        node_map, latch_map = self(inputs, latches=latches, lift=lift)
        return AIG(
            comments=self.comments,
            inputs=self.inputs,
            node_map={k: v.node for k, v in node_map.items()},
            # TODO: change when these become PMaps.
            latch_map=frozenset({(k, v.node) for k, v in latch_map.items()}),
            latch2init=frozenset(self.latch2init.items()),
        )

    def __rshift__(self, other: AIG_Like) -> LazyAIG:
        """Cascading composition. Feeds self into other."""
        other = lazy(other)
        interface = self.outputs & other.inputs
        assert not (self.outputs - interface) & other.outputs
        assert not self.latches & other.latches

        passthrough = omit(self.node_map, interface)

        def iter_nodes():
            yield from self.__iter_nodes__()

            def add_shims(node_batch):
                for node in node_batch:
                    if isinstance(node, Input) and (node.name in interface):
                        yield Shim(new=node, old=self.node_map[node.name])
                    else:
                        yield node

            yield from map(add_shims, other.__iter_nodes__())

        return LazyAIG(
            inputs=self.inputs | (other.inputs - interface),
            latch_map=self.latch_map + other.latch_map,
            latch2init=self.latch2init + other.latch2init,
            node_map=other.node_map + passthrough,
            iter_nodes=iter_nodes,
            comments=self.comments + other.comments,
        )

    def __lshift__(self, other: AIG_Like) -> LazyAIG:
        """Cascading composition. Feeds other into self."""
        return lazy(other) >> self

    def __or__(self, other: AIG_Like) -> LazyAIG:
        """Parallel composition between self and other."""
        other = lazy(other)
        assert not self.latches & other.latches
        assert not self.outputs & other.outputs

        def iter_nodes():
            seen = set()  # which inputs have already been emitted.

            def filter_seen(node_batch):
                nonlocal seen
                for node in node_batch:
                    if node in seen:
                        continue
                    elif isinstance(node, Input):
                        seen.add(node)
                    yield node

            batches = fn.chain(self.__iter_nodes__(), other.__iter_nodes__())
            yield from map(filter_seen, batches)

        return LazyAIG(
            inputs=self.inputs | other.inputs,
            latch_map=self.latch_map + other.latch_map,
            latch2init=self.latch2init + other.latch2init,
            node_map=self.node_map + other.node_map,
            iter_nodes=iter_nodes,
            comments=self.comments + other.comments,
        )

    def cutlatches(self, latches=None, renamer=None) -> Tuple[LazyAIG, Labels]:
        """Returns LazyAIG where the latches specified
        in `latches` have been converted into inputs/outputs.

        - If `latches` is `None`, then all latches are cut.
        - `renamer`: is a function from strings to strings which
           determines how to rename latches to avoid name collisions.
        """
        if latches is None:
            latches = self.latches
        assert latches <= self.latches

        if renamer is None:
            def renamer(_):
                return A.common._fresh()

        l_map = {
            n: (renamer(n), init) for (n, init) in self.latch2init.items()
            if n in latches
        }

        assert len(
            set(fn.pluck(0, l_map.values())) & (self.inputs | self.outputs)
        ) == 0

        latch_map = omit(self.latch_map, latches)
        latch2init = omit(self.latch2init, latches)

        # Rename cut latches and add to node_map and inputs.
        renamed_node_map = walk_keys(
            lambda k: l_map[k][0],
            project(self.latch_map, latches)
        )
        new_inputs = set(renamed_node_map.keys())

        assert (self.inputs & new_inputs) == set()

        inputs = self.inputs | new_inputs
        node_map = self.node_map + renamed_node_map

        def iter_nodes():
            def cut_latches(node_batch):
                for node in node_batch:
                    if isinstance(node, LatchIn) and node.name in latches:
                        node2 = Input(l_map[node.name][0])
                        yield node2
                        yield Shim(new=node, old=node2)
                    else:
                        yield node

            return map(cut_latches, self.__iter_nodes__())

        circ = LazyAIG(
            inputs=inputs, node_map=node_map, latch_map=latch_map,
            latch2init=latch2init, iter_nodes=iter_nodes,
            comments=self.comments,
        )
        return circ, l_map

    def loopback(self, *wirings) -> LazyAIG:
        """Returns result of feeding outputs specified in `*wirings` to
        inputs specified in `wirings`.

        Each positional argument (element of wirings) should have the following
        schema:

           {
              'input': str,
              'output': str,
              'latch': str,         # what to name the new latch.
              'init': bool,         # new latch's initial value.
              'keep_output': bool,  # whether output is consumed by feedback.
            }
        """
        for wire in wirings:
            wire.setdefault('latch', wire['input'])
            wire.setdefault('init', False)
            wire.setdefault('keep_output', True)

        in2wire = {w['input']: w for w in wirings}
        out2wire = {w['output']: w for w in wirings}
        latch2wire = {w['latch']: w for w in wirings}
        assert len(in2wire) == len(latch2wire) == len(wirings)
        assert (self.latches & set(latch2wire.keys())) == set()

        latch2init = {k: v['init'] for k, v in latch2wire.items()}
        latch2init = self.latch2init + latch2init

        latch_map = project(self.node_map, out2wire.keys())
        latch_map = walk_keys(lambda k: out2wire[k]['latch'], latch_map)
        latch_map = self.latch_map + latch_map

        dropped = {k for k, w in out2wire.items() if not w['keep_output']}
        node_map = omit(self.node_map, dropped)

        inputs = self.inputs - set(in2wire.keys())

        def iter_nodes():
            def latch_inputs(node_batch):
                for node in node_batch:
                    if not (isinstance(node, Input) and node.name in in2wire):
                        yield node
                    else:
                        wire = in2wire[node.name]
                        node2 = LatchIn(wire['latch'])
                        yield node2
                        yield Shim(new=node, old=node2)

            yield from map(latch_inputs, self.__iter_nodes__())

        return LazyAIG(
            inputs=inputs, node_map=node_map, iter_nodes=iter_nodes,
            latch_map=latch_map, latch2init=latch2init, comments=self.comments
        )

    def unroll(self, horizon, *, init=True, omit_latches=True,
               only_last_outputs=False) -> LazyAIG:
        """
        Returns circuit which computes the same function as
        the sequential circuit after `horizon` many inputs.

        Each input/output has `##time_{time}` appended to it to
        distinguish different time steps.
        """
        circ = lazy(self.aig)                   # Make single node_batch.

        if not omit_latches:
            assert (circ.latches & circ.outputs) == set()

        if not init:
            assert (circ.latches & circ.inputs) == set()

        inputs, node_map = set(), pmap()        # Get timed inputs/outputs.
        for t in range(horizon):
            inputs |= {f'{i}##time_{t}' for i in circ.inputs}

            if only_last_outputs and (t != horizon - 1):
                continue

            tmp = circ.node_map.items()
            if not omit_latches:
                tmp = fn.chain(tmp, circ.latch_map.items())

            node_map += {f'{k}##time_{t+1}': (t, v) for k, v in tmp}

        latch_map = dict(circ.latch_map)  # TODO: remove when latch_map: dict.
        boundary = set(circ.node_map.values())
        if not omit_latches:
            boundary |= set(latch_map.values())

        def iter_nodes():
            @fn.curry
            def timed_iter(time, node_batch):
                for node in node_batch:
                    if isinstance(node, Input):
                        node2 = Input(f"{node.name}##time_{time}")
                        yield from [node2, Shim(new=node, old=node2)]
                    elif isinstance(node, LatchIn):
                        if time > 0:
                            node2 = (time - 1, latch_map[node.name])
                        elif init:  # yield constant.
                            node2 = ConstFalse()
                            yield node2

                            if self.latch2init[node.name]:
                                node2 = Inverter(node2)
                                yield node2
                        else:  # Turn Initial Latch into input.
                            assert time == 0
                            node2 = Input(f"{node.name}##time_{0}")
                            yield node2
                        yield Shim(new=node, old=node2)
                    else:
                        yield node

                    if node in boundary:  # This is an eventual output.
                        yield Shim(new=(time, node), old=node)

            for time in range(horizon):
                yield from map(timed_iter(time), circ.__iter_nodes__())

        return LazyAIG(
            inputs=inputs, node_map=node_map, iter_nodes=iter_nodes,
            comments=circ.comments
        )

    def __getitem__(self, others):
        """Relabel inputs, outputs, or latches.

        `others` is a tuple, (kind, relabels), where

          1. kind in {'i', 'o', 'l'}
          2. relabels is a mapping from old names to new names.

        Note: The syntax is meant to resemble variable substitution
        notations, i.e., foo[x <- y] or foo[x / y].
        """
        assert isinstance(others, tuple) and len(others) == 2
        kind, relabels = others

        if kind == 'i':
            relabels_ = {v: [k] for k, v in relabels.items()}
            return lazy(A.tee(relabels_)) >> self

        def relabel(k):
            return relabels.get(k, k)

        if kind == 'o':
            node_map = walk_keys(relabel, self.node_map)
            return attr.evolve(self, node_map=node_map)

        # Latches
        assert kind == 'l'
        latch_map = walk_keys(relabel, self.latch_map)
        latch2init = walk_keys(relabel, self.latch2init)

        def iter_nodes():
            def rename_latches(node_batch):
                for node in node_batch:
                    if isinstance(node, LatchIn) and node.name in relabels:
                        node2 = LatchIn(relabel(node.name))
                        yield node2
                        yield Shim(new=node, old=node2)
                    else:
                        yield node

            return map(rename_latches, self.__iter_nodes__())

        return attr.evolve(
            self,
            latch_map=latch_map,
            latch2init=latch2init,
            iter_nodes=iter_nodes
        )


AIG_Like = Union[AIG, LazyAIG]
Labels = Mapping[str, str]


def lazy(circ: Union[AIG, LazyAIG]) -> LazyAIG:
    """Lifts AIG to a LazyAIG."""
    return LazyAIG(
        inputs=circ.inputs,
        latch_map=pmap(circ.latch_map),
        node_map=pmap(circ.node_map),
        latch2init=pmap(circ.latch2init),
        iter_nodes=circ.__iter_nodes__,
        comments=circ.comments,
    )


def walk_keys(func, mapping):
    return fn.walk_keys(func, dict(mapping))


def omit(mapping, keys):
    return fn.omit(dict(mapping), keys)


def project(mapping, keys):
    return fn.project(dict(mapping), keys)


@attr.s(frozen=True, auto_attribs=True)
class Parallel:
    left: AIG_Like
    right: AIG_Like

    aig = LazyAIG.aig

    def __call__(self, inputs, latches=None, *, lift=None):
        out_l, lmap_l = self.left(inputs, latches=latches, lift=lift)
        out_r, lmap_r = self.right(inputs, latches=latches, lift=lift)
        return fn.merge(out_l, out_r), fn.merge(lmap_l, lmap_r)

    def _merge_maps(self, key):
        map1, map2 = [pmap(getattr(c, key)) for c in (self.left, self.right)]
        return map1 + map2
    
    @property
    def latch2init(self):
        return self._merge_maps('latch2init')

    @property
    def inputs(self):
        return self.left.inputs | self.right.inputs

    @property
    def outputs(self):
        return self.left.outputs | self.right.outputs

    @property
    def latches(self):
        return frozenset(self.latch2init.keys())

    @property
    def lazy_aig(self):
        return self

    @property
    def comments(self):
        return self.left.comments + self.right.comments


@attr.s(frozen=True, auto_attribs=True)
class Wire:
    input: str
    output: str
    latch: str
    keep_output: bool = True
    init: bool = True


def convert_wirings(wirings):
    for wire in wirings:
        wire.setdefault('latch', wire['input'])

    return tuple(Wire(**w) for w in wirings)


@attr.s(frozen=True, auto_attribs=True)
class LoopBack:
    circ: AIG_Like
    wirings: Sequence[Wire] = attr.ib(converter=convert_wirings)

    aig = LazyAIG.aig

    def __call__(self, inputs, latches=None, *, lift=None):
        if latches is None:
            latches = pmap()
        latches = dict(self.latch2init + latches)  # Override initial values.

        for wire in self.wirings:
            inputs[wire.input] = latches[wire.latch]
            del latches[wire.latch]

        omap, lmap = self.circ(inputs, latches=latches, lift=lift)

        for wire in self.wirings:
            out, latch = wire.output, wire.latch
            lmap[latch] = omap[out]
            if not wire.keep_output:
                del omap[out]

        return omap, lmap

    @property
    def latch2init(self):
        latch2init = pmap(self.circ.latch2init).evolver()
        for wire in self.wirings:
            latch2init[wire.latch] = wire.init
        return latch2init.persistent()

    @property
    def inputs(self):
        return self.circ.inputs - {w.input for w in self.wirings}

    @property
    def outputs(self):
        omitted = {w.output for w in self.wirings if not w.keep_output}
        return self.circ.outputs - omitted

    @property
    def latches(self):
        return frozenset(self.latch2init.keys())

    @property
    def lazy_aig(self):
        return self

    @property
    def comments(self):
        return self.circ.comments


def convert_renamer(renamer):
    if renamer is None:
        def renamer(*_):
            return A.common._fresh()
    return fn.memoize(renamer)


@attr.s(frozen=True, auto_attribs=True)
class CutLatches:
    circ: AIG_Like
    renamer: Callable[[str], str] = attr.ib(converter=convert_renamer)
    cut: Union[FrozenSet[str]] = None

    aig = LazyAIG.aig

    def __call__(self, inputs, latches=None, *, lift=None):
        if latches is None:
            latches = pmap()
        latches = dict(self.latch2init + latches)  # Override initial values.

        for latch in self.cut_latches:
            new_name = self.renamer(latch)
            latches[latch] = inputs[new_name]
            del inputs[new_name]

        omap, lmap = self.circ(inputs, latches=latches, lift=lift)

        for latch in self.cut_latches:
            new_name = self.renamer(latch)
            omap[new_name] = lmap[latch]
            del lmap[latch]

        return omap, lmap

    @property
    def cut_latches(self):
        return self.circ.latches if (self.cut is None) else self.cut

    @property
    def latch2init(self):
        return pmap(omit(self.circ.latch2init, self.cut_latches))

    @property
    def inputs(self):
        return self.circ.inputs | set(map(self.renamer, self.cut_latches))

    @property
    def outputs(self):
        return self.circ.outputs | set(map(self.renamer, self.cut_latches))

    @property
    def latches(self):
        return frozenset(self.latch2init.keys())

    @property
    def lazy_aig(self):
        return self

    @property
    def comments(self):
        return self.circ.comments


@attr.s(frozen=True, auto_attribs=True)
class Cascading:
    left: AIG_Like
    right: AIG_Like

    aig = LazyAIG.aig

    def __call__(self, inputs, latches=None, *, lift=None):
        inputs_l = project(inputs, self.left.inputs)
        omap_l, lmap_l = self.left(inputs_l, latches=latches, lift=lift)

        inputs_r = project(inputs, self.right.inputs)
        inputs_r.update(omap_l)  # <--- Cascade setup happens here.
        omap_l = omit(omap_l, self._interface)

        omap_r, lmap_r = self.right(inputs_r, latches=latches, lift=lift)
        return fn.merge(omap_l, omap_r), fn.merge(lmap_l, lmap_r)

    def _merge_maps(self, key):
        map1, map2 = [pmap(getattr(c, key)) for c in (self.left, self.right)]
        return map1 + map2

    @property
    def latch2init(self):
        return self._merge_maps('latch2init')

    @property
    def _interface(self):
        return self.left.outputs & self.right.inputs

    @property
    def inputs(self):
        return self.left.inputs | (self.right.inputs - self._interface)

    @property
    def outputs(self):
        return self.right.outputs | (self.left.outputs - self._interface)

    @property
    def latches(self):
        return frozenset(self.latch2init.keys())

    @property
    def lazy_aig(self):
        return self

    @property
    def comments(self):
        return self.left.comments + self.right.comments


def _relabel_map(relabels, mapping):
    return pmap(walk_keys(lambda x: relabels.get(x, x), mapping))


@attr.s(frozen=True, auto_attribs=True)
class Relabeled:
    circ: AIG_Like
    input_relabels: PMap[str, str] = pmap()
    latch_relabels: PMap[str, str] = pmap()
    output_relabels: PMap[str, str] = pmap()

    aig = LazyAIG.aig

    def __call__(self, inputs, latches=None, *, lift=None):
        if latches is None:
            latches = pmap()
        latches = dict(self.latch2init + latches)  # Override initial values.
        
        new2old_i = bidict(self.input_relabels).inv
        new2old_l = bidict(self.input_relabels).inv
        inputs = _relabel_map(new2old_i, inputs)
        latches = _relabel_map(new2old_l, latches)

        omap, lmap = self.circ(inputs, latches=latches, lift=lift)
        
        omap = _relabel_map(self.output_relabels, omap)
        lmap = _relabel_map(self.latch_relabels, lmap)
        return omap, lmap

    @property
    def latch2init(self):
        return _relabel_map(self.latch_relabels, self.circ.latch2init)

    @property
    def inputs(self):
        old_inputs = self.circ.inputs
        return frozenset(self.input_relabels.get(i, i) for i in old_inputs)

    @property
    def outputs(self):
        old_output = self.circ.outputs
        return frozenset(self.output_relabels.get(i, i) for i in old_outputs)

    @property
    def latches(self):
        return frozenset(self.latch2init.keys())

    @property
    def lazy_aig(self):
        return self

    @property
    def comments(self):
        return self.circ.comments


@attr.s(frozen=True, auto_attribs=True)
class Unrolled:
    circ: AIG_Like
    horizon: int
    init: bool = True
    omit_latches: bool = True
    only_last_outputs: bool = False

    aig = LazyAIG.aig

    def __call__(self, inputs, latches=None, *, lift=None):
        circ, omit_latches, init = self.circ, self.omit_latches, self.init
        horizon, only_last_outputs = self.horizon, self.only_last_outputs
        
        if not omit_latches:
            assert (circ.latches & circ.outputs) == set()

        if not init:
            assert (circ.latches & circ.inputs) == set()

        latches = circ.latch2init if init else project(inputs, circ.inputs)
        if init:
            inputs = omit(inputs, circ.inputs)

        outputs = {}
        for time in range(horizon):
            omap, latches = circ(
                inputs={i: inputs[f'{i}##time_{time}'] for i in circ.inputs},
                latches=latches,
                lift=lift
            )

            if (not only_last_outputs) or (time + 1 == horizon):
                template = '{}' + f'##time_{time + 1}'
                outputs.update(walk_keys(template.format, omap))

                if not self.omit_latches:
                    outputs.update(walk_keys(template.format, latches))

        return outputs, latches
            

    @property
    def latch2init(self):
        return pmap()

    def __with_times(self, keys, times):
        for time in times:
            template = '{}' + f'##time_{time}'
            yield from map(template.format, keys)

    def _with_times(self, keys, times):
        return frozenset(self.__with_times(keys, times))

    @property
    def inputs(self):
        base = set() if self.init else self.circ.latches
        base |= self.circ.inputs
        return self._with_times(base, times=range(self.horizon))

    @property
    def outputs(self):
        start = horizon if self.only_last_outputs else 0
        base = set() if self.omit_latches else self.circ.latches
        base |= self.circ.outputs
        return self._with_times(base, times=range(start, self.horizon + 1))

    @property
    def latches(self):
        return frozenset(self.latch2init.keys())

    @property
    def lazy_aig(self):
        return self

    @property
    def comments(self):
        return self.circ.comments



__all__ = ['lazy', 'LazyAIG', 'Parallel', 'LoopBack', 'CutLatches', 'Cascading',
           'Relabeled', 'Unrolled']
