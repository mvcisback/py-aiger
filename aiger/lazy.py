"""
Abstractions for lazy compositions/manipulations of And Inverter
Graphs.
"""

from __future__ import annotations

from typing import (Union, FrozenSet, Callable, Tuple,
                    Mapping, Sequence, Optional)

import attr
import funcy as fn
from bidict import bidict
from pyrsistent import pmap
from pyrsistent.typing import PMap

import aiger as A
from aiger.aig import AIG, Node, Input, LatchIn
from aiger.aig import ConstFalse


@attr.s(frozen=True, auto_attribs=True)
class LazyAIG:
    def __call__(self, inputs, latches=None, *, lift=None):
        pass

    @property
    def latch2init(self):
        pass

    @property
    def inputs(self):
        pass

    @property
    def outputs(self):
        pass

    @property
    def comments(self):
        pass

    def write(self, path):
        self.aig.write(path)

    relabel = AIG.relabel
    simulator = AIG.simulator
    simulate = AIG.simulate

    @property
    def latches(self) -> FrozenSet[str]:
        return frozenset(self.latch2init.keys())

    @property
    def lazy_aig(self) -> LazyAIG:
        return self

    @property
    def aig(self) -> AIG:
        """Return's flattened AIG represented by this LazyAIG."""
        false = ConstFalse()
        inputs = {i: Input(i) for i in self.inputs}
        latches = {i: LatchIn(i) for i in self.latches}

        def lift(obj):
            if isinstance(obj, Node):
                return obj
            assert isinstance(obj, bool)
            return ~false if obj else false

        node_map, latch_map = self(inputs, latches=latches, lift=lift)
        return AIG(
            comments=self.comments,
            inputs=self.inputs,
            node_map=node_map,
            latch_map=latch_map,
            latch2init=self.latch2init,
        )

    def __rshift__(self, other: AIG_Like) -> LazyAIG:
        """Cascading composition. Feeds self into other."""
        return Cascading(self, other)

    def __lshift__(self, other: AIG_Like) -> LazyAIG:
        """Cascading composition. Feeds other into self."""
        return lazy(other) >> self

    def __or__(self, other: AIG_Like) -> LazyAIG:
        """Parallel composition between self and other."""
        assert not self.latches & other.latches
        assert not self.outputs & other.outputs
        return Parallel(self, other)

    def cutlatches(self, latches=None, renamer=None) -> Tuple[LazyAIG, Labels]:
        """Returns LazyAIG where the latches specified
        in `latches` have been converted into inputs/outputs.

        - If `latches` is `None`, then all latches are cut.
        - `renamer`: is a function from strings to strings which
           determines how to rename latches to avoid name collisions.
        """
        lcirc = CutLatches(self, renamer=renamer, cut=latches)
        l2init = dict(self.latch2init)
        lmap = {k: (lcirc.renamer(k), l2init[k]) for k in lcirc.cut}
        return lcirc, lmap

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
        return LoopBack(self, wirings=wirings)

    def unroll(self, horizon, *, init=True, omit_latches=True,
               only_last_outputs=False) -> LazyAIG:
        """
        Returns circuit which computes the same function as
        the sequential circuit after `horizon` many inputs.

        Each input/output has `##time_{time}` appended to it to
        distinguish different time steps.
        """
        return A.Unrolled(
            self, horizon, init, omit_latches, only_last_outputs
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
        assert kind in {'i', 'o', 'l'}
        key = {
            'i': 'input_relabels',
            'l': 'latch_relabels',
            'o': 'output_relabels',
        }.get(kind)

        return A.Relabeled(self, **{key: relabels})

    def reinit(self, latch2init) -> LazyAIG:
        """Update late initial values based on mapping provided."""
        assert set(latch2init.keys()) <= self.latches
        return UpdatedLatchInits(circ=self, latch2init=latch2init)


AIG_Like = Union[AIG, LazyAIG]
Labels = Mapping[str, str]


def walk_keys(func, mapping):
    return fn.walk_keys(func, dict(mapping))


def omit(mapping, keys):
    return fn.omit(dict(mapping), keys)


def project(mapping, keys):
    return fn.project(dict(mapping), keys)


@attr.s(frozen=True, auto_attribs=True)
class Parallel(LazyAIG):
    left: AIG_Like
    right: AIG_Like

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
class LoopBack(LazyAIG):
    circ: AIG_Like
    wirings: Sequence[Wire] = attr.ib(converter=convert_wirings)

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
    def comments(self):
        return self.circ.comments


def convert_renamer(renamer):
    if renamer is None:
        def renamer(*_):
            return A.common._fresh()
    return fn.memoize(renamer)


@attr.s(frozen=True, auto_attribs=True)
class CutLatches(LazyAIG):
    circ: AIG_Like
    renamer: Callable[[str], str] = attr.ib(converter=convert_renamer)
    cut: Optional[FrozenSet[str]] = None

    def __attrs_post_init__(self):
        if self.cut is None:
            object.__setattr__(self, "cut", self.circ.latches)

    def __call__(self, inputs, latches=None, *, lift=None):
        inputs = dict(inputs)
        if latches is None:
            latches = pmap()
        latches = dict(self.latch2init + latches)  # Override initial values.

        for latch in self.cut:
            new_name = self.renamer(latch)
            latches[latch] = inputs[new_name]
            del inputs[new_name]

        omap, lmap = self.circ(inputs, latches=latches, lift=lift)

        for latch in self.cut:
            new_name = self.renamer(latch)
            omap[new_name] = lmap[latch]
            del lmap[latch]

        return omap, lmap

    @property
    def latch2init(self):
        return pmap(omit(self.circ.latch2init, self.cut))

    @property
    def inputs(self):
        return self.circ.inputs | set(map(self.renamer, self.cut))

    @property
    def outputs(self):
        return self.circ.outputs | set(map(self.renamer, self.cut))

    @property
    def comments(self):
        return self.circ.comments


@attr.s(frozen=True, auto_attribs=True)
class Cascading(LazyAIG):
    left: AIG_Like
    right: AIG_Like

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
    def comments(self):
        return self.left.comments + self.right.comments


def _relabel_map(relabels, mapping):
    return pmap(walk_keys(lambda x: relabels.get(x, x), mapping))


@attr.s(frozen=True, auto_attribs=True)
class UpdatedLatchInits(LazyAIG):
    circ: AIG_Like
    _latch2init: PMap[str, bool] = pmap()

    def __call__(self, inputs, latches=None, *, lift=None):
        if latches is None:
            latches = pmap()

        latches = dict(self.latch2init + latches)  # Override initial values.
        return self.circ(inputs, latches=latches, lift=lift)

    @property
    def latch2init(self):
        return self.circ.latch2init + self._latch2init

    @property
    def inputs(self):
        return self.circ.inputs

    @property
    def outputs(self):
        return self.circ.outputs

    @property
    def comments(self):
        return self.circ.comments


@attr.s(frozen=True, auto_attribs=True)
class Relabeled(LazyAIG):
    circ: AIG_Like
    input_relabels: PMap[str, str] = pmap()
    latch_relabels: PMap[str, str] = pmap()
    output_relabels: PMap[str, str] = pmap()

    def __call__(self, inputs, latches=None, *, lift=None):
        if latches is None:
            latches = pmap()

        latches = dict(self.latch2init + latches)  # Override initial values.

        new2old_i = bidict(self.input_relabels).inv
        new2old_l = bidict(self.latch_relabels).inv
        inputs = _relabel_map(new2old_i, inputs)
        latches = _relabel_map(new2old_l, latches)

        omap, lmap = self.circ(inputs, latches=latches, lift=lift)

        omap = _relabel_map(self.output_relabels, omap)
        lmap = _relabel_map(self.latch_relabels, lmap)
        return dict(omap), dict(lmap)

    @property
    def latch2init(self):
        return _relabel_map(self.latch_relabels, self.circ.latch2init)

    @property
    def inputs(self):
        old_inputs = self.circ.inputs
        return frozenset(self.input_relabels.get(i, i) for i in old_inputs)

    @property
    def outputs(self):
        old_outputs = self.circ.outputs
        return frozenset(self.output_relabels.get(i, i) for i in old_outputs)

    @property
    def comments(self):
        return self.circ.comments


@attr.s(frozen=True, auto_attribs=True)
class Unrolled(LazyAIG):
    circ: AIG_Like
    horizon: int
    init: bool = True
    omit_latches: bool = True
    only_last_outputs: bool = False

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

        assert set(outputs.keys()) == self.outputs

        return dict(outputs), {}

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
        start = self.horizon if self.only_last_outputs else 1
        base = set() if self.omit_latches else self.circ.latches
        base |= self.circ.outputs
        return self._with_times(base, times=range(start, self.horizon + 1))

    @property
    def comments(self):
        return self.circ.comments


@attr.s(frozen=True, auto_attribs=True)
class Lifted(LazyAIG):
    circ: AIG_Like

    def __call__(self, inputs, latches=None, *, lift=None):
        return self.circ(inputs=inputs, latches=latches, lift=lift)

    @property
    def latch2init(self):
        return self.circ.latch2init

    @property
    def inputs(self):
        return self.circ.inputs

    @property
    def outputs(self):
        return self.circ.outputs

    @property
    def comments(self):
        return self.circ.comments


def lazy(circ: Union[AIG, LazyAIG]) -> LazyAIG:
    """Lifts AIG to a LazyAIG."""
    return Lifted(circ)


__all__ = ['lazy', 'LazyAIG', 'Parallel', 'LoopBack', 'CutLatches',
           'Cascading', 'Relabeled', 'Unrolled', 'AIG_Like']
