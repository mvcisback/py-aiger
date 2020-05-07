"""
Abstractions for lazy compositions/manipulations of And Inverter
Graphs.
"""

from __future__ import annotations

from typing import Union, FrozenSet, Callable, Sequence, Tuple, Mapping

import attr
import funcy as fn
from pyrsistent import pmap
from pyrsistent.typing import PMap

from aiger.aig import AIG, Node, Shim, Input


@attr.s(frozen=True, auto_attribs=True)
class LazyAIG:
    __iter_nodes__: Callable[[], Sequence[Sequence[Node]]]

    inputs: FrozenSet[str] = frozenset()
    latch2init: PMap[str, bool] = pmap()

    # Note: Unlike in aig.AIG, here Nodes **only** serve as keys.
    node_map: PMap[str, Node] = pmap()
    latch_map: FrozenSet[Tuple[str, Node]] = pmap()
    comments: Sequence[str] = ()

    __call__ = AIG.__call__

    @property
    def outputs(self) -> FrozenSet[str]:
        return frozenset(self.node_map.keys())

    @property
    def latches(self) -> FrozenSet[str]:
        return frozenset(self.latch_map.keys())

    @property
    def aig(self) -> AIG:
        """Return's flattened AIG represented by this LazyAIG."""
        raise NotImplementedError

    def __rshift__(self, other: AIG_Like) -> LazyAIG:
        """Cascading composition. Feeds self into other."""
        interface = self.outputs & other.inputs
        assert not (self.outputs - interface) & other.outputs
        assert not self.latches & other.latches

        passthrough = fn.omit(dict(self.node_map), interface)

        def iter_nodes():
            yield from self.__iter_nodes__()

            def add_shims(node_batch):
                for node in node_batch:
                    if isinstance(node, Input) and (node.name in interface):
                        name = node.name
                        yield Shim(name=name, node=self.node_map[name])
                    else:
                        yield node                

            yield from map(add_shims, other.__iter_nodes__())

        return LazyAIG(
            inputs=self.inputs | (other.inputs - interface),
            latch_map=self.latch_map + other.latch_map,
            latch2init=self.latch2init + other.latch2init,
            node_map=other.node_map + passthrough,
            iter_nodes__=iter_nodes,
            comments=self.comments + other.comments,
        )

    def __lshift__(self, other: AIG_Like) -> LazyAIG:
        """Cascading composition. Feeds other into self."""
        return lazy(other) >> self

    def __or__(self, other: AIG_Like) -> LazyAIG:
        """Parallel composition between self and other."""
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
            iter_nodes__=iter_nodes,
            comments=self.comments + other.comments,
        )

    def cutlatches(self, latches=None, renamer=None) -> Tuple[LazyAIG, Labels]:
        """Returns LazyAIG where the latches specified
        in `latches` have been converted into inputs/outputs.

        - If `latches` is `None`, then all latches are cut.
        - `renamer`: is a function from strings to strings which
           determines how to name latches to avoid name collisions.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def unroll(self, horizon, *, init=True, omit_latches=True,
               only_last_outputs=False) -> LazyAIG:
        """
        Returns circuit which computes the same function as
        the sequential circuit after `horizon` many inputs.

        Each input/output has `##time_{time}` appended to it to
        distinguish different time steps.
        """
        raise NotImplementedError

    def __getitem__(self, others):
        raise NotImplementedError

    relabel = AIG.relabel


AIG_Like = Union[AIG, LazyAIG]
Labels = Mapping[str, str]


def lazy(circ: Union[AIG, LazyAIG]) -> LazyAIG:
    """Lifts AIG to a LazyAIG."""
    return LazyAIG(
        inputs=circ.inputs,
        latch_map=pmap(circ.latch_map),
        node_map=pmap(circ.node_map),
        latch2init=pmap(circ.latch2init),
        iter_nodes__=circ.__iter_nodes__,
        comments=circ.comments,
    )


__all__ = ['lazy', 'LazyAIG']
