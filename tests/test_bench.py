from pathlib import Path

import pytest
import funcy as fn

import aiger


@pytest.mark.timeout(10)
def test_fast_call():
    """Based on https://github.com/mvcisback/py-aiger/issues/135."""

    def gates(circ):
        gg = []
        count = 0

        class NodeAlg:
            def __init__(self, lit: int):
                self.lit = lit

            @fn.memoize
            def __and__(self, other):
                nonlocal count
                nonlocal gg
                count += 1
                new = NodeAlg(count << 1)
                right, left = sorted([self.lit, other.lit])
                gg.append((new.lit, left, right))
                return new

            @fn.memoize
            def __invert__(self):
                return NodeAlg(self.lit ^ 1)

        def lift(obj) -> NodeAlg:
            if isinstance(obj, bool):
                return NodeAlg(int(obj))
            elif isinstance(obj, NodeAlg):
                return obj
            raise NotImplementedError

        start = 1
        inputs = {k: NodeAlg(i << 1) for i, k in enumerate(sorted(circ.inputs), start)}
        count += len(inputs)

        omap, _ = circ(inputs=inputs, lift=lift)

        return gg

    path = Path(__file__).parent / "bench1.aag"

    circ = aiger.to_aig(path)
    gg = gates(circ)
    assert gg is not None
