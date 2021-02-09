import funcy as fn

import attr
from bidict import bidict
from sortedcontainers import SortedDict


AAG_HEADER = "aag {} {} {} {} {}\n"


def header(inputs, outputs, latchins, latchouts, inits, count):
    n_in, n_lin, n_out = len(inputs), len(latchins), len(outputs)
    n_and = count - n_in - n_lin
    assert len(latchouts) == n_lin == len(inits)

    buff = AAG_HEADER.format(count, n_in, n_lin, n_out, n_and)

    # Note: lits sorted for stable output order.
    if inputs:
        buff += "\n".join(map(str, inputs.values())) + "\n"
    for key in sorted(latchins, key=latchins.get):
        buff += f"{latchins[key]} {latchouts[key]} {int(inits[key])}\n"
    if outputs:
        buff += "\n".join(map(str, sorted(outputs.values()))) + "\n"
    return buff


def footer(inputs, latches, outputs, comments):
    # Note: relies on the fact that dictionaries respect insertion order.
    buff = ""
    for pre, elems in zip("iol", (inputs, outputs, latches)):
        if not elems:
            continue
        buff += "\n".join(f"{pre}{i} {k}" for i, k in enumerate(elems)) + "\n"
    if comments:
        buff += "c\n"
        buff += "\n".join(comments) + "\n"
    return buff


def dump(circ):
    # Create Algebra to write to a string buffer.
    buff = ""
    count = 0

    @attr.s(auto_attribs=True, frozen=True, repr=False)
    class NodeAlg:
        lit: int

        def __repr__(self):
            return str(self.lit)

        @fn.memoize
        def __and__(self, other):
            nonlocal count
            nonlocal buff

            count += 1
            new = NodeAlg(count << 1)

            right, left = sorted([self.lit, other.lit])
            buff += f"{new} {left} {right}\n"
            return new

        def __invert__(self):
            return NodeAlg(self.lit ^ 1)

    def lift(obj) -> NodeAlg:
        if isinstance(obj, bool):
            return NodeAlg(int(obj))
        elif isinstance(obj, NodeAlg):
            return obj
        raise NotImplementedError

    # Inputs and Latches are first indices.

    start = count + 1
    inputs = bidict(
        {k: NodeAlg(i << 1) for i, k in enumerate(sorted(circ.inputs), start)}
    )
    count += len(inputs)
    start = count + 1
    latches = bidict(
        {k: NodeAlg(i << 1) for i, k in enumerate(sorted(circ.latches), start)}
    )
    count += len(latches)

    # Interpret circ over Algebra.
    omap, lmap = circ(inputs=inputs, latches=latches, lift=lift)
    omap = SortedDict(omap.get, omap)  # Force outputs to be ordered.

    # Add header and footer.
    head = header(inputs, omap, latches, lmap, circ.latch2init, count)
    foot = footer(inputs, latches, omap, circ.comments)

    return head + buff + foot


__all__ = ['dump']
