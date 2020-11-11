import funcy as fn

import attr
from bidict import bidict


AAG_HEADER = "aag {} {} {} {} {}\n"


def header(inputs, outputs, latchins, latchouts, inits, count):
    n_in, n_lin, n_out = len(inputs), len(latchins), len(outputs)
    n_and = count - n_in - n_lin - 1
    assert len(latchouts) == n_lin == len(inits)

    buff = AAG_HEADER.format(count, n_in, n_lin, n_out, n_and)

    if inputs:
        buff += "\n".join(map(str, inputs.values())) + "\n"
    for key in latchins:
        buff += f"{latchins[key]} {latchouts[key]} {int(inits[key])}\n"
    if outputs:
        buff += "\n".join(map(str, outputs.values())) + "\n"
    return buff


def footer(inputs, latches, outputs, comments):
    # Note: relies on the fact that dictionaries respect insertion order.
    buff = ""
    if inputs:
        buff += "\n".join(f"i{i} {k}" for i, k in enumerate(inputs)) + "\n"
    if outputs:
        buff += "\n".join(f"o{i} {k}" for i, k in enumerate(outputs)) + "\n"
    if latches:
        buff += "\n".join(f"l{i} {k}" for i, k in enumerate(latches)) + "\n"
    if comments:
        buff += "c\n"
        buff += "\n".join(comments) + "\n"
    return buff


def dump(circ):
    # Create Algebra to write to a string buffer.
    buff = ""

    @attr.s(auto_attribs=True, frozen=True, repr=False)
    class NodeAlg:
        lit: int

        def __repr__(self):
            return str(self.lit)

        @fn.memoize
        def __and__(self, other):
            nonlocal count
            nonlocal buff

            new = NodeAlg(count << 1)
            count += 1
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
    count = 1
    inputs = bidict(
        {k: NodeAlg(i << 1) for i, k in enumerate(sorted(circ.inputs), count)}
    )
    count += len(inputs)
    latches = bidict(
        {k: NodeAlg(i << 1) for i, k in enumerate(sorted(circ.latches), count)}
    )
    count += len(latches)

    # Interpret circ over Algebra.
    omap, lmap = circ(inputs=inputs, latches=latches, lift=lift)

    # Add header and footer
    head = header(inputs, omap, latches, lmap, circ.latch2init, count)
    foot = footer(inputs, latches, omap, circ.comments)

    return head + buff + foot


__all__ = ['dump']
