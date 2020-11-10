import io
import re
from collections import defaultdict
from itertools import chain
from typing import NamedTuple, Mapping, List, Tuple, Sequence, Optional

import attr
import funcy as fn
from bidict import bidict
from toposort import toposort
from uuid import uuid1

from aiger import aig


def _to_idx(lit):
    """AAG format uses least significant bit to encode an inverter.
    The index is thus the interal literal shifted by one bit."""
    return lit >> 1


def _polarity(i):
    return aig.Inverter if i & 1 == 1 else lambda x: x


class AAG(NamedTuple):
    inputs: Mapping[str, int]
    latches: Mapping[str, Tuple[int, int, int]]
    outputs: Mapping[str, int]
    gates: List[Tuple[int, int, int]]
    comments: Sequence[str]

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

        out = str(self.header) + '\n'
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
            {0: aig.ConstFalse()},
            {_to_idx(l): aig.Input(n) for n, l in self.inputs.items()},
            {
                _to_idx(l): aig.LatchIn(n)
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
            lookup[_to_idx(gate)] = aig.AndGate(*sources)

        latch_dependencies = {
            i: (n, dep) for n, (i, dep, _) in self.latches.items()
        }
        for gate in fn.cat(latch_order):
            assert _to_idx(gate) in lookup
            if not isinstance(lookup[_to_idx(gate)], aig.LatchIn):
                continue

            name, dep = latch_dependencies[gate]
            source = _polarity(dep)(lookup[_to_idx(dep)])
            latches.add((name, source))

        def get_output(v):
            idx = _to_idx(v)
            return _polarity(v)(lookup[idx])

        top_level = ((k, get_output(v)) for k, v in self.outputs.items())
        return aig.AIG(
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


def aig2aag(circ) -> AAG:
    aag, max_idx, lit_map = _to_aag(
        circ.cones | circ.latch_cones,
        AAG({}, {}, {}, [], circ.comments),
    )

    # Check that all inputs have a lit.
    for name in filter(lambda x: x not in aag.inputs, circ.inputs):
        aag.inputs[name] = lit_map[name] = 2 * max_idx
        max_idx += 1

    # Update cone maps.
    aag.outputs.update(
        {k: lit_map[cone] for k, cone in circ.node_map.items()}
    )
    latch2init = dict(circ.latch2init)
    for name, cone in circ.latch_map.items():
        latch = aig.LatchIn(name)
        if latch not in lit_map:
            lit = lit_map[latch] = 2 * max_idx
            max_idx += 1
        else:
            lit = lit_map[latch]

        init = int(latch2init[name])
        ilit = lit_map[cone]
        aag.latches[name] = lit, ilit, init

    return aag


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

        if isinstance(gate, aig.Inverter):
            input_lit = lit_map[gate.input]
            lit_map[gate] = (input_lit & -2) | (1 ^ (input_lit & 1))
            continue
        elif isinstance(gate, aig.ConstFalse):
            lit_map[gate] = 0
            continue

        # Must be And, Latch, or Input
        lit_map[gate] = 2 * max_idx
        max_idx += 1
        if isinstance(gate, aig.AndGate):
            encoded = tuple(map(lit_map.get, (gate, gate.left, gate.right)))
            aag.gates.append(encoded)

        elif isinstance(gate, aig.Input):
            aag.inputs[gate.name] = lit_map[gate]
        else:
            assert isinstance(gate, aig.LatchIn)

    return aag, max_idx, lit_map


NOT_DONE_PARSING_ERROR = "Lines exhausted before parsing ended!\n{}"
DONE_PARSING_ERROR = "Lines exhausted before parsing ended!\n{}"


@attr.s(auto_attribs=True, repr=False)
class Header:
    max_var_index: int
    num_inputs: int
    num_latches: int
    num_outputs: int
    num_ands: int

    def __repr__(self):
        return f"aag {self.max_var_index} {self.num_inputs} " \
            f"{self.num_latches} {self.num_outputs} {self.num_ands}"


class Latch(NamedTuple):
    id: int
    input: int
    init: bool


class And(NamedTuple):
    id: int
    left: int
    right: int


@attr.s(auto_attribs=True, frozen=True)
class Symbol:
    kind: str
    name: str
    index: int


def fresh():
    return str(uuid1())


@attr.s(auto_attribs=True, frozen=True)
class SymbolTable:
    inputs: Mapping[int, str] = attr.ib(factory=lambda: defaultdict(fresh))
    outputs: Mapping[int, str] = attr.ib(factory=lambda: defaultdict(fresh))
    latches: Mapping[int, str] = attr.ib(factory=lambda: defaultdict(fresh))


@attr.s(auto_attribs=True)
class State:
    header: Optional[Header] = None
    inputs: List[int] = attr.ib(factory=list)
    outputs: List[int] = attr.ib(factory=list)
    ands: List[And] = attr.ib(factory=list)
    latches: List[Latch] = attr.ib(factory=list)
    symbols: SymbolTable = attr.ib(factory=SymbolTable)
    comments: Optional[List[str]] = None

    @property
    def remaining_ands(self):
        return self.header.num_ands - len(self.ands)

    @property
    def remaining_latches(self):
        return self.header.num_latches - len(self.latches)

    @property
    def remaining_outputs(self):
        return self.header.num_outputs - len(self.outputs)

    @property
    def remaining_inputs(self):
        return self.header.num_inputs - len(self.inputs)


def parse_header(state, lines) -> bool:
    if state.header is not None:
        return False

    try:
        kind, *ids = lines.split()
        ids = fn.lmap(int, ids)

        if any(x < 0 for x in ids):
            raise ValueError("Indicies must be positive!")

        max_idx, nin, nlatch, nout, nand = ids
        if nin + nlatch + nand > max_idx:
            raise ValueError("Sum of claimed indices greater than max.")

        state.header = Header(
            max_var_index=max_idx,
            num_inputs=nin,
            num_latches=nlatch,
            num_outputs=nout,
            num_ands=nand,
        )

    except ValueError as err:
        raise ValueError(f"Failed parsing the header: {err}")
    return True


def parse_input(state, line) -> bool:
    idx, *rest = line.split()
    if rest or state.remaining_inputs <= 0:
        return False
    state.inputs.append(int(line))
    return True


def parse_output(state, line) -> bool:
    idx, *rest = line.split()
    if rest or state.remaining_outputs <= 0:
        return False
    state.outputs.append(int(line))
    return True


LATCH_PATTERN = re.compile(r"(\d+) (\d+)(?: (\d+))?\n")


def parse_latch(state, line) -> bool:
    if state.remaining_latches <= 0:
        return False

    match = LATCH_PATTERN.match(line)
    if match is None:
        return False
    elems = match.groups()

    if elems[2] is None:
        elems = elems[:2] + (0,)

    state.latches.append(Latch(*map(int, elems)))
    return True


AND_PATTERN = re.compile(r"(\d+) (\d+) (\d+)\n")


def parse_and(state, line) -> bool:
    if state.header.num_ands <= 0:
        return False

    match = AND_PATTERN.match(line)
    if match is None:
        return False

    state.ands.append(And(*map(int, match.groups())))
    return True


SYM_PATTERN = re.compile(r"([ilo])(\d+) (.*)\n")


def parse_symbol(state, line) -> bool:
    match = SYM_PATTERN.match(line)
    if match is None:
        return False

    kind, idx, name = match.groups()

    table = {
        'i': state.symbols.inputs,
        'o': state.symbols.outputs,
        'l': state.symbols.latches
    }.get(kind)
    table[int(idx)] = name
    return True


def parse_comment(state, line) -> bool:
    if state.comments is not None:
        state.comments.append(line.rstrip())
    elif line.rstrip() == 'c':
        state.comments = []
    else:
        raise ValueError("Expected 'c' to start of comment section.")
    return True


def parse_seq():
    yield parse_header
    yield parse_input
    yield parse_latch
    yield parse_output
    yield parse_and
    yield parse_symbol
    yield parse_comment


def finish_table(table, keys):
    assert len(table) <= len(keys)
    return bidict({table[i]: key for i, key in enumerate(keys)})


def parse(lines, to_aig: bool = True):
    if isinstance(lines, str):
        lines = io.StringIO(lines)

    state = State()
    parsers = parse_seq()
    parser = next(parsers)

    for line in lines:
        while not parser(state, line):
            parser = next(parsers)

            if parser is None:
                raise ValueError(NOT_DONE_PARSING_ERROR.format(state))

    if parser not in (parse_comment, parse_symbol):
        raise ValueError(DONE_PARSING_ERROR.format(state))

    assert state.remaining_ands == 0
    assert state.remaining_inputs == 0
    assert state.remaining_outputs == 0
    assert state.remaining_latches == 0

    aag = AAG(
        inputs=finish_table(state.symbols.inputs, state.inputs),
        outputs=finish_table(state.symbols.outputs, state.outputs),
        latches=finish_table(state.symbols.latches, state.latches),
        gates=state.ands,
        comments=tuple([] if state.comments is None else state.comments),
    )
    return aag._to_aig() if to_aig else aag


def load(path: str, to_aig: bool = True):
    with open(path, 'r') as f:
        return parse(''.join(f.readlines()), to_aig=to_aig)
