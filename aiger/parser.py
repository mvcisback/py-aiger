import re
from collections import defaultdict
from functools import reduce
from typing import Mapping, List, Optional
from uuid import uuid1

import attr
import funcy as fn
from bidict import bidict
from sortedcontainers import SortedDict
from toposort import toposort_flatten

import aiger as A


@attr.s(auto_attribs=True, repr=False)
class Header:
    binary_mode: bool
    max_var_index: int
    num_inputs: int
    num_latches: int
    num_outputs: int
    num_ands: int

    def __repr__(self):
        mode = 'aig' if self.binary_mode else 'aag'
        return f"{mode} {self.max_var_index} {self.num_inputs} " \
               f"{self.num_latches} {self.num_outputs} {self.num_ands}"


NOT_DONE_PARSING_ERROR = "Parsing rules exhausted at line {}!\n{}"
DONE_PARSING_ERROR = "Lines exhausted before parsing ended!\n{}"


@attr.s(auto_attribs=True, frozen=True)
class Latch:
    id: int
    input: int
    init: bool = attr.ib(converter=bool)


@attr.s(auto_attribs=True, frozen=True)
class And:
    lhs: int
    rhs0: int
    rhs1: int


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
    latches: List[Latch] = attr.ib(factory=list)
    ands: List[And] = attr.ib(factory=list)
    symbols: SymbolTable = attr.ib(factory=SymbolTable)
    comments: Optional[List[str]] = None
    nodes: SortedDict = attr.ib(factory=SortedDict)

    @property
    def remaining_latches(self):
        return self.header.num_latches - len(self.latches)

    @property
    def remaining_outputs(self):
        return self.header.num_outputs - len(self.outputs)

    @property
    def remaining_inputs(self):
        return self.header.num_inputs - len(self.inputs)

    @property
    def remaining_ands(self):
        return self.header.num_ands - len(self.ands)


def _consume_stream(stream, delim) -> str:
    line = bytearray()
    ch = -1
    delim = ord(delim)
    while ch != delim:
        ch = next(stream, delim)
        line.append(ch)
    return line.decode('ascii')


HEADER_PATTERN = re.compile(r"(a[ai]g) (\d+) (\d+) (\d+) (\d+) (\d+)\n")


def parse_header(state, stream) -> bool:
    if state.header is not None:
        return False

    line = _consume_stream(stream, '\n')
    match = HEADER_PATTERN.match(line)
    if not match:
        raise ValueError(f"Failed to parse aag/aig HEADER. {line}")

    try:
        binary_mode = match.group(1) == 'aig'
        ids = fn.lmap(int, match.groups()[1:])

        if any(x < 0 for x in ids):
            raise ValueError("Indices must be positive!")

        max_idx, nin, nlatch, nout, nand = ids
        if nin + nlatch + nand > max_idx:
            raise ValueError("Sum of claimed indices greater than max.")

        state.header = Header(
            binary_mode=binary_mode,
            max_var_index=max_idx,
            num_inputs=nin,
            num_latches=nlatch,
            num_outputs=nout,
            num_ands=nand,
        )

    except ValueError as err:
        raise ValueError(f"Failed parsing the header: {err}")
    return True


IO_PATTERN = re.compile(r"(\d+)\s*\n")


def _add_input(state, lit):
    state.inputs.append(lit)
    state.nodes[lit] = set()


def parse_input(state, stream) -> bool:
    if state.remaining_inputs <= 0:
        return False

    if state.header.binary_mode:
        for lit in range(2, 2 * (state.header.num_inputs + 1), 2):
            _add_input(state, lit)
        return False

    line = _consume_stream(stream, '\n')
    match = IO_PATTERN.match(line)

    if match is None:
        raise ValueError(f"Expecting an input: {line}")

    _add_input(state, int(line))
    return True


def parse_output(state, stream) -> bool:
    if state.remaining_outputs <= 0:
        return False

    line = _consume_stream(stream, '\n')
    match = IO_PATTERN.match(line)
    if match is None:
        raise ValueError(f"Expecting an output: {line}")
    lit = int(line)
    state.outputs.append(lit)
    if lit & 1:
        state.nodes[lit] = {lit ^ 1}
    return True


LATCH_PATTERN = re.compile(r"(\d+) (\d+)(?: (\d+))?\n")
LATCH_PATTERN_BINARY = re.compile(r"(\d+)(?: (\d+))?\n")


def parse_latch(state, stream) -> bool:
    if state.remaining_latches <= 0:
        return False

    line = _consume_stream(stream, '\n')

    if state.header.binary_mode:
        match = LATCH_PATTERN_BINARY.match(line)
        if match is None:
            raise ValueError(f"Expecting a latch: {line}")
        idx = state.header.num_inputs + len(state.latches) + 1
        lit = 2 * idx
        elems = (lit,) + match.groups()
    else:
        match = LATCH_PATTERN.match(line)
        if match is None:
            raise ValueError(f"Expecting a latch: {line}")
        elems = match.groups()

    if elems[2] is None:
        elems = elems[:2] + (0,)
    elems = fn.lmap(int, elems)

    state.latches.append(Latch(*elems))
    state.nodes[elems[0]] = set()  # Add latch in as source.
    lit = elems[1]  # Add latchout to dependency graph.
    if lit & 1:
        state.nodes[lit] = {lit ^ 1}
    return True


AND_PATTERN = re.compile(r"(\d+) (\d+) (\d+)\s*\n")


def _read_delta(data):
    ch = next(data)
    i = 0
    delta = 0
    while (ch & 0x80) != 0:
        if i == 5:
            raise ValueError("Invalid byte in delta encoding")
        delta |= (ch & 0x7f) << (7 * i)
        i += 1
        ch = next(data)
    if i == 5 and ch >= 8:
        raise ValueError("Invalid byte in delta encoding")

    delta |= ch << (7 * i)
    return delta


def _add_and(state, elems):
    lhs, rhs0, rhs1 = fn.lmap(int, elems)
    state.ands.append(And(lhs, rhs0, rhs1))
    deps = {rhs0, rhs1}
    state.nodes[lhs] = deps
    for dep in deps:
        if dep & 1:
            state.nodes[dep] = {dep ^ 1}


def parse_and(state, stream) -> bool:
    if state.remaining_ands <= 0:
        return False

    if state.header.binary_mode:
        idx = state.header.num_inputs + state.header.num_latches + len(state.ands) + 1
        lhs = 2 * idx
        delta = _read_delta(stream)
        if delta > lhs:
            raise ValueError(f"Invalid lhs {lhs} or delta {delta}")
        rhs0 = lhs - delta
        delta = _read_delta(stream)
        if delta > rhs0:
            raise ValueError(f"Invalid rhs0 {rhs0} or delta {delta}")
        rhs1 = rhs0 - delta
    else:
        line = _consume_stream(stream, '\n')
        match = AND_PATTERN.match(line)
        if match is None:
            raise ValueError(f"Expecting an and: {line}")
        lhs, rhs0, rhs1 = match.groups()

    _add_and(state, (lhs, rhs0, rhs1))
    return True


SYM_PATTERN = re.compile(r"([ilo])(\d+) (.*)\s*\n")


def parse_symbol(state, stream) -> bool:
    line = _consume_stream(stream, '\n')
    match = SYM_PATTERN.match(line)
    if match is None:
        # We might have consumed the 'c' starting the comments section
        if line.rstrip() == 'c':
            state.comments = []
        return False

    kind, idx, name = match.groups()

    table = {
        'i': state.symbols.inputs,
        'o': state.symbols.outputs,
        'l': state.symbols.latches
    }.get(kind)
    table[int(idx)] = name
    return True


def parse_comment(state, stream) -> bool:
    line = _consume_stream(stream, '\n')
    if state.comments is not None:
        state.comments.append(line.rstrip())
    elif line.rstrip() == 'c':
        state.comments = []
    else:
        return False
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
    return {table[i]: key for i, key in enumerate(keys)}


def parse(stream):
    if isinstance(stream, list):
        stream = ''.join(stream)
    if isinstance(stream, str):
        stream = bytes(stream, 'ascii')
    stream = iter(stream)

    state = State()
    parsers = parse_seq()
    parser = next(parsers)

    i = 0
    while stream.__length_hint__() > 0:
        i += 1
        while not parser(state, stream):
            parser = next(parsers, None)

            if parser is None:
                raise ValueError(NOT_DONE_PARSING_ERROR.format(i, state))

    done = state.remaining_ands == 0
    done |= parser in (parse_header, parse_output, parse_comment, parse_symbol)

    if not done:
        raise ValueError(DONE_PARSING_ERROR.format(state))

    assert state.remaining_ands == 0
    assert state.remaining_inputs == 0
    assert state.remaining_outputs == 0
    assert state.remaining_latches == 0

    if len(state.inputs) != len(set(state.inputs)):
        raise ValueError("Duplicated inputs detected.")

    # Complete Symbol Table.
    inputs = bidict(finish_table(state.symbols.inputs, state.inputs))
    outputs = finish_table(state.symbols.outputs, state.outputs)
    latches = bidict(finish_table(state.symbols.latches, state.latches))

    # Create expression DAG.
    latch_ids = {latch.id: name for name, latch in latches.items()}
    and_ids = {and_.lhs: and_ for and_ in state.ands}
    lit2expr = {0: A.aig.ConstFalse()}
    for lit in toposort_flatten(state.nodes):
        if lit == 0:
            continue
        elif lit in state.inputs:
            lit2expr[lit] = A.aig.Input(inputs.inv[lit])
        elif lit in latch_ids:
            name = latch_ids[lit]
            lit2expr[lit] = A.aig.LatchIn(name)
        elif lit & 1:
            lit2expr[lit] = A.aig.Inverter(lit2expr[lit & -2])
        else:
            assert lit in and_ids
            nodes = [lit2expr[lit2] for lit2 in state.nodes[lit]]
            lit2expr[lit] = reduce(A.aig.AndGate, nodes)

    return A.aig.AIG(
        inputs=set(inputs),
        node_map={n: lit2expr[lit] for n, lit in outputs.items()},
        latch_map={n: lit2expr[latch.input] for n, latch in latches.items()},
        latch2init={n: latch.init for n, latch in latches.items()},
        comments=tuple(state.comments if state.comments else []),
    )


def load(path: str):
    with open(path, 'rb') as f:
        return parse(f.read())


__all__ = ['load', 'parse']
