import io
import re
from collections import defaultdict
from functools import reduce
from typing import Mapping, List, Optional

import attr
import funcy as fn
from bidict import bidict
from toposort import toposort_flatten
from uuid import uuid1
from sortedcontainers import SortedList, SortedSet, SortedDict

import aiger as A


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


NOT_DONE_PARSING_ERROR = "Parsing rules exhausted at line {}!\n{}"
DONE_PARSING_ERROR = "Lines exhausted before parsing ended!\n{}"


@attr.s(auto_attribs=True, frozen=True)
class Latch:
    id: int
    input: int
    init: bool = attr.ib(converter=bool)


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
    inputs: List[str] = attr.ib(factory=list)
    outputs: List[str] = attr.ib(factory=list)
    latches: List[str] = attr.ib(factory=list)
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


HEADER_PATTERN = re.compile(r"aag (\d+) (\d+) (\d+) (\d+) (\d+)\n")


def parse_header(state, line) -> bool:
    if state.header is not None:
        return False

    match = HEADER_PATTERN.match(line)
    if not match:
        raise ValueError(f"Failed to parse aag HEADER. {line}")

    try:
        ids = fn.lmap(int, match.groups())

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


IO_PATTERN = re.compile(r"(\d+)\s*\n")


def parse_input(state, line) -> bool:
    match = IO_PATTERN.match(line)

    if match is None or state.remaining_inputs <= 0:
        return False
    lit = int(line)
    state.inputs.append(lit)
    state.nodes[lit] = set()
    return True


def parse_output(state, line) -> bool:
    match = IO_PATTERN.match(line)
    if match is None or state.remaining_outputs <= 0:
        return False
    lit = int(line)
    state.outputs.append(lit)
    if lit & 1:
        state.nodes[lit] = {lit ^ 1}
    return True


LATCH_PATTERN = re.compile(r"(\d+) (\d+)(?: (\d+))?\n")


def parse_latch(state, line) -> bool:
    if state.remaining_latches <= 0:
        return False

    match = LATCH_PATTERN.match(line)
    if match is None:
        raise ValueError("Expecting a latch: {line}")

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


def parse_and(state, line) -> bool:
    if state.header.num_ands <= 0:
        return False

    match = AND_PATTERN.match(line)
    if match is None:
        return False

    elems = fn.lmap(int, match.groups())
    state.header.num_ands -= 1
    deps = set(elems[1:])
    state.nodes[elems[0]] = deps
    for dep in deps:
        if dep & 1:
            state.nodes[dep] = {dep ^ 1}
    return True


SYM_PATTERN = re.compile(r"([ilo])(\d+) (.*)\s*\n")


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


def parse(lines, to_aig: bool = True):
    if isinstance(lines, str):
        lines = io.StringIO(lines)

    state = State()
    parsers = parse_seq()
    parser = next(parsers)

    for i, line in enumerate(lines):
        while not parser(state, line):
            parser = next(parsers, None)

            if parser is None:
                raise ValueError(NOT_DONE_PARSING_ERROR.format(i + 1, state))

    if parser not in (parse_header, parse_output, parse_comment, parse_symbol):
        raise ValueError(DONE_PARSING_ERROR.format(state))

    assert state.header.num_ands == 0
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
            nodes = [lit2expr[lit2] for lit2 in state.nodes[lit]]
            lit2expr[lit] = reduce(A.aig.AndGate, nodes)

    return A.aig.AIG(
        inputs=set(inputs),
        node_map={n: lit2expr[lit] for n, lit in outputs.items()},
        latch_map={n: lit2expr[latch.input] for n, latch in latches.items()},
        latch2init={n: latch.init for n, latch in latches.items()},
        comments=tuple(state.comments if state.comments else []),
    )


def load(path: str, to_aig: bool = True):
    with open(path, 'r') as f:
        return parse(''.join(f.readlines()), to_aig=to_aig)


__all__ = ['load', 'parse']
