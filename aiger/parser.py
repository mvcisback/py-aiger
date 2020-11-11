import io
import re
import operator as op
from collections import defaultdict
from functools import reduce
from typing import Mapping, List, Optional

import attr
import funcy as fn
from bidict import bidict
from toposort import toposort_flatten as toposort
from uuid import uuid1
from sortedcontainers import SortedSet, SortedDict

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


NOT_DONE_PARSING_ERROR = "Lines exhausted before parsing ended!\n{}"
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
    inputs: SortedSet = attr.ib(factory=SortedSet)
    outputs: SortedSet = attr.ib(factory=SortedSet)
    latches: SortedSet = attr.ib(factory=SortedSet)
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


IO_PATTERN = re.compile(r"(\d+)\n")


def parse_input(state, line) -> bool:
    match = IO_PATTERN.match(line)
    if match is None or state.remaining_inputs <= 0:
        return False
    lit = int(line)
    state.inputs.add(lit)
    state.nodes[lit] = set()
    return True


def parse_output(state, line) -> bool:
    match = IO_PATTERN.match(line)
    if match is None or state.remaining_outputs <= 0:
        return False

    lit = int(line)
    state.outputs.add(lit)
    if lit & 1:
        state.nodes[lit] = {lit ^ 1}
    return True


LATCH_PATTERN = re.compile(r"(\d+) (\d+)(?: (\d+))?\n")


def parse_latch(state, line) -> bool:
    if state.remaining_latches <= 0:
        return False

    match = LATCH_PATTERN.match(line)
    if match is None:
        return False
    elems = fn.lmap(int, match.groups())

    if elems[2] is None:
        elems = elems[:2] + (0,)

    state.latches.add(Latch(*elems))
    state.nodes[elems[0]] = set()  # Add latch in as source.
    lit = elems[1]  # Add latchout to dependency graph.
    if lit & 1:
        state.nodes[lit] = {lit ^ 1}
    return True


AND_PATTERN = re.compile(r"(\d+) (\d+) (\d+)\n")


def parse_and(state, line) -> bool:
    if state.header.num_ands <= 0:
        return False

    match = AND_PATTERN.match(line)
    if match is None:
        return False

    elems = fn.lmap(int, match.groups())
    state.header.num_ands -= 1
    deps = SortedSet(elems[1:])
    state.nodes[elems[0]] = deps
    for dep in deps:
        if dep & 1:
            state.nodes[dep] = {dep ^ 1}
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

    assert state.header.num_ands == 0
    assert state.remaining_inputs == 0
    assert state.remaining_outputs == 0
    assert state.remaining_latches == 0

    # Complete Symbol Table.
    inputs = bidict(finish_table(state.symbols.inputs, state.inputs))
    outputs = finish_table(state.symbols.outputs, state.outputs)
    latches = bidict(finish_table(state.symbols.latches, state.latches))

    # Create expression DAG.
    latch_lits = {latch.id for latch in state.latches}
    lit2expr = {}
    for lit in toposort(state.nodes):
        if lit in state.inputs:
            lit2expr[lit] = A.atom(inputs.inv[lit])
        elif lit in latch_lits:
            lit2expr[lit] = A.atom(None)
        elif lit & 1:
            lit2expr[lit] = ~lit2expr[lit & -2]
        else:
            exprs = (lit2expr[lit2] for lit2 in state.nodes[lit])
            lit2expr[lit] = reduce(op.and_, exprs)

    circ = A.sink(set(inputs))  # Make sure circ depends on all inputs.

    for name, lit in outputs.items():  # Add output cones.
        lit2expr[lit] = lit2expr[lit].with_output(name)
        circ |= lit2expr[lit].aig

    for latch in state.latches:  # Mark latch outs cones.
        if latch.input not in state.outputs:
            circ |= lit2expr[latch.input].aig

    # Make DAG cyclic via latch feedback.
    wires = []
    for latch in state.latches:
        wires.append({
            'input': fn.first(lit2expr[latch.id].inputs),
            'output': lit2expr[latch.input].output,
            'keep_output': True,
            'init': latch.init,
            'latch': latches.inv[latch]
        })
    circ = circ.loopback(*wires)
    circ >>= A.sink(circ.outputs - set(outputs))  # Remove unused outputs.

    comments = tuple(state.comments if state.comments else [])
    return attr.evolve(circ, comments=comments)


def load(path: str, to_aig: bool = True):
    with open(path, 'r') as f:
        return parse(''.join(f.readlines()), to_aig=to_aig)


__all__ = ['load', 'parse']
