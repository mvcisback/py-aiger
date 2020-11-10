import io
from collections import defaultdict
from typing import Optional, Tuple, List, NamedTuple, Mapping
from uuid import uuid1

import attr
import funcy as fn
from bidict import bidict

import aiger as A
import aiger.parser as P


NOT_DONE_PARSING_ERROR = "Lines exhausted before parsing ended!\n{}"
DONE_PARSING_ERROR = "Lines exhausted before parsing ended!\n{}"


@attr.s(auto_attribs=True, frozen=True)
class Header:
    max_var_index: int
    num_inputs: int
    num_latches: int
    num_outputs: int
    num_ands: int


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
    inputs: Mapping[int, str] = defaultdict(fresh)
    outputs: Mapping[int, str] = defaultdict(fresh)
    latches: Mapping[int, str] = defaultdict(fresh)


@attr.s(auto_attribs=True)
class State:
    header: Optional[Header] = None
    inputs: List[int] = attr.ib(factory=list)
    outputs: List[int] = attr.ib(factory=list)
    ands: List[And] = attr.ib(factory=list)
    latches: List[Latch] = attr.ib(factory=list)
    symbols: SymbolTable = attr.ib(factory=SymbolTable)
    comments: Optional[List[str]] = None


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
    if not rest:
        state.inputs.append(int(line))
        return True
    return False


def parse_output(state, line) -> bool:
    idx, *rest = line.split()
    if not rest:
        state.outputs.append(int(line))
        return True
    return False


def parse_latch(state, line) -> bool:
    elems = line.split()
    
    if not (2 <= len(elems) <= 3):
        return False

    elems = fn.lmap(int, elems)
    if len(elems) == 2:
        elems.append(0)
    assert len(elems) == 3
    state.latches.append(Latch(id=elems[0], input=elems[1], init=elems[2]))
    return True


def parse_and(state, line) -> bool:
    elems = line.split()
    
    if len(elems) != 3:
        return False

    elems = fn.lmap(int, elems)
    state.ands.append(And(id=elems[0], left=elems[1], right=elems[2]))
    return True


def parse_symbol(state, line) -> bool:
    elems = line.split()

    if len(elems) != 2:
        return False

    kind_idx, name = elems

    if len(kind_idx) <= 1:
        raise ValueError("Expected symbol id {i,l,o}{id}." f"Got {kind_idx}")
    if kind_idx[0] not in {'i', 'l', 'o'}:
        raise ValueError("Symbol must start with {i,l,o}{id}.")

    kind, idx = kind_idx[0], kind_idx[1:]
    
    if kind == 'i':
        table = state.symbols.inputs
    elif kind == 'o':
        table = state.symbols.outputs
    else:
        table = state.symbols.latches
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


def parse(lines):
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

    assert len(state.ands) == state.header.num_ands
    assert len(state.inputs) == state.header.num_inputs
    assert len(state.outputs) == state.header.num_outputs
    assert len(state.latches) == state.header.num_latches

    return P.AAG(
        inputs=finish_table(state.symbols.inputs, state.inputs),
        outputs=finish_table(state.symbols.outputs, state.outputs),
        latches=finish_table(state.symbols.latches, state.latches),
        gates=state.ands,
        comments=tuple(state.comments),
    )
