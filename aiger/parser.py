from collections import namedtuple
from itertools import chain
from typing import NamedTuple, Mapping, List, Tuple

import funcy as fn
from bidict import bidict
from parsimonious import Grammar, NodeVisitor
from toposort import toposort

from aiger import aig

_Symbol = namedtuple('Symbol', ['kind', 'index', 'name'])
_SymbolTable = namedtuple('SymbolTable', ['inputs', 'outputs', 'latches'])

AAG_GRAMMAR = Grammar(u'''
aag = header ios latches ios gates symbols comments?
header = "aag" _ id _ id _ id _ id _ id EOL

ios = io*
io = id EOL

latch_or_gate = id _ id _ id EOL?

latches = (latch / latch_or_gate)*
latch = id _ id EOL

gates = latch_or_gate*

symbols = symbol*
symbol =  symbol_kind id _ symbol_name EOL
symbol_kind = ("i" / "o" / "l")
symbol_name = (~r".")+

comments = "c" EOL comment+
comment = (~r".")* EOL?

_ = ~r" "+
id = ~r"\\d"+
EOL = "\\n"
''')


class AAGVisitor(NodeVisitor):
    def generic_visit(self, _, children):
        return children

    def visit_id(self, node, children):
        return int(node.text)

    def visit_header(self, _, children):
        return Header(*map(int, children[2::2]))

    def visit_io(self, _, children):
        return int(children[::2][0])

    def visit_latches(self, _, children):
        return list(fn.pluck(0, children))

    def visit_latch_or_gate(self, _, children):
        return list(map(int, children[::2]))

    visit_latch = visit_latch_or_gate

    def visit_aag(self, _, children):
        header, ios1, lgs1, ios2, lgs2, symbols, comments = children
        ios, lgs = ios1 + ios2, lgs1 + lgs2
        assert len(ios) == header.num_inputs + header.num_outputs
        inputs, outputs = ios[:header.num_inputs], ios[header.num_inputs:]
        assert len(lgs) == header.num_ands + header.num_latches

        latches, gates = lgs[:header.num_latches], lgs[header.num_latches:]

        # TODO: need to allow for inputs, outputs, latches not in
        # symbol table.
        inputs = {
            symbols.inputs.inv.get(idx, f'i{idx}'): i
            for idx, i in enumerate(inputs)
        }
        outputs = {
            symbols.outputs.inv.get(idx, f'o{idx}'): i
            for idx, i in enumerate(outputs)
        }

        latches = {
            symbols.latches.inv.get(idx, f'l{idx}'): tuple(i)
            for idx, i in enumerate(latches)
        }
        latches = fn.walk_values(lambda l: (l + (0, ))[:3], latches)

        if len(comments) > 0:
            assert comments[0].startswith('c\n')
            comments[0] = comments[0][2:]
        return AAG(
            inputs=inputs,
            outputs=outputs,
            latches=fn.walk_values(tuple, latches),
            gates=fn.lmap(tuple, gates),
            comments=tuple(comments))

    def visit_symbols(self, node, children):
        children = {(k, int(i), n) for k, i, n in children}

        def to_dict(kind):
            return bidict({n: i for k, i, n in children if k == kind})

        return _SymbolTable(to_dict('i'), to_dict('o'), to_dict('l'))

    def visit_symbol(self, node, children):
        return _Symbol(children[0], int(children[1]), children[3])

    def node_text(self, node, _):
        return node.text

    visit_symbol_kind = node_text
    visit_symbol_name = node_text
    visit_comments = node_text


class Header(NamedTuple):
    max_var_index: int
    num_inputs: int
    num_latches: int
    num_outputs: int
    num_ands: int


def _to_idx(lit):
    """AAG format uses least significant bit to encode an inverter.
    The index is thus the interal literal shifted by one bit."""
    return lit >> 1


def _polarity(i):
    return aig.Inverter if i & 1 == 1 else lambda x: x


class AAG(NamedTuple):
    inputs: Mapping[str, int]
    latches: Mapping[str, Tuple[int]]
    outputs: Mapping[str, int]
    gates: List[Tuple[int]]
    comments: Tuple[str]

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

        out = f"aag " + " ".join(map(str, self.header)) + '\n'
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
    for name, cone in circ.latch_map:
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


def parse(aag_str: str, rule: str = "aag", to_aig=True):
    aag = AAGVisitor().visit(AAG_GRAMMAR[rule].parse(aag_str))
    return aag._to_aig() if to_aig else aag


def load(path: str, rule: str = "aag", to_aig=True):
    with open(path, 'r') as f:
        return parse(''.join(f.readlines()), to_aig=to_aig)
