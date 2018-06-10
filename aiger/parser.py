from collections import namedtuple

import funcy as fn
from bidict import bidict
from parsimonious import Grammar, NodeVisitor

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
comment = (~r".")+ EOL?

_ = ~r" "+
id = ~r"\d"+
EOL = "\\n"
''')


class AAGVisitor(NodeVisitor):
    def generic_visit(self, _, children):
        return children

    def visit_id(self, node, children):
        return int(node.text)

    def visit_header(self, _, children):
        return aig.Header(*map(int, children[2::2]))

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
        return aig.AAG(
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


def parse(aag_str: str, rule: str = "aag", to_aig=True):
    aag = AAGVisitor().visit(AAG_GRAMMAR[rule].parse(aag_str))
    return aag._to_aig() if to_aig else aag


def load(path: str, rule: str = "aag"):
    with open(path, 'r') as f:
        return parse(''.join(f.readlines()))
