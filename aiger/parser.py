import funcy as fn
from bidict import bidict

from aiger.common import AAG, Header, Symbol, SymbolTable
from parsimonious import Grammar, NodeVisitor


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

        inputs = {n: inputs[i] for n, i in symbols.inputs.items()}
        outputs = {n: outputs[i] for n, i in symbols.outputs.items()}
        latches = {n: latches[i] for n, i in symbols.latches.items()}
        latches = fn.walk_values(lambda l: (l + [0])[:3], latches)

        if len(comments) > 0:
            assert comments[0].startswith('c\n')
            comments[0] = comments[0][2:]
        return AAG(header, inputs, outputs, latches, gates, comments)

    def visit_symbols(self, node, children):
        children = {(k, int(i), n) for k, i, n in children}

        def to_dict(kind):
            return bidict({n: i for k, i, n in children if k == kind})

        return SymbolTable(to_dict('i'), to_dict('o'), to_dict('l'))

    def visit_symbol(self, node, children):
        return Symbol(children[0], int(children[1]), children[3])

    def node_text(self, node, _):
        return node.text

    visit_symbol_kind = node_text
    visit_symbol_name = node_text
    visit_comments = node_text


def parse(aag_str: str, rule: str = "aag"):
    return AAGVisitor().visit(AAG_GRAMMAR[rule].parse(aag_str))


def load(path: str, rule: str = "aag"):
    with open(path, 'r') as f:
        return parse(''.join(f.readlines()))
