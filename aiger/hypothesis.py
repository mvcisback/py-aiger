import hypothesis.strategies as st
from hypothesis_cfg import ContextFreeGrammarStrategy
from parsimonious import Grammar, NodeVisitor

from aiger import common, expr

CIRC_GRAMMAR = Grammar(u'''
phi =  and / neg / vyest / AP
and = "(" _ phi _ "&" _ phi _ ")"
neg = "~" _ phi
vyest = "Z" _ phi

_ = ~r" "*
AP = ~r"[a-zA-z]" ~r"[a-zA-Z\d]*"
EOL = "\\n"
''')


class CircVisitor(NodeVisitor):
    def generic_visit(self, _, children):
        return children

    def visit_phi(self, _, children):
        return children[0]

    def visit_AP(self, node, _):
        return expr.atom(node.text)

    def visit_and(self, _, children):
        return children[2] & children[6]

    def visit_neg(self, _, children):
        return ~children[2]

    def visit_vyest(self, _, children):
        _, _, phi = children
        aig = phi.aig >> common.delay(
            inputs=phi.aig.outputs,
            initials=[True],
            latches=[common._fresh()],
            outputs=[common._fresh()]
        )
        return expr.BoolExpr(aig)


def parse(circ_str: str):
    return CircVisitor().visit(CIRC_GRAMMAR.parse(circ_str)).aig


GRAMMAR = {
    'psi': (('(', 'psi', ' & ', 'psi', ')'), ('~ ', 'psi'), ('Z ', 'psi'),
            ('AP', )),
    'AP': (
        ('a', ),
        ('b', ),
        ('c', ),
        ('d', ),
        ('e', ),
        ('f', ),
        ('g', ),
        ('h', ),
        ('i', ),
        ('j', ),
        ('k', ),
        ('l', ),
    ),
}


def make_circuit(term):
    circ_str = ''.join(term)
    return parse(circ_str).evolve(comments=(circ_str, ))


Circuits = st.builds(make_circuit,
                     ContextFreeGrammarStrategy(
                         GRAMMAR, max_length=15, start='psi'))
