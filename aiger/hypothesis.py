from uuid import uuid1

import hypothesis.strategies as st
from hypothesis_cfg import ContextFreeGrammarStrategy
from parsimonious import Grammar, NodeVisitor

from aiger import common

CIRC_GRAMMAR = Grammar(u'''
phi =  and / neg / vyest / zero / AP
and = "(" _ phi _ "&" _ phi _ ")"
neg = "~" _ phi
vyest = "Z" _ phi
zero = "0"

_ = ~r" "*
AP = ~r"[a-zA-z]" ~r"[a-zA-Z\d]*"
EOL = "\\n"
''')


def atomic_pred(a):
    return common.identity([a], outputs=[str(uuid1())])


def vyesterday(a):
    return common.delay({a: True})


class CircVisitor(NodeVisitor):
    def generic_visit(self, _, children):
        return children

    def visit_phi(self, _, children):
        return children[0]

    def visit_AP(self, node, _):
        return atomic_pred(node.text)

    def visit_zero(self, node, _):
        return common.source({str(uuid1()): False})

    def visit_and(self, _, children):
        _, _, left, _, _, _, right, _, _ = children
        combined = left | right
        return combined >> common.and_gate(combined.outputs)

    def visit_neg(self, _, children):
        _, _, phi = children
        return phi >> common.bit_flipper(phi.outputs)

    def visit_vyest(self, _, children):
        _, _, phi = children
        (out, ) = phi.outputs
        return phi >> vyesterday(out)


def parse(circ_str: str):
    return CircVisitor().visit(CIRC_GRAMMAR.parse(circ_str))


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
        ('0', ),
    ),
}


def make_circuit(term):
    circ_str = ''.join(term)
    return parse(circ_str)._replace(comments=(circ_str,))


Circuits = st.builds(make_circuit,
                     ContextFreeGrammarStrategy(
                         GRAMMAR, max_length=5, start='psi'))
