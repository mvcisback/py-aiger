from uuid import uuid1

import hypothesis.strategies as st
from hypothesis_cfg import ContextFreeGrammarStrategy
from lenses import bind
from parsimonious import Grammar, NodeVisitor

import aiger
from aiger import aig


CIRC_GRAMMAR = Grammar(u'''
phi =  and / neg / vyest / AP
and = "(" _ phi _ "&" _ phi _ ")"
neg = "~" _ phi
vyest = "Z" _ phi

_ = ~r" "*
AP = ~r"[a-zA-z]" ~r"[a-zA-Z\d]*"
EOL = "\\n"
''')


def atomic_pred(a, out=None):
    if out is None:
        out = f'l{uuid1()}'

    return aig.AIG(
        inputs=frozenset([a]),
        top_level=frozenset([(out, aig.Input(a))]),
        comments=())


def vyesterday(a, out, latch_name=None):
    if latch_name is None:
        latch_name = f'l{uuid1()}'

    latch = aig.Latch(latch_name, aig.Input(a), True)
    return aig.AIG(
        inputs=frozenset([a]),
        top_level=frozenset([(out, latch)]),
        comments=())


class CircVisitor(NodeVisitor):
    def generic_visit(self, _, children):
        return children

    def visit_phi(self, _, children):
        return children[0]

    def visit_AP(self, node, _):
        return atomic_pred(node.text)

    def visit_and(self, _, children):
        _, _, left, _, _, _, right, _, _ = children
        combined = left | right
        return combined >> aig.and_gate(combined.outputs, str(uuid1()))

    def visit_neg(self, _, children):
        _, _, phi = children
        return phi >> aig.bit_flipper(phi.outputs)

    def visit_vyest(self, _, children):
        _, _, phi = children
        (out,) = phi.outputs
        return phi >> vyesterday(out, str(uuid1()))


def parse(circ_str: str):
    return CircVisitor().visit(CIRC_GRAMMAR.parse(circ_str))


GRAMMAR = {
    'psi': (('(', 'psi', ' & ', 'psi', ')'), ('~ ', 'psi'),  ('Z ', 'psi'),
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
    return bind(parse(circ_str)).comments.set([circ_str])


Circuits = st.builds(make_circuit,
                     ContextFreeGrammarStrategy(
                         GRAMMAR, max_length=15, start='psi'))
