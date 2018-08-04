
"""
This module provides basic operations on aig circuits, such as
satisfiability queries, model counting, and quantifier elimination.
"""

import aiger
from aiger.expr import BoolExpr
from pysat.formula import CNF
from pysat.solvers import Lingeling

import funcy as fn


def tseitin(e):
    assert isinstance(e, aiger.AIG) or isinstance(e, BoolExpr)
    if isinstance(e, aiger.AIG):
        aig = e
    else:  # isinstance(e, BoolExpr)
        aig = e.aig

    assert len(aig.outputs) == 1
    assert len(aig.latches) == 0

    node_map = dict(aig.node_map)

    output = node_map[fn.first(aig.outputs)]

    clauses = []
    symbol_table = {}  # maps input names to tseitin variables
    gates = {}         # maps gates to tseitin variables

    max_var = 0

    def fresh_var():
        nonlocal max_var
        max_var += 1
        return max_var

    true_var = None  # Reserved variable name for constant True

    for gate in fn.cat(aig._eval_order):
        if isinstance(gate, aiger.aig.ConstFalse):
            if true_var is None:
                true_var = fresh_var()
                clauses.append([true_var])
            gates[gate] = - true_var
        elif isinstance(gate, aiger.aig.Inverter):
            gates[gate] = - gates[gate.input]
        elif isinstance(gate, aiger.aig.Input):
            if gate.name not in symbol_table:
                symbol_table[gate.name] = fresh_var()
                gates[gate] = symbol_table[gate.name]
        elif isinstance(gate, aiger.aig.AndGate):
            gates[gate] = fresh_var()
            clauses.append([-gates[gate.left], -gates[gate.right],  gates[gate]])  # noqa
            clauses.append([ gates[gate.left],                     -gates[gate]])  # noqa
            clauses.append([                    gates[gate.right], -gates[gate]])  # noqa

    clauses.append([gates[output]])

    return clauses, symbol_table, max_var


def satisfiable(aig):
    formula = CNF()
    clauses, _, _ = tseitin(aig)
    for clause in clauses:
        formula.append(clause)
    with Lingeling(bootstrap_with=formula.clauses) as ling:
        return ling.solve()


# def count(aig, variables=None):


# def quantify(aig, quantifiers):
    # quantify(aig, [('a', [1, 2, 3]), ('e', [4, 5, 6])])
