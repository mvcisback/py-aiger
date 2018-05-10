"""Auxiliary functions for working with bitvectors that require ABC to be installed."""

import parser
import bv

import tempfile
from subprocess import call, PIPE
import re


def simplify(expr):
    f = tempfile.NamedTemporaryFile()
    f.write(str(expr).encode())
    f.seek(0)

    call(["cp", f.name, f.name + ".aag"])
    call(["aigtoaig", f.name + ".aag", f.name + ".aig"])
    call(["abc", "-c", "read {}; print_stats; dc2; dc2; dc2; print_stats; write {}".format(f.name + ".aig", f.name + ".aig")], stdout=PIPE)
    call(["aigtoaig", f.name + ".aig", f.name + ".aag"])

    simplified = open(f.name + ".aag")
    simp_aig_string = simplified.read()
    simplified.close()
    f.close()
    aig = parser.parse(simp_aig_string)
    aig.comments.pop()  # remove ABC's comments
    return bv.BV(expr.size, (expr.variables, aig))


def _bit_value(expr, name):
    signal = expr.aig.outputs[name]
    if signal not in [True, False]:  # relying on the fact that True and False are represented as 1 and 0
        raise ValueError('Value of {} not constant'.format(name))
    return signal


def unsigned_value(expr):
    """Assumes that the expression is a BV"""
    expr = simplify(expr)
    value = 0
    for i in range(expr.size):
        value += 2**i * _bit_value(expr, expr.name(i))
    return value


def value(expr):
    """Assumes that the expression is an AIG that has a single output word"""
    expr = simplify(expr)
    sign_bit_name = expr.name() + '[{}]'.format(str(expr.size - 1))
    if not _bit_value(expr, sign_bit_name):
        return unsigned_value(expr)  # positive number
    else:
        return - unsigned_value(-expr)


