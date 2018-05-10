"""Auxiliary functions for working with bitvectors that require ABC to be installed."""

import parser
import bv

import tempfile
from subprocess import call
import re


def simplify(aig):
    f = tempfile.NamedTemporaryFile()
    f.write(str(aig).encode())
    f.seek(0)

    call(["cp", f.name, f.name + ".aag"])
    call(["aigtoaig", f.name + ".aag", f.name + ".aig"])
    call(["abc", "-c", "read {}; print_stats; dc2; dc2; dc2; print_stats; write {}".format(f.name + ".aig", f.name + ".aig")])
    call(["aigtoaig", f.name + ".aig", f.name + ".aag"])

    simplified = open(f.name + ".aag")
    simp_aig_string = simplified.read()
    simplified.close()
    f.close()
    aig = parser.parse(simp_aig_string)
    aig.comments.pop()  # remove ABC's comments
    return aig


def _bit_value(expr, name):
    signal = expr.outputs[name]
    if signal not in [True, False]:  # relying on the fact that True and False are represented as 1 and 0
        raise ValueError('Value of {} not constant'.format(name))
    return signal


def _bv_base_name(expr):
    """Assumes that the expression is an AIG that has a single output word"""
    name = list(expr.outputs.keys())[0]
    m = re.search(r"(.*)\[[0-9_]+\]", name)  # match everything before the brackets '[i]'
    return m.group(1) # TODO: may fail if not matched anything


def value(wordlen, expr):
    """Assumes that the expression is an AIG that has a single output word"""
    name = _bv_base_name(expr)
    value = 0
    for i in range(len(expr.outputs)):
        value += 2**i * _bit_value(expr, name + '[{}]'.format(i))
    return value

assert value(4, bv.const(4, 6)) == 6

def signed_value(wordlen, expr):
    """Assumes that the expression is an AIG that has a single output word"""
    sign_bit_name = _bv_base_name(expr) + '[{}]'.format(str(wordlen - 1))
    if not _bit_value(expr, sign_bit_name):
        return value(expr)  # positive number
    else:
        return - value(simplify(expr >> bv.negative(4)))

minus_six = bv.const(4, 6) >> bv.negative(4)
print("Minus 6:", minus_six)
print(signed_value(4, simplify(minus_six)))

def assign(wordlen, expr, assignment={}):
    for name, value in assignment:
        expr = bv.const(wordlen) >> expr
    return expr

def eval(wordlen, expr, assignment={}):
    assign(wordlen, expr, assignment)
    return value(simplify(expr))
