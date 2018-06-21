"""Auxiliary functions for working with bitvectors that require ABC to
be installed."""

import tempfile
from subprocess import PIPE, call

from aiger import bv
from aiger import parser


def simplify(expr):
    f = tempfile.NamedTemporaryFile()
    f.write(str(expr).encode())
    f.seek(0)

    call(["cp", f.name, f.name + ".aag"])
    call(["aigtoaig", f.name + ".aag", f.name + ".aig"], stdout=PIPE)
    call(
        [
            "abc", "-c",
            "read {}; print_stats; dc2; dc2; dc2; print_stats; write {}".
            format(f.name + ".aig", f.name + ".aig")
        ],
        stdout=PIPE
    )  # this ensures that ABC is not too verbose, but still prints errors
    simplified_filename = f.name + ".simp.aag"
    call(["aigtoaig", f.name + ".aig", simplified_filename], stdout=PIPE)

    try:
        simplified = open(simplified_filename)
        simp_aig_string = simplified.read()
        simplified.close()
    except IOError as e:
        print(f'I/O error({e.errno}): {e.strerror}')
        simp_aig_string = None
    finally:
        f.close()

    if not simp_aig_string:  # could not simplify file
        return None

    aig = parser.parse(simp_aig_string)._replace(
        # remove ABC's comments
        comments=[f'simplified'] + bv._indent(expr.aig.comments)
    )
    return bv.BV(expr.size, (expr.variables, aig), name=expr.name())


def _bit_value(expr, name):
    signal = expr.aig.outputs[name]
    # Relying on the fact that True and False are represented as 1 and 0
    if signal not in [True, False]:
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
    assert expr.size > 1  # signed values with 1 bit don't make sense
    expr = simplify(expr)
    sign_bit_name = expr.name(expr.size - 1)
    if not _bit_value(expr, sign_bit_name):
        return unsigned_value(expr)  # positive number
    else:
        return -unsigned_value(-expr)
