from itertools import chain
from math import exp, log

import click

from aiger import parser
from aiger.common import AAG

try:
    from dd.cudd import BDD
except ImportError:
    from dd.autoref import BDD


def to_bdd(aag: AAG, output):
    assert len(aag.outputs) == 1 or (output is not None)
    assert len(aag.latches) == 0

    if output is None:
        output = list(aag.outputs.keys())[0]

    eval_order, gate_lookup = aag.eval_order_and_gate_lookup

    inputs = aag.inputs.values()
    assert eval_order[0] <= set(inputs) | {0, 1}

    bdd = BDD()
    bdd.declare(*(f'x{i}' for i in inputs))
    gate_nodes = {i: bdd.add_expr(f'x{i}') for i in inputs}
    gate_nodes[0] = bdd.add_expr('False')
    gate_nodes[1] = bdd.add_expr('True')
    for gate in chain(*eval_order[1:]):
        out, i1, i2 = gate_lookup[gate]
        f1 = ~gate_nodes[i1 & -2] if i1 & 1 else gate_nodes[i1 & -2]
        f2 = ~gate_nodes[i2 & -2] if i2 & 1 else gate_nodes[i2 & -2]
        gate_nodes[out] = f1 & f2

    out = aag.outputs[output]
    return (~gate_nodes[out & -2] if out & 1 else gate_nodes[out & -2]), bdd


def count(aag, output=None):
    f, bdd = to_bdd(aag, output)

    n = aag.header.num_inputs
    return f.count(n)


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--percent', is_flag=True)
def parse_and_count(path, percent):
    aag = parser.load(path)
    num_models = count(aag)
    if percent:
        print(exp(log(num_models) - aag.header.num_inputs))
    else:
        print(num_models)


@click.command()
@click.argument('path1', type=click.Path(exists=True))
@click.argument('path2', type=click.Path(exists=True))
@click.argument('dest', type=click.Path(exists=False))
def parse_and_compose(path1, path2, dest):
    aag1, aag2 = map(parser.load, (path1, path2))

    with open(dest, 'w') as f:
        f.write((aag1 >> aag2).dump())


@click.command()
@click.argument('path1', type=click.Path(exists=True))
@click.argument('path2', type=click.Path(exists=True))
@click.argument('dest', type=click.Path(exists=False))
def parse_and_parcompose(path1, path2, dest):
    aag1, aag2 = map(parser.load, (path1, path2))

    with open(dest, 'w') as f:
        f.write((aag1 | aag2).dump())
