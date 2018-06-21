from math import exp, log

import click
import funcy as fn

import aiger

try:
    from dd.cudd import BDD
except ImportError:
    from dd.autoref import BDD


def to_bdd(aag, output):
    assert len(aag.outputs) == 1 or (output is not None)
    assert len(aag.latches) == 0

    node_map = dict(aag.node_map)

    if output is None:
        output = node_map[fn.first(aag.outputs)]
    else:
        output = node_map[output]  # By name instead.

    bdd = BDD()
    input_refs_to_var = {ref: f'x{i}' for i, ref in enumerate(aag.inputs)}
    bdd.declare(*input_refs_to_var.values())

    gate_nodes = {}
    for gate in fn.cat(aag._eval_order):
        if isinstance(gate, aiger.aig.ConstFalse):
            gate_nodes[gate] = bdd.add_expr('False')
        elif isinstance(gate, aiger.aig.Inverter):
            gate_nodes[gate] = ~gate_nodes[gate.input]
        elif isinstance(gate, aiger.aig.Input):
            gate_nodes[gate] = bdd.add_expr(input_refs_to_var[gate.name])
        elif isinstance(gate, aiger.aig.AndGate):
            gate_nodes[gate] = gate_nodes[gate.left] & gate_nodes[gate.right]

    return gate_nodes[output], bdd


def count(aag, output=None):
    f, bdd = to_bdd(aag, output)

    n = len(aag.inputs)
    return f.count(n)


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--percent', is_flag=True)
def parse_and_count(path, percent):
    aag = aiger.parser.load(path)
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
    aag1, aag2 = map(aiger.parser.load, (path1, path2))

    with open(dest, 'w') as f:
        f.write((aag1 >> aag2).dump())


@click.command()
@click.argument('path1', type=click.Path(exists=True))
@click.argument('path2', type=click.Path(exists=True))
@click.argument('dest', type=click.Path(exists=False))
def parse_and_parcompose(path1, path2, dest):
    aag1, aag2 = map(aiger.parser.load, (path1, path2))

    with open(dest, 'w') as f:
        f.write((aag1 | aag2).dump())
