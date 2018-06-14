from math import exp, log

import click
from toposort import toposort_flatten as toposort

import aiger


try:
    from dd.cudd import BDD
except ImportError:
    from dd.autoref import BDD


def to_bdd(aag, output):
    assert len(aag.outputs) == 1 or (output is not None)
    assert len(aag.latches) == 0

    if output is None:
        output = list(aag.output_map.values())[0]
    else:
        output = aag.output_map[output]  # Use uuid reference instead.

    bdd = BDD()
    input_refs_to_var = {
        ref: f'x{i}' for i, ref in enumerate(aag.input_map.values())
    }
    bdd.declare(*input_refs_to_var.values())

    deps = {k: v.children for k, v in aag.node_map.items()}
    gate_nodes = {}
    for ref in toposort(deps):
        gate = aag.node_map[ref]

        if isinstance(gate, aiger.aig.ConstFalse):
            gate_nodes[ref] = bdd.add_expr('False')
        elif isinstance(gate, aiger.aig.Inverter):
            gate_nodes[ref] = ~gate_nodes[gate.input]
        elif isinstance(gate, aiger.aig.Input):
            gate_nodes[ref] = bdd.add_expr(input_refs_to_var[ref])
        elif isinstance(gate, aiger.aig.AndGate):
            gate_nodes[ref] = gate_nodes[gate.left] & gate_nodes[gate.right]

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
