import click

import aiger


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
