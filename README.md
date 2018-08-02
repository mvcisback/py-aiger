<figure>
  <img src="assets/logo_text.svg" alt="py-aiger logo" width=300px>
  <figcaption>pyAiger: A python library for manipulating sequential and inverter gates.</figcaption>
</figure>

[![Build Status](https://travis-ci.org/mvcisback/py-aiger.svg?branch=master)](https://travis-ci.org/mvcisback/py-aiger)
[![codecov](https://codecov.io/gh/mvcisback/py-aiger/branch/master/graph/badge.svg)](https://codecov.io/gh/mvcisback/py-aiger)
[![PyPI version shields.io](https://img.shields.io/pypi/v/py-aiger.svg)](https://pypi.python.org/pypi/py-aiger/)
[![PyPI license](https://img.shields.io/pypi/l/py-aiger.svg)](https://pypi.python.org/pypi/py-aiger/)

# Table of Contents
- [About](#about-py-aiger)
- [Installation](#installation)
- [Boolean Expr DSL](#boolean-expression-dsl)
- [Sequential Circuit DSL](#sequential-circuit-dsl)
- [Scripts](#scripts)


# About Py-Aiger

1. Q: How is Py-Aiger pronounced? A: Like "pie" + "grrr".
2. Q: Why python? Aren't you worried about performance?! A: No. The goals of this library are expressivity, ease of use, and hackability.

# Installation

`$ pip install py-aiger`

or as a developer:

`$ python setup.py develop`



# Boolean Expression DSL
While powerful, when writing combinatorial circuits, the Sequential
Circuit DSL came be somewhat clumsy. For this common usecase, we have
developed the Boolean Expression DSL. All circuits generated this way
have a single output.

```python
import aiger
x, y = aiger.atom('x'), aiger.atom('y')
expr1 = x & y  # circuit with inputs 'x', 'y' and 1 output computing x AND y.
expr2 = x | y  # logical or.
expr3 = x ^ y  # logical xor.
expr4 = x == y  # logical ==, xnor.
expr5 = x.implies(y)
expr6 = ~x  # logical negation.

# Atoms can be constants.
expr7 = x & aiger.atom(True)  # Equivilent to just x.
expr8 = x & aiger.atom(False)  # Equivilent to const False.

# And you can inspect the AIG if needed.
circ = x.aig
```


# Sequential Circuit DSL

```python
import aiger
from aiger import utils


aag1 = aiger.load(path_to_aag1_file)
aag2 = aiger.load(path_to_aag2_file)
```

## Sequential composition
```python
aag3 = aag1 >> aag2
```

## Parallel composition
```python
aig4 = aag1 | aag2
```

## Adding Feedback (inserts a delay)
```python
# Connect output y to input x with delay (initialized to True).
aig5 = aig1.feedback(
    inputs=['x'],
    outputs=['y'],
    initials=[True],
    keep_outputs=True
)
```

## Count solutions
```python
# Assume 1 output. This could be passed as an argument in the future.
print(utils.count(aag3))
```

## Relabeling
```python
# Relabel input 'x' to 'z'.
aig1['i', {'x': 'z'}]

# Relabel output 'y' to 'w'.
aig1['o', {'y': 'w'}]

# Relabel latches 'l1' to 'l2'.
aig1['o', {'l1': 'l2'}]
```

## Evaluation
```python
# Combinatoric evaluation.
aig3(inputs={'x':True, 'y':False})

# Sequential evaluation.
sim = aig3.simulate({'x': 0, 'y': 0}, 
                    {'x': 1, 'y': 2},
                    {'x': 3, 'y': 4})

# Simulation Coroutine
sim = aig3.simulator()  # Coroutine
next(sim)  # Initialize
print(sim.send({'x': 0, 'y': 0}))
print(sim.send({'x': 1, 'y': 2}))
print(sim.send({'x': 3, 'y': 4}))


# Unroll
aig4 = aig3.unroll(steps=10, init=True)
```

## Useful circuits
```python
# Fix input x to be False.
aag4 = aiger.source({'x': False}) >> aag3

# Remove output y. 
aag4 = aag3 >> aiger.sink(['y'])

# Create duplicate w of output y.
aag4 = aag3 >> aiger.tee({'y': ['y', 'w']})

# Make an AND gate.
aiger.and_gate(['x', 'y'], out='name')

# Make an OR gate.
aiger.or_gate(['x', 'y'])  # Default output name is #or_output.

# And outputs.
aig1 >> aiger.and_gate(aag1.outputs) # Default output name is #and_output.

# Or outputs.
aig1 >> aiger.or_gate(inputs=aag1.outputs, output='my_output')

# Flip outputs.
aig1 >> aiger.bit_flipper(inputs=aag1.outputs)

# Flip inputs.
aiger.bit_flipper(inputs=aag1.inputs) >> aig1

# ITE circuit
# ['o1', 'o2'] = ['i1', 'i2'] if 'test' Else ['i3', 'i4'] 
aiger.ite('test', ['i1', 'i2'], ['i3', 'i4'], outputs=['o1', 'o2'])
```


# Scripts

Installing py-aiger should install two commandline scripts:

- aigseqcompose
- aigparcompose
- aigcount

These are meant to augment the
[aiger](fmv.jku.at/aiger/aiger-1.9.9.tar.gz) library. Ideally, we
would someday like feature parity.
