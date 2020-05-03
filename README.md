<figure>
  <img src="assets/logo_text.svg" alt="py-aiger logo" width=300px>
  <figcaption>
      pyAiger: A python library for manipulating sequential and
      combinatorial circuits.
  </figcaption>

</figure>


[![Build Status](https://cloud.drone.io/api/badges/mvcisback/py-aiger/status.svg)](https://cloud.drone.io/mvcisback/py-aiger)
[![codecov](https://codecov.io/gh/mvcisback/py-aiger/branch/master/graph/badge.svg)](https://codecov.io/gh/mvcisback/py-aiger)
[![Updates](https://pyup.io/repos/github/mvcisback/py-aiger/shield.svg)](https://pyup.io/repos/github/mvcisback/py-aiger/)

[![PyPI version](https://badge.fury.io/py/py-aiger.svg)](https://badge.fury.io/py/py-aiger)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1405781.svg)](https://doi.org/10.5281/zenodo.1405781)

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-generate-toc again -->
**Table of Contents**

- [About PyAiger](#about-pyaiger)
- [Installation](#installation)
- [Boolean Expression DSL](#boolean-expression-dsl)
- [Sequential Circuit DSL](#sequential-circuit-dsl)
    - [Sequential composition](#sequential-composition)
    - [Parallel composition](#parallel-composition)
    - [Circuits with Latches/Feedback/Delay](#circuits-with-latchesfeedbackdelay)
    - [Relabeling](#relabeling)
    - [Evaluation](#evaluation)
    - [Useful circuits](#useful-circuits)
- [Extra](#extra)
- [Ecosystem](#ecosystem)
- [Related Projects](#related-projects)
- [Citing](#citing)

<!-- markdown-toc end -->


# About PyAiger

1. Q: How is Py-Aiger pronounced? A: Like "pie" + "grrr".
2. Q: Why python? Aren't you worried about performance?! A: No. The goals of this library are ease of use and hackability. 
3. Q: No, I'm really concerned about performance! A: This library is not suited to implement logic solvers. For everything else, such as the creation and manipulation of circuits with many thousands of gates in between solver calls, the library is really fast enough.
4. Q: Where does the name come from? A: <a href="http://fmv.jku.at/aiger/">Aiger</a> is a popular circuit format. The format is used in <a href="http://fmv.jku.at/hwmcc17/">hardware model checking</a>, <a href="http://www.syntcomp.org/">synthesis</a>, and is supported by <a href="https://github.com/berkeley-abc/abc">ABC</a>. The name is a combination of AIG (standing for <a href="https://en.wikipedia.org/wiki/And-inverter_graph">And-Inverter-Graph</a>) and the mountain <a href="https://en.wikipedia.org/wiki/Eiger">Eiger</a>.

# Installation

If you just need to use `aiger`, you can just run:

`$ pip install py-aiger`

For developers, note that this project uses the
[poetry](https://poetry.eustace.io/) python package/dependency
management tool. Please familarize yourself with it and then
run:

`$ poetry install`

# Boolean Expression DSL
While powerful, when writing combinatorial circuits, the Sequential
Circuit DSL can be somewhat clumsy. For this common usecase, we have
developed the Boolean Expression DSL. All circuits generated this way
have a single output.

```python
import aiger
x, y, z = aiger.atoms('x', 'y', 'z')
expr1 = x & y  # circuit with inputs 'x', 'y' and 1 output computing x AND y.
expr2 = x | y  # logical or.
expr3 = x ^ y  # logical xor.
expr4 = x == y  # logical ==, xnor.
expr5 = x.implies(y)
expr6 = ~x  # logical negation.
expr7 = aiger.ite(x, y, z)  # if x then y else z.

# Atoms can be constants.
expr8 = x & aiger.atom(True)  # Equivalent to just x.
expr9 = x & aiger.atom(False)  # Equivalent to const False.

# Specifying output name of boolean expression.
# - Output is a long uuid otherwise.
expr10 = expr5.with_output('x_implies_y')
assert expr10.output == 'x_implies_y'

# And you can inspect the AIG if needed.
circ = x.aig

# And of course, you can get a BoolExpr from a single output aig.
expr10 = aiger.BoolExpr(circ)
```


# Sequential Circuit DSL

```python
import aiger
from aiger import utils

# Parser for ascii AIGER format.
aig1 = aiger.load(path_to_aig1_file.aag)
aig2 = aiger.load(path_to_aig2_file.aag)
```

## Sequential composition
```python
aig3 = aig1 >> aig2
```

## Parallel composition
```python
aig4 = aig1 | aig2
```

## Circuits with Latches and Delayed Feedback
Sometimes one requires some outputs to be feed back into the circuits
as time delayed inputs. This can be accomplished using the `loopback`
method. This method takes in a variable number of dictionaries
encoding how to wire an input to an output. The wiring dictionaries
with the following keys and default values:

| Key         | Default | Meaning                          |
| ----------- | ------- | -------------------------------- |
| input       |         | Input port                       |
| output      |         | Output port                      |
| latch       | input   | Latch name                       |
| init        | True    | Initial latch value              |
| keep_output | True    | Keep loopbacked output as output |


```python
# Connect output y to input x with delay, initialized to True.
# (Default initialization is False.)
aig5 = aig1.loopback({
  "input": "x", "output": "y",
})

aig6 = aig1.loopback({
  "input": "x", "output": "y",
}, {
  "input": "z", "output": "z", latch="z_latch",
  "init": False, "keep_outputs": False
})

```

## Relabeling

There are two syntaxes for relabeling. The first uses indexing
syntax in a nod to [notation often used variable substition in math](https://mathoverflow.net/questions/243084/history-of-the-notation-for-substitution).

The syntax is the relabel method, which is preferable when one wants
to be explicit, even for those not familar with `py-aiger`.

```python
# Relabel input 'x' to 'z'.
aig1['i', {'x': 'z'}]
aig1.relabel('input', {'x': 'z'})

# Relabel output 'y' to 'w'.
aig1['o', {'y': 'w'}]
aig1.relabel('output', {'y': 'w'})

# Relabel latches 'l1' to 'l2'.
aig1['l', {'l1': 'l2'}]
aig1.relabel('latch', {'l1': 'l2'})
```

## Evaluation
```python
# Combinatoric evaluation.
aig3(inputs={'x':True, 'y':False})

# Sequential evaluation.
sim = aig3.simulate([{'x': 0, 'y': 0}, 
                    {'x': 1, 'y': 2},
                    {'x': 3, 'y': 4}])

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
aig4 = aiger.source({'x': False}) >> aig3

# Remove output y. 
aig4 = aig3 >> aiger.sink(['y'])

# Create duplicate w of output y.
aig4 = aig3 >> aiger.tee({'y': ['y', 'w']})

# Make an AND gate.
aiger.and_gate(['x', 'y'], out='name')

# Make an OR gate.
aiger.or_gate(['x', 'y'])  # Default output name is #or_output.

# And outputs.
aig1 >> aiger.and_gate(aig1.outputs) # Default output name is #and_output.

# Or outputs.
aig1 >> aiger.or_gate(inputs=aig1.outputs, output='my_output')

# Flip outputs.
aig1 >> aiger.bit_flipper(inputs=aig1.outputs)

# Flip inputs.
aiger.bit_flipper(inputs=aig1.inputs) >> aig1

# ITE circuit
# ['o1', 'o2'] = ['i1', 'i2'] if 'test' Else ['i3', 'i4'] 
aiger.common.ite('test', ['i1', 'i2'], ['i3', 'i4'], outputs=['o1', 'o2'])
```

# Extra
```python
aiger.common.eval_order(aig1)  # Returns topological ordering of circuit gates.

# Convert object into an AIG from multiple formats including BoolExpr, AIG, str, and filepaths.
aiger.to_aig(aig1)
```

# Ecosystem

### Stable
- [py-aiger-bv](https://github.com/mvcisback/py-aiger-bv): Extension of pyAiger for manipulating sequential bitvector circuits.
- [py-aiger-cnf](https://github.com/mvcisback/py-aiger-cnf): BoolExpr to Object representing CNF. Mostly used for interfacing with py-aiger-sat.
- [py-aiger-past-ltl](https://github.com/mvcisback/py-aiger-past-ltl): Converts Past Linear Temporal Logic to aiger circuits.
- [py-aiger-coins](https://github.com/mvcisback/py-aiger-coins): Library for creating circuits that encode discrete distributions.
- [py-aiger-sat](https://github.com/mvcisback/py-aiger-sat): Bridge between py-aiger and py-sat.
- [py-aiger-bdd](https://github.com/mvcisback/py-aiger-bdd): Aiger <-> BDD bridge.
- [py-aiger-gridworld](https://github.com/mvcisback/py-aiger-gridworld): Create aiger circuits representing gridworlds.
- [py-aiger-dfa](https://pypi.org/project/py-aiger-dfa/): Convert between finite automata and sequential circuits.


### Underdevelopment


- [py-aiger-spectral](https://github.com/mvcisback/py-aiger-spectral): A tool for performing (Fourier) Analysis of Boolean Functions.
- [py-aiger-abc](https://pypi.org/project/py-aiger-abc/): Aiger and abc bridge.

# Related Projects
- [pyAig](https://bitbucket.org/sterin/pyaig): Another python library
  for working with AIGER circuits.


# Citing

```
@misc{pyAiger,
  author       = {Marcell Vazquez-Chanlatte},
  title        = {mvcisback/py-aiger},
  month        = aug,
  year         = 2018,
  doi          = {10.5281/zenodo.1326224},
  url          = {https://doi.org/10.5281/zenodo.1326224}
}
```
