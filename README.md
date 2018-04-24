# Installation

`$ python setup.py develop`

Note that this currently assumes dd is installed with cudd.

See https://github.com/johnyf/dd#cython-bindings for details.

TODO: make dd an optional dependency.
TODO: Fall by to python dd engine if dd not available.


# Usage

Installing py-aiger should install two commandline scripts:

- aigcompose
- aigcount

These are meant to augment the [aiger](fmv.jku.at/aiger/aiger-1.9.9.tar.gz) library. Ideally, we would like
feature parity.


Library usage

```python
from aiger import parser
from aiger import utils

aag1 = parser.load(path_to_aag1_file)
aag2 = parser.load(path_to_aag2_file)

# Sequential composition
aag3 = aag1 >> aag2

# Count solutions
# Assume 1 output. This could be passed as an argument in the future.
print(utils.count(aag3))

# Parallel composition
# TODO: not implemented
aig4 = aag1 @ aag2

# Conjunction of outputs.
# Assumes only one output each.
aig_and = aag1 & aag2

# Inverts all outputs
# TODO
aig_neg = ~aag1

# Useful compositions.
# TODO
aig1 >> aiger.and_gate(aag1.outputs)
aig1 >> aiger.or_gate(aag1.outputs)

# Relabel input x to z and output o1 to out1
# TODO
aig1[{'x': 'z', 'o1': 'out1'}]

# Evaluation
# TODO: not implemented
# Circuit has inputs x, y and latches l1 and l2
# Circuit has outputs o1 and o2
# latches default to 0 if aiger doesn't specify.
aig3(x=2, y=2)

# Partial evaluation
# TODO
aig3(x=2) # New aiger circuit

# Simulate
# TODO
sim = aig3.simulate()  # Coroutine
print(next(sim))  # Initialize
print(sim.send({'x': 0, 'y': 0}))
print(sim.send({'x': 1, 'y': 2}))
print(sim.send({'x': 3, 'y': 4}))


# Unroll
# TODO
aig4 = aig3.unroll(steps=10, init=True)
```


# TODO

- [ ] Implement features in demo.
- [ ] Document.
- [ ] Publish on pypi.
- [ ] Setup continuous integration
- [ ] Support parser full the new aiger features 1.9.3.
  - [X] Latch Initialization
  - [ ] TODO: fill out with other feaures.
- [ ] Symbolic circuits: Composition returns a function that composes using the rules defined.
