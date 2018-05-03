# Installation

`$ python setup.py develop`

Note that this currently assumes dd is installed with cudd.

See https://github.com/johnyf/dd#cython-bindings for details.

TODO: make dd an optional dependency.
TODO: Fall by to python dd engine if dd not available.


# Usage

Installing py-aiger should install two commandline scripts:

- aigseqcompose
- aigparcompose
- aigcount

These are meant to augment the
[aiger](fmv.jku.at/aiger/aiger-1.9.9.tar.gz) library. Ideally, we
would someday like feature parity.


## Implemented API

```python
from aiger import common
from aiger import parser
from aiger import utils

aag1 = parser.load(path_to_aag1_file)
aag2 = parser.load(path_to_aag2_file)

# Sequential composition
aag3 = aag1 >> aag2

# Parallel composition
aig4 = aag1 | aag2

# Evaluation
aig3(inputs={x:True, y:False})

# Count solutions
# Assume 1 output. This could be passed as an argument in the future.
print(utils.count(aag3))

# Relabel input 'x' to 'z'.
aig1['i', {'x': 'z'}]

# Relabel output 'y' to 'w'.
aig1['o', {'y': 'w'}]

# Relabel latches 'l1' to 'l2'.
aig1['o', {'l1': 'l2'}]

# Simulator
sim = aig3.simulator()  # Coroutine
next(sim)  # Initialize
print(sim.send({'x': 0, 'y': 0}))
print(sim.send({'x': 1, 'y': 2}))
print(sim.send({'x': 3, 'y': 4}))

# Simulate
sim = aig3.simulate({'x': 0, 'y': 0}, 
                    {'x': 1, 'y': 2},
                    {'x': 3, 'y': 4})

# Unroll
aig4 = aig3.unroll(steps=10, init=True)

# Fix input.
aag4 = common.source({'x': False}) >> aag3

# Remove output. 
aag4 = aag3 >> common.sink(['y'])

# Duplicate outputs.
aag4 = aag3 >> common.tee(['y', 'w'])

# And outputs.
aig1 >> common.and_gate(aag1.outputs) # Default output name is #and_output.

# Or outputs.
aig1 >> common.or_gate(inputs=aag1.outputs, output='my_output')
```


# TODO
- [ ] Document.
- [ ] Publish on pypi.
- [ ] Setup continuous integration
- [ ] Support parser full the new aiger features 1.9.3.
  - [X] Latch Initialization
  - [ ] TODO: fill out with other feaures.
- [ ] Symbolic circuits: Composition returns a function that composes using the rules defined.
- [ ] qaiger

# Proposed API

```
# Partial evaluation
# TODO
aig3({x=True}) # New aiger circuit
```
