# TODO: factor out common parts of seq_compose and par_compose
from collections import namedtuple
from functools import reduce
from typing import List, Mapping, NamedTuple

import funcy as fn
from lenses import bind
from toposort import toposort

Header = namedtuple(
    'Header',
    ['max_var_index', 'num_inputs', 'num_latches', 'num_outputs', 'num_ands'])
Symbol = namedtuple('Symbol', ['kind', 'index', 'name'])
SymbolTable = namedtuple('SymbolTable', ['inputs', 'outputs', 'latches'])


def to_idx(lit):
    return lit >> 1


@fn.curry
def _relabel(relabels, syms):
    return fn.walk_keys(lambda k: relabels.get(k, k), syms)


class AAG(NamedTuple):
    header: Header
    inputs: Mapping[str, int]
    outputs: Mapping[str, int]
    latches: Mapping[str, List[int]]
    gates: List[List[int]]
    comments: List[str]

    def __rshift__(self, other):
        return seq_compose(self, other)

    def __or__(self, other):
        return par_compose(self, other)

    def __getitem__(self, others):
        if not isinstance(others, tuple):
            return super().__getitem__(others)

        name, relabels = others
        if name not in {'i', 'o', 'l'}:
            raise NotImplemented

        name = {'i': 'inputs', 'o': 'outputs', 'l': 'latches'}.get(name)
        return bind(self).GetAttr(name).modify(_relabel(relabels))

    def __repr__(self):
        if self.inputs:
            input_names, input_lits = zip(*list(self.inputs.items()))
        if self.outputs:
            output_names, output_lits = zip(*list(self.outputs.items()))
        if self.latches:
            latch_names, latch_lits = zip(*list(self.latches.items()))

        def str_idx(lit):
            return str(to_idx(lit))

        out = f"aag " + " ".join(map(str, self.header)) + '\n'
        if self.inputs:
            out += '\n'.join(map(str, input_lits)) + '\n'
        if self.latches:
            out += '\n'.join([' '.join(map(str, xs))
                              for xs in latch_lits]) + '\n'
        if self.outputs:
            out += '\n'.join(map(str, output_lits)) + '\n'
        if self.gates:
            out += '\n'.join([' '.join(map(str, xs))
                              for xs in self.gates]) + '\n'
        if self.inputs:
            out += '\n'.join(f"i{idx} {name}"
                             for idx, name in enumerate(input_names)) + '\n'
        if self.outputs:
            out += '\n'.join(f"o{idx} {name}"
                             for idx, name in enumerate(output_names)) + '\n'
        if self.latches:
            out += '\n'.join(f"l{idx} {name}"
                             for idx, name in enumerate(latch_names)) + '\n'
        if self.comments:
            out += 'c\n' + '\n'.join(self.comments) + '\n'
        return out

    def dump(self):
        return repr(self)

    def write(self, location):
        with open(location, "w") as f:
            f.write(self.dump())

    def __call__(self, inputs, latches=None):
        # TODO: implement partial evaluation.
        # TODO: implement setting latch values
        eval_order, gate_lookup = self.eval_order_and_gate_lookup
        latches = dict() if latches is None else latches

        def latch_init(latch):
            return False if len(latch) < 3 else bool(latch[2])

        gate_nodes = fn.merge(
            {v: inputs[k]
             for k, v in self.inputs.items()},
            {
                v[0]: latches.get(k, latch_init(v))
                for k, v in self.latches.items()
            },
            {0: False,
             1: True},
        )

        def gate_output(lit):
            return (not gate_nodes[lit & -2]) if lit & 1 else gate_nodes[lit
                                                                         & -2]

        for gate in fn.cat(eval_order[1:]):
            out, i1, i2 = gate_lookup[gate]
            gate_nodes[out] = gate_output(i1) and gate_output(i2)

        outputs = {k: gate_output(v) for k, v in self.outputs.items()}

        latches = {k: gate_output(v) for k, (_, v, _) in self.latches.items()}
        return outputs, latches

    @property
    def eval_order_and_gate_lookup(self):
        gate_deps = {a & -2: {b & -2, c & -2} for a, b, c in self.gates}
        gate_lookup = {a & -2: (a, b, c) for a, b, c in self.gates}
        return list(toposort(gate_deps)), gate_lookup

    def simulator(self, latches=None):
        inputs = yield
        while True:
            outputs, latches = self(inputs, latches)
            inputs = yield outputs, latches

    def simulate(self, input_seq, latches=None):
        sim = self.simulator()
        next(sim)
        return [sim.send(inputs) for inputs in input_seq]

    def unroll(self, horizon, *, init=True, omit_latches=True):
        # TODO:
        # - Check for name collisions.
        aag0 = cutlatches(self, self.latches.keys())

        def _unroll():
            prev = aag0
            for t in range(1, horizon + 1):
                tmp = prev['i',
                           {k: f"{k}##time_{t-1}"
                            for k in aag0.inputs.keys()}]
                yield tmp['o',
                          {k: f"{k}##time_{t}"
                           for k in aag0.outputs.keys()}]

        unrolled = reduce(seq_compose, _unroll())
        if init:
            latch_source = {
                f"{k}##time_0": val
                for k, (_, _, val) in self.latches.items()
            }
            unrolled = source(latch_source) >> unrolled

        if omit_latches:
            latch_names = [f"{k}##time_{horizon}" for k in self.latches.keys()]

            unrolled = unrolled >> sink(latch_names)

        return unrolled


def cutlatches(aag, latches):
    # TODO: assert relabels won't collide with existing labels.

    # Make latch an input.
    new_inputs = fn.merge(
        aag.inputs, {f"{name}": aag.latches[name][0]
                     for name in latches})

    # Make latch an output.
    new_outputs = fn.merge(
        aag.outputs, {f"{name}": aag.latches[name][1]
                      for name in latches})

    nlatches = len(latches)
    return AAG(
        header=Header(aag.header.max_var_index,
                      aag.header.num_inputs + nlatches,
                      aag.header.num_latches - nlatches,
                      aag.header.num_outputs + nlatches, aag.header.num_ands),
        inputs=new_inputs,
        outputs=new_outputs,
        latches=fn.omit(aag.latches, latches),
        gates=aag.gates,
        comments=aag.comments,
    )


def seq_compose(aag1, aag2, check_precondition=True):
    output1_names = set(aag1.outputs.keys())
    input2_names = set(aag2.inputs.keys())
    interface = output1_names & input2_names

    if check_precondition:
        input1_names = set(aag1.inputs.keys())
        output2_names = set(aag2.outputs.keys())

        assert len((input2_names - interface) & input1_names) == 0
        assert len((output1_names - interface) & output2_names) == 0
        assert len(set(aag1.latches.keys()) & set(aag2.latches.keys())) == 0

    idx_to_name = {
        to_idx(lit): n
        for n, lit in aag2.inputs.items() if n in interface
    }
    n = aag1.header.max_var_index

    def new_lit(lit):
        if lit in (0, 1):
            return lit

        key = to_idx(lit)
        if key not in idx_to_name:
            return lit + 2 * n

        lit2 = aag1.outputs[idx_to_name[key]]
        return (lit2 & -2) + ((lit2 & 1) ^ (lit & 1))

    def new_lits(lits):
        return fn.lmap(new_lit, lits)

    inputs3 = fn.merge(aag1.inputs,
                       fn.walk_values(new_lit, fn.omit(aag2.inputs,
                                                       interface)))
    outputs3 = fn.merge(
        fn.omit(aag1.outputs, interface), fn.walk_values(
            new_lit, aag2.outputs))
    latches3 = fn.merge(aag1.latches, fn.walk_values(new_lits, aag2.latches))
    gates3 = aag1.gates + fn.lmap(new_lits, aag2.gates)

    lits = fn.flatten(
        fn.concat(inputs3.values(),
                  outputs3.values(), latches3.values(), gates3))
    header3 = Header(
        max(map(to_idx, lits)),
        len(inputs3), len(latches3), len(outputs3), len(gates3))

    return AAG(header3, inputs3, outputs3, latches3, gates3,
               aag1.comments + aag2.comments)


def par_compose(aag1, aag2, check_precondition=True):
    input1_names = set(aag1.inputs.keys())
    input2_names = set(aag2.inputs.keys())
    interface = input1_names & input2_names

    if check_precondition:
        assert len(set(aag1.latches.keys()) & set(aag2.latches.keys())) == 0
        assert len(set(aag1.outputs.keys()) & set(aag2.outputs.keys())) == 0

    idx_to_name = {
        to_idx(lit): n
        for n, lit in aag2.inputs.items() if n in interface
    }
    n = aag1.header.max_var_index

    def new_lit(lit):
        if lit in (0, 1):
            return lit

        key = to_idx(lit)
        if key not in idx_to_name:
            return lit + 2 * n

        lit2 = aag1.inputs[idx_to_name[key]]
        return (lit2 & -2) + ((lit2 & 1) ^ (lit & 1))

    def new_lits(lits):
        return fn.lmap(new_lit, lits)

    inputs3 = fn.merge(aag1.inputs,
                       fn.walk_values(new_lit, fn.omit(aag2.inputs,
                                                       interface)))
    outputs3 = fn.merge(aag1.outputs, fn.walk_values(new_lit, aag2.outputs))
    latches3 = fn.merge(aag1.latches, fn.walk_values(new_lits, aag2.latches))
    gates3 = aag1.gates + fn.lmap(new_lits, aag2.gates)

    lits = fn.flatten(
        fn.concat(inputs3.values(),
                  outputs3.values(), latches3.values(), gates3))
    header3 = Header(
        max(map(to_idx, lits)),
        len(inputs3), len(latches3), len(outputs3), len(gates3))

    return AAG(header3, inputs3, outputs3, latches3, gates3,
               aag1.comments + aag2.comments)


def empty():
    return sink([])


def source(outputs):
    return AAG(
        header=Header(0, 0, 0, len(outputs), 0),
        inputs={},
        latches={},
        outputs={key: int(value)
                 for key, value in outputs.items()},
        gates=[],
        comments=[])


def sink(inputs):
    return AAG(
        header=Header(len(inputs), len(inputs), 0, 0, 0),
        inputs={name: 2 * (i + 1)
                for i, name in enumerate(inputs)},
        latches={},
        outputs={},
        gates=[],
        comments=[])


def tee(outputs):
    # TODO: add precondition check.

    def default_name(i, name, key):
        return f"{key}##copy{i}" if name is None else name

    def fix_names(key_val):
        key, val = key_val
        return key, [default_name(i, name, key) for i, name in enumerate(val)]

    outputs = fn.walk(fix_names, outputs)

    num_inputs = len(outputs.keys())
    num_outputs = sum(map(len, outputs.values())) + num_inputs

    orig = list(outputs.keys())
    return AAG(
        header=Header(num_outputs, num_inputs, 0, num_outputs, 0),
        inputs={name: 2 * (i + 1)
                for i, name in enumerate(orig)},
        latches={},
        outputs=fn.merge(*({name: 2 * (i + 1)
                            for name in outputs[key]}
                           for i, key in enumerate(orig)), ),
        gates=[],
        comments=[])


def _make_tree(num, idx=1):
    indices = range(idx, idx + num)

    remain = 0
    while num > 1:
        left, right = indices[::2], indices[1::2]

        idx += num - remain

        gates = [(idx + i, l, r) for i, (l, r) in enumerate(zip(left, right))]

        yield from gates

        if num % 2 == 1:
            remain = 1
            gates.append((indices[-1], None, None))
        else:
            remain = 0

        indices = list(fn.pluck(0, gates))
        num = len(indices)


def bit_flipper(inputs, outputs=None):
    if outputs is None:
        outputs = inputs
    else:
        assert len(outputs) == len(inputs)

    return AAG(
        header=Header(len(inputs), len(inputs), 0, len(inputs), 0),
        inputs={name: 2 * (i + 1)
                for i, name in enumerate(inputs)},
        latches={},
        outputs={name: 2 * (i + 1) + 1
                 for i, name in enumerate(outputs)},
        gates=[],
        comments=[])


def and_gate(inputs, output=None):
    def _and_gate(gate):
        return fn.lmap(lambda x: 2 * x, gate)

    output = '#and_output' if output is None else output
    gates = fn.lmap(_and_gate, _make_tree(len(inputs)))

    return AAG(
        header=Header(len(gates) + len(inputs), len(inputs), 0, 1, len(gates)),
        inputs={name: 2 * (i + 1)
                for i, name in enumerate(inputs)},
        latches={},
        outputs={
            output:
            gates[-1][0] if len(inputs) > 1 else 2
            if len(inputs) == 1 else int(True)
        },
        gates=gates,
        comments=[])


def or_gate(inputs, output=None):
    output = '#or_output' if output is None else output
    aag = and_gate(inputs, output)
    return bit_flipper(inputs) >> aag >> bit_flipper([output])
