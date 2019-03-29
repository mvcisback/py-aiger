from typing import Tuple, NamedTuple, Mapping, Union
from uuid import uuid1

import attr
import funcy as fn
from pyrsistent import pmap, thaw
from toposort import toposort

from aiger import common as cmn
from aiger import parser


NODE_ID = str


class AndGate(NamedTuple):
    left: NODE_ID
    right: NODE_ID


class Inverter(NamedTuple):
    input: NODE_ID


class Latch(NamedTuple):
    id: NODE_ID
    input: NODE_ID
    initial: bool


NODE = Union[AndGate, Inverter, bool]


def seq_compose(circ1, circ2):
    interface = circ1.outputs & circ2.inputs
    assert not (circ1.outputs - interface) & circ2.outputs
    assert not circ1.latches & circ2.latches

    dag = (circ1.dag + circ2.dag).evolver() 
    omap1, imap2 = circ1.omap.evolver(), circ2.imap.evolver()
    for i in interface:
        dag[imap2[i]] = dag[omap1[i]]  # Wire output to input.
        del dag[omap1[i]]  # No longer a node.
        del omap1[i]  # No longer an output.
        del imap2[i]  # No longer an input.

    return AIG2(
        dag=dag.persistent(),
        imap=circ1.imap + imap2.persistent(),
        omap=circ2.omap + omap1.persistent(),
        lmap=circ1.lmap + circ2.lmap,
        comments=circ1.comments + circ2.comments
    )


def par_compose(aig1, aig2):
    assert not aig1.latches & aig2.latches
    assert not aig1.outputs & aig2.outputs

    # TODO: handle shared inputs
    assert not aig1.inputs & aig2.inputs

    return AIG2(
        imap=aig1.imap + aig2.imap,
        outputs=aig1.omap + aig2.omap,
        latches=aig1.lmap + aig2.lmap,
        dag=aig1.dag + aig2.dag,
        comments=aig1.comments + aig2.comments
    )


def _eval(circ, inputs, latches=None):
    if latches is None:
        latches = dict()

    lookup = fn.merge(
        {k: inputs[name] for name, k in circ.imap.items()},
        {k: latches.get(name, l.initial) for name, l in circ.lmap.items()}
    )

    for key in fn.concat(*circ.eval_order()):
        if key in lookup:
            continue

        node = circ.dag[key]
        if isinstance(node, bool):
            lookup[key] = node
        elif isinstance(node, AndGate):
            lookup[key] = lookup[node.left] and lookup[node.right]
        elif isinstance(node, Inverter):
            lookup[key] = not lookup[node.input]
        else:
            lookup[key] = lookup[node]

    outputs = {name: lookup[key] for name, key in circ.omap.items()}
    latch_outputs = {name: lookup[l.id] for name, l in circ.lmap.items()}
    return outputs, latch_outputs


def to_aag(circ):
    aag = parser.AAG({}, {}, {}, [], circ.comments)
    max_idx = 1

    lit_map = {}
    for name, key in circ.imap.items():
        aag.inputs[name] = 2*max_idx
        lit_map[key] = 2*max_idx
        max_idx += 1

    for name, l in circ.lmap.items():
        lit_map[l.input] = 2*max_idx
        max_idx += 1

    # Update aag with current level.
    for key in fn.concat(*circ.eval_order()):
        if key in lit_map:
            continue

        node = circ.dag[key]
        if isinstance(node, Inverter):
            input_lit = lit_map[node.input]
            lit_map[key] = (input_lit & -2) | (1 ^ (input_lit & 1))
            continue

        elif node is False:
            lit_map[key] = 0
            continue

        # Must be And or Latch
        lit_map[key] = 2 * max_idx
        max_idx += 1
        if isinstance(node, AndGate):
            encoded = tuple(map(lit_map.get, (key, node.left, node.right)))
            aag.gates.append(encoded)

    for name, l in circ.lmap.items():
        if l.id in lit_map:
            lit = 2*max_idx
            max_idx += 1
        else:
            lit = lit_map[l.id]

        ilit = lit_map[l.input]
        aag.latches[name] = lit, ilit, int(l.initial)

    for name, key in circ.omap.items():
        aag.outputs[name] = lit_map[key]

    return aag


@attr.s(frozen=True, slots=True, auto_attribs=True, repr=False)
class AIG2:
    imap: Mapping[str, NODE_ID] = pmap()  # input -> id.
    omap: Mapping[str, NODE_ID] = pmap()  # output -> id.
    lmap: Mapping[str, Latch] = pmap()  # latch -> id.
    dag: Mapping[NODE_ID, NODE] = pmap()  # id -> node.
    comments: Tuple[str] = ()
    is_aag: bool = False

    __or__ = par_compose
    __rshift__ = seq_compose
    __call__ = _eval

    def __getitem__(self, others):
        assert isinstance(others, tuple) and len(others) == 2
        kind, relabels = others

        assert kind in {'i', 'o', 'l'}
        kind = {'i': 'imap', 'o': 'omap', 'l': 'lmap'}.get(kind)

        e = getattr(self, kind).evolver()
        for k, v in relabels.items():
            if k not in e:
                continue

            e[v] = e[k]
            del e[k]
        
        return attr.evolve(self, **{kind: e.persistent()})


    def __repr__(self):
        return repr(to_aag(self))

    @property
    def inputs(self):
        return frozenset(self.imap.keys())

    @property
    def outputs(self):
        return frozenset(self.omap.keys())

    @property
    def latches(self):
        return frozenset(self.lmap.keys())
    
    def eval_order(self):
        def to_deps(node):
            if isinstance(node, (AndGate, Inverter)):
                return set(node)
            return set()

        graph = {k: to_deps(v) for k, v in self.dag.items()}
        latch_ids = {l.input for l in self.lmap.values()}
        return fn.chain([latch_ids], toposort(graph))

    def simulator(self, latches=None):
        inputs = yield
        while True:
            outputs, latches = self(inputs, latches)
            inputs = yield outputs, latches

    def simulate(self, input_seq, latches=None):
        sim = self.simulator()
        next(sim)
        return [sim.send(inputs) for inputs in input_seq]

    def write(self, path):
        with open(path, 'w') as f:
            f.write(repr(self))

    def cutlatches(self, latches):
        assert not (self.inputs & set(latches))
        assert not (self.outputs & set(latches))

        lmap = self.lmap.evolver()
        imap = self.imap.evolver()
        omap = self.omap.evolver()

        for name in latches:
            latch = lmap[name]
            del lmap[name]

            imap[name] = latch.input
            omap[name] = latch.id

        return attr.evolve(self, 
            imap=imap.persistent(),
            omap=omap.persistent(),
            lmap=lmap.persistent()
        )

    def feedback(
        self, inputs, outputs, initials=None, latches=None, keep_outputs=False
    ):
        if latches is None:
            latches = inputs

        if initials is None:
            initials = [False for _ in inputs]

        assert len(inputs) == len(initials) == len(outputs) == len(latches)
        assert len(set(inputs) & self.inputs) != 0
        assert len(set(outputs) & self.outputs) != 0
        
        imap = self.imap.evolver()
        lmap = self.lmap.evolver()
        omap = self.omap.evolver()
        rekey = {}
        for i, o, l, v in  zip(inputs, outputs, latches, initials):
            lmap[l] = Latch(id=omap[o], input=imap[i], initial=v)
            del imap[i]
            if not keep_outputs:
                del omap[o]

        return attr.evolve(self, 
            imap=imap.persistent(),
            omap=omap.persistent(),
            lmap=lmap.persistent(),
        )


# Core set of gates.


def _fresh():
    return str(uuid1())


def and_gate(left, right, output):
    lnode, rnode, onode = _fresh(), _fresh(), _fresh()
    return AIG2(
        imap=pmap({left: lnode, right: rnode}),
        omap=pmap({output: onode}),
        dag=pmap({onode: AndGate(lnode, rnode),})
    )


def source(outputs):
    omap = pmap({k: _fresh() for k in outputs.keys()})
    dag=pmap({k: v for k, v in zip(omap.values(), outputs.values())})
    return AIG2(omap=omap, dag=dag)


def sink(inputs):
    return AIG2(imap=pmap({k: _fresh() for k in inputs}))


def bit_flipper(inputs, outputs=None):
    if outputs is None:
        outputs = inputs
    else:
        assert len(outputs) == len(inputs)

    imap = pmap({k: _fresh() for k in inputs})
    omap = pmap({k: _fresh() for k in outputs})
    dag=pmap({omap[o]: Inverter(imap[i]) for i, o in zip(inputs, outputs)})
    return AIG2(imap=imap, omap=omap, dag=dag)
