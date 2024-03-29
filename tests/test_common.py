import funcy as fn
import hypothesis.strategies as st
import pytest
from hypothesis import given
from bidict import bidict

import aiger
from aiger import hypothesis as aigh
from aiger import common


@given(st.integers(2, 10), st.data())
def test_and(n_inputs, data):
    aag = common.and_gate([f'x{i}' for i in range(n_inputs)], "out")
    test_input = {f'x{i}': data.draw(st.booleans()) for i in range(n_inputs)}
    out, _ = aag(test_input)
    assert out['out'] == all(test_input.values())


@given(aigh.Circuits, aigh.Circuits, st.data())
def test_and2(aag1, aag2, data):
    aag3 = aag1 | aag2
    aag3 >>= common.and_gate(aag3.outputs)

    test_input = {f'{i}': data.draw(st.booleans()) for i in aag3.inputs}

    out1, _ = aag1(test_input)
    out2, _ = aag2(test_input)
    out3, _ = aag3(test_input)

    v12 = list(out1.values())[0] and list(out2.values())[0]
    v3 = list(out3.values())[0]
    assert v12 == v3


@given(st.integers(2, 10), st.data())
def test_or(n_inputs, data):
    aag = common.or_gate([f'x{i}' for i in range(n_inputs)], "out")
    test_input = {f'x{i}': data.draw(st.booleans()) for i in range(n_inputs)}
    out, _ = aag(test_input)
    assert out['out'] == any(test_input.values())


@given(aigh.Circuits, aigh.Circuits, st.data())
def test_or2(aag1, aag2, data):
    aag3 = aag1 | aag2
    aag3 >>= common.or_gate(aag3.outputs)

    test_input = {f'{i}': data.draw(st.booleans()) for i in aag3.inputs}

    out1, _ = aag1(test_input)
    out2, _ = aag2(test_input)
    out3, _ = aag3(test_input)

    v12 = list(out1.values())[0] or list(out2.values())[0]
    v3 = list(out3.values())[0]
    assert v12 == v3


@given(st.data())
def test_source(data):
    circ = common.source({'x': True, 'y': False})
    assert len(circ.inputs) == 0
    assert circ({})[0] == {'x': True, 'y': False}


@given(st.data())
def test_sink(data):
    circ = common.sink(['x', 'y'])
    test_input = {i: data.draw(st.booleans()) for i in circ.inputs}
    assert circ(test_input)[0] == dict()


@given(st.integers(1, 5), st.data())
def test_ite(n, data):
    inputs0 = [f'i0_{idx}' for idx in range(n)]
    inputs1 = [f'i1_{idx}' for idx in range(n)]
    outputs = [f'o{idx}' for idx in range(n)]

    circ = common.ite('test', inputs1, inputs0, outputs)
    assert len(circ.outputs) == n

    _inputs = {'test': data.draw(st.booleans())}
    _inputs.update({i: data.draw(st.booleans()) for i in inputs0})
    _inputs.update({i: data.draw(st.booleans()) for i in inputs1})

    res, _ = circ(_inputs)
    for i0, i1, out in zip(inputs0, inputs1, outputs):
        if _inputs['test']:
            assert res[out] == _inputs[i1]
        else:
            assert res[out] == _inputs[i0]


@given(aigh.Circuits, st.data())
def test_flipper(aag1, data):
    aag2 = aag1 >> common.bit_flipper(aag1.outputs)
    aag3 = aag2 >> common.bit_flipper(aag2.outputs)

    test_input = {f'{i}': data.draw(st.booleans()) for i in aag1.inputs}
    out1, _ = aag1(test_input)
    out2, _ = aag2(test_input)
    out3, _ = aag3(test_input)

    v1 = list(out1.values())[0]
    v2 = list(out2.values())[0]
    v3 = list(out3.values())[0]

    assert v1 == (not v2)
    assert v1 == v3


@given(aigh.Circuits, st.data())
def test_tee(aag1, data):
    # TODO
    aag2 = aag1 >> common.tee({k: ['a', 'b', 'c'] for k in aag1.outputs})

    assert len(aag2.outputs) == 3
    assert len(aag2.inputs) == len(aag1.inputs)

    test_input = {f'{i}': data.draw(st.booleans()) for i in aag1.inputs}
    out1, _ = aag1(test_input)
    out2, _ = aag2(test_input)
    v1 = list(out1.values())[0]
    for v in out2.values():
        assert v1 == v


@given(aigh.Circuits, st.data())
def test_loopback(aag1, data):
    aag2 = aag1 | (common.identity(['##test##']))
    assert len(aag2.outputs) == 1 + len(aag1.outputs)
    assert len(aag2.inputs) == 1 + len(aag1.inputs)

    aag3 = aag2.loopback({
        'input': '##test##', 'output': '##test##',
        'init': True, 'keep_output': False
    })

    assert aag3.outputs == aag1.outputs
    assert aag3.inputs == aag1.inputs
    test_input = {f'{i}': data.draw(st.booleans()) for i in aag3.inputs}
    assert aag1(test_input)[0] == aag3(test_input)[0]


@given(aigh.Circuits)
def test_relabel(aag1):
    # TODO
    new_inputs = {k: f'{k}#2' for k in aag1.inputs}
    assert set(aag1['i', new_inputs].inputs) == set(new_inputs.values())
    assert aag1['i', new_inputs].inputs == \
        aag1.relabel('input', new_inputs).inputs

    new_outputs = {k: f'{k}#2' for k in aag1.outputs}
    assert set(aag1['o', new_outputs].outputs) == set(new_outputs.values())
    assert aag1['o', new_inputs].outputs == \
        aag1.relabel('output', new_inputs).outputs

    with pytest.raises(AssertionError):
        aag1['z', {}]


@given(aigh.Circuits, st.sampled_from(['inputs', 'outputs', 'latches']))
def test_relabel_undo_relabel(circ, kind):
    new_inputs = bidict({k: f'{k}#2' for k in getattr(circ, kind)})
    key = kind[0]
    circ2 = circ[key, new_inputs][key, new_inputs.inv]
    assert circ.inputs == circ2.inputs
    assert circ.outputs == circ2.outputs
    assert circ.latches == circ2.latches


@given(aigh.Circuits, st.data())
def test_cutlatches(aag1, data):
    aag2, lmap = aag1.cutlatches(aag1.latches)

    assert aag2.inputs >= aag1.inputs
    assert aag2.outputs >= aag1.outputs
    assert len(aag2.latches) == 0

    test_inputs = {i: data.draw(st.booleans()) for i in aag1.inputs}
    test_latch_ins = {i: data.draw(st.booleans()) for i in aag1.latches}
    test_inputs2 = fn.merge(test_inputs,
                            {lmap[k][0]: v
                             for k, v in test_latch_ins.items()})
    out_vals, latch_vals = aag1(test_inputs, latches=test_latch_ins)
    out_vals2, _ = aag2(test_inputs2)
    assert fn.project(out_vals2, aag1.outputs) == out_vals
    latch_vals2 = {k: out_vals2[v] for k, (v, _) in lmap.items()}
    assert latch_vals == latch_vals2


@given(aigh.Circuits, st.data())
def test_lazy_cutlatches(aag1, data):
    aag2, lmap2 = aag1.cutlatches(renamer=lambda x: x)
    aag3, lmap3 = aag1.lazy_aig.cutlatches(renamer=lambda x: x)

    test_inputs = {i: data.draw(st.booleans()) for i in aag2.inputs}

    out_vals2, _ = aag2(test_inputs)
    out_vals3, _ = aag3(test_inputs)

    assert out_vals2 == out_vals3


@given(aigh.Circuits, st.integers(min_value=1, max_value=4), st.data())
def test_unroll_simulate(aag1, horizon, data):
    # TODO
    aag2 = aag1.unroll(horizon)
    assert horizon * len(aag1.inputs) == len(aag2.inputs)
    assert horizon * len(aag1.outputs) == len(aag2.outputs)

    test_inputs = [{f'{i}': data.draw(st.booleans())
                    for i in aag1.inputs} for _ in range(horizon)]

    time = -1

    def unroll_keys(inputs):
        nonlocal time
        time += 1

        def unroll_key(key):
            return f"{key}##time_{time}"

        return fn.walk_keys(unroll_key, inputs)

    # Check the number of trues and falses match up.
    sum1 = sum(sum(x.values()) for x, _ in aag1.simulate(test_inputs))
    sum2 = sum(aag2(fn.merge(*map(unroll_keys, test_inputs)))[0].values())
    assert sum1 == sum2


def test_unroll_without_init():
    circ = aiger.parse('''aag 3 1 1 1 1
2
4 6 1
6
6 4 2
i0 x
o0 x
l0 y
''')
    circ2 = circ.unroll(2, init=False)
    assert circ2.inputs == {'x##time_0', 'x##time_1', 'y##time_0'}
    assert circ2.outputs == {'x##time_1', 'x##time_2'}
    out, _ = circ2({'x##time_0': 1, 'x##time_1': 1, 'y##time_0': 1})
    assert out['x##time_2'] and out['x##time_1']
    out, _ = circ2({'x##time_0': 1, 'x##time_1': 1, 'y##time_0': 0})
    assert not out['x##time_2'] and not out['x##time_1']


@given(aigh.Circuits, st.data())
def test_feedback_then_cut(circ, data):
    def renamer(_):
        return "##test"

    wire = {
        'input': fn.first(circ.inputs),
        'output': fn.first(circ.outputs),
        'keep_output': False,
        'init': True,
        'latch': '##test',
    }
    circ1 = circ.loopback(wire)
    assert '##test' in circ1.latches
    circ2 = circ1.cutlatches(latches={'##test'}, renamer=renamer)[0]
    circ3 = circ2.relabel('input', {'##test': wire['input']}) \
                 .relabel('output', {'##test': wire['output']})

    assert circ3.inputs == circ.inputs
    assert circ3.outputs == circ.outputs
    assert circ3.latches == circ.latches

    test_input = {f'{i}': data.draw(st.booleans()) for i in circ.inputs}
    assert circ(test_input) == circ3(test_input)


def test_eval_order_smoke():
    circ = common.and_gate([f'x{i}' for i in range(16)], "out")
    assert len(common.eval_order(circ)) == 2*16 - 1


def test_delay():
    circ = common.delay(['x', 'y'], initials=[False, False])
    assert circ.inputs == circ.outputs == circ.latches
    circ2 = circ['l', {'x': 'z'}]
    assert circ2.latches == {'z', 'y'}
    assert set(dict(circ2.latch2init).keys()) == {'z', 'y'}


EXAMPLE = """aag 3 2 1 2 0
2
6
4 6 1
2
4
i0 a
i1 c
o0 a
o1 c
l0 c
"""


def test_unroll_keep_inputs():
    circ = aiger.parse(EXAMPLE)
    unrolled = circ.unroll(3)
    assert unrolled.inputs == {
        "a##time_0", "a##time_1", "a##time_2",
        "c##time_0", "c##time_1", "c##time_2",
    }


@given(aigh.Circuits)
def test_iter_nodes(circ):
    nodes = set(fn.cat(circ.__iter_nodes__()))
    assert set(circ.node_map.values()) <= nodes
    assert set(dict(circ.latch_map).values()) <= nodes


def assert_sample_equiv(circ1, circ2, data):
    assert circ1.inputs == circ2.inputs
    assert circ1.outputs == circ2.outputs
    assert circ1.latches == circ2.latches

    test_input = {f'{i}': data.draw(st.booleans()) for i in circ1.inputs}
    assert circ1(test_input) == circ2(test_input)


def fresh_io(circ):
    _fresh = common._fresh
    return circ.relabel('input', {i: _fresh() for i in circ.inputs}) \
               .relabel('output', {o: _fresh() for o in circ.outputs})


@given(aigh.Circuits, aigh.Circuits, st.data())
def test_seq_compose(circ1, circ2, data):
    circ1, circ2 = map(fresh_io, (circ1, circ2))

    # 1. Check >> same as eager | on disjoint interfaces.
    assert_sample_equiv(circ1 | circ2, circ1 >> circ2, data)

    # 2. Force common interface.
    circ1 = circ1['o', {fn.first(circ1.outputs): '##test'}]
    circ2 = circ2['i', {fn.first(circ2.inputs): '##test'}]

    # Compose and check sample equivilence.
    circ12 = circ1 >> circ2

    assert (circ1.latches | circ2.latches) == circ12.latches
    assert circ1.inputs <= circ12.inputs
    assert circ2.outputs <= circ12.outputs
    assert '##test' not in circ12.inputs
    assert '##test' not in circ12.outputs

    # 3. Check cascading inputs work as expected.
    test_input1 = {f'{i}': data.draw(st.booleans()) for i in circ1.inputs}
    test_input2 = {f'{i}': data.draw(st.booleans()) for i in circ2.inputs}

    omap1, lmap1 = circ1(test_input1)
    test_input2['##test'] = omap1['##test']
    omap2, lmap2 = circ2(test_input2)

    # 3a. Combine outputs/latch outs.
    omap12_expected = fn.merge(fn.omit(omap1, '##test'), omap2)
    lmap12_expected = fn.merge(lmap1, lmap2)

    test_input12 = fn.merge(test_input1, test_input2)
    omap12, lmap12 = circ12(test_input12)

    assert lmap12 == lmap12_expected
    assert omap12 == omap12_expected


@given(aigh.Circuits, st.data())
def test_reinit(circ1, data):
    latch2init_1 = circ1.latch2init
    latch2init_2 = {k: data.draw(st.booleans()) for k in circ1.latches}

    circ2 = circ1.reinit(latch2init_2)
    assert circ2.latch2init == latch2init_2

    circ3 = circ2.reinit(latch2init_1)
    assert circ1.latch2init == circ3.latch2init

    circ4 = circ3.reinit(latch2init_1)
    assert circ1.latch2init == circ4.latch2init

    circ5 = circ3.reinit({})
    assert circ1.latch2init == circ5.latch2init
