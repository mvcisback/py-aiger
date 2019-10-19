import funcy as fn
import hypothesis.strategies as st
import pytest
from hypothesis import given

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
def test_feedback(aag1, data):
    aag2 = aag1 | (common.identity(['##test##']))
    assert len(aag2.outputs) == 1 + len(aag1.outputs)
    assert len(aag2.inputs) == 1 + len(aag1.inputs)

    aag3 = aag2.feedback(
        inputs=['##test##'],
        outputs=['##test##'],
        initials=[True],
        keep_outputs=False)

    assert aag3.outputs == aag1.outputs
    assert aag3.inputs == aag1.inputs

    test_input = {f'{i}': data.draw(st.booleans()) for i in aag3.inputs}
    assert aag1(test_input)[0] == aag3(test_input)[0]


@given(aigh.Circuits)
def test_relabel(aag1):
    # TODO
    new_inputs = {k: f'{k}#2' for k in aag1.inputs}
    assert set(aag1['i', new_inputs].inputs) == set(new_inputs.values())

    new_outputs = {k: f'{k}#2' for k in aag1.outputs}
    assert set(aag1['o', new_outputs].outputs) == set(new_outputs.values())

    with pytest.raises(AssertionError):
        aag1['z', {}]


@given(aigh.Circuits, st.data())
def test_cutlatches(aag1, data):
    aag2, lmap = aag1.cutlatches(aag1.latches)

    assert aag2.inputs >= aag1.inputs
    assert aag2.outputs >= aag1.outputs
    assert len(aag2.latches) == 0

    test_inputs = {i: data.draw(st.booleans()) for i in aag1.inputs}
    test_latch_ins = {l: data.draw(st.booleans()) for l in aag1.latches}
    test_inputs2 = fn.merge(test_inputs,
                            {lmap[k][0]: v
                             for k, v in test_latch_ins.items()})
    out_vals, latch_vals = aag1(test_inputs, latches=test_latch_ins)
    out_vals2, _ = aag2(test_inputs2)
    assert fn.project(out_vals2, aag1.outputs) == out_vals
    latch_vals2 = {k: out_vals2[v] for k, (v, _) in lmap.items()}
    assert latch_vals == latch_vals2


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


def test_eval_order_smoke():
    circ = common.and_gate([f'x{i}' for i in range(16)], "out")
    assert len(common.eval_order(circ)) == 2*16 - 1


def test_delay():
    circ = common.delay(['x', 'y'], initials=[False, False])
    assert circ.inputs == circ.outputs == circ.latches
    circ2 = circ['l', {'x': 'z'}]
    assert circ2.latches == {'z', 'y'}
    assert set(dict(circ2.latch2init).keys()) == {'z', 'y'}
