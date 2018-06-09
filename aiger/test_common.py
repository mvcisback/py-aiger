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


@given(aigh.Circuits)
def test_relabel(aag1):
    # TODO
    new_inputs = {k: f'{k}#2' for k in aag1.inputs}
    assert set(aag1['i', new_inputs].inputs) == set(new_inputs.values())

    new_outputs = {k: f'{k}#2' for k in aag1.outputs}
    assert set(aag1['o', new_outputs].outputs) == set(new_outputs.values())

    with pytest.raises(NotImplementedError):
        aag1['z', {}]


@given(aigh.Circuits, st.integers(min_value=1, max_value=4), st.data())
def test_unroll_simulate(aag1, horizon, data):
    # TODO
    aag2 = aag1.unroll(horizon)

    test_inputs = [{f'{i}': data.draw(st.booleans())
                    for i in aag1.inputs} for _ in range(horizon)]

    time = -1

    def unroll_keys(inputs):
        nonlocal time
        time += 1

        def unroll_key(key):
            return f"{key}##time_{time}"

        return fn.walk_keys(unroll_key, inputs)

    *_, (out1, _) = aag1.simulate(test_inputs)
    out2, _ = aag2(fn.merge(*map(unroll_keys, test_inputs)))
