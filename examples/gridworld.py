import aiger as aig
from aiger.bv import BV


def decode_start(num, n):
    return [x == '1' for x in format(num, f'0{n}b')[::-1]]


def _gridworld1d(n, state_name='x', start=0):
    # Create feedback adder AIG.
    a0, a1 = f'a{state_name}0', f'a{state_name}1'
    state_bv = BV(n, state_name, state_name)
    delta_bv = BV(n, "delta", "delta")
    adder_bv = (state_bv + delta_bv).rename(state_name)

    start = decode_start(start, n)

    adder_aig = adder_bv.aig.feedback(
        inputs=state_bv.names,
        outputs=adder_bv.names,
        initials=start,
        keep_outputs=True,
    )

    # Create action mux.
    inc, stay, dec = BV(n, 1, 'inc'), BV(n, 0, 'stay'), BV(n, -1, 'dec')
    circ1 = aig.ite(a0, stay.names, delta_bv.names, delta_bv.names)
    circ2 = aig.ite(a1, inc.names, dec.names, delta_bv.names)
    amux = ((inc.aig | dec.aig) >> circ2) >> (stay.aig >> circ1)

    return amux >> adder_aig


def gridworld2d(n, starts=None):
    xworld = _gridworld1d(n, 'x', starts[0])
    yworld = _gridworld1d(n, 'y', starts[1])
    return xworld | yworld
