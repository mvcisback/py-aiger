import aiger as aig
from aiger.bv import BV


def gridworld1d(n, state_name, a0, a1, initials=None):
    # Create feedback adder AIG.
    state_bv = BV(n, state_name, state_name)
    delta_bv = BV(n, "delta", "delta")
    adder_bv = state_bv + delta_bv

    adder_aig = adder_bv.aig.feedback(
        inputs=state_bv.names, outputs=adder_bv.names, initials=initials,
    )

    # Create action mux.
    inc, stay, dec = BV(n, 1, 'inc'), BV(n, 0, 'stay'), BV(n, -1, 'dec')
    circ1 = aig.ite(a0, stay.names, delta_bv.names, delta_bv.names)
    import pdb; pdb.set_trace()
    circ2 = aig.ite(a1, inc.names, dec.names, delta_bv.names)
    amux = ((inc.aig | dec.aig) >>  circ2) >> (stay.aig >> circ1)

    return amux >> adder_aig

    
if __name__ == '__main__':
    g = gridworld1d(2, 'x', 'ax0', 'ax1')



