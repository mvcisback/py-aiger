from hypothesis import given

from aiger import hypothesis as aigh


@given(aigh.Circuits)
def test_aig_to_aag(circ):
    aag = circ._to_aag()
    assert repr(aag) == repr(aag._to_aig())
    
