from tempfile import NamedTemporaryFile

import hypothesis.strategies as st
from hypothesis import given

from aiger import hypothesis as aigh
from aiger import parser as aigp


@given(aigh.Circuits)
def test_load(aag):
    with NamedTemporaryFile() as f:
        aag.write(f.name)
        aag2 = aigp.load(f.name)
        
        assert aag == aag2
