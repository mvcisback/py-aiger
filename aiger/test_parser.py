from tempfile import NamedTemporaryFile

from hypothesis import given
from lenses import bind

from aiger import hypothesis as aigh
from aiger import parser as aigp


@given(aigh.Circuits)
def test_load(aag):
    with NamedTemporaryFile() as f:
        aag.write(f.name)
        aag2 = aigp.load(f.name)

        assert bind(aag).comments.set('') == bind(aag2).comments.set('')
