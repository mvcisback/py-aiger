import pathlib
import tempfile

import aiger


def test_to_aig():
    x = aiger.atom('x')
    c1 = aiger.to_aig(x)
    c2 = aiger.to_aig(c1)
    c3 = aiger.to_aig(str(c2))

    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "foo.aag"
        c3.write(path)
        c4 = aiger.to_aig(path)

    for c in [c1, c2, c3, c4]:
        assert isinstance(c1, aiger.AIG)
        assert isinstance(c2, aiger.AIG)
        assert isinstance(c3, aiger.AIG)
        assert isinstance(c4, aiger.AIG)
