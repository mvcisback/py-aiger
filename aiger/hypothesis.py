import hypothesis.strategies as st

from aiger_ptltl.test_ptltl import PTLTL_STRATEGY


Circuits = st.builds(lambda expr: expr.aig, PTLTL_STRATEGY)
