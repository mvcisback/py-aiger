import aiger_sat
from typing import Dict, Text
import generator


def is_model(polish_formula: Text, model: Dict[Text, int]):
  formula = generator.to_expression(polish_formula.split())
  solver = aiger_sat.SolverWrapper()
  solver.add_expr(~formula)
  return not solver.is_sat(assumptions=model)


assert is_model('and var00 neg var01', model={'var00': True, 'var01': False})
assert not is_model('and var00 neg var01', model={'var00': True})
assert is_model('or var00 neg var01', model={'var00': True})
