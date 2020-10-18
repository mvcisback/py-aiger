import aiger
import aiger_sat
import generator


def test_preorder():
  flattened = list(generator.pre_order([['1'], '2', ['3', '4']]))
  assert flattened == ['1', '2', '3', '4'], flattened


def test_to_expression_drops_surplus_tokens():
  # This is not necessarily desired behavior and should be changed eventually.
  testexpr = generator.to_expression(iter(['var01', 'var00', 'var02']))
  assert aiger_sat.are_equiv(testexpr, aiger.atom('var01'))


def to_expression_test():
  testexpr = generator.to_expression(iter(['and', 'var00', 'var02']))
  assert aiger_sat.are_equiv(testexpr, aiger.atom('var00') & aiger.atom('var02'))


def get_model_test():
  testformula = aiger.atom('x') | ~ aiger.atom('x')
  testformulamodel = get_model(testformula)
  assert len(testformulamodel) == 1, testformulamodel


def minimized_model_test():
  testformula = aiger.atom('x') | ~ aiger.atom('x')
  testformulamodel = get_model(testformula)
  minimized_testformulamodel = minimize_model(testformula, testformulamodel)
  assert len(minimized_testformulamodel) == 0, minimized_testformulamodel
