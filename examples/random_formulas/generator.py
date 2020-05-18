import aiger
from aiger_cnf import aig2cnf
import numpy.random as random
from typing import List, Iterable, Text


NUM_VARIABLES = 5
variables = ['var%02d' % v for v in range(NUM_VARIABLES)]
bool_constants = ['True', 'False']
zero_arity_ops = variables + bool_constants
unary_operators = ['neg']
zero_and_unary = zero_arity_ops + unary_operators
binary_operators = ['or', 'and', 'xor', 'eq']
blocks = zero_and_unary + binary_operators


def random_formula_tree(max_depth=10):
  if max_depth <= 0:
    op = random.choice(zero_arity_ops)
  if max_depth == 1:
    op = random.choice(zero_and_unary)
  else:
    op = random.choice(blocks)
  if op in zero_arity_ops:
    return [op]
  if op in unary_operators:
    return [op, random_formula_tree(max_depth - 1)]
  assert op in binary_operators, 'Unknown op: %s zeros: %s' % (op, zero_arity_ops)
  return [op, random_formula_tree(max_depth - 1), random_formula_tree(max_depth - 1)]


def pre_order(tree):
  to_process = [tree]
  while to_process:
    elem = to_process.pop()
    if isinstance(elem, list):
      to_process.extend(elem[::-1])
    else:
      assert isinstance(elem, str) or isinstance(elem, bool)
      yield elem


def flatten_to_string(f):
  return ' '.join(map(str, pre_order(f)))


def to_expression(token_sequence):
  if isinstance(token_sequence, list):
    token_sequence = iter(pre_order(token_sequence))
  elem = next(token_sequence, None)
  if elem is None:
    raise ValueError('Sequence ends before expression is complete.')
  if elem in variables:
    return aiger.atom(elem)
  if elem in bool_constants:
    return aiger.atom(elem == 'True')
  if elem in unary_operators:
    assert elem == 'neg'
    return ~ to_expression(token_sequence)
  assert elem in binary_operators, 'Unknown op: %s' % elem
  left = to_expression(token_sequence)
  right = to_expression(token_sequence)
  if elem == 'or':
    return left | right
  if elem == 'and':
    return left & right
  if elem == 'xor':
    return left ^ right
  if elem == 'eq':
    return left == right
  raise ValueError('Should not reach this point')
