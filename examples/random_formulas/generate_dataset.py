import aiger_sat
import argparse
import os
import tensorflow as tf

import generator
import tf_converter


EXAMPLES_FILE_NAME = 'examples.tfrecordio'
VOCAB_FILE_NAME = 'vocab.txt'


def get_model(formula):
  solver = aiger_sat.SolverWrapper()
  solver.add_expr(formula)
  return solver.get_model()


def minimize_model(formula, model):
  if model is None:
    return None
  solver = aiger_sat.SolverWrapper()
  solver.add_expr(~formula)
  if solver.is_sat(assumptions=model):
    raise ValueError('UNSAT core generation failed.')
  minimized = solver.get_unsat_core()
  if minimized is None:
    minimized = {}
  return minimized


def generate_example(converter) -> tf.train.Example:
  f = generator.random_formula_tree()
  polish = generator.flatten_to_string(f)  # formula in Polish notation
  print('Polish notation: %s' % polish)
  expr = generator.to_expression(f)
  aag = str(expr.aig)
  aag = aag.split('\n')[:-2]  # drop last line which contains a hash
  aag = '\n'.join(aag)
  # print(aag)

  model = get_model(expr)
  # print('Model:     %s' % model)
  model_str = 'None'
  if model:
    model_word_pairs = ['%s %s' % x for x in sorted(model.items(), key=lambda x: x[0])]
    model_str = ' '.join(model_word_pairs)

  minimized = minimize_model(expr, model)
  # print('Minimized model: %s' % minimized)
  minimized_str = 'None'
  if minimized:
    minimized_word_pairs = ['%s %s' % x for x in sorted(minimized.items(), key=lambda x: x[0])]
    minimized_str = ' '.join(minimized_word_pairs)
  # print('Minimized model: %s' % minimized_str)

  features = {
    'formula_polish': polish,
    'aag': aag,
    'model': model_str,
    'minimized': minimized_str,
    # 'cnf': str(aig2cnf(expr.aig)),
  }
  example = converter.convert(features)
  # print(example)
  return example


def main():
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--num_examples', metavar='N', type=int, nargs=1, dest='num_examples',
                      help='Number of examples to generate.', required=True)
  parser.add_argument('--target_directory', type=str, nargs=1, dest='target_directory',
                      help='Directory to which to write the tfexamples and the vocabulary.',
                      required=True)
  args = parser.parse_args()

  converter = tf_converter.Converter()
  # primitive evaluation
  examples_path = os.path.join(args.target_directory[0], EXAMPLES_FILE_NAME)
  with tf.io.TFRecordWriter(examples_path) as file_writer:
    for _ in range(args.num_examples[0]):
      example = generate_example(converter)
      file_writer.write(example.SerializeToString())

  vocab_path = os.path.join(args.target_directory[0], VOCAB_FILE_NAME)
  vocab = sorted(converter.vocab.items(), key=lambda x: x[1])
  with open(vocab_path, 'w') as vocab_file:
    vocab_file.write('\n'.join([word for word, _ in vocab]))


if __name__ == "__main__":
    # execute only if run as a script
    main()
