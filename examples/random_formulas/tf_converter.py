import collections
import itertools
import tensorflow as tf
from typing import Dict, List, Iterable, Text
Token = int

PAD_TOKEN = '<PAD>'
START_TOKEN = '<START>'
END_TOKEN = '<END>'


def fresh_vocab(initial_vocab=None):
  if initial_vocab is None:
    initial_vocab = {}
    max_token = 0
  else:
    max_token = max(initial_vocab.values()) + 1
  incrementer = itertools.count(max_token)

  def fresh_token():
    return next(incrementer)

  return collections.defaultdict(fresh_token, initial_vocab)


def pad_to_length(sequence: List[Token], length: int,
                  pad_token: int) -> List[Token]:
  if len(sequence) > length:
    raise ValueError('Given sequence is longer than specified padding length.')
  sequence.extend([pad_token] * (length - len(sequence)))


def int_feature(int_list: List[int]):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=int_list))


def string_feature(strings: Iterable[Text]):
  return tf.train.Feature(
      bytes_list=tf.train.BytesList(
          value=[s.encode(encoding='utf-8') for s in strings]))


class Converter:
  """Converts theorems and proof logs to tensorflow examples."""

  def __init__(self, length=None, vocab=None):
    self.length = length
    # This defaultdict will remember all the token assignments.
    self.vocab = fresh_vocab(vocab)  # type: DefaultDict[Text, Token]
    _ = self.vocab[PAD_TOKEN]
    _ = self.vocab[START_TOKEN]
    _ = self.vocab[END_TOKEN]
    # Keep track of statistics by a dict of counters:
    self.stats = collections.defaultdict(int)  # type: DefaultDict[Text, int]

  def tokenize(self, string_to_tokenize) -> List[Token]:
    tokens = [self.vocab[START_TOKEN]]
    for word in string_to_tokenize.split():
      tokens.append(self.vocab[word])
    tokens.append(self.vocab[END_TOKEN])
    return tokens

  def convert(self, features_dict: Dict[str, str]) -> tf.train.Example:
    """Converts a dictionary over strings to a tfexample."""
    for key, value in list(features_dict.items()):
      assert isinstance(value, str)
      tokens = self.tokenize(value)
      if self.length is not None:
        tokens = pad_to_length(tokens, length)
      token_key = f'{key}_tokens'
      assert token_key not in features_dict
      features_dict[key] = string_feature([value])
      features_dict[token_key] = int_feature(tokens)
    return tf.train.Example(features=tf.train.Features(feature=features_dict))
