"""Simulate from the prior, or parent-conditioned on partial traces.

blaze run simulate_prior_traces -- \
  --npzfile=german_partial_prior.npz \
  --model=german_credit_lognormalcentered --num_samples=10000 --num_chains=10 \
  --source_npz=german_credit_lognormalcentered_data/i2.npz \
  --source_vars=overall_log_scale
"""
from absl import app
from absl import flags
import io

import numpy as np
import tensorflow as tf
from tensorflow_probability import edward2 as ed

import models

flags.DEFINE_integer('num_samples', default=100, help='')
flags.DEFINE_integer('num_chains', default=10, help='')
flags.DEFINE_string('model', default='', help='')
flags.DEFINE_string('dataset', default='', help='')
flags.DEFINE_string('npzfile', default='', help='')

flags.DEFINE_string('source_npz', default='', help='')
flags.DEFINE_string('source_vars', default='', help='')

FLAGS = flags.FLAGS


def main(_):
  model_config = models.get_model_by_name(FLAGS.model, dataset=FLAGS.dataset)

  def prior_sample(_, vars_to_fix={}):
    # Get ordered list of latent variable names for this model.
    with ed.tape() as model_tape:
      with ed.interception(ed.make_value_setter(**vars_to_fix)):
        model_config.model(*model_config.model_args)
    return {k: v for (k, v) in model_tape.items()
            if k not in model_config.observed_data}

  fixed_vars = {}
  if FLAGS.source_npz is not None:
    with tf.io.gfile.GFile(FLAGS.source_npz, 'rb') as f:
      source_traces = np.load(f)
    source_vars = FLAGS.source_vars.split(',')
    fixed_vars = {k: np.reshape(source_traces[k],
                                [-1] + list(source_traces[k].shape[2:]))
                  for k in source_vars}

  # problem: prior samples are likely crazy?
  # maybe I'll want an option to load a previous trace and use its set values
  # for some variables. but I'll cross that bridge when we come to it. Should
  # be easily doable here since we can always work in CP.
  num_samples = FLAGS.num_samples * FLAGS.num_chains
  sample_idx = np.arange(num_samples)

  sampled_tape = tf.vectorized_map(prior_sample, (sample_idx, fixed_vars))
  add_chain_dim = lambda v: tf.reshape(v, tf.concat(
      [[FLAGS.num_samples, FLAGS.num_chains], tf.shape(input=v)[1:]], axis=0))
  tape_with_chains = tf.nest.map_structure(add_chain_dim, sampled_tape)

  with tf.compat.v1.Session() as sess:
    tape_with_chains_ = sess.run(tape_with_chains)

  np_path = FLAGS.npzfile
  with tf.io.gfile.GFile(np_path, 'wb') as out_f:
    io_buffer = io.BytesIO()
    np.savez(io_buffer, **tape_with_chains_)
    out_f.write(io_buffer.getvalue())

if __name__ == '__main__':
  app.run(main)
