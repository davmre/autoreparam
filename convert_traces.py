"""Convert a CP trace to NCP or VIP.

blaze run convert_traces -- \
  --tracefile=german_partial_prior.npz \
  --model=german_credit_lognormalcentered \
  --vip_json=german_credit_lognormalcentered_data/cVIP_exp_tied.json
"""
from absl import app
from absl import flags
import io
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow_probability import edward2 as ed

import models as models

flags.DEFINE_string('tracefile', default='', help='')
flags.DEFINE_string('vip_json', default='', help='')
flags.DEFINE_string('model', default='', help='')
flags.DEFINE_string('dataset', default='', help='')
FLAGS = flags.FLAGS


def main(_):
  model_config = models.get_model_by_name(FLAGS.model, dataset=FLAGS.dataset)

  if FLAGS.vip_json:
    if tf.io.gfile.exists(FLAGS.vip_json):
      with tf.io.gfile.GFile(FLAGS.vip_json, 'r') as f:
        prev_results = json.load(f)
    else:
      raise Exception('Run VI first to find initial step sizes')
    vip_reparam = prev_results['learned_reparam']
    new_method = 'cVIP'
    to_noncentered = model_config.make_to_partially_noncentered(**vip_reparam)
  else:
    new_method = 'NCP'
    to_noncentered = model_config.to_noncentered

  with tf.io.gfile.GFile(FLAGS.tracefile) as f:
    traces = dict(np.load(f))

  # Get ordered list of latent variable names for this model.
  with ed.tape() as model_tape:
    model_config.model(*model_config.model_args)
  param_names = [
      k for k in list(model_tape.keys()) if k not in model_config.observed_data
  ]
  traces_as_list = [traces[k] for k in param_names]
  initial_shape = traces_as_list[0].shape[:2]  # [num_results x num_chains]
  flattened_traces = [np.reshape(v, [-1] + list(v.shape[2:]))
                      for v in traces_as_list]
  transformed_traces = tf.vectorized_map(to_noncentered, flattened_traces)
  unflattened_traces = {k: tf.reshape(v, initial_shape + v.shape[1:])
                        for (k, v) in zip(param_names, transformed_traces)}

  with tf.compat.v1.Session() as sess:
    unflattened_traces_ = sess.run(unflattened_traces)

  np_path = FLAGS.tracefile[:-4] + '_{}.npz'.format(new_method)
  with tf.io.gfile.GFile(np_path, 'wb') as out_f:
    io_buffer = io.BytesIO()
    np.savez(io_buffer, **unflattened_traces_)
    out_f.write(io_buffer.getvalue())

if __name__ == '__main__':
  app.run()
