# coding=utf-8
# Copyright 2018 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformations of Edward2 programs."""
# pylint: disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import collections
import inspect
import six
import tensorflow as tf

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.experimental.edward2.generated_random_variables import Normal
from tensorflow_probability.python.experimental.edward2.interceptor import interceptable
from tensorflow_probability.python.experimental.edward2.interceptor import interception

from tensorflow_probability.python import edward2
from tensorflow_probability.python.internal import prefer_static

from tensorflow.python.ops.parallel_for.pfor import RegisterPFor, RegisterPForWithArgs, wrap

__all__ = [
    'make_log_joint_fn', 'make_variational_model', 'make_value_setter', 'ncp',
    'get_trace'
]


flags.DEFINE_boolean(
    'float64',
    default=False,
    help='Whether to do linear algebra calculations (eigh / svd / etc) '
         'in float64.'
    )

FLAGS = flags.FLAGS


def make_log_joint_fn(model):
  """Takes Edward probabilistic program and returns its log joint function.

  Args:
    model: Python callable which executes the generative process of a computable
      probability distribution using `ed.RandomVariable`s.

  Returns:
    A log-joint probability function. Its inputs are `model`'s original inputs
    and random variables which appear during the program execution. Its output
    is a scalar tf.Tensor.

  #### Examples

  Below we define Bayesian logistic regression as an Edward program,
  representing the model's generative process. We apply `make_log_joint_fn` in
  order to represent the model in terms of its joint probability function.

  ```python
  from tensorflow_probability import edward2 as ed

  def logistic_regression(features):
    coeffs = ed.Normal(loc=0., scale=1.,
                       sample_shape=features.shape[1], name='coeffs')
    outcomes = ed.Bernoulli(logits=tf.tensordot(features, coeffs, [[1], [0]]),
                            name='outcomes')
    return outcomes

  log_joint = ed.make_log_joint_fn(logistic_regression)

  features = tf.random_normal([3, 2])
  coeffs_value = tf.random_normal([2])
  outcomes_value = tf.round(tf.random_uniform([3]))
  output = log_joint(features, coeffs=coeffs_value, outcomes=outcomes_value)
  ```

  """

  def log_joint_fn(*args, **kwargs):
    """Log-probability of inputs according to a joint probability distribution.

    Args:
      *args: Positional arguments. They are the model's original inputs and can
        alternatively be specified as part of `kwargs`.
      **kwargs: Keyword arguments, where for each key-value pair `k` and `v`,
        `v` is passed as a `value` to the random variable(s) whose keyword
        argument `name` during construction is equal to `k`.

    Returns:
      Scalar tf.Tensor, which represents the model's log-probability summed
      over all Edward random variables and their dimensions.

    Raises:
      TypeError: If a random variable in the model has no specified value in
        `**kwargs`.
    """
    log_probs = []

    def interceptor(rv_constructor, *rv_args, **rv_kwargs):
      """Overrides a random variable's `value` and accumulates its log-prob."""
      # Set value to keyword argument indexed by `name` (an input tensor).
      rv_name = rv_kwargs.get('name')
      if rv_name is None:
        raise KeyError('Random variable constructor {} has no name '
                       'in its arguments.'.format(rv_constructor.__name__))
      value = kwargs.get(rv_name)
      if value is None:
        raise LookupError('Keyword argument specifying value for {} is '
                          'missing.'.format(rv_name))
      rv_kwargs['value'] = value

      rv = rv_constructor(*rv_args, **rv_kwargs)
      log_prob = tf.reduce_sum(input_tensor=rv.distribution.log_prob(rv.value))
      log_probs.append(log_prob)
      return rv

    model_kwargs = _get_function_inputs(model, **kwargs)
    with interception(interceptor):
      try:
        model(*args, **model_kwargs)
      except TypeError as err:
        raise Exception(
            'Wrong number of arguments in log_joint function definition. {}'
            .format(err))
    log_prob = sum(log_probs)
    return log_prob

  return log_joint_fn


def make_value_setter(*positional_args, **model_kwargs):
  """Creates a value-setting interceptor.

  Args:
    *positional_args: Optional positional `Tensor` values. If provided, these
      will be used to initialize variables in the order they occur in the
      program's execution trace.
    **model_kwargs: dict of str to Tensor. Keys are the names of random variable
      in the model to which this interceptor is being applied. Values are
      Tensors to set their value to.

  Returns:
    set_values: Function which sets the value of intercepted ops.
  """

  consumable_args = [x for x in positional_args]
  if len(consumable_args) and len(model_kwargs):
    raise ValueError('make_value_setter does not support simultaneous '
                     'use of positional and keyword args.')

  def set_values(f, *args, **kwargs):
    """Sets random variable values to its aligned value."""
    name = kwargs.get('name')
    if name in model_kwargs:
      kwargs['value'] = model_kwargs[name]
    elif consumable_args:
      kwargs['value'] = consumable_args.pop(0)
    return interceptable(f)(*args, **kwargs)

  return set_values


def get_trace(model, *args, **kwargs):

  trace_result = {}

  def trace(rv_constructor, *rv_args, **rv_kwargs):
    rv = interceptable(rv_constructor)(*rv_args, **rv_kwargs)
    name = rv_kwargs['name']
    trace_result[name] = rv.value
    return rv

  with interception(trace):
    model(*args, **kwargs)

  return trace_result


def make_variational_model(model, *args, **kwargs):

  variational_parameters = collections.OrderedDict()

  def get_or_init(name, shape=None):

    loc_name = name + '_loc'
    scale_name = name + '_scale'

    if loc_name in variational_parameters.keys() and \
       scale_name in variational_parameters.keys():
      return (variational_parameters[loc_name],
              variational_parameters[scale_name])
    else:
      # shape must not be None
      variational_parameters[loc_name] = \
          tf.compat.v1.get_variable(
              name=loc_name,
              initializer=1e-2 * tf.random.normal(shape, dtype=tf.float32))

      variational_parameters[scale_name] = tf.nn.softplus(
          tf.compat.v1.get_variable(
              name=scale_name,
              initializer=-2 * tf.ones(shape, dtype=tf.float32)))
      return (variational_parameters[loc_name],
              variational_parameters[scale_name])

  def mean_field(rv_constructor, *rv_args, **rv_kwargs):

    name = rv_kwargs['name']
    if name not in kwargs.keys():
      rv = rv_constructor(*rv_args, **rv_kwargs)
      loc, scale = get_or_init(name, rv.shape)

      # NB: name must be the same as original variable,
      # in order to be able to do black-box VI (setting
      # parameters to variational values obtained via trace).
      return Normal(loc=loc, scale=scale, name=name)

    else:
      rv_kwargs['value'] = kwargs[name]
      return rv_constructor(*rv_args, **rv_kwargs)

  def variational_model(*args):
    with interception(mean_field):
      return model(*args)

  _ = variational_model(*args)

  return variational_model, variational_parameters


# FIXME: Assumes the name of the data starts with y... Need to fix so that
# it works with user-specified data.
def ncp(rv_constructor, *rv_args, **rv_kwargs):
  if (rv_constructor.__name__ == 'Normal' and
      not rv_kwargs['name'].startswith('y')):
    loc = rv_kwargs['loc']
    scale = rv_kwargs['scale']
    name = rv_kwargs['name']

    shape = rv_constructor(*rv_args, **rv_kwargs).shape

    kwargs_std = {}
    kwargs_std['loc'] = tf.zeros(shape)
    kwargs_std['scale'] = tf.ones(shape)
    kwargs_std['name'] = name + '_std'

    b = tfb.AffineScalar(scale=scale, shift=loc)
    if 'value' in rv_kwargs:
      kwargs_std['value'] = b.inverse(rv_kwargs['value'])

    rv_std = interceptable(rv_constructor)(*rv_args, **kwargs_std)

    return b.forward(rv_std)

  elif ((rv_constructor.__name__.startswith('MultivariateNormal')
            or rv_constructor.__name__.startswith('GaussianProcess'))
            and not rv_kwargs['name'].startswith('y')):

    name = rv_kwargs['name']

    if rv_constructor.__name__.startswith('GaussianProcess'):
      gp_dist = rv_constructor(*rv_args, **rv_kwargs).distribution
      X = gp_dist._get_index_points()
      x_loc = gp_dist.mean_fn(X)
      x_cov = gp_dist._compute_covariance(index_points=X)
      shape = tfd.MultivariateNormalFullCovariance(x_loc, x_cov).event_shape

    else:
      x_loc = rv_kwargs['loc']
      x_cov = rv_kwargs['covariance_matrix']
      shape = rv_constructor(*rv_args, **rv_kwargs).shape

    kwargs_std = {}

    kwargs_std['loc'] = tf.zeros(shape)
    kwargs_std['scale_diag'] = tf.ones(shape[0])
    kwargs_std['name'] = name + '_std'

    scale = tf.linalg.cholesky(x_cov + 1e-6 * tf.eye(tf.shape(x_cov)[-1]))
    b = tfb.AffineLinearOperator(
        scale=tf.linalg.LinearOperatorLowerTriangular(scale),
        shift=x_loc)

    if 'value' in rv_kwargs:
      kwargs_std['value'] = b.inverse(rv_kwargs['value'])

    rv_std = edward2.MultivariateNormalDiag(*rv_args, **kwargs_std)
    return b.forward(rv_std)

  else:
    return interceptable(rv_constructor)(*rv_args, **rv_kwargs)


class LinearOperatorEigenScale(tf.linalg.LinearOperator):
  """LinearOperator representing Q D where Q is Hermitian and D is diagonal.

  This enables efficient calculations with the matrix square roots arising
  from eigendecomposition. That is, if `X = Q R Q'` where `R` is the diagonal
  matrix of eigenvalues and `Q` has eigenvectors as its columns, then
  `LinearOperatorEigenScale(Q, d=sqrt(diag_part(R)))` represents a matrix
  `M` such that `M M.T = X`.

  In particular, of all such matrices, we choose `Q sqrt(R) Q'` because it
  is stable under optimization. Note that Q is not smooth wrt X -- small
  changes in eigenvalues lead to discontinuities in the sorted eigenvectors --
  but `Q sqrt(R) Q'` is.
  """

  def __init__(self, Q, d, name='LinearOperatorEigenScale'):
    """
    Args:
      Q: `float` `Tensor` of shape `[..., N, N]` satisfying `Q @ Q.T = I`.
      d: `float` `Tensor` of shape `[..., N]`.
    """
    self.Q = Q
    self.d = d
    super(LinearOperatorEigenScale, self).__init__(
          dtype=self.Q.dtype,
          graph_parents=[],
          is_non_singular=None,
          is_self_adjoint=None,
          is_positive_definite=None,
          is_square=True,
          name=name)

  def _shape(self):
    return self.Q.shape

  def _shape_tensor(self):
    return tf.shape(self.Q)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    x = tf.linalg.adjoint(x) if adjoint_arg else x
    return tf.matmul(self.Q, self.d[..., tf.newaxis] * tf.matmul(
        self.Q, x, adjoint_a=True))

  def _matvec(self, x, adjoint=False):
    return tf.linalg.matvec(self.Q, self.d * tf.linalg.matvec(
        self.Q, x, adjoint_a=True))

  def _determinant(self):
    return tf.reduce_prod(self.d, axis=[-1])

  def _log_abs_determinant(self):
    log_det = tf.reduce_sum(
        tf.log(tf.abs(self.d)), axis=[-1])
    if self.dtype.is_complex:
      log_det = tf.cast(log_det, dtype=self.dtype)
    return log_det

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    rhs = tf.linalg.adjoint(rhs) if adjoint_arg else rhs
    # multiply by (Q L Q')^-1 = Q'^-1 L^-1 Q^-1 = Q L^-1 Q'
    return tf.matmul(self.Q,
        tf.matmul(self.Q, rhs, adjoint_a=True) / self.d[..., tf.newaxis])

  def _to_dense(self):
    return tf.matmul(self.Q * self.d[..., tf.newaxis, :],
                     self.Q,
                     adjoint_b=True)


@RegisterPFor("SelfAdjointEigV2")
def _convert_self_adjoint_eig(pfor_input):
  t = pfor_input.stacked_input(0)
  return [wrap(x, True) for x in eigh_with_safe_gradient(t)]


@tf.custom_gradient
def eigh_with_safe_gradient(x):
  """Like tf.linalg.eigh, but avoids NaN gradients from repeated eigenvalues."""
  e, v = tf.linalg.eigh(x)

  def grad(grad_e, grad_v):
    """Gradient for SelfAdjointEigV2."""
    with tf.control_dependencies([grad_e, grad_v]):
      ediffs = tf.expand_dims(e, -2) - tf.expand_dims(e, -1)

      # Avoid NaNs from reciprocals when eigenvalues are close.
      safe_recip = tf.where(ediffs**2 < 1e-10,
                            tf.zeros_like(ediffs),
                            tf.reciprocal(ediffs))
      f = tf.matrix_set_diag(
          safe_recip,
          tf.zeros_like(e))
      grad_a = tf.matmul(
          v,
          tf.matmul(
              tf.matrix_diag(grad_e) +
              f * tf.matmul(v, grad_v, adjoint_a=True),
              v,
              adjoint_b=True))
    # The forward op only depends on the lower triangular part of a, so here we
    # symmetrize and take the lower triangle
    grad_a = tf.linalg.band_part(grad_a + tf.linalg.adjoint(grad_a), -1, 0)
    grad_a = tf.linalg.set_diag(grad_a, 0.5 * tf.matrix_diag_part(grad_a))
    return grad_a

  return (e, v), grad


def make_learnable_parametrisation(init_val_loc=0.,
                                   init_val_scale=0.,
                                   learnable_parameters=None,
                                   tau=1.,
                                   parameterisation_type='unspecified',
                                   tied_pparams=False):
  allow_new_variables = False
  if learnable_parameters is None:
    learnable_parameters = collections.OrderedDict()
    allow_new_variables = True

  def get_or_init(name, loc_shape, scale_shape,
                  parameterisation_type=None):
    loc_name = name + '_a'
    scale_name = name + '_b'
    alternate_scale_name = name + '_c'

    if tied_pparams:
      loc_shape = prefer_static.broadcast_shape(loc_shape, scale_shape)

    if loc_name in learnable_parameters.keys():
      scale_param = learnable_parameters.get(scale_name, 1.)
      alternate_param = learnable_parameters.get(alternate_scale_name, 1.)
      return (learnable_parameters[loc_name],
              scale_param,
              alternate_param)
    else:
      if not allow_new_variables:
        raise Exception('trying to create a variable for {}, but '
                        'parameterization was already passed in ({})'.format(
                            name, learnable_parameters))

      learnable_parameters[loc_name] = tf.sigmoid(
          tau * tf.compat.v1.get_variable(
              name=loc_name + '_unconstrained',
              initializer=tf.ones(loc_shape) * init_val_loc))

      alternate_param = 0.
      if tied_pparams:
        scale_param = learnable_parameters[loc_name]
      else:
        scale_param = 1.
        if (('eig' in parameterisation_type) or
            ('chol' in parameterisation_type) or
            ('scalar' in parameterisation_type)):
          scale_param = tf.sigmoid(tau * tf.compat.v1.get_variable(
              name=scale_name + '_unconstrained',
              initializer=tf.ones(scale_shape) * init_val_scale))
          learnable_parameters[scale_name] = scale_param

        if 'indep' in parameterisation_type:
          alternate_param = 1e-4 + tf.nn.softplus(
              tf.compat.v1.get_variable(
                  name=alternate_scale_name + '_unconstrained',
                  initializer=tf.ones(scale_shape)))
          learnable_parameters[alternate_scale_name] = alternate_param
      return (learnable_parameters[loc_name],
              scale_param,
              alternate_param)

  bijectors = collections.OrderedDict()

  def recenter(rv_constructor, *rv_args, **rv_kwargs):

    if (rv_constructor.__name__ == 'Normal' and
        not rv_kwargs['name'].startswith('y')):

      # NB: assume everything is kwargs for now.
      x_loc = rv_kwargs['loc']
      x_scale = rv_kwargs['scale']

      name = rv_kwargs['name']
      a, b, _ = get_or_init(name,
                             loc_shape=tf.shape(x_loc),
                             scale_shape=tf.shape(x_scale),
                             parameterisation_type='scalar')

      kwargs_std = {}
      kwargs_std['loc'] = tf.multiply(x_loc, a)
      kwargs_std['scale'] = tf.pow(x_scale,
                                   b)  # tf.multiply(x_scale - 1., b) + 1.
      kwargs_std['name'] = name

      scale = x_scale / kwargs_std['scale']  # tf.pow(x_scale, 1. - b)
      shift = x_loc - tf.multiply(scale, kwargs_std['loc'])
      b = tfb.AffineScalar(scale=scale, shift=shift)
      if 'value' in rv_kwargs:
        kwargs_std['value'] = b.inverse(rv_kwargs['value'])
      learnable_parameters[name + '_prior_mean'] = tf.convert_to_tensor(x_loc)
      learnable_parameters[name + '_prior_scale'] = tf.convert_to_tensor(x_scale)

      rv_std = interceptable(rv_constructor)(*rv_args, **kwargs_std)
      bijectors[name] = b
      return b.forward(rv_std)

    if ((rv_constructor.__name__.startswith('MultivariateNormal')
           or rv_constructor.__name__.startswith('GaussianProcess'))
          and not rv_kwargs['name'].startswith('y')):

      name = rv_kwargs['name']

      if rv_constructor.__name__.startswith('GaussianProcess'):
        gp_dist = rv_constructor(*rv_args, **rv_kwargs).distribution
        X = gp_dist._get_index_points()
        x_loc = gp_dist.mean_fn(X)
        x_cov = gp_dist._compute_covariance(index_points=X)
      else:
        x_loc = rv_kwargs['loc']
        x_cov = rv_kwargs['covariance_matrix']

      a, b, c = get_or_init(name,
                           loc_shape=tf.shape(x_loc),
                           scale_shape=tf.shape(x_cov)[:-1],
                           parameterisation_type=parameterisation_type)
      ndims = tf.shape(x_cov)[-1]
      x_loc = tf.broadcast_to(x_loc, tf.shape(x_cov)[:-1])
      cov_dtype = tf.float64 if FLAGS.float64 else x_cov.dtype
      x_cov = tf.cast(x_cov, cov_dtype)
      if parameterisation_type == 'eig':
        Lambda, Q = eigh_with_safe_gradient(x_cov)
        Lambda = tf.abs(Lambda)
        Lambda = tf.cast(Lambda, tf.float32)
        Q = tf.cast(Q, tf.float32)
        Lambda_hat_b = tf.pow(Lambda, b)
        if tied_pparams:
          # If the scale parameterization is in the eigenbasis,
          # apply it to the mean in the same basis.
          loc_in_eigenbasis = tf.linalg.matvec(Q, x_loc, adjoint_a=True)
          reparam_loc = tf.linalg.matvec(Q, tf.multiply(loc_in_eigenbasis, a))
        else:
          reparam_loc = tf.multiply(x_loc, a)

        kwargs_std = {}
        kwargs_std['loc'] = reparam_loc
        kwargs_std['scale'] = LinearOperatorEigenScale(
            Q, d=tf.sqrt(Lambda_hat_b))
        kwargs_std['name'] = name

        L = tf.matmul(Q, tf.linalg.diag(tf.sqrt(Lambda)))
        L_hat_inv = tf.matmul(
            tf.linalg.diag(1. / tf.sqrt(Lambda_hat_b + 1e-10)), Q,
            adjoint_b=True)
        scale = tf.matmul(L, L_hat_inv)
        shift = x_loc - tf.linalg.matvec(scale, reparam_loc)
        b = tfb.AffineLinearOperator(
            scale=tf.linalg.LinearOperatorFullMatrix(scale), shift=shift)
        if 'value' in rv_kwargs:
          kwargs_std['value'] = b.inverse(rv_kwargs['value'])

      elif parameterisation_type == 'chol':
        L = tf.linalg.cholesky(x_cov + 1e-6 * tf.eye(ndims, dtype=x_cov.dtype))
        L = tf.cast(L, tf.float32)

        reparam_loc = x_loc * a
        reparam_scale = tf.linalg.LinearOperatorLowerTriangular(
            tf.linalg.diag(1 - b) + b[..., tf.newaxis] * L)
        kwargs_std = {}
        kwargs_std['loc'] = reparam_loc
        kwargs_std['scale'] = reparam_scale
        kwargs_std['name'] = name

        Dinv = tf.linalg.triangular_solve(tf.cast(reparam_scale.to_dense(),
                                                  cov_dtype),
                                          tf.eye(ndims, dtype=cov_dtype))
        Dinv = tf.cast(Dinv, tf.float32)
        scale = tf.matmul(L, Dinv)
        shift = x_loc - tf.linalg.matvec(scale, reparam_loc)
        b = tfb.AffineLinearOperator(
            scale=tf.linalg.LinearOperatorFullMatrix(scale), shift=shift)
        if 'value' in rv_kwargs:
          kwargs_std['value'] = b.inverse(rv_kwargs['value'])

      elif parameterisation_type=='indep':
        # Assumes `C^-1 = diag(c)` is a learned diagonal matrix of 'evidence
        # precisions'. This approximates the true posterior under an iid
        # Gaussian observation model:
        prior_chol = tf.linalg.cholesky(x_cov)
        prior_inv = tf.linalg.cholesky_solve(
            prior_chol, tf.eye(ndims, dtype=prior_chol.dtype))
        approx_posterior_prec = prior_inv + tf.cast(
            tf.linalg.diag(c), prior_inv.dtype)
        approx_posterior_prec_chol = tf.linalg.cholesky(approx_posterior_prec)
        approx_posterior_cov = tf.linalg.cholesky_solve(
            approx_posterior_prec_chol, tf.eye(
                ndims, dtype=approx_posterior_prec_chol.dtype))
        cov_chol = tf.linalg.cholesky(approx_posterior_cov)

        cov_chol = tf.cast(cov_chol, tf.float32)
        prior_chol = tf.cast(prior_chol, tf.float32)
        scale_linop = tf.linalg.LinearOperatorLowerTriangular(cov_chol)

        reparam_loc = x_loc * a
        reparam_scale = tf.linalg.LinearOperatorComposition([
            tf.linalg.LinearOperatorInversion(scale_linop),
            tf.linalg.LinearOperatorLowerTriangular(prior_chol)])
        kwargs_std = {}
        kwargs_std['loc'] = reparam_loc
        kwargs_std['scale'] = reparam_scale
        kwargs_std['name'] = name

        shift = x_loc - scale_linop.matvec(reparam_loc)
        b = tfb.AffineLinearOperator(scale=scale_linop, shift=shift)
        if 'value' in rv_kwargs:
          kwargs_std['value'] = b.inverse(rv_kwargs['value'])

      elif parameterisation_type == 'eigindep':
        # Combines 'eig' and 'indep' parameterizations, modeling the posterior
        # as
        # (V D**(-b) V' + diag(c))^-1
        # where VDV' is the eigendecomposition of the prior cov, and b and c
        # are learned vectors.
        b, c = [tf.cast(x, cov_dtype) for x in (b, c)]
        Lambda, Q = eigh_with_safe_gradient(x_cov)
        Lambda = tf.abs(Lambda)
        Lambda_hat_b = 1e-6 + tf.pow(Lambda, b)
        prior = tf.matmul(Q, tf.matmul(
            tf.linalg.diag(Lambda_hat_b), Q, adjoint_b=True))
        prior_chol = tf.linalg.cholesky(prior  + 1e-6 * tf.eye(
            ndims, dtype=prior.dtype))
        prior_prec = tf.linalg.cholesky_solve(
            prior_chol + 1e-6 * tf.eye(ndims, dtype=prior_chol.dtype),
            tf.eye(ndims, dtype=prior_chol.dtype))

        approx_posterior_prec = prior_prec + tf.linalg.diag(c)
        approx_posterior_prec_chol = tf.linalg.cholesky(approx_posterior_prec)
        approx_posterior_cov = tf.linalg.cholesky_solve(
            approx_posterior_prec_chol + 1e-6 * tf.eye(
                ndims, dtype=approx_posterior_prec_chol.dtype),
            tf.eye(ndims, dtype=approx_posterior_prec_chol.dtype))
        cov_chol = tf.linalg.cholesky(approx_posterior_cov + 1e-6 * tf.eye(
            ndims, dtype=approx_posterior_cov.dtype))
        cov_chol = tf.cast(cov_chol, tf.float32)
        prior_chol = tf.cast(prior_chol, tf.float32)
        scale_linop = tf.linalg.LinearOperatorLowerTriangular(cov_chol)

        reparam_loc = tf.multiply(x_loc, a)

        reparam_scale = tf.linalg.LinearOperatorComposition([
            tf.linalg.LinearOperatorInversion(scale_linop),
            tf.linalg.LinearOperatorLowerTriangular(prior_chol)])
        kwargs_std = {}
        kwargs_std['loc'] = reparam_loc
        kwargs_std['scale'] = reparam_scale
        kwargs_std['name'] = name

        shift = x_loc - scale_linop.matvec(reparam_loc)
        b = tfb.AffineLinearOperator(scale=scale_linop, shift=shift)
        if 'value' in rv_kwargs:
          kwargs_std['value'] = b.inverse(rv_kwargs['value'])
      else:
        raise Exception('unrecognized reparameterization strategy!')

      if rv_constructor.__name__.startswith('GaussianProcess'):
        rv_std = edward2.MultivariateNormalLinearOperator(
            *rv_args, **kwargs_std)
      else:
        rv_std = interceptable(rv_constructor)(*rv_args, **kwargs_std)

      bijectors[name] = b
      return b.forward(rv_std)
    else:
      return interceptable(rv_constructor)(*rv_args, **rv_kwargs)

  return learnable_parameters, recenter, bijectors


def _get_function_inputs(f, **kwargs):
  """Filters inputs to be compatible with function `f`'s signature.

  Args:
    f: Function according to whose input signature we filter arguments.
    **kwargs: Keyword arguments to filter according to `f`.

  Returns:
    Dict of key-value pairs in `kwargs` which exist in `f`'s signature.
  """
  if hasattr(f, '_func'):  # functions returned by tf.make_template
    f = f._func  # pylint: disable=protected-access

  try:  # getargspec was deprecated in Python 3.6
    argspec = inspect.getfullargspec(f)
  except AttributeError:
    argspec = inspect.getargspec(f)

  fkwargs = {k: v for k, v in six.iteritems(kwargs) if k in argspec.args}
  return fkwargs
