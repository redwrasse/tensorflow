import tensorflow as tf
import numpy as np
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg_impl
from tensorflow.python.framework import errors
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


def np_expm(x):
  num_terms = 40  # TODO: justify 40 terms
  y = np.zeros(x.shape, dytpe=x.dtype)
  xn = np.eye(x.shape[0], dtype=x.dtype)
  for n in range(40):
    if n > 0:
      xn /= float(n)
    y += xn
    xn = np.dot(xn, x)
  return y


# TODO: test behavior at different dimension scales, and also matrix norm scales
# ref https://personales.upv.es/serblaza/2018Expm_Rev.pdf
# 'the scaling and squaring
# procedure is perhaps the most popular when the dimension of the corresponding
# matrix runs well into the hundreds.'

class ExponentialOpTest(test.TestCase):

  def test_foo(self):
    self.assertEqual(1, 1)

  def test_euler_relation(self):
    pass

  def test_incorrect_dims(self):  # todo: rename to test_unsupported_shapes
    # todo: more thorough shapes edge cases
    x = constant_op.constant([1., 2.])
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                r'Matrix size-incompatible'):
      linalg_impl.matrix_exponential(x)


if __name__ == "__main__":
  test.main()
