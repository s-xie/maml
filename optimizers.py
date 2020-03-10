import tensorflow as tf
from tensorflow.python.training import optimizer
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer


class SGD(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, use_locking=False, name="SGD"):
        super(SGD, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._use_locking = use_locking

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)

        eps = 1e-7 #cap for moving average

        m = self.get_slot(var, "m")
        #m_t = m.assign(tf.maximum(beta_t * m + eps, tf.abs(grad)))

        var_update = state_ops.assign_sub(var, lr_t*grad, use_locking=self._use_locking) #Update 'ref' by subtracting 'value
        #Create an op that groups multiple operations.
        #When this op finishes, all ops in input have finished
        return control_flow_ops.group(*[var_update])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
