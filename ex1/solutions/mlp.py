import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

########### TO-DO ###########
# Implement the methods marked with "BEGINNING OF YOUR CODE" and "END OF YOUR CODE"
# loss(), delta(), sigmoid(), dsigmoid(), 
# fc_forward(), fc_backward(), act_forward(), act_backward()
# Do not change the function signatures
# Do not change any other code
#############################


class SquareLoss(object):
    """
    Square loss function.
    """

    def __init__(self):
        pass

    def loss(self, y_true, y_pred):
        """
        Compute the squared error loss between true and predicted values.

        The loss is defined as half the squared difference between the 
        prediction and the ground truth, applied elementwise.

        Intuitively:
        * Take the difference between prediction and target.
        * Square this difference so that errors are always positive.
        * Multiply by 1/2 (a constant factor that simplifies derivatives).

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth values (targets), shape matches predictions.
        y_pred : np.ndarray
            Predicted values from the network.

        Returns
        -------
        np.ndarray
            Squared error for each element of the prediction.
        """
        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        raise NotImplementedError("Provide your solution here")
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    def delta(self, y_true, y_pred):
        """
        Compute the derivative of the squared error loss with respect 
        to the predictions.

        Starting from the squared error definition, the gradient should:
        * compare the prediction to the target,
        * reflect how the prediction must change to reduce the error,
        * have the same shape as the prediction.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth values (targets), shape matches predictions.
        y_pred : np.ndarray
            Predicted values from the network.

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to `y_pred`.
        """
        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        raise NotImplementedError("Provide your solution here")
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    def calculate_accuracy(self, y_true, y_pred):
        """
        Computes the accuracy of the model.

        Args:
            y_pred (numpy.ndarray): Predicted values.
            y (numpy.ndarray): True values.

        Returns:
            float: Accuracy value.
        """
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)

        assert y_true.size == y_pred.size
        y_pred = y_pred > 0.5
        return (y_true == y_pred).sum().item() / y_true.size

# ---------------------- Activations ----------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Apply the logistic sigmoid activation function elementwise.

    Conceptually:
      * For each input value, take its negative.
      * Exponentiate that value.
      * Add 1.
      * Take the reciprocal.

    This maps any real number to the open interval (0, 1).

    Parameters
    ----------
    x : np.ndarray
        Input array of any shape containing real values.

    Returns
    -------
    np.ndarray
        Output array of the same shape as `x`, where each element has
        been transformed by sigma(x).
    """
    # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    raise NotImplementedError("Provide your solution here")
    # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

def dsigmoid(a: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the sigmoid activation function.

    If the forward pass produced activations a = sigma(x),
    then the derivative can be written directly in terms of a:
      * Multiply a by (1 - a), elementwise.

    This gives the slope of the sigmoid at each activation value.

    Parameters
    ----------
    a : np.ndarray
        Array of activations (output of the sigmoid function).

    Returns
    -------
    np.ndarray
        Array of the same shape as `a`, containing the derivative values.
    """
    # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    raise NotImplementedError("Provide your solution here")
    # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def dtanh(a: np.ndarray) -> np.ndarray:
    return 1.0 - a**2

ACT_FUNCS = {
    "sigmoid": (sigmoid, dsigmoid),
    "tanh": (tanh, dtanh),
}

# ------------------Forward & Backward --------------------

def fc_forward(a, layer):
    """
    Forward pass through a fully connected (linear) layer.

    Mathematically:
        z = x W + b

    where
      * x = input vector of shape (in_dim,)
      * W = weight matrix of shape (in_dim, out_dim)
      * b = bias vector of shape (out_dim,)

    Parameters
    ----------
    a : np.ndarray, shape (in_dim,)
        Input activation vector from the previous layer.
    layer : dict
        Dictionary with keys:
          "type": "FC"
          "W": weight matrix
          "b": bias vector

    Returns
    -------
    z : np.ndarray, shape (out_dim,)
        Linear output before activation.
    cache : dict
        Contains values needed for backward pass (e.g. input vector).
    """
    # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    raise NotImplementedError("Provide your solution here")
    # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    return z, cache

def fc_backward(grad, layer, cache, lr):
    """
    Backward pass for a fully connected (linear) layer.

    Mathematically, if z = x W + b, then for upstream gradient g = ∂L/∂z:
      * dW = xᵀ g
      * db = g
      * grad_prev = g Wᵀ

    Parameter update (SGD):
      W ← W - lr * dW
      b ← b - lr * db

    Parameters
    ----------
    grad : np.ndarray, shape (out_dim,)
        Gradient of loss with respect to layer output.
    layer : dict
        Dictionary with keys "W" and "b"; updated in place.
    cache : dict
        Must contain the forward input vector x.
    lr : float
        Learning rate.

    Returns
    -------
    grad_prev : np.ndarray, shape (in_dim,)
        Gradient of loss with respect to input x.
    """
    x = cache["x"]                               # (in_dim,)
    W = layer["W"]                               # (in_dim, out_dim)

    # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    raise NotImplementedError("Provide your solution here")
    # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    return grad_prev

def act_forward(a, func, dfunc):
    """
    Forward pass through an elementwise non-linear activation.

    Applies the activation function φ to each element of the input:
        out = φ(a)

    Parameters
    ----------
    a : np.ndarray
        Input tensor, can be vector (for FC) or multi-dimensional (for CNN).
    func : callable
        Activation function φ, applied elementwise.
    dfunc : callable
        Derivative of activation function φ', used for backward pass.

    Returns
    -------
    out : np.ndarray, same shape as a
        Activated output.
    cache : dict
        Contains activation derivative function and the output,
        used for computing gradients during backprop.
    """

    # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    raise NotImplementedError("Provide your solution here")
    # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    return out, cache

def act_backward(grad, cache):
    """
    Backward pass through an activation function.

    Given upstream gradient g and activation derivative φ'(a),
    the local gradient is:
        g_out = g ⊙ φ'(a)

    Parameters
    ----------
    grad : np.ndarray, same shape as cache["a"]
        Upstream gradient ∂L/∂out.
    cache : dict
        From act_forward, containing "dfunc" and activation output "a".

    Returns
    -------
    grad_out : np.ndarray
        Gradient of loss with respect to activation input a.
    """

    # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    raise NotImplementedError("Provide your solution here")
    # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    return grad_out


# ------------------------- Model -------------------------

class MultiLayerPerceptron:
    """
    Multi-layer perceptron for binary classification.

    Initialize with an ordered dict like:
        {
            "FC1": (in_dim, hidden_dim),
            "Sigmoid1": (),
            "FC2": (hidden_dim, hidden_dim2),
            "Sigmoid2": (),
            "FC3": (hidden_dim2, out_dim),
            "SigmoidOut": ()
        }
    """

    def __init__(self, layers_spec):
        self.layers = []
        self.bias_hidden_value = 0
        self.loss = SquareLoss()

        def _add_fc(in_dim, out_dim):
            in_dim, out_dim = int(in_dim), int(out_dim)
            W = self.init_weights(in_dim, out_dim)
            b = self.init_bias(out_dim)
            self.layers.append({"type": "FC", "W": W, "b": b})

        for i, layer in enumerate(layers_spec):
            if not isinstance(layer, dict):
                raise ValueError(f"Layer at index {i} must be a dict")
            t = str(layer.get("type", "")).lower()
            if t in ("fc", "linear", "conv"):
                if "in_dim" in layer and "out_dim" in layer:
                    in_dim, out_dim = layer["in_dim"], layer["out_dim"]
                elif "in_channels" in layer and "out_channels" in layer:
                    in_dim, out_dim = layer["in_channels"], layer["out_channels"]
                else:
                    raise ValueError(
                        f'Layer at index {i} requires either ("in_dim","out_dim") or ("in_channels","out_channels")'
                    )
                _add_fc(in_dim, out_dim)
            elif t in ("act", "activation"):
                act_name = layer.get("name", None)
                if act_name not in ACT_FUNCS:
                    raise ValueError(f"Unknown activation: {act_name}")
                f, df = ACT_FUNCS[act_name]
                self.layers.append({"type": "ACT", "func": f, "dfunc": df})
            else:
                raise ValueError(f"Unknown layer type: {layer.get('type')}")

        self.loss_train_plot = []
        self.loss_test_plot = []
        self.acc_train_plot = []
        self.acc_test_plot = []

    def init_bias(self, x, mode="default"):
        if mode == "random":
            bias = np.random.random((x,)).astype(np.float64) * 0.001
        elif mode == "xavier":
            limit = 1.0 / math.sqrt(x)
            bias = np.random.uniform(-limit, limit, (x,)).astype(np.float64)
        else:
            bias = np.zeros((x,), dtype=np.float64)
        return bias

    def init_weights(self, x, y, random=False):
        if random:
            weight = np.random.random((x, y)).astype(np.float32) * 0.001
        else:
            limit = 1 / math.sqrt(y)
            weight = np.random.uniform(-limit, limit, (x, y))
        return weight

    def forward(self, inputs):
        """
        Sequential FC + ACT using shared ops. Caches per-sample steps in self._cache.
        """
        a = inputs
        self._cache = []

        for layer in self.layers:
            if layer["type"] == "FC":
                z, cache = fc_forward(a, layer)
                self._cache.append({"kind": "FC", "cache": cache, "layer": layer})
                a = z
            else:
                out, cache = act_forward(a, layer["func"], layer["dfunc"])
                self._cache.append({"kind": "ACT", "cache": cache})
                a = out

        return a

    def backward(self, input, pred, gt, lr):
        """
        Backprop using shared ops. Updates parameters in place.
        """
        grad = self.loss.delta(gt, pred)

        for step in reversed(self._cache):
            if step["kind"] == "ACT":
                grad = act_backward(grad, step["cache"])
            else:
                grad = fc_backward(grad, step["layer"], step["cache"], lr)

    def fit(self, X_train, y_train, X_test, y_test, epochs, lr):
        n = len(X_train)

        self.loss_train_plot = []
        self.loss_test_plot = []
        self.acc_train_plot = []
        self.acc_test_plot = []

        X_train = X_train / 255
        X_test = X_test / 255

        training_idx = np.arange(n)
        pbar = trange(epochs)

        for current_epoch in pbar:
            epoch_loss_train, epoch_acc_train = [], []
            epoch_loss_test, epoch_acc_test = [], []

            np.random.shuffle(training_idx)

            for idx in training_idx:
                pred = self.forward(X_train[idx])
                epoch_loss_train.append(self.loss.loss(y_train[idx], pred))
                epoch_acc_train.append(self.loss.calculate_accuracy(y_train[idx], pred))
                self.backward(X_train[idx], pred, y_train[idx], lr)

            for idx, inputs in enumerate(X_test):
                pred = self.forward(inputs)
                epoch_loss_test.append(self.loss.loss(y_test[idx], pred))
                epoch_acc_test.append(self.loss.calculate_accuracy(y_test[idx], pred))

            pbar.set_description(
                f"Epoch {current_epoch + 1} - Loss (Train) {np.mean(epoch_loss_train):.5f}"
            )
            self.loss_train_plot.append(np.mean(epoch_loss_train))
            self.loss_test_plot.append(np.mean(epoch_loss_test))
            self.acc_train_plot.append(np.mean(epoch_acc_train))
            self.acc_test_plot.append(np.mean(epoch_acc_test))

    def predict(self, X):
        X = X / 255
        return self.forward(X)