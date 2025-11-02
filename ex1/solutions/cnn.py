import math
import numpy as np
from collections import OrderedDict

from .mlp import MultiLayerPerceptron, ACT_FUNCS, act_forward, act_backward, fc_backward

########### TO-DO ###########
# Implement the methods marked with "BEGINNING OF YOUR CODE" and "END OF YOUR CODE"
# _conv2d_forward(), _conv2d_backward()
# _maxpool2d_forward(), _maxpool2d_backward()
# Do not change the function signatures
# Do not change any other code
#############################

# ---------- helpers (vectorized) ----------

def _pad2d(x, pad):
    """
    Apply symmetric zero-padding to a 3D tensor (C, H, W).

    Parameters
    ----------
    x : np.ndarray, shape (C, H, W)
        Input image or feature map.
    pad : int
        Number of zeros to add to each border (top, bottom, left, right).
        No padding is applied across channels.

    Returns
    -------
    xp : np.ndarray, shape (C, H+2*pad, W+2*pad)
        Zero-padded version of the input.
    pad_config : tuple
        The padding configuration passed to np.pad, useful for reversing.
    """
    if pad == 0:
        return x, ((0,0),(0,0),(0,0))
    C, H, W = x.shape
    xp = np.pad(x, ((0,0),(pad,pad),(pad,pad)), mode="constant")
    return xp, ((0,0),(pad,pad),(pad,pad))

def _get_im2col_indices(C, H_p, W_p, k, stride):
    """
    Compute index arrays to convert a padded image into sliding windows.

    This prepares indices so that each kxk region of each channel can be 
    flattened into a column. The stride determines how far the window 
    shifts in each step.

    Parameters
    ----------
    C : int
        Number of channels.
    H_p, W_p : int
        Height and width of the padded image.
    k : int
        Kernel (filter) size, assumed square.
    stride : int
        Step size between adjacent windows.

    Returns
    -------
    i, j, c : np.ndarray
        Arrays of row indices, column indices, and channel indices.
        Together they select all kxk patches per channel.
    H_out, W_out : int
        Spatial dimensions of the output after applying kernel+stride.
    """
    H_out = (H_p - k) // stride + 1
    W_out = (W_p - k) // stride + 1

    i0 = np.repeat(np.arange(k), k)          # (k*k,)
    i0 = np.tile(i0, C)                      # (C*k*k,)
    j0 = np.tile(np.arange(k), k)            # (k*k,)
    j0 = np.tile(j0, C)                      # (C*k*k,)

    i1 = stride * np.repeat(np.arange(H_out), W_out)  # (H_out*W_out,)
    j1 = stride * np.tile(np.arange(W_out), H_out)    # (H_out*W_out,)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)         # (C*k*k, H_out*W_out)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)         # (C*k*k, H_out*W_out)
    c = np.repeat(np.arange(C), k * k).reshape(-1, 1) # (C*k*k, 1)

    return i, j, c, H_out, W_out

def _im2col_from_padded(xp, k, stride):
    """
    Convert a padded image tensor into a 2D matrix of patches (im2col).

    Each kxk patch (per channel) is unrolled into a column. This trick 
    turns convolution into matrix multiplication.

    Parameters
    ----------
    xp : np.ndarray, shape (C, H_p, W_p)
        Padded input tensor.
    k : int
        Kernel size.
    stride : int
        Step size between adjacent windows.

    Returns
    -------
    cols : np.ndarray, shape (C*k*k, H_out*W_out)
        Matrix of flattened patches.
    idx : tuple of (i, j, c)
        Indices used to reconstruct the image in col2im.
    H_out, W_out : int
        Output spatial dimensions for the convolution/pooling result.
    """
    C, H_p, W_p = xp.shape
    i, j, c, H_out, W_out = _get_im2col_indices(C, H_p, W_p, k, stride)
    cols = xp[c, i, j]  # (C*k*k, H_out*W_out)
    return cols, (i, j, c), H_out, W_out

def _col2im_into_padded(cols, xp_shape, idx):
    """
    Inverse of im2col: scatter-add columns back into the padded image.

    This takes gradients (or reconstructed values) arranged in column 
    format and accumulates them into the correct spatial positions of 
    the padded image.

    Parameters
    ----------
    cols : np.ndarray, shape (C*k*k, H_out*W_out)
        Column representation, e.g. from backward pass.
    xp_shape : tuple (C, H_p, W_p)
        Shape of the padded image to scatter into.
    idx : tuple of (i, j, c)
        Indices that specify where each column entry belongs.

    Returns
    -------
    xp : np.ndarray, shape (C, H_p, W_p)
        Padded image with scattered values, summing where windows overlap.
    """
    C, H_p, W_p = xp_shape
    i, j, c = idx
    xp = np.zeros((C, H_p, W_p), dtype=cols.dtype)
    # Accumulate overlapping positions
    np.add.at(xp, (c, i, j), cols)
    return xp

# ---------- convolution (vectorized) ----------

def _conv2d_forward(x, W, b, stride=1, pad=0):
    """
    2D convolution forward pass for a single sample.

    Inputs
    -------
    x : array of shape (C_in, H, W)
        Input feature map per channel.
    W : array of shape (C_out, C_in, k, k)
        Convolution kernels.
    b : array of shape (C_out,)
        Bias per output channel.
    stride : int
        Spatial step between neighboring convolution windows.
    pad : int
        Amount of zero padding applied to top, bottom, left, right.

    Task
    ----
    Compute y = Conv(x; W, b) with stride and symmetric zero padding.
    For each output channel c_out and spatial location (u, v):
        y[c_out, u, v] = sum_{c_in} sum_{i=0}^{k-1} sum_{j=0}^{k-1}
                         x_padded[c_in, u*stride + i, v*stride + j] * W[c_out, c_in, i, j]
                         + b[c_out]

    You should also cache everything needed for the backward pass:
      original x shape, W, b, stride, pad
      any intermediate representation you create for efficient gradient computation
      output spatial sizes H_out and W_out

    Returns
    -------
    y : array of shape (C_out, H_out, W_out)
        Convolved output.
    cache : dict
        Data needed for the backward pass. It must be sufficient to compute
        dL/dx, dL/dW, dL/db.

    Notes
    -----
    Implement as vectorized operations, no explicit loops over pixels.

    """
    # x: (C_in,H,W)
    # W: (C_out,C_in,k,k), b: (C_out,)
    C_out, C_in, k, _ = W.shape

    # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    raise NotImplementedError("Provide your solution here")
    # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    return out, cache

def _conv2d_backward(dout, cache):
    """
    2D convolution backward pass for a single sample.

    Inputs
    -------
    dout : array of shape (C_out, H_out, W_out)
        Upstream gradient dL/dy.
    cache : dict
        The cache produced by _conv2d_forward.

    Task
    ----
    Compute gradients with respect to x, W, b.

    Using y = Conv(x; W, b), for each output location (u, v):
      dL/db[c_out] accumulates dout[c_out, u, v].
      dL/dW[c_out, c_in, i, j] accumulates
          x_padded[c_in, u*stride + i, v*stride + j] * dout[c_out, u, v].
      dL/dx at padded coordinates accumulates
          W[c_out, c_in, i, j] * dout[c_out, u, v]
      at positions aligned with the sliding window. After accumulation,
      remove the padding region to obtain dL/dx with shape equal to x.

    Be careful with:
      correct spatial indexing with stride
      summation over all output locations that reuse the same kernel element
      trimming padded borders when forming dx

    Returns
    -------
    dx : array of shape like x
        Gradient with respect to input.
    dW : array of shape like W
        Gradient with respect to kernels.
    db : array of shape like b
        Gradient with respect to bias.

    Notes
    -----
    Implement as vectorized operations, no explicit loops over pixels.

    """
    # dout: (C_out,H_out,W_out)
    x_shape = cache["x_shape"]
    W = cache["W"]
    b = cache["b"]
    stride = cache["stride"]
    pad = cache["pad"]
    xp_shape = cache["xp_shape"]
    cols = cache["cols"]
    idx = cache["idx"]

    # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    raise NotImplementedError("Provide your solution here")
    # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    return dx.astype(np.float64), dW.astype(np.float64), db.astype(np.float64)

# ---------- maxpooling (vectorized) ----------

def _maxpool2d_forward(x, kernel=2, stride=2):
    """
    2D max pooling forward pass for a single sample.

    Inputs
    -------
    x : array of shape (C, H, W)
        Input feature map.
    kernel : int
        Pooling window size k. Windows are k by k.
    stride : int
        Spatial step between neighboring pooling windows.

    Task
    ----
    For each channel independently and for each window,
    output the maximum value inside that k by k window.

    For each channel c and output location (u, v):
        y[c, u, v] = max_{0 <= i, j < k} x[c, u*stride + i, v*stride + j]

    Cache the data needed for backward:
      input shape, kernel size, stride
      the argmax position inside each window or an equivalent mask
      any indexing you create to reconstruct positions during backward

    Returns
    -------
    y : array of shape (C, H_out, W_out)
        Pooled output.
    cache : dict
        Data required to route gradients back to the maxima in backward.

    Notes
    -----
    Implement as vectorized operations, no explicit loops over pixels.

    """
    # x: (C,H,W)
    C, H, W = x.shape

    # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    raise NotImplementedError("Provide your solution here")
    # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    return out.astype(np.float64), cache

def _maxpool2d_backward(dout, cache):
    """
    2D max pooling backward pass for a single sample.

    Inputs
    -------
    dout : array of shape (C, H_out, W_out)
        Upstream gradient dL/dy.
    cache : dict
        The cache produced by _maxpool2d_forward.

    Task
    ----
    Route each upstream gradient value to the unique input position
    that achieved the maximum within its pooling window. All other
    positions in that window receive zero.

    For each channel c and output location (u, v):
      Let (i*, j*) be the argmax inside the window corresponding to (u, v).
      Then:
        dx[c, u*stride + i*, v*stride + j*] += dout[c, u, v]
      All other entries in that window get zero contribution.

    Return dx with the same shape as the original input.

    Returns
    -------
    dx : array of shape like x
        Gradient with respect to the input of the pooling layer.

    Notes
    -----
    Implement as vectorized operations, no explicit loops over pixels.

    """
    # dout: (C,H_out,W_out)
    C, H, W = cache["x_shape"]
    kernel = cache["kernel"]
    stride = cache["stride"]
    i, j, c = cache["idx"]
    max_idx = cache["max_idx"]
    H_out = cache["H_out"]
    W_out = cache["W_out"]

    # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    raise NotImplementedError("Provide your solution here")
    # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    return dx.astype(np.float64)
    

# ---------- subclass that reuses FC stack from your MLP and adds CNN front ----------

class ConvolutionalNeuralNetwork(MultiLayerPerceptron):
    def __init__(self, conv_spec, fc_layers_spec, input_shape):
        self.conv_layers = []
        self._conv_cache = []
        self._conv_out_shape = None
        self._last_conv_output_shape = None

        self._build_conv_layers(conv_spec)
        super().__init__(fc_layers_spec)

    def _xavier(self, fan_in, fan_out, k=None):
        limit = 1.0 / math.sqrt(fan_out)
        if k is None:
            return np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float64)
        else:
            return np.random.uniform(-limit, limit, (fan_out, fan_in, k, k)).astype(np.float64)

    def _build_conv_layers(self, conv_spec):
        for layer in conv_spec:
            t = layer["type"].lower()
            if t == "conv":
                Cin = int(layer["in_channels"])
                Cout = int(layer["out_channels"])
                k = int(layer.get("kernel", 3))
                s = int(layer.get("stride", 1))
                p = int(layer.get("pad", 0))
                W = self._xavier(Cin, Cout, k=k)
                b = np.zeros((Cout,), dtype=np.float64)
                self.conv_layers.append({"type": "conv", "W": W, "b": b, "stride": s, "pad": p})
            elif t == "act":
                name = layer.get("name", "tanh")
                if name not in ACT_FUNCS:
                    raise ValueError(f"Unknown activation: {name}")
                f, df = ACT_FUNCS[name]
                self.conv_layers.append({"type": "act", "func": f, "dfunc": df})
            elif t == "pool":
                k = int(layer.get("kernel", 2))
                s = int(layer.get("stride", 2))
                self.conv_layers.append({"type": "pool", "kernel": k, "stride": s})
            else:
                raise ValueError(f"Unknown conv layer type: {layer['type']}")

    def forward(self, x_img):
        a = x_img.reshape((1, 16, 16))
        self._conv_cache = []
        for layer in self.conv_layers:
            if layer["type"] == "conv":
                z, cache = _conv2d_forward(a, layer["W"], layer["b"], layer["stride"], layer["pad"])
                self._conv_cache.append({"type": "conv", "cache": cache, "layer": layer})
                a = z
            elif layer["type"] == "act":
                out, cache = act_forward(a, layer["func"], layer["dfunc"])
                self._conv_cache.append({"type": "act", "cache": cache})
                a = out
            else:
                z, cache = _maxpool2d_forward(a, kernel=layer["kernel"], stride=layer["stride"])
                self._conv_cache.append({"type": "pool", "cache": cache})
                a = z

        flat = a.reshape(-1)
        y = super().forward(flat)
        self._last_conv_output_shape = a.shape
        return y

    def backward(self, x_img, pred, gt, lr):
        grad = self.loss.delta(gt, pred)

        # Reuse the same ops as the MLP for the FC stack
        for step in reversed(self._cache):
            if step["kind"] == "ACT":
                grad = act_backward(grad, step["cache"])
            else:
                grad = fc_backward(grad, step["layer"], step["cache"], lr)

        grad = grad.reshape(self._last_conv_output_shape)

        for step in reversed(self._conv_cache):
            if step["type"] == "act":
                grad = act_backward(grad, step["cache"])
            elif step["type"] == "pool":
                grad = _maxpool2d_backward(grad, step["cache"])
            else:
                cache = step["cache"]
                layer = step["layer"]
                dx, dW, db = _conv2d_backward(grad, cache)
                layer["W"] = layer["W"] - lr * dW
                layer["b"] = layer["b"] - lr * db
                grad = dx

    def predict(self, X):
        return self.forward(X)
