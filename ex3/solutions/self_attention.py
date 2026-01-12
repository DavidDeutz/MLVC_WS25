# File: self_attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

########### TO-DO ###########
# Implement the methods marked with "BEGINNING OF YOUR CODE" and "END OF YOUR CODE":
# __init__(), forward()
# Do not change the function signatures
# Do not change any other code
#############################

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model=64, num_heads=4):
        """
        Multi-Head Self-Attention (MHSA) over a sequence of token embeddings.

        Purpose
        -------
        Implements the "scaled dot-product attention" with multiple heads,
        allowing the model to jointly attend to information from different
        representation subspaces.

        Inputs
        ------
        x : torch.Tensor, shape (B, S, d_model)
            Batch of input sequences.
            B = batch size, S = sequence length, d_model = embedding dimension.

        Outputs
        -------
        out : torch.Tensor, shape (B, S, d_model)
            Sequence of transformed embeddings after self-attention.

        Stored Attributes
        -----------------
        d_model : int
            Dimension of the input embeddings (size of each token vector).
        num_heads : int
            Number of attention heads.
        d_head : int
            Dimension of each attention head (d_model / num_heads).

        q_proj : nn.Linear
            Linear layer projecting inputs to queries (Q).
        k_proj : nn.Linear
            Linear layer projecting inputs to keys (K).
        v_proj : nn.Linear
            Linear layer projecting inputs to values (V).
        out_proj : nn.Linear
            Linear layer combining outputs from all heads back to d_model.

        last_attn : torch.Tensor, shape (B, H, S, S)
            Stores the attention weights (softmax probabilities) from
            the most recent forward pass. Useful for visualization and analysis.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        raise NotImplementedError("Provide your solution here")
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        self.last_attn = None  # (B, H, S, S)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-head self-attention.

        Inputs
        ------
        x : torch.Tensor, shape (B, S, d_model)
            Input sequence of embeddings.
            B = batch size, S = sequence length, d_model = model dimension.

        Computation
        -----------
        1. Project inputs into query, key, and value spaces:
              Q = x W_Q,   K = x W_K,   V = x W_V
          Reshape so each of H heads has dimension d_head = d_model / H.

        2. Compute scaled dot-product attention for each head:
              scores = Q K^T / sqrt(d_head)
              A = softmax(scores)   along the last dimension.

        3. Weight values:
              context = A V

        4. Concatenate the heads:
              context ∈ R^{B × S × d_model}

        5. Apply a final linear projection to produce the output.

        Outputs
        -------
        out : torch.Tensor, shape (B, S, d_model)
            Sequence of embeddings after attention mixing.

        Notes
        -----
        * Stores the attention matrix A in self.last_attn for later plotting.
        * This is the standard scaled dot-product multi-head self-attention
          as introduced in the Transformer architecture.
        """
        B, S, D = x.shape
        assert D == self.d_model

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        raise NotImplementedError("Provide your solution here")
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        return out
