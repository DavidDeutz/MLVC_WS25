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
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Linear projections for Q, K, V (all from d_model to d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)

        # Final projection after concatenating heads
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
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
        # 1) Project inputs to Q, K, V
        q = self.q_proj(x)  # (B, S, D)
        k = self.k_proj(x)  # (B, S, D)
        v = self.v_proj(x)  # (B, S, D)

        # 2) Split into heads: (B, S, H, d_head) and move heads forward -> (B, H, S, d_head)
        H = self.num_heads
        dh = self.d_head

        q = q.view(B, S, H, dh).transpose(1, 2)  # (B, H, S, dh)
        k = k.view(B, S, H, dh).transpose(1, 2)  # (B, H, S, dh)
        v = v.view(B, S, H, dh).transpose(1, 2)  # (B, H, S, dh)

        # 3) Scaled dot-product attention
        # scores: (B, H, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh)
        attn = F.softmax(scores, dim=-1)  # softmax over keys dimension

        # store attention weights for visualization
        self.last_attn = attn

        # 4) Apply attention weights to values -> context: (B, H, S, dh)
        context = torch.matmul(attn, v)

        # 5) Concatenate heads back: (B, S, D)
        context = context.transpose(1, 2).contiguous().view(B, S, D)

        # 6) Final linear projection
        out = self.out_proj(context)
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        return out
