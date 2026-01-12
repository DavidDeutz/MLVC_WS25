# File: vit_classifier.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

########### TO-DO ###########
# Implement the methods marked with "BEGINNING OF YOUR CODE" and "END OF YOUR CODE":
# __init__() in PatchEmbed, forward() in PatchEmbed, 
# __init__() in PositionalEncoding, forward() in PositionalEncoding
# __init__() in ViTClassifier, forward() in ViTClassifier
# Do not change the function signatures
# Do not change any other code
#############################

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=4, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads

        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True
        )
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.last_attn = None  # will store (B, H, S, S)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        assert D == self.d_model
        out, attn = self.mha(x, x, x, need_weights=True, average_attn_weights=False)
        out = self.proj_dropout(out)

        self.last_attn = attn.detach()
        return out


class ViTBlock(nn.Module):
    def __init__(self, d_model=64, mlp_hidden=128, num_heads=4, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.ln1(x))
        h = h + self.mlp(self.ln2(h))
        return h
    
class PatchEmbed(nn.Module):
    """
    Patch embedding using a Conv2d with kernel size and stride equal to patch size.

    Converts an input image into a sequence of patch tokens of dimension D.

    Input
    -----
    x : Tensor, shape (B, C, H, W)

    Output
    ------
    tokens : Tensor, shape (B, N, D)
        Where H' = H // P, W' = W // P, N = H' * W' is the number of patches,
        and D = embed_dim is the embedding dimension.
    """
    def __init__(self, img_size=16, patch_size=4, in_chans=1, embed_dim=64):
        """
        Initialize the patch embedding layer.

        Parameters
        ----------
        img_size : int
            Input image size (assumed square).
        patch_size : int
            Size of each square patch (P x P).
        in_chans : int
            Number of input channels (1 for grayscale, 3 for RGB).
        embed_dim : int
            Dimension D of the patch embeddings.

        Task
        ----
        * Compute grid_h and grid_w = number of patches along each axis.
        * Store total number of patches = grid_h x grid_w.
        * Define a Conv2d projection with:
              in_channels = in_chans
              out_channels = embed_dim
              kernel_size = stride = patch_size
          so that each patch is mapped directly to a D-dimensional token.
        """

        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        raise NotImplementedError("Provide your solution here")
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: split image into patches and project to embeddings.

        Parameters
        ----------
        x : Tensor, shape (B, C, H, W)
            Input batch of images. H and W must equal img_size.

        Returns
        -------
        tokens : Tensor, shape (B, N, D)
            Sequence of patch embeddings, where N = (H // P) x (W // P).

        Task
        ----
        * Apply the Conv2d projection to produce shape (B, D, H', W').
        * Flatten spatial dimensions to length N = H' x W'.
        * Transpose to get (B, N, D), the token sequence.
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        raise NotImplementedError("Provide your solution here")
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        return x


class PositionalEncoding(nn.Module):
    """
    Learnable or sinusoidal positional encodings.
    """
    def __init__(self, seq_len, d_model, learnable=True):
        """
        Initialize positional encodings.

        Parameters
        ----------
        seq_len : int
            Maximum sequence length to support.
        d_model : int
            Embedding dimension (must match model dimension).
        learnable : bool, default=True
            If True, use learnable positional embeddings initialized randomly.
            If False, use the deterministic sinusoidal encoding as in
            "Attention is All You Need".

        Task
        ----
        * For learnable encodings: create a parameter tensor of shape
          (1, seq_len, d_model) that can be trained.
        * For sinusoidal encodings: precompute a matrix of shape
          (1, seq_len, d_model) with alternating sine and cosine terms:
              PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
              PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
          Register this as a non-trainable buffer.
        """
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        raise NotImplementedError("Provide your solution here")
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to the input embeddings.

        Parameters
        ----------
        x : torch.Tensor, shape (B, S, d_model)
            Input embeddings for a batch of sequences,
            where B = batch size, S = sequence length.

        Returns
        -------
        x_pos : torch.Tensor, shape (B, S, d_model)
            Input embeddings with positional encodings added.

        Notes
        -----
        * Adds the stored positional encoding (broadcasted over batch).
        """

        B, S, D = x.shape
        assert S == self.seq_len and D == self.d_model

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        raise NotImplementedError("Provide your solution here")
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        return x_pos


class ViTClassifier(nn.Module):
    """
    ViT-style classifier for 16x16 grayscale images.

    Modes:
      - patchify=True: images -> patch tokens via Conv2d
      - patchify=False: images -> 256 pixel tokens via Linear

    Options:
      - use_cls_token: prepend CLS and pool it
      - pos_emb: None | "learnable" | "sinusoidal"
    """
    def __init__(
        self,
        d_model=64,
        mlp_hidden=128,
        num_classes=2,
        patch_size=4,
        num_heads=4,
        num_blocks=1,
        use_cls_token=False,
        pos_emb=None,       # None, "learnable", "sinusoidal"
        patchify=False
    ):
        """
        Initialize a minimal ViT encoder + classifier head.

        Parameters
        ----------
        d_model : int
            Token embedding dimension D.
        mlp_hidden : int
            Hidden size of the classification head MLP.
        num_classes : int
            Number of output classes.
        patch_size : int
            Square patch edge length P (allowed here: 4 or 8).
        num_heads : int
            Number of attention heads H in each transformer block.
        num_blocks : int
            Number of transformer blocks stacked.
        use_cls_token : bool
            If True, a learnable [CLS] token is prepended and pooled.
        pos_emb : {None, "learnable", "sinusoidal"}
            Type of positional encoding applied to the token sequence.
        patchify : bool
            If True, tokenize image into non-overlapping PxP patches.
            If False, tokenize per pixel (S = 256 tokens).

        Task
        ----
        1) Patch tokenization setup (only when patchify=True):
           - Create a projection that maps 1 input channel to D, using stride=P
             so that each PxP window becomes one token.
           - Compute base_len = number of tokens S = (16 / P)^2.

        2) [CLS] token setup (only when use_cls_token=True):
           - Create a learnable parameter of shape (1, 1, D).
           - Decide total sequence length seq_len = base_len + 1.
           - If no [CLS], use seq_len = base_len.

        Notes
        -----
        - PositionalEncoding is added later in forward, but needs seq_len here.
        - Blocks, normalization, and classifier head are provided for you.
        """

        super().__init__()
        assert patch_size in (4, 8)
        assert pos_emb in (None, "learnable", "sinusoidal")

        self.d_model = d_model
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cls_token = use_cls_token
        self.do_patchify = patchify

        if self.do_patchify:
            self.patch_embed = PatchEmbed(img_size=16, patch_size=patch_size, in_chans=1, embed_dim=d_model)
            base_len = self.patch_embed.num_patches
        else:
            self.patch_conv = None
            base_len = 16 * 16
            self.pixel_embed = nn.Linear(1, d_model, bias=True)

        if use_cls_token:
            # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
            raise NotImplementedError("Provide your solution here")
            # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        else:
            self.cls_token = None
            seq_len = base_len

        if pos_emb is None:
            self.pos_enc = None
        elif pos_emb == "learnable":
            self.pos_enc = PositionalEncoding(seq_len=seq_len, d_model=d_model, learnable=True)
        else:
            self.pos_enc = PositionalEncoding(seq_len=seq_len, d_model=d_model, learnable=False)

        self.blocks = nn.ModuleList([
            ViTBlock(d_model=d_model, mlp_hidden=mlp_hidden, num_heads=num_heads)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_classes),
        )

        self.last_attn = None

    def _embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert an input image batch into a sequence of D-dimensional tokens.

        Input
        -----
        x : Tensor, shape (B, 1, 16, 16)
            Grayscale images.

        Output
        ------
        tokens : Tensor, shape (B, S, D)
            Token sequence where S depends on tokenization mode:
            - patchify=True: S = (16 / P)^2
            - patchify=False: S = 256

        """
        if self.do_patchify:
            tokens = self.patch_embed(x) 
        else:
            B = x.size(0)
            seq = x.view(B, 16 * 16, 1)
            tokens = self.pixel_embed(seq)
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ViT classifier.

        Input
        -----
        x : Tensor, shape (B, 1, 16, 16)

        Output
        ------
        logits : Tensor, shape (B, num_classes)

        Task
        ----

        1) Optional [CLS] prepend:
           - Expand the learned [CLS] to batch size.
           - Concatenate at sequence start to get shape (B, S+1, D).

        2) Pooling for classification:
           - If [CLS] is used,
             take the first token h[:, 0] as the pooled representation.
           - Otherwise, mean-pool over the sequence dimension.

        """
        tokens = self._embed_tokens(x)                       # (B, S, D)
        B, _, D = tokens.shape
        assert D == self.d_model

        if self.use_cls_token:
            # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
            raise NotImplementedError("Provide your solution here")
            # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        h = tokens if self.pos_enc is None else self.pos_enc(tokens)

        for blk in self.blocks:
            h = blk(h)
            self.last_attn = blk.attn.last_attn

        h = self.norm(h)

        if self.use_cls_token:
            # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
            raise NotImplementedError("Provide your solution here")
            # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        else:
            pooled = h.mean(dim=1)

        logits = self.head(pooled)
        return logits
