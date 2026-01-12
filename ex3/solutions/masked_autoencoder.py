import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from solutions.vision_transformer import PositionalEncoding, ViTBlock, MultiHeadSelfAttention, PatchEmbed

########### TO-DO ###########
# Implement the methods marked with "BEGINNING OF YOUR CODE" and "END OF YOUR CODE":
# patchify(), unpatchify(), 
# random_mask_indices(), block_mask_indices(), grid_mask_indices()
# reconstruction_loss()
# Do not change the function signatures
# Do not change any other code
#############################

def assert_shape(x, shape, msg_prefix=""):
    if isinstance(shape, int):
        shape = (shape,)
    assert len(x.shape) == len(shape), f"{msg_prefix} rank mismatch: got {tuple(x.shape)} expected rank {len(shape)}"
    for i, (a, b) in enumerate(zip(x.shape, shape)):
        if b != -1:
            assert a == b, f"{msg_prefix} dim {i} mismatch: got {a} expected {b}"


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Split images into non-overlapping square patches.

    Parameters
    ----------
    images : torch.Tensor, shape (B, 1, 16, 16)
        Batch of grayscale images. Height and width must be 16.
    patch_size : int
        Edge length of each square patch. Allowed values: 4 or 8.

    Returns
    -------
    patches : torch.Tensor, shape (B, N, P)
        Flattened patches where:
          * N = (16 // patch_size) ** 2 is the number of patches per image,
          * P = patch_size * patch_size is the number of pixels in each patch.

    Task
    ----
    * Reshape the image tensor into a grid of patches.
    * Rearrange dimensions so patches are contiguous in memory.
    * Flatten each patch into a 1D vector of length P.
    * Ensure the final shape is (B, N, P).
    """
    B, C, H, W = images.shape
    assert C == 1 and H == 16 and W == 16
    assert patch_size in (4, 8)

    # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    raise NotImplementedError("Provide your solution here")
    # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    return patches


def unpatchify(patches: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Reconstruct images from flattened patches.

    Parameters
    ----------
    patches : torch.Tensor, shape (B, N, P)
        Flattened patches where:
          * N = (16 // patch_size) ** 2 must equal the number of patches per image,
          * P = patch_size * patch_size must equal the patch area.
    patch_size : int
        Edge length of each square patch. Allowed values: 4 or 8.

    Returns
    -------
    images : torch.Tensor, shape (B, 1, 16, 16)
        Reconstructed grayscale images.

    Task
    ----
    * Reshape the flattened patches back into a grid (gh x gw) of square patches.
    * Rearrange axes so spatial dimensions align correctly.
    * Stitch patches together to form the original 16x16 image.
    * Verify output shape is (B, 1, 16, 16).
    """
    B, N, P = patches.shape
    assert patch_size in (4, 8)
    assert P == patch_size * patch_size

    # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    raise NotImplementedError("Provide your solution here")
    # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    return x


def random_mask_indices(num_patches: int, mask_ratio: float) -> torch.Tensor:
    """
    Create a random boolean mask over patches.

    Parameters
    ----------
    num_patches : int
        Total number of patches N in the sequence.
    mask_ratio : float
        Fraction of patches to mask (between 0 and 1).

    Returns
    -------
    mask : torch.Tensor of shape (N,), dtype=bool
        Boolean mask with exactly round(N * mask_ratio) entries set to True.

    Task
    ----
    * Sample k = round(num_patches * mask_ratio).
    * Randomly permute indices from 0..N-1.
    * Mark the first k indices as True in the mask.

    ASCII illustration (N=8, k=3):
        patches: [0][1][2][3][4][5][6][7]
        mask:     T  F  T  F  F  F  T  F
    """
    # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    raise NotImplementedError("Provide your solution here")
    # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    return mask


def block_mask_indices(grid_h: int, grid_w: int, block_h: int, block_w: int) -> torch.Tensor:
    """
    Create a block mask over a 2D patch grid.

    Parameters
    ----------
    grid_h : int
        Height of patch grid.
    grid_w : int
        Width of patch grid.
    block_h : int
        Height of block to mask (≤ grid_h).
    block_w : int
        Width of block to mask (≤ grid_w).

    Returns
    -------
    mask : torch.Tensor of shape (grid_h * grid_w,), dtype=bool
        Boolean mask with a contiguous block set to True, flattened row-major.

    Task
    ----
    * Randomly choose a valid top-left corner inside the grid.
    * Set the rectangle [top:top+block_h, left:left+block_w] to True.
    * Flatten the 2D mask to shape (grid_h*grid_w,).

    ASCII illustration (grid_h=4, grid_w=4, block_h=2, block_w=2):
        grid mask (T=True, F=False):
        F F F F
        F T T F
        F T T F
        F F F F
    """
    assert 1 <= block_h <= grid_h and 1 <= block_w <= grid_w
    # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    raise NotImplementedError("Provide your solution here")
    # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    return mask


def grid_mask_indices(grid_h: int, grid_w: int, step_h: int, step_w: int) -> torch.Tensor:
    """
    Create a grid-like sampling mask.

    Parameters
    ----------
    grid_h : int
        Height of patch grid.
    grid_w : int
        Width of patch grid.
    step_h : int
        Row step size for marking True entries.
    step_w : int
        Column step size for marking True entries.

    Returns
    -------
    mask : torch.Tensor of shape (grid_h * grid_w,), dtype=bool
        Boolean mask with True values in a regular grid pattern.

    Task
    ----
    * Start with all False entries.
    * Mark positions [0::step_h, 0::step_w] as True.
    * Flatten mask to shape (grid_h*grid_w,).

    ASCII illustration (grid_h=5, grid_w=5, step_h=2, step_w=2):
        T F T F T
        F F F F F
        T F T F T
        F F F F F
        T F T F T
    """
    assert step_h >= 1 and step_w >= 1
    # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    raise NotImplementedError("Provide your solution here")
    # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    return mask


class TinyMAE(nn.Module):
    """
    Minimal MAE for 16x16 grayscale images with masking strategies.
    """
    def __init__(
        self,
        d_model=64,
        mlp_hidden=128,
        num_heads=4,
        num_enc_blocks=2,
        num_dec_blocks=1,
        patch_size=4,
        pos_emb="learnable",      # "learnable" or "sinusoidal"
        masking="random",         # "random", "block", "grid"
        mask_ratio=0.5,
        block_h=2,
        block_w=2,
        step_h=2,
        step_w=2
    ):
        super().__init__()
        assert patch_size in (4, 8)
        assert pos_emb in ("learnable", "sinusoidal")
        assert masking in ("random", "block", "grid")

        self.d_model = d_model
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.masking = masking
        self.mask_ratio = float(mask_ratio)
        self.block_h = int(block_h)
        self.block_w = int(block_w)
        self.step_h = int(step_h)
        self.step_w = int(step_w)

        self.patch_embed = PatchEmbed(img_size=16, patch_size=patch_size, in_chans=1, embed_dim=d_model)
        self.grid_h = self.patch_embed.grid_h
        self.grid_w = self.patch_embed.grid_w
        self.N = self.patch_embed.num_patches
        self.P = patch_size * patch_size

        self.enc_pos = PositionalEncoding(seq_len=self.N, d_model=d_model, learnable=(pos_emb == "learnable"))
        self.dec_pos = PositionalEncoding(seq_len=self.N, d_model=d_model, learnable=(pos_emb == "learnable"))

        self.encoder_blocks = nn.ModuleList([
            ViTBlock(d_model=d_model, mlp_hidden=mlp_hidden, num_heads=num_heads)
            for _ in range(num_enc_blocks)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.enc_to_dec = nn.Identity()

        self.decoder_blocks = nn.ModuleList([
            ViTBlock(d_model=d_model, mlp_hidden=mlp_hidden, num_heads=num_heads)
            for _ in range(num_dec_blocks)
        ])
        self.dec_norm = nn.LayerNorm(d_model)
        self.pred = nn.Linear(d_model, self.P)

        self.last_enc_attn = None
        self.last_dec_attn = None

    def _make_mask_batch(self, B: int, device: torch.device) -> torch.Tensor:
        masks = []
        for _ in range(B):
            if self.masking == "random":
                m = random_mask_indices(self.N, self.mask_ratio)
            elif self.masking == "block":
                m = block_mask_indices(self.grid_h, self.grid_w, self.block_h, self.block_w)
            else:
                m = grid_mask_indices(self.grid_h, self.grid_w, self.step_h, self.step_w)
            masks.append(m)
        mask = torch.stack(masks, dim=0).to(device)
        num_masked = mask.sum(dim=1)
        assert torch.all(num_masked > 0), "At least one patch must be masked per sample"
        assert torch.all(num_masked < self.N), "At least one patch must remain visible per sample"
        return mask

    def forward(self, images: torch.Tensor):
        """
        images: (B, 1, 16, 16)
        returns:
          recon: (B, N, P)
          mask:  (B, N) boolean True for masked patches
        """
        assert images.dim() == 4 and images.shape[1:] == (1, 16, 16)
        device = images.device
        B = images.size(0)

        tokens = self.patch_embed(images)                     # (B, N, D)

        mask = self._make_mask_batch(B, device)              # (B, N) True=masked
        tokens_pos = self.enc_pos(tokens)                    # (B, N, D)

        S_vis = (~mask[0]).sum().item()
        x_vis = torch.empty(B, S_vis, self.d_model, device=device)
        keep_indices = []
        for i in range(B):
            keep_i = (~mask[i]).nonzero(as_tuple=False).squeeze(1)
            keep_indices.append(keep_i)
            x_vis[i] = tokens_pos[i, keep_i, :]

        h = x_vis
        for blk in self.encoder_blocks:
            h = blk(h)
            self.last_enc_attn = blk.attn.last_attn
        h = self.enc_norm(h)                                  # (B, S_vis, D)

        dec_input = self.mask_token.expand(B, self.N, self.d_model).clone()
        for i in range(B):
            dec_input[i, keep_indices[i], :] = self.enc_to_dec(h[i])

        dec_input = self.dec_pos(dec_input)                   # (B, N, D)

        z = dec_input
        for blk in self.decoder_blocks:
            z = blk(z)
            self.last_dec_attn = blk.attn.last_attn
        z = self.dec_norm(z)                                  # (B, N, D)

        recon = self.pred(z)                                  # (B, N, P)
        assert_shape(recon, (B, self.N, self.P), "recon")
        return recon, mask

    @torch.no_grad()
    def reconstruct_image(self, recon: torch.Tensor) -> torch.Tensor:
        """
        Reassemble an image from its patch-level reconstruction.

        Parameters
        ----------
        recon : torch.Tensor, shape (B, N, P)
            Reconstructed patches from the decoder,
            where N = number of patches, P = patch_size^2.

        Returns
        -------
        images : torch.Tensor, shape (B, 1, H, W)
            Reconstructed full images, obtained by unpatchifying.

        Notes
        -----
        * Uses the stored patch_size from the class.
        * This is mainly for visualization (to inspect the MAE outputs).
        """
        return unpatchify(recon, self.patch_size)

    def reconstruction_loss(self, images: torch.Tensor, recon: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute mean squared error loss on masked patches only.

        Parameters
        ----------
        images : torch.Tensor, shape (B, 1, H, W)
            Original input images.
        recon : torch.Tensor, shape (B, N, P)
            Reconstructed patches (decoder output).
        mask : torch.Tensor, shape (B, N)
            Boolean mask, True for patches that were masked (to be predicted).

        Returns
        -------
        loss : torch.Tensor, scalar
            Average squared error per pixel over masked patches.

        Task
        ------------------------------------------
        * Convert input images into target patches via patchify (shape (B, N, P)).
        * Compute squared error: (recon - target)^2, shape (B, N, P).
        * Multiply error by mask so only masked patches contribute.
        * Normalize by total number of masked pixels (avoid div-by-zero).
        """
        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        raise NotImplementedError("Provide your solution here")
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        return loss

    @torch.no_grad()
    def mae_encode_features(self, images: torch.Tensor, device="cpu", pool="mean") -> torch.Tensor:
        """
        Extract a single D-dim feature per image from the encoder.
        Uses a fresh random mask per batch call.
        """
        self.eval()
        images = images.to(device, non_blocking=True)

        tokens = self.patch_embed(images)
        tokens_pos = self.enc_pos(tokens)

        B = images.size(0)
        mask = self._make_mask_batch(B, images.device)
        S_vis = (~mask[0]).sum().item()
        x_vis = torch.empty(B, S_vis, self.d_model, device=images.device)
        for i in range(B):
            keep_i = (~mask[i]).nonzero(as_tuple=False).squeeze(1)
            x_vis[i] = tokens_pos[i, keep_i, :]

        h = x_vis
        for blk in self.encoder_blocks:
            h = blk(h)
        h = self.enc_norm(h)
        feats = h.mean(dim=1) if pool != "cls" else h.mean(dim=1)
        feats = F.normalize(feats, dim=1)
        return feats
