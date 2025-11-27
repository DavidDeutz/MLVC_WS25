import torch

def pick_device(preferred=("cuda", "mps", "cpu"), cuda_index=0):
    """
    Return a torch.device with priority order given in `preferred`.
    Args:
        preferred (tuple[str]): ordered device backends to try.
        cuda_index (int): which CUDA device index to use if CUDA is available.
    Returns:
        torch.device
    """
    for device in preferred:
        # CUDA (NVIDIA GPU)
        if device == "cuda" and torch.cuda.is_available():
            return torch.device(f"cuda:{cuda_index}")
        # Apple Silicon MPS (Metal Performance Shaders)
        # Note: requires PyTorch built with MPS and macOS 12.3+
        elif device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        # CPU fallback
        elif device == "cpu":
            return torch.device("cpu")
    
    # If no preferred device is available, default to CPU
    return torch.device("cpu")


def print_device_info(device: torch.device):
    """Nice one-liner to see what you got."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        cap  = torch.cuda.get_device_capability(device)
        print(f"Using CUDA device: {name} (capability {cap[0]}.{cap[1]})")
    elif device.type == "mps":
        # PyTorch doesn't expose a name string for MPS; keep it simple.
        print("Using Apple MPS device")
    else:
        print("Using CPU")