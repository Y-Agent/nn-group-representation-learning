"""Wide neural network with spherical parameterization.

The model is parameterized by:
- scale_in, scale_out: Shared scalar scales
- W_in: Each positional row block lies on its own unit sphere
- W_out: Direction matrix with columns on the unit sphere
"""

from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WideNetworkScaleSphere(nn.Module):
    """Wide network with separate scale (shared) and direction (unit sphere) parameters.

    Architecture:
        input -> one_hot_embed -> scale_in * (embed @ W_in.T) -> activation -> scale_out * (h @ W_out.T) / M -> output

    The key insight is that W_in positional blocks and W_out columns lie
    on the unit sphere, while scale_in and scale_out are shared scalars that
    control the overall magnitude.
    """

    def __init__(
        self,
        d_vocab: int,
        width: int,
        act_type: Literal["ReLU", "Quad", "Abs", "GeLU"] = "Quad",
        init_scale: float = 0.5,
        share_embed: bool = True,
    ):
        """Initialize the network.

        Args:
            d_vocab: Vocabulary size (input/output dimension)
            width: Number of neurons (M)
            act_type: Activation function type
            init_scale: Initial value for scale_in and scale_out
            share_embed: If True, sum one-hot embeddings (commutative: e_x + e_y).
                        If False, concatenate embeddings (non-commutative: [e_x; e_y]).
        """
        super().__init__()
        if d_vocab <= 0 or width <= 0:
            raise ValueError("Vocabulary size and width must be positive")
        self.d_vocab = d_vocab
        self.width = width
        self.act_type = act_type
        self.share_embed = share_embed

        # Input dimension depends on embedding mode
        self.d_input = d_vocab if share_embed else 2 * d_vocab

        # Direction parameters (on unit sphere)
        # W_in: each positional block of each row has unit norm
        w_in = torch.randn(width, self.d_input)
        blocks = w_in.reshape(width, -1, d_vocab)
        w_in = (blocks / blocks.norm(dim=-1, keepdim=True)).reshape(width, self.d_input)
        self.W_in = nn.Parameter(w_in)

        # W_out: each column has unit norm
        w_out = torch.randn(d_vocab, width)
        w_out = w_out / w_out.norm(dim=0, keepdim=True)
        self.W_out = nn.Parameter(w_out)

        # Scale parameters (single scalar each, stored as 1D tensor)
        self.scale_in = nn.Parameter(torch.tensor([float(init_scale)]))
        self.scale_out = nn.Parameter(torch.tensor([float(init_scale)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 2) containing token indices

        Returns:
            Logits of shape (batch_size, d_vocab)
        """
        # One-hot embedding
        one_hot = F.one_hot(x, num_classes=self.d_vocab).to(self.W_in.dtype)  # (batch, 2, d_vocab)

        if self.share_embed:
            # Sum embeddings (commutative): e_x + e_y
            embed = one_hot.sum(dim=1)  # (batch, d_vocab)
        else:
            # Concatenate embeddings (non-commutative): [e_x; e_y]
            embed = one_hot.view(x.shape[0], -1)  # (batch, 2*d_vocab)

        # Hidden layer with scale
        h = self.scale_in * (embed @ self.W_in.T)  # (batch, width)

        # Activation
        if self.act_type == "ReLU":
            h = F.relu(h)
        elif self.act_type == "Quad":
            h = h**2
        elif self.act_type == "Abs":
            h = torch.abs(h)
        elif self.act_type == "GeLU":
            h = F.gelu(h)
        else:
            raise ValueError(f"Unknown activation: {self.act_type}")

        # Output with scale (normalized by width)
        out = self.scale_out * (h @ self.W_out.T) / self.width  # (batch, d_vocab)
        return out

    def input_blocks(self):
        """Separate positional spheres; shared input has one block."""
        return self.W_in.reshape(self.width, -1, self.d_vocab)

    @torch.no_grad()
    def normalize_directions(self) -> None:
        """Retract each positional embedding and output onto its own sphere."""
        blocks = self.input_blocks()
        if torch.any(blocks.norm(dim=-1) == 0) or torch.any(self.W_out.norm(dim=0) == 0):
            raise ValueError("Cannot normalize a zero direction")
        blocks.div_(blocks.norm(dim=-1, keepdim=True))
        self.W_out.div_(self.W_out.norm(dim=0, keepdim=True))

    @torch.no_grad()
    def project_direction_gradients(self) -> None:
        """Project each sphere separately. Shared theta's 1/2 is applied by LR."""
        if self.W_in.grad is not None:
            w = self.input_blocks()
            g = self.W_in.grad.reshape_as(w)
            g.sub_((g * w).sum(dim=-1, keepdim=True) * w)
        if self.W_out.grad is not None:
            g, w = self.W_out.grad, self.W_out
            g.sub_((g * w).sum(dim=0, keepdim=True) * w)

    def effective_scale(self):
        """For quadratic activation, the paper's a = scale_out * scale_in**2."""
        if self.act_type != "Quad":
            raise ValueError("Effective quadratic scale requires Quad activation")
        return self.scale_out * self.scale_in.square()

    def get_direction_params(self):
        """Get parameters for direction (W_in, W_out)."""
        return [self.W_in, self.W_out]

    def get_scale_params(self):
        """Get parameters for scale (scale_in, scale_out)."""
        return [self.scale_in, self.scale_out]

    def get_state(self) -> Dict[str, torch.Tensor]:
        """Get a copy of current model state."""
        return {
            "W_in": self.W_in.detach().cpu().clone(),
            "W_out": self.W_out.detach().cpu().clone(),
            "scale_in": self.scale_in.detach().cpu().clone(),
            "scale_out": self.scale_out.detach().cpu().clone(),
        }

    def load_state(
        self, state: Dict[str, torch.Tensor], device: Optional[torch.device] = None
    ) -> None:
        """Load model state from a state dict.

        Args:
            state: State dictionary from get_state()
            device: Device to load tensors to (default: current device)
        """
        if device is None:
            device = self.W_in.device

        expected = {
            name: getattr(self, name) for name in ("W_in", "W_out", "scale_in", "scale_out")
        }
        for name, parameter in expected.items():
            if state[name].shape != parameter.shape:
                raise ValueError(
                    f"Checkpoint shape mismatch for {name}: {state[name].shape} != {parameter.shape}"
                )
        # Validate all shapes before mutating any parameter; retain parameter
        # identity so existing optimizer references remain valid.
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(state[name].to(device=device, dtype=parameter.dtype))

    def get_direction_norms(self) -> Dict[str, torch.Tensor]:
        """Get L2 norms of direction vectors (should be ~1 after normalization)."""
        return {
            "W_in_norms": self.input_blocks().detach().norm(dim=-1),  # (width,)
            "W_out_norms": self.W_out.data.norm(dim=0),  # (width,)
        }

    def __repr__(self) -> str:
        return (
            f"WideNetworkScaleSphere(\n"
            f"  d_vocab={self.d_vocab},\n"
            f"  width={self.width},\n"
            f"  act_type='{self.act_type}',\n"
            f"  share_embed={self.share_embed},\n"
            f"  d_input={self.d_input},\n"
            f"  scale_in={self.scale_in.item():.4f},\n"
            f"  scale_out={self.scale_out.item():.4f}\n"
            f")"
        )
