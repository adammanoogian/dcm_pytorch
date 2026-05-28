from __future__ import annotations

import math

import torch
import torch.nn as nn


class ContinuousTimeRNN(nn.Module):
    """Continuous-time RNN with Euler discrete-time integration.

    Implements the CT-RNN dynamics::

        tau * dh/dt = -h + f(W_rec @ h + W_in @ u + b)

    Discretized via Euler integration as::

        h[t+1] = (1 - alpha) * h[t] + alpha * f(W_rec @ h[t] + W_in @ u[t] + b)

    where ``alpha = dt / tau``.

    This matches the Langdon & Engel (2025) ``trainRNNbrain`` formulation.
    Formal reference ID will be assigned in Phase 25 (PUB-03); cite as
    Langdon & Engel (2025) in the interim.

    Parameters
    ----------
    n_input : int
        Input dimension (e.g. neurogym ``obs_size``).
    n_hidden : int
        Number of hidden units H.
    n_output : int
        Output dimension (e.g. neurogym ``act_size``).
    tau : float, optional
        Time constant in normalized units. Default 1.0.
    dt : float, optional
        Integration step size. ``alpha = dt / tau``. Default 0.1.
    activation : str, optional
        Hidden-unit activation. ``"relu"`` (default) or ``"tanh"``.
    noise_std : float, optional
        Standard deviation of additive Gaussian noise injected onto
        hidden states during training. 0.0 disables noise. Default 0.0.

    Attributes
    ----------
    alpha : float
        Euler step fraction ``dt / tau``. Fixed (non-learnable).
    W_rec : nn.Parameter, shape (n_hidden, n_hidden)
        Recurrent weight matrix.
    W_in : nn.Parameter, shape (n_hidden, n_input)
        Input weight matrix.
    W_out : nn.Parameter, shape (n_output, n_hidden)
        Output (readout) weight matrix.
    b : nn.Parameter, shape (n_hidden,)
        Recurrent bias.

    Raises
    ------
    ValueError
        If ``activation`` is not ``"relu"`` or ``"tanh"``.

    Notes
    -----
    ``alpha`` is a fixed scalar, not a learnable parameter (v0.6.0 design).
    Noise injection is only applied when ``self.training is True`` and
    ``noise_std > 0``. The output readout is linear (softmax, if needed,
    is applied by the caller).
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        tau: float = 1.0,
        dt: float = 0.1,
        activation: str = "relu",
        noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.tau = tau
        self.dt = dt
        self.alpha = dt / tau
        self.noise_std = noise_std

        if activation == "relu":
            self.f = torch.relu
        elif activation == "tanh":
            self.f = torch.tanh
        else:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                "Expected 'relu' or 'tanh'."
            )

        # Weight initialisation: Gaussian with std ~ 1/sqrt(fan_in)
        self.W_rec = nn.Parameter(
            torch.empty(n_hidden, n_hidden).normal_(0.0, 1.0 / math.sqrt(n_hidden))
        )
        self.W_in = nn.Parameter(
            torch.empty(n_hidden, n_input).normal_(0.0, 1.0 / math.sqrt(n_input))
        )
        self.W_out = nn.Parameter(
            torch.empty(n_output, n_hidden).normal_(0.0, 1.0 / math.sqrt(n_hidden))
        )
        self.b = nn.Parameter(torch.zeros(n_hidden))

    def forward(
        self,
        u: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run CT-RNN forward pass.

        Parameters
        ----------
        u : torch.Tensor
            Input time series. Accepted shapes:

            - ``(T, B, M_in)`` — batched
            - ``(T, M_in)`` — unbatched; a batch dimension of 1 is inserted.

            Where T = timesteps, B = batch size, M_in = input dimension.
        h0 : torch.Tensor or None, optional
            Initial hidden state, shape ``(B, H)``. If ``None``,
            initialised to zeros on the same device and dtype as ``u``.

        Returns
        -------
        z : torch.Tensor, shape (T, B, M_out)
            Linear readout at each timestep (pre-softmax logits).
        h_traj : torch.Tensor, shape (T, B, H)
            Hidden-state trajectory.
        """
        # Accept unbatched (T, M_in) and promote to (T, 1, M_in)
        if u.dim() == 2:
            u = u.unsqueeze(1)

        T, B, _ = u.shape

        if h0 is None:
            h = torch.zeros(B, self.n_hidden, device=u.device, dtype=u.dtype)
        else:
            h = h0

        h_traj: list[torch.Tensor] = []
        for t in range(T):
            pre_act = h @ self.W_rec.T + u[t] @ self.W_in.T + self.b
            h = (1.0 - self.alpha) * h + self.alpha * self.f(pre_act)
            if self.training and self.noise_std > 0.0:
                h = h + self.noise_std * torch.randn_like(h)
            h_traj.append(h)

        h_traj_t = torch.stack(h_traj, dim=0)  # (T, B, H)
        z = h_traj_t @ self.W_out.T  # (T, B, M_out)
        return z, h_traj_t
