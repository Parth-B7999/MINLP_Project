"""
src/model.py
QCAC_Surrogate neural network — moved here from the notebook so that
ProcessPoolExecutor worker processes can import it cleanly.
"""

import torch
import torch.nn as nn


class QCAC_Surrogate(nn.Module):
    """
    Multi-head surrogate for the AC Unit Commitment problem.

    Predicts four quantities from a load profile (Pd, Qd):
      - u_pred   : continuous generator commitment probabilities  [N_g]
      - vr_base  : real-part voltage expansion points            [N_b]
      - vi_base  : imag-part voltage expansion points            [N_b]
      - rho      : penalty weight for the QCAC slack             [1]
      - A_cut    : learned integer-cut matrix                    [K × N_g]
      - b_cut    : learned integer-cut RHS                       [K]
    """

    def __init__(self, num_buses: int, num_gens: int, num_cuts: int = 5):
        super().__init__()
        self.num_buses = num_buses
        self.num_gens  = num_gens
        self.num_cuts  = num_cuts

        input_dim = num_buses * 2

        # Shared feature extractor
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Head 1 — Voltage Taylor expansion points
        self.head_v    = nn.Linear(64, num_buses * 2)

        # Head 2 — Dynamic rho penalty (scalar, strictly positive)
        self.head_rho  = nn.Linear(64, 1)

        # Head 3 — Learned integer cuts  (A: K×N_g,  b: K)
        self.head_cuts = nn.Linear(64, (num_cuts * num_gens) + num_cuts)

        # Initialize cuts to be inactive (0*u <= 1)
        nn.init.zeros_(self.head_cuts.weight)
        nn.init.constant_(self.head_cuts.bias, 1.0)
        
        # Initialize rho to the interior feasible region to avoid boundary gradient death
        nn.init.constant_(self.head_rho.bias, 10000.0)

    def forward(self, pd_qd_tensor):
        feat = self.trunk(pd_qd_tensor)

        v_out   = self.head_v(feat)
        vr_base = v_out[:, :self.num_buses]
        vi_base = v_out[:, self.num_buses:]

        rho     = nn.functional.softplus(self.head_rho(feat))

        cuts_out = self.head_cuts(feat)
        A_cut    = cuts_out[:, :self.num_cuts * self.num_gens].view(-1, self.num_cuts, self.num_gens)
        b_cut    = cuts_out[:, self.num_cuts * self.num_gens:]

        return vr_base, vi_base, rho, A_cut, b_cut
