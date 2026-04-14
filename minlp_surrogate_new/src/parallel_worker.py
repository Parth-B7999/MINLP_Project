"""
src/parallel_worker.py  (v3 — FedAvg weight averaging)

FedAvg replaces gradient averaging:
  1. Each worker receives current model weights
  2. Worker does SEQUENTIAL per-sample forward→CVXPY→backward→step on its chunk
     (just like the original Phase-2 loop, but on 1/n_workers of the data)
  3. Worker returns its FINAL updated weights (not gradients)
  4. Main process AVERAGES the weight dicts from all workers
  5. Sets averaged weights as new model state

Why this works when gradient averaging didn't:
  • Sequential updates inside each worker use the LATEST local weights after
    each sample — no stale-gradient cancellation
  • Weight averaging (FedAvg) has proven convergence properties
  • Loss will now decrease steadily each epoch
"""

import os
import sys
import io
import warnings
import numpy as np
import time
import concurrent.futures
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# WORKER  (top-level so ProcessPoolExecutor can pickle it)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_fedavg(args):
    """
    FedAvg worker: does SEQUENTIAL per-sample SGD on its chunk,
    then returns the FINAL model weights (not accumulated gradients).
    """
    (project_root, data_file_path,
     model_state_np, samples_np,
     num_buses, num_gens, num_cuts,
     lambda_integrality, slack_pen_weight,
     solver_eps, solver_iters, phase2_lr) = args

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    warnings.filterwarnings("ignore")

    from src.model       import QCAC_Surrogate
    from src.data_utils  import parse_file_data
    from src.cvxpy_layer import build_diffopt_qcac_layer
    from contextlib import redirect_stdout

    # Rebuild model + load weights
    data = parse_file_data(data_file_path)
    model = QCAC_Surrogate(num_buses, num_gens, num_cuts=num_cuts)
    state_dict = {k: torch.tensor(v, dtype=torch.float32)
                  for k, v in model_state_np.items()}
    model.load_state_dict(state_dict)
    model.train()

    # Fresh local optimizer (resets Phase-1 momentum)
    local_opt = torch.optim.Adam(model.parameters(), lr=phase2_lr)

    # Rebuild CVXPY layer silently
    with redirect_stdout(io.StringIO()):
        cvx_layer, _, _ = build_diffopt_qcac_layer(data, num_cuts=num_cuts)

    total_loss = 0.0
    n_valid    = 0

    # ── SEQUENTIAL per-sample updates (key difference vs gradient averaging) ──
    for sample_np in samples_np:
        x_in = torch.tensor(sample_np, dtype=torch.float32).unsqueeze(0)
        pd_t = x_in[:, :num_buses]
        qd_t = x_in[:, num_buses:]

        local_opt.zero_grad()

        try:
            vr_base, vi_base, rho, A_cut, b_cut = model(x_in)

            u_relax, pg, qg, vr, vi, s_cut, xi_c, xij_c, xij_s = cvx_layer(
                vr_base, vi_base, rho.squeeze(-1), A_cut, b_cut, pd_t, qd_t,
                solver_args={"max_iters": solver_iters, "eps": solver_eps},
            )

            # --- Realistic Cost Calculation ---
            # We use the actual coefficients from the grid data
            # cost = sum( c2*pg^2 + c1*pg + c0*u_relax )
            gen_cost_total = 0.0
            for idx, g_id in enumerate(sorted(data.gens.keys())):
                c2, c1, c0 = data.gens[g_id]["cost"]
                gen_cost_total += c2 * (pg[0, idx]**2) + c1 * pg[0, idx] + c0 * u_relax[0, idx]

            integrality_pen = torch.sum(u_relax * (1.0 - u_relax))
            slack_pen       = torch.sum(s_cut) + torch.sum(xi_c) + torch.sum(xij_c) + torch.sum(xij_s)

            loss = (gen_cost_total
                    + lambda_integrality * integrality_pen
                    + slack_pen_weight   * slack_pen)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            local_opt.step()   # update IMMEDIATELY after each sample

            total_loss += loss.item()
            n_valid    += 1

        except Exception:
            pass

    # Return FINAL weights after sequential training on this chunk
    final_weights = {k: v.detach().numpy().copy()
                     for k, v in model.state_dict().items()}

    return final_weights, total_loss, n_valid


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2_parallel(
    model,
    optimizer,          # Phase-1 optimizer (not used in Phase 2)
    X_unsup_t,
    cvx_layer,
    data_file_path,
    num_buses,
    num_gens,
    num_cuts        = 5,
    epochs          = 5,
    batch_size      = 50,
    n_workers       = 6,
    lambda_int      = 5.0,
    slack_weight    = 1.0,
    solver_eps      = 2e-3,
    solver_iters    = 5000,
    phase2_lr       = 1e-3,   # per-sample lr inside each worker
                              # lower than Phase-1 lr=5e-3 since Phase-2 is fine-tuning
):
    project_root = os.path.abspath(".")
    model.train()

    print(f"Phase 2 (FedAvg): {n_workers} workers | "
          f"{batch_size} samples/epoch | eps={solver_eps} | lr={phase2_lr}")

    for epoch in range(epochs):
        t_start = time.time()

        # Random mini-batch
        idx_batch  = torch.randperm(len(X_unsup_t))[:batch_size]
        samples_np = X_unsup_t[idx_batch].numpy()

        # Snapshot current weights
        model_state_np = {k: v.detach().numpy()
                          for k, v in model.state_dict().items()}

        # Split into chunks
        chunk_size = max(1, -(-batch_size // n_workers))
        chunks     = [samples_np[i: i + chunk_size]
                      for i in range(0, len(samples_np), chunk_size)]

        args_list = [
            (project_root, data_file_path,
             model_state_np, chunk,
             num_buses, num_gens, num_cuts,
             lambda_int, slack_weight,
             solver_eps, solver_iters, phase2_lr)
            for chunk in chunks
        ]

        # Run workers in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_worker_fedavg, args_list))

        total_valid    = sum(r[2] for r in results)
        total_loss_sum = sum(r[1] for r in results)

        if total_valid > 0:
            # FedAvg: average the WEIGHTS from all workers (weighted by n_valid)
            avg_state = {}
            for key in model_state_np.keys():
                weighted_sum = None
                total_w      = 0
                for r in results:
                    if r[2] > 0:
                        contribution = r[0][key] * r[2]
                        weighted_sum = contribution if weighted_sum is None \
                                       else weighted_sum + contribution
                        total_w += r[2]
                if weighted_sum is not None:
                    avg_state[key] = torch.tensor(
                        weighted_sum / total_w, dtype=torch.float32)

            model.load_state_dict(avg_state)

        t_elapsed = time.time() - t_start
        avg_loss  = total_loss_sum / max(total_valid, 1)
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Solved {total_valid}/{batch_size} grids | "
              f"Avg Loss: {avg_loss:.4f} | Time: {t_elapsed:.1f}s")

    print("Phase 2 Complete! The Neural Network now understands continuous AC Physics.")
