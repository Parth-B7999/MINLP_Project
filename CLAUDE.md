# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ISyE 823 research project implementing **decision-focused surrogate models for MIQCQP** (Mixed-Integer Quadratically Constrained Quadratic Programs), applied to AC Optimal Power Flow with Unit Commitment (AC OPF+UC).

The core idea: learn an MINLP using a surrogate convex QCQP, then embed that surrogate as a layer in a neural network for decision-focused learning.

## Common Commands

```bash
# Start Julia with the project environment
julia --project

# Install/update dependencies
julia --project -e "using Pkg; Pkg.instantiate()"

# Run the main entry point
julia --project src/main.jl

# From within the Julia REPL
include("src/formulation.jl")
include("src/main.jl")
```

## Architecture

### File Roles

- **`src/formulation.jl`** — Core module. Defines the `MatpowerData` struct for parsing IEEE Matpower network files, and all optimization model builders for AC OPF+UC.
- **`src/main.jl`** — Entry point. Defines `solve_convex_MINLP()`, which iterates over parameter instances and returns a results DataFrame.
- **`src/surrogates.jl`** — Stub (not functional). Intended to embed the convex QCQP as a neural network layer using Flux + DiffOpt for end-to-end decision-focused learning.
- **`src/utils.jl`** — `save_results()` helper to serialize DataFrames to JLD2.
- **`src/old.jl`** — Legacy/reference implementations; not used in the active pipeline.

### Problem Formulations (in `formulation.jl`)

Three model variants are implemented:

| Function | Description |
|---|---|
| `ac_uc()` | Single-period AC-UC (polar or rectangular coordinates) |
| `mp_ac_uc()` | Multi-period AC-UC with a demand curve |
| `convex_ac_uc()` / `build_convex_ac_uc()` | Convex QCQP approximation (active development) |

The **convex formulation** is the main research contribution. It uses rectangular coordinates with auxiliary variables `c_ii`, `c_ij`, `s_ij` (voltage products) and slack variables `xi_c`, `xij_c`, `xij_s` to relax nonlinear power flow constraints into convex ones, following Constante-Flores & Li (2026), equations 4b–4j.

### Decision Variables

- `vr[i,t]`, `vi[i,t]` — Real/imaginary nodal voltages
- `c_ii[i,t]`, `c_ij[e,t]`, `s_ij[e,t]` — Auxiliary voltage product variables
- `u[g,t]` — Binary unit commitment
- `pg[g,t]`, `qg[g,t]` — Active/reactive power generation
- `p_fr/to[e,t]`, `q_fr/to[e,t]` — Branch power flows
- `xi_c`, `xij_c`, `xij_s` — Slack variables for convex relaxation

### Solvers

- **Gurobi** — Required for MIQP/MIQCQP (needs a valid license)
- **Ipopt** — Used for NLP relaxations

### Test Case Data

IEEE Matpower cases in `data/`: `case14.m`, `case30.m`, `case85.m`, `case300.m`. Parsed via PowerModels with standardized cost terms and computed thermal limits.

## Current Status

- Convex QCQP formulation: implemented and functional
- Surrogate neural network module (`surrogates.jl`): **stub only, not working** (see commit "MAIN, SURROGATES DO NOT WORK")
- No formal test suite; validation is done manually or via `examples/case14/build_model.ipynb`
