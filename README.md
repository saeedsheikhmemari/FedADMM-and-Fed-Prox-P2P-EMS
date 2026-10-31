# FedADMM and Fed-Prox P2P EMS

**Federated deep reinforcement learning for privacy-aware peer-to-peer residential energy management**

> **Repository status:** Manuscript-based project documentation and MATLAB directory structure are available. The MATLAB implementation, experimental inputs, and results have **not yet been uploaded**. This repository does not currently provide runnable results or a reproducible software release.

## Overview

This project investigates a residential energy management system (EMS) that combines **federated deep reinforcement learning (FDRL)**, **Soft Actor-Critic (SAC)** control, **federated ADMM consensus**, and **double-auction peer-to-peer (P2P) electricity trading**. Household agents coordinate flexible resources under dynamic tariffs while retaining local operational trajectories. The manuscript compares the proposed **FedADMM–SAC** coordination method with **FedProx–SAC** and **FedAvg–SAC** baselines.

The manuscript evaluates a **24-hour scheduling horizon** for **320 households** connected to an **IEEE 33-bus radial distribution feeder**, with network constraints for voltage and line loading.

## Methodology at a glance

```text
Household demand / PV / EV / BESS / price signals
                      |
                      v
             Local HEMS SAC agents
       (EV, BESS, trade quantity and bid price)
                      |
                      v
        Double-auction P2P market clearing
         (local matches; grid residuals)
                      |
                      v
       Cost, comfort, and network feedback
                      |
                      v
      Local actor-critic policy improvement
                      |
                      v
        Periodic federated coordination
     +----------------+----------------+
     |                |                |
 FedAvg-SAC       FedProx-SAC      FedADMM-SAC
   baseline         baseline       primal-dual
                                    consensus
```

**Local learning.** Each home is modeled as an autonomous SAC agent, with continuous control of EV charging/discharging, stationary battery operation, P2P bid quantity, and bid price. Local rewards consider energy settlement costs and equipment-related penalties. The agents maintain private replay buffers.

**P2P market.** An hourly double-sided auction matches buyers and sellers based on bid/ask prices and quantities. Unmatched energy is settled with the upstream grid. Dynamic prices are informed by system demand and supply-to-demand conditions.

**Federated coordination.** The proposed FedADMM-SAC formulation synchronizes local actor and critic parameters through consensus penalties and dual updates. Periodic synchronization and partial client participation are intended to reduce communication overhead and mitigate slow-client effects. The comparison methods are FedAvg-SAC and FedProx-SAC.

**Physical feasibility.** Household net injections are assessed with a radial branch-flow representation of the IEEE 33-bus distribution network, including power balance, voltage bounds, and line-current/congestion restrictions.

## Residential case study

| Home category | Count | Installed resources |
| --- | ---: | --- |
| Type 1 | 100 | PV + BESS |
| Type 2 | 100 | PV + EV |
| Type 3 | 50 | PV + EV + BESS |
| Type 4 | 70 | Consumer only |
| **Total** | **320** | **IEEE 33-bus feeder** |

## Repository structure

```text
FedADMM-and-Fed-Prox-P2P-EMS/
├── README.md                  Project overview and current status
├── .gitignore                 MATLAB and local-output exclusions
├── CONTRIBUTING.md            Future contribution and reproduction guidance
├── docs/
│   └── methodology.md         Research workflow and evaluation dimensions
├── MATLAB/
│   ├── agents/                HEMS states, actions, rewards, SAC
│   ├── algorithms/            FedADMM, FedProx, FedAvg coordination
│   ├── market/                Dynamic pricing and double auction
│   ├── network/               IEEE 33-bus power flow/constraints
│   ├── config/                Experiment configuration
│   ├── experiments/           Training, comparisons, evaluation
│   └── utils/                 Shared utilities
├── data/
│   └── README.md              Input-data and privacy notes
├── figures/
│   └── README.md              Figures to add after verification
└── results/
    └── README.md              Result artifacts to add after verification
```

## Reproduction status

**Code release pending.** To reproduce the manuscript, the following will need to be added and verified:

1. MATLAB source (`.m`) files for the SAC agents, federated aggregation, market clearing, network calculations, and experiment entry points.
2. MATLAB version, toolbox dependencies, and any solver requirements.
3. Source and preprocessing details for household load, PV, EV availability, and grid profiles.
4. Random seeds, simulation hyperparameters, communication intervals, and baseline configurations.
5. Instructions for generating figures and numerical tables from original experimental outputs.

Do not interpret the repository scaffold as an executable implementation or an independent verification of manuscript claims.

## Associated manuscript

*Federated Learning-Based Optimization for Secure and Efficient Microgrid Energy Management with Peer-to-Peer Trading, Adaptive Demand Response, and Grid Stability Enhancement* (user-provided manuscript; publication metadata to be added after confirmation).

## Citation

Please cite the related manuscript when its final bibliographic details become available. A machine-readable `CITATION.cff` can be added after the author list, publication venue, and DOI are confirmed.

## License

A software license has **not yet been selected**. Public visibility alone does not grant permission to reuse or redistribute software. Add an appropriate `LICENSE` file before releasing the MATLAB source code for third-party use.
