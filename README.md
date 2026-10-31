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



