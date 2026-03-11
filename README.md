# app-forward-v2

Computes an MEG/EEG forward solution (lead-field matrix) using MNE-Python.

## Overview

The forward solution maps from source space (dipoles on the cortical surface) to sensor space (MEG/EEG channels). It requires a pre-computed source space (`src.fif`), a coregistration transform (`trans.fif`), and a BEM solution (`bem-sol.fif`).

## Inputs

| Input | Datatype | Config key | Required | Description |
|---|---|---|---|---|
| Sensor data | `meeg/mne/evoked`, `meeg/mne/epochs`, or `meeg/mne/raw` | `evoked` / `epochs` / `raw` | Yes (one of) | Provides channel info and projectors. If multiple provided, most downstream used: evoked > epochs > raw |
| Source space | `raw source_space` | `src` | Yes | From app-source-space-v2 |
| Trans | `meeg/mne/trans` | `trans` | Yes | Coregistration transform from app-coreg-v2 |
| BEM | `meg/fif` | `bem` | Yes | BEM solution from app-bem-v2 |

## Output

| File | Description |
|---|---|
| `out_dir/forward-fwd.fif` | Forward solution (lead-field matrix) |

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mindist` | number | `5.0` | Min distance (mm) from sources to inner skull. Removes sources too close to boundary. |

## Pipeline Position

```
app-source-space-v2  →  src.fif   ─┐
app-coreg-v2         →  trans.fif  ├→  app-forward-v2  →  forward-fwd.fif
app-bem-v2           →  bem.fif   ─┘
epochs / evoked / raw ─────────────┘
```

## Fallback

- EEG only + no trans/BEM: sphere head model (less accurate)
- MEG + no trans/BEM: fatal error

## Container

`docker://aunnikri642/app-freesurfer-mne-source-recon` (MNE 1.11)
