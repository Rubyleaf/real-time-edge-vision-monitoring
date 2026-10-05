# Real-Time Restricted Area Monitoring on Resource-Constrained GPU

## Objective
Build a real-time video monitoring system that detects target objects
inside a polygon restricted zone, logs events, and measures performance
on an NVIDIA RTX 2060 SUPER with 8 GiB VRAM.

## Status
Environment validation completed. Application implementation has not started.

## Verified Environment
- Windows 11 Home with WSL2 Ubuntu 24.04
- Python 3.12.3
- PyTorch 2.8.0+cu126
- Windows NVIDIA driver 560.94
- NVIDIA GeForce RTX 2060 SUPER, 8 GiB VRAM
- CUDA tensor computation passed a correctness check

## Known Environment Issue
systemd-binfmt.service reported a read-only binfmt_misc error.
It has not blocked virtual environment creation or the CUDA smoke test.
