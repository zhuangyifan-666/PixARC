#!/usr/bin/env bash
set -euo pipefail

# Read-only. This intentionally never reads /proc/*/environ.
nvidia-smi
ps -eo user,pid,ppid,etimes,lstart,args --sort=pid
