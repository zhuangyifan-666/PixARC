#!/usr/bin/env bash
set -euo pipefail

# Read-only inspection only. Deliberately no /proc reads, debugger attachment,
# environment scraping, signalling, or process termination.
nvidia-smi
ps -eo user,pid,ppid,etimes,lstart,args --sort=pid
