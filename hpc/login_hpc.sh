#!/usr/bin/env bash
set -euo pipefail

# Interactive helper to log into IITD HPC using the two-step flow:
#   1) ssh $HPC_SSH_USER@$HPC_SSH_BASTION
#   2) ssh $HPC_CLUSTER_HOST
# but chained so you can run just this script in a real terminal.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/hpc_config.sh"

if [ -f "$CONFIG_FILE" ]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
else
  echo "[login_hpc] Config file $CONFIG_FILE not found."
  echo "[login_hpc] Copy hpc/hpc_config_example.sh to hpc/hpc_config.sh and edit your px user."
  exit 1
fi

: "${HPC_SSH_USER:?HPC_SSH_USER must be set in hpc_config.sh}"
: "${HPC_SSH_BASTION:?HPC_SSH_BASTION must be set in hpc_config.sh}"
: "${HPC_CLUSTER_HOST:?HPC_CLUSTER_HOST must be set in hpc_config.sh}"

# -t -t forces allocation of a TTY even when running from some shells/tools.
# You should run this from a real terminal (Terminal/iTerm/gnome-terminal etc.)
# so you get an interactive shell on the HPC cluster.
ssh -tt "${HPC_SSH_USER}@${HPC_SSH_BASTION}" "ssh -tt ${HPC_CLUSTER_HOST}"
