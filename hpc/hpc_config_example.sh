#!/usr/bin/env bash

# Example config for IITD HPC access. Copy this file to hpc_config.sh
# and edit the variables for your own account.

# Your IITD HPC SSH user, e.g. px081.visitor
export HPC_SSH_USER="px114.visitor"

# SSH bastion / login host
export HPC_SSH_BASTION="sshhpc.iitd.ac.in"

# Inner HPC host reached after logging into bastion
# (on IITD this is typically just 'hpc').
export HPC_CLUSTER_HOST="hpc"

# Optional: default remote root directory where you keep this project
# on HPC (used by helper scripts that sync or run jobs).
export HPC_PROJECT_ROOT="/home/$USER/dl_project"
