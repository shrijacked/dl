#!/usr/bin/env bash
set -euo pipefail

# Append the required ssh-rsa options for sshhpc.iitd.ac.in to ~/.ssh/config
# in a safe, idempotent way.

CONFIG_FILE="$HOME/.ssh/config"
mkdir -p "$(dirname "$CONFIG_FILE")"
: > /tmp/.hpc_tmp_config_part
cat << 'EOC' > /tmp/.hpc_tmp_config_part

Host sshhpc.iitd.ac.in
  HostKeyAlgorithms +ssh-rsa
  PubkeyAcceptedKeyTypes +ssh-rsa
EOC

if [ -f "$CONFIG_FILE" ] && grep -q "Host sshhpc.iitd.ac.in" "$CONFIG_FILE"; then
  echo "[setup_hpc_ssh] Entry for sshhpc.iitd.ac.in already present in $CONFIG_FILE"
else
  touch "$CONFIG_FILE"
  chmod 600 "$CONFIG_FILE"
  cat /tmp/.hpc_tmp_config_part >> "$CONFIG_FILE"
  echo "[setup_hpc_ssh] Added sshhpc.iitd.ac.in block to $CONFIG_FILE"
fi

rm -f /tmp/.hpc_tmp_config_part
