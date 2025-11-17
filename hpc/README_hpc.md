## HPC Setup for IITD Cluster

This project includes small helper scripts so anyone with an IITD HPC account (pxXXX.visitor) can use the cluster with the same codebase.

### 1. One‑time SSH configuration (Linux / macOS)

From your local machine run:

```bash
bash hpc/setup_hpc_ssh.sh
```

This appends the required block to `~/.ssh/config`:

```text
Host sshhpc.iitd.ac.in
  HostKeyAlgorithms +ssh-rsa
  PubkeyAcceptedKeyTypes +ssh-rsa
```

If you prefer, you can do this manually as well.

### 2. Create your personal HPC config

Copy the example config and edit it:

```bash
cp hpc/hpc_config_example.sh hpc/hpc_config.sh
vim hpc/hpc_config.sh   # or any editor
```

Set at least:

- **HPC_SSH_USER**: your px user, e.g. `px114.visitor`
- **HPC_SSH_BASTION**: `sshhpc.iitd.ac.in`
- **HPC_CLUSTER_HOST**: `hpc` (default for IITD)
- **HPC_PROJECT_ROOT**: path on the cluster where you keep this repo

The personal `hpc_config.sh` file should **not** be committed; you can add it to `.gitignore` if desired.

### 3. Login helper

Once `hpc_config.sh` is set up, you can log in with the usual two‑step flow via a single command:

```bash
bash hpc/login_hpc.sh
```

This is equivalent to:

```bash
ssh $HPC_SSH_USER@$HPC_SSH_BASTION
# then on the bastion
ssh hpc
```

### 4. Using this from a fresh clone

For any collaborator who clones this repo:

1. Generate/upload their SSH key to IITD (per HPC documentation).
2. Run `bash hpc/setup_hpc_ssh.sh` once on their local machine.
3. Copy and edit `hpc/hpc_config_example.sh` to `hpc/hpc_config.sh` with their own px user.
4. Use `bash hpc/login_hpc.sh` to reach the cluster and then run the usual training / analysis scripts from this project on the HPC nodes.
