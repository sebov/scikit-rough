# Silo

A reproducible Ubuntu-based development container with pre-installed tools (uv, mise, opencode).

## Setup

### Prerequisites

- Docker and Docker Compose installed on your host machine

### Volume mounts and ownership

The container runs as the `ubuntu` user with **UID 1000** and **GID 1000**. For the bind-mounted
volumes under `~/.silo/` to work correctly, **they should be owned by UID 1000 on the host**.

The following host directories are mounted into the container:

| Host path                      | Container path                       | Purpose                  |
| ------------------------------ | ------------------------------------ | ------------------------ |
| `~/.silo/agents`               | `/home/ubuntu/.agents`               | Agent skills and plugins |
| `~/.silo/config/opencode`      | `/home/ubuntu/.config/opencode`      | OpenCode configuration   |
| `~/.silo/local/share/opencode` | `/home/ubuntu/.local/share/opencode` | OpenCode local data      |

To set up correct ownership and permissions, run on the host:

```bash
mkdir -p ~/.silo/agents ~/.silo/config/opencode ~/.silo/local/share/opencode
sudo chown -R 1000:1000 ~/.silo
chmod 700 ~/.silo/agents ~/.silo/config ~/.silo/local ~/.silo/local/share
```

Setting `chmod 700` restricts access to these directories to the owner only. If you prefer to apply
it recursively to all nested directories, you can use:

```bash
find ~/.silo -type d -exec chmod 700 {} +
```

If these directories are owned by a different UID, you may encounter permission errors when the
container tries to read or write to them.

### Running

```bash
docker compose -f silo.compose.yml up -d
docker compose -f silo.compose.yml exec silo bash
```
