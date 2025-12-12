# Environment Configuration Guide

## The .env.global File

MADSci services require environment variables to configure port numbers and other settings. These are defined in `.env.global` at the root of the MEDAL-LAB repository.

## Quick Fix for "env file not found"

If you see the error:
```
env file /path/to/MEDAL-LAB/.env.global not found
```

Run this command from the MEDAL-LAB root directory:

```bash
cd ~/MEDAL-LAB

cat > .env.global << 'EOF'
# MADSci Core Services Ports
EXPERIMENT_SERVER_PORT=8002
RESOURCE_SERVER_PORT=8003
DATA_SERVER_PORT=8004
WORKCELL_SERVER_PORT=8005
LOCATION_SERVER_PORT=8006
EVENT_SERVER_PORT=8007
LAB_SERVER_PORT=8008
MINIO_CONSOLE_PORT=9001
EOF
```

Then try starting the services again:
```bash
cd ~/MEDAL-LAB/madsci-core
docker compose up -d
```

---

## Complete .env.global Template

Create `~/MEDAL-LAB/.env.global` with this content:

```bash
# ============================================================
# MADSci Core Services Environment Configuration
# ============================================================

# Service Ports (Do not change unless you have port conflicts)
EXPERIMENT_SERVER_PORT=8002
RESOURCE_SERVER_PORT=8003
DATA_SERVER_PORT=8004
WORKCELL_SERVER_PORT=8005
LOCATION_SERVER_PORT=8006
EVENT_SERVER_PORT=8007
LAB_SERVER_PORT=8008

# MinIO Object Storage Console
MINIO_CONSOLE_PORT=9001

# Network Configuration
NETWORK_SUBNET=172.20.0.0/16

# Database Configuration (if using PostgreSQL)
POSTGRES_USER=madsci
POSTGRES_PASSWORD=madsci_dev_password
POSTGRES_DB=madsci

# Redis Configuration (if using Redis)
REDIS_PORT=6379

# Development Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=false

# Optional: Custom service URLs (usually not needed)
# EXPERIMENT_SERVER_URL=http://localhost:8002
# RESOURCE_SERVER_URL=http://localhost:8003
# DATA_SERVER_URL=http://localhost:8004
# WORKCELL_SERVER_URL=http://localhost:8005
# LOCATION_SERVER_URL=http://localhost:8006
```

---

## What Each Variable Means

### Service Ports

| Variable | Default | Service | Purpose |
|----------|---------|---------|---------|
| `EXPERIMENT_SERVER_PORT` | 8002 | Experiment Manager | Manages experiment lifecycle |
| `RESOURCE_SERVER_PORT` | 8003 | Resource Manager | Tracks resources (blocks, containers) |
| `DATA_SERVER_PORT` | 8004 | Data Manager | Stores experiment data and images |
| `WORKCELL_SERVER_PORT` | 8005 | Workcell Manager | Orchestrates workflows |
| `LOCATION_SERVER_PORT` | 8006 | Location Manager | Manages physical locations |
| `EVENT_SERVER_PORT` | 8007 | Event Manager | Logs events and messages |
| `LAB_SERVER_PORT` | 8008 | Lab Manager | Lab-level coordination |
| `MINIO_CONSOLE_PORT` | 9001 | MinIO Console | Web UI for object storage |

### When to Change Ports

**Change ports if:**
- Another service is using the default port
- You get "address already in use" errors
- You need to run multiple MADSci instances

**Example: Port conflict on 8002**
```bash
# Check what's using port 8002
sudo lsof -i :8002

# If something else is using it, change in .env.global:
EXPERIMENT_SERVER_PORT=8012  # Use different port
```

---

## Troubleshooting Environment Issues

### Error: "variable is not set. Defaulting to a blank string"

**Symptoms:**
```
WARN[0000] The "RESOURCE_SERVER_PORT" variable is not set. Defaulting to a blank string.
```

**Cause:** `.env.global` file is missing or incomplete

**Fix:**
1. Create `.env.global` in the repository root (not in `madsci-core/`)
2. Add all required port variables
3. Restart services

```bash
cd ~/MEDAL-LAB
# Create .env.global (see template above)

cd madsci-core
docker compose down
docker compose up -d
```

### Error: "env file not found"

**Symptoms:**
```
env file /path/to/MEDAL-LAB/.env.global not found
```

**Fix:**
```bash
# Make sure you're in the right directory
cd ~/MEDAL-LAB
ls -la .env.global  # Should exist

# If not, create it
cat > .env.global << 'EOF'
EXPERIMENT_SERVER_PORT=8002
RESOURCE_SERVER_PORT=8003
DATA_SERVER_PORT=8004
WORKCELL_SERVER_PORT=8005
LOCATION_SERVER_PORT=8006
EVENT_SERVER_PORT=8007
LAB_SERVER_PORT=8008
MINIO_CONSOLE_PORT=9001
EOF
```

### Error: "Address already in use"

**Symptoms:**
```
Error starting userland proxy: listen tcp 0.0.0.0:8002: bind: address already in use
```

**Fix:**
```bash
# Find what's using the port
sudo lsof -i :8002

# Option 1: Stop the conflicting service
sudo kill [PID]

# Option 2: Change the port in .env.global
nano ~/MEDAL-LAB/.env.global
# Change EXPERIMENT_SERVER_PORT=8002 to another port like 8012

# Restart services
cd ~/MEDAL-LAB/madsci-core
docker compose down
docker compose up -d
```

---

## Verifying Your Configuration

### 1. Check .env.global exists
```bash
cd ~/MEDAL-LAB
cat .env.global
# Should show all port variables
```

### 2. Check Docker Compose can read it
```bash
cd ~/MEDAL-LAB/madsci-core
docker compose config
# Should show resolved port numbers (no warnings)
```

### 3. Verify services are using correct ports
```bash
docker compose ps
# Should show ports like 0.0.0.0:8002->8002/tcp
```

### 4. Test service connectivity
```bash
curl http://localhost:8002/health  # Experiment Manager
curl http://localhost:8003/health  # Resource Manager
curl http://localhost:8004/health  # Data Manager
curl http://localhost:8005/health  # Workcell Manager
curl http://localhost:8006/health  # Location Manager
```

---

## File Location

**Correct location:**
```
~/MEDAL-LAB/
├── .env.global              [HERE - at repository root]
├── madsci-core/
│   └── docker-compose.yml   [References ../.env.global]
├── clients/
└── robot-nodes/
```

**Wrong locations:**
```
~/MEDAL-LAB/madsci-core/.env.global  [WRONG - too deep]
~/.env.global                         [WRONG - too high]
```

---

## Environment Variables in Docker Compose

The `docker-compose.yml` file references `.env.global` like this:

```yaml
# In madsci-core/docker-compose.yml
env_file:
  - ../.env.global  # Goes up one directory to find .env.global

services:
  experiment-manager:
    ports:
      - "${EXPERIMENT_SERVER_PORT}:8002"  # Uses variable from .env.global
```

This means:
1. Docker Compose looks for `.env.global` in parent directory
2. Loads all variables into environment
3. Substitutes `${VARIABLE_NAME}` with actual values

---

## Multiple Environments

If you need different configurations:

**Development:**
```bash
# .env.global
ENVIRONMENT=development
LOG_LEVEL=DEBUG
```

**Production:**
```bash
# .env.global
ENVIRONMENT=production
LOG_LEVEL=INFO
```

Or use separate files:
```bash
# Start with specific env file
docker compose --env-file ../.env.production up -d
```

---

## Security Notes

### Don't commit .env.global to git!

The `.env.global` file may contain sensitive information. Add it to `.gitignore`:

```bash
# Add to .gitignore
echo ".env.global" >> ~/MEDAL-LAB/.gitignore
```

### Use strong passwords in production

For production deployments, change default passwords:

```bash
# In .env.global
POSTGRES_PASSWORD=strong_random_password_here
MINIO_ROOT_PASSWORD=another_strong_password
```

---

## Quick Reference Commands

```bash
# Create minimal .env.global
cd ~/MEDAL-LAB
cat > .env.global << 'EOF'
EXPERIMENT_SERVER_PORT=8002
RESOURCE_SERVER_PORT=8003
DATA_SERVER_PORT=8004
WORKCELL_SERVER_PORT=8005
LOCATION_SERVER_PORT=8006
EVENT_SERVER_PORT=8007
LAB_SERVER_PORT=8008
MINIO_CONSOLE_PORT=9001
EOF

# Verify it exists
cat .env.global

# Start services
cd madsci-core
docker compose up -d

# Check services are running
docker compose ps

# Test connectivity
curl http://localhost:8002/health
curl http://localhost:8005/health
```

---

## Summary

**Essential steps:**

1. Create `.env.global` in `~/MEDAL-LAB/` (repository root)
2. Add all required port variables
3. Start services from `~/MEDAL-LAB/madsci-core/`
4. Verify services are running and responding

**Common mistakes:**
- Creating `.env.global` in wrong directory
- Missing required variables
- Port conflicts with other services
- Forgetting to restart after changes

**Quick test:**
```bash
# Everything working if all return JSON:
curl http://localhost:8002/health
curl http://localhost:8003/health
curl http://localhost:8004/health
curl http://localhost:8005/health
curl http://localhost:8006/health
```

You're now ready to continue with the experiment setup!

