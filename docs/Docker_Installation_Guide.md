# Docker Installation and Troubleshooting

## The "docker-compose not found" Error

If you see `command docker-compose not found`, this means Docker Compose isn't installed or you need to use the newer command syntax.

### Quick Fix

Docker has two command formats:

**Newer Docker (v2+):** `docker compose` (with space)
```bash
docker compose up -d
docker compose ps
docker compose down
```

**Older Docker:** `docker-compose` (with hyphen)
```bash
docker-compose up -d
docker-compose ps
docker-compose down
```

**Try the newer format first!** Modern Docker installations use `docker compose` (with space).

---

## Check What You Have

### 1. Check if Docker is installed
```bash
docker --version
```

**Expected output:**
```
Docker version 24.0.0, build xyz
```

If this fails, Docker isn't installed. Skip to [Installing Docker](#installing-docker).

### 2. Check Docker Compose version
```bash
# Try newer format
docker compose version

# OR try older format
docker-compose --version
```

**Expected output (one of):**
```
Docker Compose version v2.20.0
```
or
```
docker-compose version 1.29.2
```

If both fail, Docker Compose isn't installed. Skip to [Installing Docker Compose](#installing-docker-compose).

---

## Installing Docker

### Ubuntu/Debian Linux

```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Verify installation
docker --version
docker compose version

# Add your user to docker group (so you don't need sudo)
sudo usermod -aG docker $USER

# Log out and back in for group changes to take effect
echo "Log out and back in, then test with: docker ps"
```

### Other Linux Distributions

**CentOS/RHEL/Fedora:**
```bash
sudo yum install -y docker docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

**Arch Linux:**
```bash
sudo pacman -S docker docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

---

## Installing Docker Compose (if Docker is already installed)

### If you have Docker but not Compose

#### Option 1: Docker Compose Plugin (Recommended)
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Verify
docker compose version
```

#### Option 2: Standalone Binary
```bash
# Download latest version
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Make executable
sudo chmod +x /usr/local/bin/docker-compose

# Verify
docker-compose --version
```

---

## Post-Installation Setup

### 1. Enable Docker to start on boot
```bash
sudo systemctl enable docker
```

### 2. Start Docker service
```bash
sudo systemctl start docker
```

### 3. Verify Docker is running
```bash
sudo systemctl status docker
# Should show "active (running)"
```

### 4. Test Docker installation
```bash
docker run hello-world
```

**Expected output:**
```
Hello from Docker!
This message shows that your installation appears to be working correctly.
[...]
```

### 5. Add your user to docker group (avoid using sudo)
```bash
sudo usermod -aG docker $USER

# Log out and log back in, then verify
docker ps
# Should NOT require sudo
```

---

## Troubleshooting Docker Issues

### Error: "permission denied while trying to connect to the Docker daemon socket"

**Cause:** Your user isn't in the docker group

**Fix:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Apply group changes (choose one):
# Option 1: Log out and back in
# Option 2: Run this in current session
newgrp docker

# Verify
docker ps
```

### Error: "Cannot connect to the Docker daemon"

**Cause:** Docker service isn't running

**Fix:**
```bash
# Start Docker
sudo systemctl start docker

# Enable on boot
sudo systemctl enable docker

# Check status
sudo systemctl status docker
```

### Error: "Got permission denied while trying to connect"

**Fix:**
```bash
# Check if Docker service is running
sudo systemctl status docker

# If not running, start it
sudo systemctl start docker

# If running but still getting error, add user to group
sudo usermod -aG docker $USER
newgrp docker
```

---

## Verifying Your Setup for MADSci

Once Docker is installed, verify you can run the MADSci services:

```bash
cd ~/MEDAL-LAB/madsci-core

# Try newer format first
docker compose up -d

# Check if containers started
docker compose ps

# All containers should show "Up" or "healthy"
# Expected output:
# NAME                    STATUS
# experiment-manager      Up 5 seconds
# resource-manager        Up 5 seconds
# data-manager           Up 5 seconds
# workcell-manager       Up 5 seconds
# location-manager       Up 5 seconds

# Test services are responding
curl http://localhost:8002/health
curl http://localhost:8005/health

# If everything works, you're ready!
```

### If containers won't start

**Check logs:**
```bash
docker compose logs
# OR
docker compose logs -f  # Follow logs in real-time
```

**Check specific service:**
```bash
docker compose logs workcell-manager
```

**Restart services:**
```bash
docker compose down
docker compose up -d
```

---

## Quick Reference

### Essential Docker Commands

```bash
# Start services (detached mode)
docker compose up -d

# Stop services
docker compose down

# View running containers
docker compose ps

# View logs
docker compose logs -f

# Restart a service
docker compose restart workcell-manager

# Restart all services
docker compose restart

# Check Docker status
sudo systemctl status docker

# View Docker info
docker info
```

### Switching Between docker-compose Formats

**If you have older docker-compose installed:**
```bash
# Old format still works
docker-compose up -d
```

**If you have newer Docker with plugin:**
```bash
# Use new format
docker compose up -d
```

**Both installed? Use new format:**
```bash
# Prefer new format (it's faster)
docker compose up -d
```

---

## What to Do After Installing Docker

1. **Log out and back in** (for group permissions)
2. **Navigate to madsci-core:**
   ```bash
   cd ~/MEDAL-LAB/madsci-core
   ```
3. **Start services:**
   ```bash
   docker compose up -d
   ```
4. **Verify services:**
   ```bash
   docker compose ps
   curl http://localhost:8002/health
   ```
5. **Continue with Quick Start Checklist**

---

## Getting Help

### Docker Issues
- Docker documentation: https://docs.docker.com/
- Docker forums: https://forums.docker.com/

### MADSci Issues
- Check service logs: `docker compose logs [service-name]`
- Verify all services healthy: `docker compose ps`
- Contact lab support (see Quick Start Checklist)

---

## Summary

**Most common fix for "docker-compose not found":**

```bash
# Instead of this:
docker-compose up -d

# Use this:
docker compose up -d
```

**If that still doesn't work, install Docker:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER
# Log out and back in
```

**Then verify:**
```bash
docker --version
docker compose version
cd ~/MEDAL-LAB/madsci-core
docker compose up -d
```

You're now ready to continue with the experiment setup!

