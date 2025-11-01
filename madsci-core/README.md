# MADSci Core Server

Production-ready deployment of the complete MADSci framework stack.

## Architecture

**MADSci Services (Ports 8000-8006):**
- **Lab Manager (8000)**: Web dashboard and lab coordination
- **Event Manager (8001)**: Event logging and monitoring
- **Experiment Manager (8002)**: Experiment design and campaigns
- **Resource Manager (8003)**: Sample and consumable tracking
- **Data Manager (8004)**: Datapoint storage and retrieval
- **Workcell Manager (8005)**: Workflow execution and robot coordination
- **Location Manager (8006)**: Location and container tracking

**Databases:**
- MongoDB (27017), PostgreSQL (5432), Redis (6379), MinIO (9000, 9001)

## Prerequisites

- Docker (>= 20.10) and Docker Compose (>= 2.0)
- Configured `../.env.global` with network settings

## Configuration

**`../.env.global`** - Single source of truth for all configuration:
- Network addresses (MADSCI_CORE_HOST, robot node IPs)
- Service ports (8000-8006, database ports)
- Service URLs (automatically derived from host + port)
- MinIO console port (9001)

**`.env`** - Local service binding configuration:
- Service binding addresses (0.0.0.0 = listen on all interfaces)
- Not read directly - kept for documentation only

The Makefile uses `docker compose --env-file=../.env.global` to ensure all services use the global configuration.

Edit `../.env.global` to set your MADSci core host IP address.

## Deployment

All deployment is managed through the root Makefile for consistency.

**Start services:**
```bash
cd /path/to/medal-lab
make madsci-up
```

**Stop services:**
```bash
make madsci-down
```

**Restart services:**
```bash
make madsci-restart
```

**View logs (all services):**
```bash
make madsci-logs
```

**View logs (specific service):**
```bash
make madsci-logs SERVICE=workcell_manager
make madsci-logs SERVICE=resource_manager
make madsci-logs SERVICE=postgres
```

**Validate deployment:**
```bash
cd madsci-core
./tools/validate-deployment.sh
```

## Access

- **Lab Dashboard**: http://YOUR_HOST_IP:8000
- **MinIO Console**: http://YOUR_HOST_IP:9001
- **API Endpoints**: http://YOUR_HOST_IP:8000-8006

## Troubleshooting

**Services won't start:**
```bash
make madsci-logs  # Check for errors
make madsci-logs SERVICE=workcell_manager  # Check specific service
```
Common issues: Port conflicts, permission issues, incorrect `.env.global`

**Database connection issues:**
```bash
# Test MongoDB
docker exec -it mongodb mongosh --eval "db.adminCommand('ping')"

# Test PostgreSQL
docker exec -it postgres psql -U madsci -d resources -c "SELECT 1;"

# Test Redis
docker exec -it redis redis-cli ping
```

**Service not accessible from other devices:**
```bash
# Verify services listening on 0.0.0.0
ss -tlnp | grep -E ':(8000|8001|8002|8003|8004|8005|8006)'

# Test from another device
curl http://YOUR_HOST_IP:8005/
```

**Fresh database after version upgrade:**
```bash
# If upgrading MADSci versions, clear old databases
make madsci-down
sudo rm -rf .madsci
make madsci-up
```

## Development

**Manager definitions** are in `./managers/`:
- `lab.manager.yaml`, `event.manager.yaml`, `experiment.manager.yaml`
- `resource.manager.yaml`, `data.manager.yaml`, `workcell.manager.yaml`, `location.manager.yaml`

**After updating a manager definition:**
```bash
make madsci-restart
# Or restart specific service:
cd madsci-core
docker compose restart workcell_manager
```

**Add workcell definitions** in `./managers/`:
```yaml
# example.workcell.yaml
workcell_id: my_workcell
workcell_name: My Laboratory Workcell
nodes:
  - node_name: robot_1
    node_url: http://192.168.1.200:2000
```

## Related

- **Client**: `../clients/README.md` - Submit workflows and experiments
- **Robot Nodes**: `../robot-nodes/dofbot-pro-ros/README.md` - Robot setup
- **MADSci Framework**: https://github.com/AD-SDL/MADSci
