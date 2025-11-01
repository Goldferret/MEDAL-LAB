# MADSci Client Workstation

This directory provides a containerized environment for interacting with the MEDAL-LAB MADSci infrastructure.

## Purpose

The client workstation is used for:
- **Submitting workflows** to the workcell manager
- **Managing experiments** through the experiment manager
- **Querying data** from completed workflows
- **Monitoring events** and system status
- **Development and testing** of new workflows and experiments

## Usage

### Container Management

```bash
make client-up          # Start container
make client-down        # Stop container  
make client-restart     # Restart container
make client-logs        # View logs
make client-shell       # Enter interactive shell
```

### Running Scripts

```bash
# Inside container (after make client-shell)
python workflows/demo_workflow.py
python workflows/my_workflow.py
python experiments/demo_experiment.py
python experiments/my_experiment.py
```

**Development workflow:** Edit scripts on host → `make client-shell` → run script → edit → re-run (no restart needed)

## Directory Structure

```
clients/
├── docker-compose.yml    # Client container definition
├── README.md             # This file
├── workflows/            # Workflow scripts
└── experiments/          # Experiment scripts
```

## Configuration

All configuration is inherited from `../.env.global`:
- MADSci service URLs (workcell, experiment, data, etc.)
- Robot node URLs
- Lab identification

No local configuration needed - the client connects to production services.

## Available MADSci Clients

Inside the container, you have access to:
- `WorkcellClient` - Submit and monitor workflows
- `ExperimentClient` - Manage experiments
- `DataClient` - Upload and query data
- `EventClient` - Query system events
- `ResourceClient` - Query resources
- `RestNodeClient` - Direct node communication

See MADSci documentation for client usage examples.

## Related

- **MADSci Core:** `../madsci-core/` - Framework services
- **Robot Nodes:** `../robot-nodes/dofbot-pro-ros/` - Robot setup
- **Workflows:** `./workflows/README.md` - Available workflows
