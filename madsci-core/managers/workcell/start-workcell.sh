#!/bin/bash
set -e

# Substitute environment variables in the template
envsubst < /home/madsci/medal-lab/managers/workcell/workcell.manager.yaml > /tmp/workcell.manager.yaml

# Start the workcell server with the processed config
exec python -m madsci.workcell_manager.workcell_server \
  --definition /tmp/workcell.manager.yaml \
  --host ${WORKCELL_HOST} \
  --port ${WORKCELL_PORT}
