#!/bin/bash

# MADSci Workcell Manager Launcher Script
# Usage: ./start_workcell.sh
# Configuration is loaded from .env file in the root directory

set -e  # Exit on any error

# Load environment variables from root .env file
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

if [ -f "$ENV_FILE" ]; then
    echo "📋 Loading environment variables from $ENV_FILE..."
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "⚠️  No .env file found at $ENV_FILE. Please copy .env.example to .env and configure it."
    exit 1
fi

# Set default values if not specified in .env
MADSCI_PATH=${MADSCI_PATH:-"/path/to/your/MADSci/repository"}
WORKCELL_DEFINITION=${WORKCELL_DEFINITION:-"example_wc.workcell.yaml"}

# Check if MADSci installation exists
if [ ! -d "$MADSCI_PATH" ]; then
    echo "Error: MADSci installation not found at $MADSCI_PATH"
    echo "Please set MADSCI_PATH in your .env file to point to your MADSci installation"
    exit 1
fi

# Check if the workcell definition file exists
WORKCELL_FILE="$ROOT_DIR/managers/$WORKCELL_DEFINITION"
if [ ! -f "$WORKCELL_FILE" ]; then
    echo "Error: Workcell definition file '$WORKCELL_DEFINITION' not found in managers/"
    echo "Available workcell definitions:"
    ls -1 "$ROOT_DIR/managers"/*.workcell.yaml 2>/dev/null | xargs -n1 basename || echo "No workcell definition files found"
    exit 1
fi

echo "🚀 Starting MADSci Workcell Manager with: $WORKCELL_DEFINITION"
echo "📁 Workcell file: $WORKCELL_FILE"
echo "📁 MADSci path: $MADSCI_PATH"

# Stop and remove existing containers if they exist
echo "🛑 Stopping any existing MADSci containers..."
docker compose -f "$MADSCI_PATH/compose.yaml" down 2>/dev/null || true

# Create a temporary directory for .madsci with proper permissions
echo "🔧 Setting up temporary MADSci directories..."
TEMP_MADSCI_DIR="/tmp/madsci-$(date +%s)"
mkdir -p $TEMP_MADSCI_DIR/logs
chmod 777 -R $TEMP_MADSCI_DIR
echo "📁 Using temporary directory: $TEMP_MADSCI_DIR"

# Copy and configure the workcell definition with IP addresses from .env
echo "📋 Configuring workcell definition with IP addresses from .env..."
mkdir -p "$MADSCI_PATH/example_lab/managers"

# Extract IP and port from WORKCELL_MANAGER_URL (e.g., http://192.168.4.91:8005)
WORKCELL_HOST=$(echo "$WORKCELL_MANAGER_URL" | sed -E 's|^https?://([^:]+):.*|\1|')
WORKCELL_PORT=$(echo "$WORKCELL_MANAGER_URL" | sed -E 's|^https?://[^:]+:([0-9]+).*|\1|')

echo "🔧 Configuring workcell manager to bind to: $WORKCELL_HOST:$WORKCELL_PORT"
echo "🔧 Robot node URL: $ROBOT_NODE_URL"

# Create configured workcell definition by substituting IP addresses
# These patterns work whether the file has localhost or existing IP addresses
sed -e "s|^  host: .*|  host: $WORKCELL_HOST|g" \
    -e "s|^  port: .*|  port: $WORKCELL_PORT|g" \
    -e "s|DOFBOT_Pro_1: http://.*|DOFBOT_Pro_1: $ROBOT_NODE_URL|g" \
    "$WORKCELL_FILE" > "$MADSCI_PATH/example_lab/managers/example_workcell.yaml"

echo "📋 Workcell definition configured with your network settings"

# Start the workcell manager and dependencies using docker compose
echo "🔄 Starting workcell manager and dependencies..."
cd "$MADSCI_PATH"
MADSCI_DIR=$TEMP_MADSCI_DIR docker compose up workcell_manager mongodb redis -d

# Wait a moment for container to start
sleep 3

# Check if container started successfully
if docker ps --format 'table {{.Names}}' | grep -q "^workcell_manager$"; then
    echo "✅ Workcell Manager started successfully!"
    echo "🌐 Access at: ${WORKCELL_MANAGER_URL:-http://localhost:8005}"
    echo "🛑 To stop: docker compose -f $MADSCI_PATH/compose.yaml down"
    echo "📁 Temporary MADSci directory: $TEMP_MADSCI_DIR"
    echo "⚠️  Press Ctrl+C to stop following logs (containers will keep running)"
    echo ""
    echo "📄 Following live logs:"
    echo "----------------------------------------"
    
    # Follow logs in real-time
    docker logs -f workcell_manager
else
    echo "❌ Failed to start workcell manager. Check logs:"
    docker logs workcell_manager 2>&1 || echo "No logs available"
    exit 1
fi
