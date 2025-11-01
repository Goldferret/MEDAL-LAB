#!/bin/bash
# Validate MADSci Core Deployment
# Checks all services and databases are running and accessible

set -e

echo "========================================="
echo "MADSci Core Deployment Validation"
echo "========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: Must run from madsci-core/ directory"
    exit 1
fi

# Container Status
echo "Container Status:"
if docker compose ps | grep -qE "Exited|Restarting"; then
    echo "  ❌ Some containers not running:"
    docker compose ps --format "  {{.Name}}: {{.Status}}" | grep -vE "Up [0-9]"
    echo ""
else
    echo "  ✅ All containers running"
    echo ""
fi

# Service Connectivity
echo "Service Connectivity:"
services=("8000:Lab" "8001:Event" "8002:Experiment" "8003:Resource" "8004:Data" "8005:Workcell")

for service in "${services[@]}"; do
    port="${service%%:*}"
    name="${service##*:}"
    status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/ 2>/dev/null)
    if [ "$status" = "200" ] || [ "$status" = "404" ]; then
        echo "  ✅ $name Manager (port $port)"
    else
        echo "  ❌ $name Manager (port $port) - HTTP $status"
    fi
done
echo ""

# Database Status
echo "Database Status:"

# MongoDB
if docker exec mongodb mongosh --eval "db.adminCommand('ping')" --quiet >/dev/null 2>&1; then
    echo "  ✅ MongoDB"
else
    echo "  ❌ MongoDB"
fi

# Redis
if docker exec redis redis-cli ping 2>/dev/null | grep -q PONG; then
    echo "  ✅ Redis"
else
    echo "  ❌ Redis"
fi

# PostgreSQL
if docker exec postgres pg_isready -U madsci 2>/dev/null | grep -q "accepting"; then
    echo "  ✅ PostgreSQL"
else
    echo "  ❌ PostgreSQL"
fi

# MinIO
if curl -s http://localhost:9000/minio/health/live >/dev/null 2>&1; then
    echo "  ✅ MinIO"
else
    echo "  ❌ MinIO"
fi

echo ""
echo "========================================="
echo "Validation Complete"
echo "========================================="
