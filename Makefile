.PHONY: help list-robots madsci-up madsci-down madsci-logs madsci-restart \
        client-up client-down client-logs client-restart client-shell \
        robot-up robot-down robot-logs robot-restart \
        up down clean

# Paths
ENV_FILE := .env.global
MADSCI_DIR := madsci-core
CLIENT_DIR := clients
ROBOT_DIR := robot-nodes

# Auto-discover robot nodes (exclude anything with 'archived' in name or 'shared' directory)
ROBOT_NODES := $(shell find $(ROBOT_DIR) -maxdepth 1 -type d ! -name '*archived*' ! -name 'robot-nodes' ! -name 'shared' -exec basename {} \;)

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Default target
help:
	@echo "$(CYAN)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(CYAN)║           MEDAL-LAB Docker Management                      ║$(NC)"
	@echo "$(CYAN)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Configuration:$(NC)"
	@echo "  Environment file: $(ENV_FILE)"
	@echo "  Edit $(ENV_FILE) to change IP addresses and ports"
	@echo ""
	@echo "$(GREEN)MADSci Core Services:$(NC)"
	@echo "  make madsci-up       - Start MADSci core services"
	@echo "  make madsci-down     - Stop MADSci core services"
	@echo "  make madsci-logs     - View MADSci core logs (all services)"
	@echo "  make madsci-logs SERVICE=<name> - View logs for specific service"
	@echo "  make madsci-restart  - Restart MADSci core services"
	@echo ""
	@echo "$(GREEN)Client Services:$(NC)"
	@echo "  make client-up       - Start client container"
	@echo "  make client-down     - Stop client container"
	@echo "  make client-logs     - View client logs"
	@echo "  make client-restart  - Restart client container"
	@echo "  make client-shell    - Open interactive shell in client container"
	@echo ""
	@echo "$(GREEN)Robot Nodes:$(NC)"
	@echo "  make list-robots              - List available robot nodes"
	@echo "  make robot-up NODE=<name>     - Start specific robot node"
	@echo "  make robot-down NODE=<name>   - Stop specific robot node"
	@echo "  make robot-logs NODE=<name>   - View robot node logs"
	@echo "  make robot-restart NODE=<name> - Restart robot node"
	@echo ""
	@echo "$(GREEN)Utility:$(NC)"
	@echo "  make up              - Start MADSci core + client (not robots)"
	@echo "  make down            - Stop all local services"
	@echo "  make clean           - Stop all and remove volumes"
	@echo ""
	@echo "$(YELLOW)Note: Robot nodes typically run on remote Jetson devices.$(NC)"
	@echo "$(YELLOW)SSH to the Jetson and run 'make robot-up NODE=<name>' there.$(NC)"

# List available robot nodes
list-robots:
	@echo "$(GREEN)Available robot nodes:$(NC)"
	@if [ -z "$(ROBOT_NODES)" ]; then \
		echo "  $(RED)No robot nodes found in $(ROBOT_DIR)/$(NC)"; \
	else \
		for node in $(ROBOT_NODES); do \
			echo "  - $$node"; \
		done; \
	fi
	@echo ""
	@echo "$(YELLOW)Usage: make robot-up NODE=<name>$(NC)"

# MADSci Core Services
madsci-up:
	@echo "$(GREEN)Starting MADSci core services...$(NC)"
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "$(RED)ERROR: $(ENV_FILE) not found!$(NC)"; \
		exit 1; \
	fi
	cd $(MADSCI_DIR) && docker compose --env-file=../$(ENV_FILE) up -d
	@echo "$(GREEN)✓ MADSci core services started$(NC)"

madsci-down:
	@echo "$(YELLOW)Stopping MADSci core services...$(NC)"
	cd $(MADSCI_DIR) && docker compose down
	@echo "$(GREEN)✓ MADSci core services stopped$(NC)"

madsci-logs:
	@if [ -n "$(SERVICE)" ]; then \
		echo "$(GREEN)Viewing logs for: $(SERVICE)$(NC)"; \
		cd $(MADSCI_DIR) && docker compose logs -f $(SERVICE); \
	else \
		echo "$(GREEN)Viewing logs for all MADSci services$(NC)"; \
		echo "$(YELLOW)Tip: Use 'make madsci-logs SERVICE=<name>' to view specific service$(NC)"; \
		cd $(MADSCI_DIR) && docker compose logs -f; \
	fi

madsci-restart: madsci-down madsci-up

# Client Services
client-up:
	@echo "$(GREEN)Starting client container...$(NC)"
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "$(RED)ERROR: $(ENV_FILE) not found!$(NC)"; \
		exit 1; \
	fi
	cd $(CLIENT_DIR) && docker compose --env-file=../$(ENV_FILE) up -d
	@echo "$(GREEN)✓ Client container started$(NC)"

client-down:
	@echo "$(YELLOW)Stopping client container...$(NC)"
	cd $(CLIENT_DIR) && docker compose down
	@echo "$(GREEN)✓ Client container stopped$(NC)"

client-logs:
	cd $(CLIENT_DIR) && docker compose logs -f

client-restart: client-down client-up

client-shell:
	@echo "$(GREEN)Opening shell in client container...$(NC)"
	@if ! docker ps --format '{{.Names}}' | grep -q '^madsci_client$$'; then \
		echo "$(RED)ERROR: Client container is not running$(NC)"; \
		echo "Start it with: make client-up"; \
		exit 1; \
	fi
	@docker exec -it madsci_client bash

# Robot Node Services
robot-up:
	@if [ -z "$(NODE)" ]; then \
		echo "$(RED)ERROR: NODE parameter required$(NC)"; \
		echo "Usage: make robot-up NODE=<name>"; \
		echo ""; \
		echo "Available nodes:"; \
		for node in $(ROBOT_NODES); do \
			echo "  - $$node"; \
		done; \
		exit 1; \
	fi
	@if [ ! -d "$(ROBOT_DIR)/$(NODE)" ]; then \
		echo "$(RED)ERROR: Robot node '$(NODE)' not found in $(ROBOT_DIR)/$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Starting robot node: $(NODE)$(NC)"
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "$(RED)ERROR: $(ENV_FILE) not found!$(NC)"; \
		exit 1; \
	fi
	cd $(ROBOT_DIR)/$(NODE) && docker compose --env-file=../../$(ENV_FILE) up -d
	@echo "$(GREEN)✓ Robot node $(NODE) started$(NC)"

robot-down:
	@if [ -z "$(NODE)" ]; then \
		echo "$(RED)ERROR: NODE parameter required$(NC)"; \
		echo "Usage: make robot-down NODE=<name>"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Stopping robot node: $(NODE)$(NC)"
	cd $(ROBOT_DIR)/$(NODE) && docker compose down
	@echo "$(GREEN)✓ Robot node $(NODE) stopped$(NC)"

robot-logs:
	@if [ -z "$(NODE)" ]; then \
		echo "$(RED)ERROR: NODE parameter required$(NC)"; \
		echo "Usage: make robot-logs NODE=<name>"; \
		exit 1; \
	fi
	cd $(ROBOT_DIR)/$(NODE) && docker compose logs -f

robot-restart: robot-down robot-up

# Convenience targets
up: madsci-up client-up
	@echo "$(GREEN)✓ Local services started (MADSci core + client)$(NC)"
	@echo "$(YELLOW)Note: Robot nodes must be started separately on their devices$(NC)"

down: madsci-down client-down
	@echo "$(GREEN)✓ All local services stopped$(NC)"

clean:
	@echo "$(RED)Stopping all services and removing volumes...$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		cd $(MADSCI_DIR) && docker compose down -v; \
		cd ../$(CLIENT_DIR) && docker compose down -v; \
		echo "$(GREEN)✓ Cleanup complete$(NC)"; \
	else \
		echo "$(YELLOW)Cancelled$(NC)"; \
	fi
