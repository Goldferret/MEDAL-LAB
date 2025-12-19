.PHONY: help list-robots madsci-up madsci-down madsci-logs madsci-restart \
        client-up client-down client-logs client-restart client-shell client-build \
        robot-up robot-down robot-logs robot-restart \
        ros-up ros-attach ros-down ros-status \
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
	@echo "  make client-build    - Rebuild client image (after Dockerfile changes)"
	@echo ""
	@echo "$(GREEN)Robot Nodes:$(NC)"
	@echo "  make list-robots              - List available robot nodes"
	@echo "  make robot-up NODE=<name>     - Start specific robot node"
	@echo "  make robot-down NODE=<name>   - Stop specific robot node"
	@echo "  make robot-logs NODE=<name>   - View robot node logs"
	@echo "  make robot-restart NODE=<name> - Restart robot node"
	@echo ""
	@echo "$(GREEN)ROS Services (Jetson only):$(NC)"
	@echo "  make ros-up          - Start all ROS services with verified dependencies"
	@echo "  make ros-attach      - Attach to ROS services tmux session"
	@echo "  make ros-down        - Stop all ROS services"
	@echo "  make ros-status      - Check ROS service status and diagnostics"
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
	cd $(MADSCI_DIR) && docker compose --env-file=../$(ENV_FILE) --env-file=.env up -d
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

client-build:
	@echo "$(GREEN)Building client image...$(NC)"
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "$(RED)ERROR: $(ENV_FILE) not found!$(NC)"; \
		exit 1; \
	fi
	cd $(CLIENT_DIR) && docker compose build
	@echo "$(GREEN)✓ Client image built$(NC)"

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

# ROS Services (Jetson only)
ros-up:
	@if tmux has-session -t ros_services 2>/dev/null; then \
		echo "$(RED)❌ ros_services already running$(NC)"; \
		echo "   Stop with: make ros-down"; \
		exit 1; \
	fi
	@echo "$(GREEN)Starting ROS services with verified dependencies...$(NC)"
	@echo "$(CYAN)[1/4] Starting MoveIt (this takes ~100 seconds)...$(NC)"
	@tmux new-session -d -s ros_services -n moveit
	@tmux send-keys -t ros_services:moveit 'roslaunch dofbot_pro_config demo.launch' C-m
	@echo "$(YELLOW)Waiting for MoveIt to initialize (checking every 5 seconds)...$(NC)"
	@bash -c 'for i in {1..30}; do \
		if timeout 2 rostopic list 2>/dev/null | grep -q "/move_group/goal"; then \
			echo -e "\033[0;32m✓ MoveIt action server detected (after $$((i*5)) seconds)\033[0m"; \
			break; \
		fi; \
		if [ $$i -eq 30 ]; then \
			echo -e "\033[0;31mERROR: MoveIt failed to start after 150 seconds\033[0m"; \
			exit 1; \
		fi; \
		echo "  ... still waiting (attempt $$i/30)"; \
		sleep 5; \
	done'
	@echo "$(YELLOW)Waiting for MoveIt to complete initialization...$(NC)"
	@sleep 10
	@bash -c 'if timeout 5 rostopic echo -n 1 /move_group/status >/dev/null 2>&1; then \
		echo -e "\033[0;32m✓ MoveIt is fully operational\033[0m"; \
	else \
		echo -e "\033[0;33m⚠ MoveIt started but status topic not responsive (continuing anyway)\033[0m"; \
	fi'
	@echo "$(CYAN)[2/4] Starting arm_driver...$(NC)"
	@tmux new-window -t ros_services -n arm_driver
	@tmux send-keys -t ros_services:arm_driver 'rosrun dofbot_pro_info arm_driver.py' C-m
	@sleep 3
	@echo "$(GREEN)✓ arm_driver started$(NC)"
	@echo "$(CYAN)[3/4] Starting sim_bridge...$(NC)"
	@tmux new-window -t ros_services -n sim_bridge
	@tmux send-keys -t ros_services:sim_bridge 'rosrun arm_moveit_demo SimulationToMachine.py' C-m
	@sleep 3
	@echo "$(GREEN)✓ sim_bridge started$(NC)"
	@echo "$(CYAN)[4/4] Starting camera...$(NC)"
	@tmux new-window -t ros_services -n camera
	@tmux send-keys -t ros_services:camera 'roslaunch orbbec_camera dabai_dcw2.launch' C-m
	@sleep 8
	@echo "$(YELLOW)Verifying camera stream...$(NC)"
	@bash -c 'if timeout 10 rostopic list 2>/dev/null | grep -q "/camera/color/image_raw"; then \
		echo -e "\033[0;32m✓ Camera topic exists\033[0m"; \
		if timeout 5 rostopic hz /camera/color/image_raw --window=5 -n 1 >/dev/null 2>&1; then \
			echo -e "\033[0;32m✓ Camera is publishing data\033[0m"; \
		else \
			echo -e "\033[0;33m⚠ Camera topic exists but no data detected yet (may need more time)\033[0m"; \
		fi; \
	else \
		echo -e "\033[0;33m⚠ Camera topic not found (launch may have failed)\033[0m"; \
	fi'
	@echo ""
	@echo "$(GREEN)════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)✓ ROS services startup complete$(NC)"
	@echo "$(GREEN)════════════════════════════════════════════════════════$(NC)"
	@echo "  View logs: $(CYAN)make ros-attach$(NC)"
	@echo "  Stop services: $(YELLOW)make ros-down$(NC)"

ros-attach:
	@tmux attach -t ros_services

ros-down:
	@tmux kill-session -t ros_services 2>/dev/null || true
	@echo "$(GREEN)✓ ROS services stopped$(NC)"

# Diagnostic target to check service status
ros-status:
	@echo "$(CYAN)Checking ROS service status...$(NC)"
	@echo ""
	@echo "$(GREEN)Topics:$(NC)"
	@bash -c 'rostopic list 2>/dev/null | grep -E "(move_group|camera|joint)" || echo "  No relevant topics found"'
	@echo ""
	@echo "$(GREEN)Active nodes:$(NC)"
	@bash -c 'rosnode list 2>/dev/null || echo "  ROS master not running"'
	@echo ""
	@if tmux has-session -t ros_services 2>/dev/null; then \
		echo "$(GREEN)✓ Tmux session 'ros_services' is active$(NC)"; \
	else \
		echo "$(RED)✗ Tmux session 'ros_services' not found$(NC)"; \
	fi

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
