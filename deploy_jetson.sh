#!/bin/bash
# Deployment script for AnyLoc Drone Navigation on Jetson Orin Nano
# Handles complete setup, optimization, and deployment

set -e

echo "🚁 AnyLoc Drone Navigation - Jetson Orin Nano Deployment"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="anyloc-drone:jetson-latest"
CONTAINER_NAME="anyloc-drone-nav"
MISSION_DIR="./missions"
CONFIG_DIR="./config"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_jetson() {
    log_info "Checking Jetson environment..."
    
    # Check if running on Jetson
    if [ ! -f /etc/nv_tegra_release ]; then
        log_warning "Not running on Jetson device - some optimizations will be skipped"
        return 1
    fi
    
    # Check Jetson model
    if grep -q "Orin" /proc/device-tree/model 2>/dev/null; then
        log_success "Detected Jetson Orin"
    else
        log_warning "Jetson model not detected as Orin"
    fi
    
    # Check memory
    total_mem=$(free -g | awk '/^Mem:/ {print $2}')
    if [ "$total_mem" -lt 6 ]; then
        log_warning "Less than 6GB RAM detected (${total_mem}GB)"
    else
        log_success "Memory: ${total_mem}GB"
    fi
    
    return 0
}

check_docker() {
    log_info "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found! Please install Docker first:"
        echo "curl -fsSL https://get.docker.com -o get-docker.sh"
        echo "sudo sh get-docker.sh"
        echo "sudo usermod -aG docker $USER"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose not found! Please install Docker Compose"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon not running! Please start Docker service"
        exit 1
    fi
    
    log_success "Docker installation verified"
}

check_nvidia_docker() {
    log_info "Checking NVIDIA Docker runtime..."
    
    if ! docker info | grep -q nvidia; then
        log_error "NVIDIA Docker runtime not found! Please install nvidia-container-toolkit:"
        echo "distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
        echo "curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -"
        echo "curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list"
        echo "sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
        echo "sudo systemctl restart docker"
        exit 1
    fi
    
    log_success "NVIDIA Docker runtime available"
}

optimize_jetson() {
    log_info "Applying Jetson optimizations..."
    
    # Enable max performance mode
    if [ -f /usr/bin/jetson_clocks ]; then
        log_info "Enabling jetson_clocks..."
        sudo /usr/bin/jetson_clocks 2>/dev/null || log_warning "Failed to enable jetson_clocks"
    fi
    
    # Set power mode to MAXN
    if [ -f /usr/sbin/nvpmodel ]; then
        log_info "Setting power mode to MAXN..."
        sudo /usr/sbin/nvpmodel -m 0 2>/dev/null || log_warning "Failed to set power mode"
    fi
    
    # Optimize GPU
    if [ -d /sys/devices/gpu.0/devfreq ]; then
        log_info "Setting GPU to performance mode..."
        echo performance | sudo tee /sys/devices/gpu.0/devfreq/*/governor > /dev/null 2>&1 || true
    fi
    
    log_success "Jetson optimizations applied"
}

create_directories() {
    log_info "Creating directory structure..."
    
    # Create required directories
    mkdir -p "$MISSION_DIR"
    mkdir -p "$CONFIG_DIR"
    mkdir -p logs
    mkdir -p cache
    
    log_success "Directories created"
}

create_default_config() {
    log_info "Creating default configuration..."
    
    # Create default mission config
    cat > "$CONFIG_DIR/mission.yaml" << 'EOF'
model:
  name: "dinov2_vitl14"
  layer: 23
  facet: "value"
  resolution: [224, 224]
  quantization: true

vlad:
  clusters: 16
  feature_dim: 1024

navigation:
  confidence_threshold: 0.3
  max_fps: 30
  search_top_k: 5

jetson:
  enable_optimizations: true
  memory_fraction: 0.8
  enable_tensorrt: false

mission:
  package_path: "/opt/anyloc/missions/current_mission.pkl"
  log_level: "INFO"
  performance_monitoring: true
EOF
    
    # Create example mission package info
    cat > "$MISSION_DIR/README.md" << 'EOF'
# Mission Packages

Place your offline-processed mission packages here.

Example mission package structure:
- `current_mission.pkl` - Active mission data
- `backup_mission.pkl` - Backup mission data
- `test_mission.pkl` - Testing mission data

## Mission Package Contents:
- `database_descriptors`: Pre-computed VLAD descriptors
- `database_names`: Reference image names  
- `database_coordinates`: GPS coordinates for each reference
- `vlad_vocabulary`: VLAD cluster centers
- `config`: Processing configuration
- `metadata`: Mission metadata

## Creating Mission Packages:
Use the offline preprocessing script:
```bash
python3 offline_vs_realtime_processing.py --create-mission
```
EOF
    
    log_success "Default configuration created"
}

build_image() {
    log_info "Building Docker image..."
    
    # Build the image
    docker build -f Dockerfile.jetson -t "$IMAGE_NAME" .
    
    if [ $? -eq 0 ]; then
        log_success "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

deploy_containers() {
    log_info "Deploying containers..."
    
    # Stop existing containers
    docker-compose -f docker-compose.jetson.yml down 2>/dev/null || true
    
    # Deploy with docker-compose
    docker-compose -f docker-compose.jetson.yml up -d
    
    if [ $? -eq 0 ]; then
        log_success "Containers deployed successfully"
    else
        log_error "Failed to deploy containers"
        exit 1
    fi
}

check_deployment() {
    log_info "Checking deployment status..."
    
    # Wait for containers to start
    sleep 10
    
    # Check container status
    if docker ps | grep -q "$CONTAINER_NAME"; then
        log_success "Main navigation container is running"
    else
        log_error "Main navigation container failed to start"
        docker logs "$CONTAINER_NAME"
        exit 1
    fi
    
    # Check health
    log_info "Waiting for health check..."
    for i in {1..30}; do
        if docker inspect "$CONTAINER_NAME" | grep -q '"Health".*"healthy"'; then
            log_success "Container is healthy"
            break
        fi
        
        if [ $i -eq 30 ]; then
            log_warning "Health check timeout - container may still be starting"
        fi
        
        sleep 2
    done
}

show_status() {
    echo ""
    echo "🎉 DEPLOYMENT COMPLETED!"
    echo "======================="
    
    # Show container status
    docker-compose -f docker-compose.jetson.yml ps
    
    echo ""
    echo "📡 Service Endpoints:"
    echo "   Navigation API:     http://localhost:8080"
    echo "   Performance Monitor: http://localhost:8081" 
    echo "   Mission Upload:      http://localhost:8082"
    
    echo ""
    echo "🔧 Management Commands:"
    echo "   View logs:     docker logs $CONTAINER_NAME"
    echo "   Stop services: docker-compose -f docker-compose.jetson.yml down"
    echo "   Restart:       docker-compose -f docker-compose.jetson.yml restart"
    echo "   Shell access:  docker exec -it $CONTAINER_NAME /bin/bash"
    
    echo ""
    echo "📊 Performance Monitoring:"
    echo "   Container stats: docker stats $CONTAINER_NAME"
    echo "   Jetson stats:    tegrastats"
    echo "   GPU memory:      nvidia-smi"
    
    echo ""
    echo "🚁 Ready for drone navigation!"
}

# Main deployment flow
main() {
    log_info "Starting deployment process..."
    
    # Pre-flight checks
    check_jetson || log_warning "Jetson optimizations may not work fully"
    check_docker
    check_nvidia_docker
    
    # Jetson optimizations
    if check_jetson; then
        optimize_jetson
    fi
    
    # Setup
    create_directories
    create_default_config
    
    # Build and deploy
    build_image
    deploy_containers
    check_deployment
    
    # Show final status
    show_status
}

# Command line options
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "build")
        build_image
        ;;
    "start")
        docker-compose -f docker-compose.jetson.yml up -d
        ;;
    "stop")
        docker-compose -f docker-compose.jetson.yml down
        ;;
    "restart") 
        docker-compose -f docker-compose.jetson.yml restart
        ;;
    "logs")
        docker logs -f "$CONTAINER_NAME"
        ;;
    "shell")
        docker exec -it "$CONTAINER_NAME" /bin/bash
        ;;
    "status")
        docker-compose -f docker-compose.jetson.yml ps
        ;;
    "clean")
        docker-compose -f docker-compose.jetson.yml down
        docker rmi "$IMAGE_NAME" 2>/dev/null || true
        docker system prune -f
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  deploy   - Full deployment (default)"
        echo "  build    - Build Docker image only"
        echo "  start    - Start containers"
        echo "  stop     - Stop containers"
        echo "  restart  - Restart containers"
        echo "  logs     - Show container logs"
        echo "  shell    - Access container shell"
        echo "  status   - Show container status"
        echo "  clean    - Clean up containers and images"
        echo "  help     - Show this help"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac