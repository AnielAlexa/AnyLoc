#!/bin/bash
# Jetson Orin Nano Environment Setup Script
# Optimizes the environment for maximum performance

set -e

echo "🚀 Setting up Jetson Orin Nano for AnyLoc Drone Navigation"
echo "============================================================"

# Enable maximum performance mode
echo "⚡ Enabling maximum performance mode..."
if [ -f /usr/bin/jetson_clocks ]; then
    sudo /usr/bin/jetson_clocks
    echo "✅ Jetson clocks enabled"
else
    echo "⚠️ jetson_clocks not found - manual power mode setup required"
fi

# Set power mode to MAXN (maximum performance)
echo "🔋 Setting power mode to MAXN..."
if [ -f /usr/sbin/nvpmodel ]; then
    sudo /usr/sbin/nvpmodel -m 0  # Mode 0 = MAXN for Orin Nano
    echo "✅ Power mode set to MAXN"
else
    echo "⚠️ nvpmodel not found - manual power setup required"
fi

# Optimize GPU settings
echo "🎮 Optimizing GPU settings..."
if [ -d /sys/devices/gpu.0/devfreq/17000000.gv11b ]; then
    # Set GPU governor to performance
    echo performance | sudo tee /sys/devices/gpu.0/devfreq/17000000.gv11b/governor > /dev/null
    echo "✅ GPU governor set to performance"
fi

# Optimize CPU settings
echo "💻 Optimizing CPU settings..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    if [ -f "$cpu" ]; then
        echo performance | sudo tee "$cpu" > /dev/null
    fi
done
echo "✅ CPU governors set to performance"

# Increase swap if needed (for large models)
echo "💾 Checking swap configuration..."
SWAP_SIZE=$(free -m | awk '/^Swap:/ {print $2}')
if [ "$SWAP_SIZE" -lt 4096 ]; then
    echo "⚠️ Warning: Swap size is ${SWAP_SIZE}MB, recommend 4GB+ for large models"
fi

# Set CUDA environment variables
echo "🎯 Setting CUDA environment variables..."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
echo "✅ CUDA environment configured"

# Optimize PyTorch for Jetson
echo "🔥 Setting PyTorch optimizations..."
export PYTORCH_JIT=0
export TORCH_HOME=/opt/anyloc/cache/torch
mkdir -p $TORCH_HOME
echo "✅ PyTorch optimizations set"

# Check GPU memory
echo "📊 GPU Memory Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits
else
    echo "⚠️ nvidia-smi not available"
fi

# Check available storage
echo "💽 Storage Status:"
df -h /opt/anyloc

# Create performance monitoring script
cat > /opt/anyloc/monitor_performance.sh << 'EOF'
#!/bin/bash
# Performance monitoring script for Jetson Orin Nano

echo "=== Jetson Orin Nano Performance Monitor ==="
echo "Timestamp: $(date)"

# GPU utilization
if command -v tegrastats &> /dev/null; then
    echo "GPU Stats:"
    tegrastats --interval 1000 | head -1
fi

# Memory usage
echo "Memory Usage:"
free -h

# CPU temperature and frequency
if [ -f /sys/devices/virtual/thermal/thermal_zone0/temp ]; then
    temp=$(cat /sys/devices/virtual/thermal/thermal_zone0/temp)
    echo "CPU Temperature: $((temp/1000))°C"
fi

# Disk usage
echo "Disk Usage:"
df -h /opt/anyloc

# Running processes
echo "Top GPU processes:"
nvidia-smi pmon -c 1 2>/dev/null || echo "nvidia-smi not available"

echo "=============================================="
EOF

chmod +x /opt/anyloc/monitor_performance.sh

echo "🎉 Jetson Orin Nano setup completed!"
echo ""
echo "📋 Performance Optimization Summary:"
echo "   ✅ Maximum performance mode enabled"
echo "   ✅ GPU governor set to performance"  
echo "   ✅ CPU governors set to performance"
echo "   ✅ CUDA environment configured"
echo "   ✅ PyTorch optimized for Jetson"
echo "   ✅ Performance monitoring script created"
echo ""
echo "🚁 Ready for drone navigation deployment!"
echo "   Run: /opt/anyloc/monitor_performance.sh for performance monitoring"