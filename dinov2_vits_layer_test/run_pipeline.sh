#!/bin/bash
# Quick-start script to run the entire pipeline

echo "=========================================="
echo "DINOv2 ViT-S Layer Testing Pipeline"
echo "=========================================="
echo ""

# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo "❌ Error: config.yaml not found!"
    echo "Please create config.yaml first."
    exit 1
fi

# Check if flight_100m.mp4 exists
if [ ! -f "../flight_100m.mp4" ]; then
    echo "⚠️  Warning: ../flight_100m.mp4 not found!"
    echo "Make sure the video file is in the parent directory."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Step 1: Downloading satellite imagery..."
echo "=========================================="
python 1_download_satellite.py
if [ $? -ne 0 ]; then
    echo "❌ Step 1 failed!"
    exit 1
fi

echo ""
echo "Step 2: Computing descriptors for all configurations..."
echo "=========================================="
python 2_compute_descriptors.py
if [ $? -ne 0 ]; then
    echo "❌ Step 2 failed!"
    exit 1
fi

echo ""
echo "Step 3: Matching video (GUI will open)..."
echo "=========================================="
echo "Note: This step requires manual interaction with the GUI."
echo "Process all configurations and close the GUI when done."
echo ""
read -p "Press Enter to launch GUI..."
python 3_match_video.py
if [ $? -ne 0 ]; then
    echo "❌ Step 3 failed!"
    exit 1
fi

echo ""
echo "Step 4: Analyzing results..."
echo "=========================================="
python 4_analyze_results.py
if [ $? -ne 0 ]; then
    echo "❌ Step 4 failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Pipeline completed successfully!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Patches: data/patches/"
echo "  - Descriptors: data/descriptors/"
echo "  - Results: data/results/flight_100m_results.json"
echo "  - Plots: data/visualizations/"
echo ""
echo "Check data/visualizations/ for comparison plots!"
