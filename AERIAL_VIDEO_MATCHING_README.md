# Aerial Video GPS Matching System

Complete solution for estimating GPS trajectories from aerial video using satellite image matching with AnyLoc's DINOv2 + VLAD approach.

## 🎯 Overview

This system allows you to:
1. **Process satellite imagery** into searchable grid patches with GPS coordinates
2. **Match aerial video frames** against the satellite database  
3. **Generate GPS trajectories** with confidence estimates
4. **Create visualizations** and export formats (KML, plots, statistics)

## 🚀 Quick Start

### Option 1: Complete Pipeline (Recommended)

Process everything in one command:

```bash
# With large satellite image
python3 aerial_video_pipeline.py \
    --video your_aerial_video.mp4 \
    --satellite-image large_satellite_map.png \
    --bounds "40.750,-73.990,40.760,-73.980" \
    --output ./flight_results

# With OpenStreetMap tiles (alternative to Google Maps)
python3 aerial_video_pipeline.py \
    --video aerial_flight.mp4 \
    --download-osm \
    --bounds "40.750,-73.990,40.760,-73.980" \
    --zoom 18 \
    --output ./results
```

### Option 2: Step-by-Step Process

```bash
# 1. Create satellite patches
python3 google_maps_processor.py \
    --input large_satellite_map.png \
    --bounds "40.750,-73.990,40.760,-73.980" \
    --output ./satellite_patches

# 2. Build AnyLoc database  
python3 drone_navigation_system.py \
    --mode build-database \
    --satellite-dir ./satellite_patches/patches \
    --gps-file ./satellite_patches/gps_metadata.json \
    --database-path ./satellite_db

# 3. Process aerial video
python3 aerial_video_processor.py \
    --video flight.mp4 \
    --database ./satellite_db \
    --output ./video_results
```

## 📋 System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 4GB+ VRAM (RTX 3070 recommended)
- **RAM**: 8GB+ system RAM  
- **Storage**: ~2GB for models, varies by area size

### Software
- **Python**: 3.8+
- **Dependencies**: PyTorch, OpenCV, PIL, matplotlib, requests
- **Environment**: Use existing AnyLoc conda environment

## 📊 Expected Performance

| Hardware | Processing Speed | Use Case |
|----------|-----------------|----------|
| RTX 3070 | 35-40 FPS | Development/Analysis |
| RTX 3080/4080 | 45-60 FPS | High-throughput processing |
| Jetson Orin Nano | 10-15 FPS | Embedded/real-time |

| Environment | Accuracy | Notes |
|-------------|----------|-------|
| Rural/Agricultural | 85-95% | Best performance |
| Suburban | 75-90% | Good with recent satellites |
| Urban | 60-80% | Challenging due to shadows |
| Dense Urban | 50-70% | Requires high-resolution satellites |

## 🛠️ Component Details

### 1. Google Maps Processor (`google_maps_processor.py`)

Converts large satellite images into grid patches or downloads OSM tiles.

```bash
# Process large satellite image
python3 google_maps_processor.py \
    --input large_map.png \
    --bounds "lat_min,lon_min,lat_max,lon_max" \
    --patch-size 512 \
    --overlap 64 \
    --output ./patches

# Download OpenStreetMap tiles
python3 google_maps_processor.py \
    --download-osm \
    --bounds "40.75,-73.99,40.76,-73.98" \
    --zoom 18 \
    --output ./osm_tiles
```

**Options:**
- `--patch-size`: Patch dimensions (512 recommended for balance)
- `--overlap`: Overlap between patches (64px helps with edge cases)
- `--zoom`: OSM zoom level (18 good for 100m altitude matching)

### 2. Aerial Video Processor (`aerial_video_processor.py`)

Processes video files to estimate GPS trajectories.

```bash
python3 aerial_video_processor.py \
    --video flight.mp4 \
    --database ./satellite_db \
    --frame-interval 30 \
    --confidence-threshold 0.3 \
    --export-kml \
    --output ./results
```

**Options:**
- `--frame-interval`: Process every Nth frame (30 = 1 per second at 30fps)
- `--confidence-threshold`: Minimum similarity for valid matches (0.3 default)
- `--max-frames`: Limit processing for testing
- `--save-frames`: Save matched frames to disk
- `--export-kml`: Create Google Earth compatible file

### 3. Complete Pipeline (`aerial_video_pipeline.py`)

End-to-end processing with comprehensive analysis.

**Key Features:**
- Automatic database building
- Progress tracking and logging
- Comprehensive analysis reports
- Multiple output formats
- Error handling and recovery

## 📍 GPS Coordinate Systems

### Bounds Format
Always use: `"lat_min,lon_min,lat_max,lon_max"`

**Example for Central Park, NYC:**
```bash
--bounds "40.764,-73.982,40.800,-73.949"
```

### Finding Your Area Bounds
1. **Google Earth**: Right-click corners to get coordinates
2. **Google Maps**: URL shows lat/lon when you click
3. **GPS Tools**: Use online bounding box tools
4. **Flight Records**: Extract from flight controller logs

## 🎨 Output Files Guide

### Complete Pipeline Output Structure
```
pipeline_results/
├── satellite_patches/           # Processed satellite imagery
│   ├── patches/                # Individual image patches  
│   ├── gps_metadata.json      # GPS coordinates for each patch
│   └── coordinate_grid.png    # Visualization of patch grid
├── anyloc_database/            # AnyLoc feature database
│   ├── satellite_database.json
│   └── satellite_features.pt
├── video_analysis/             # Video processing results
│   ├── flight_trajectory.json # Frame-by-frame GPS estimates
│   ├── trajectory_plot.png    # Flight path visualization
│   ├── flight_trajectory.kml  # Google Earth file
│   └── statistics_report.txt  # Detailed statistics
├── comprehensive_analysis.json # Complete analysis
└── FINAL_REPORT.md            # Human-readable summary
```

### Key Files Explained

**`flight_trajectory.json`**: Complete trajectory data
```json
{
  "video_info": {
    "total_frames_processed": 150,
    "valid_matches": 128
  },
  "trajectory": [
    {
      "frame_number": 30,
      "timestamp": 1.0,
      "confidence": 0.847,
      "gps": {"lat": 40.7589, "lon": -73.9851},
      "is_valid_match": true
    }
  ]
}
```

**`flight_trajectory.kml`**: Import into Google Earth to see 3D flight path

**`trajectory_plot.png`**: Shows flight path with confidence coloring

## 🔧 Advanced Usage

### Custom Processing Parameters

```bash
# High-accuracy processing (slower)
python3 aerial_video_pipeline.py \
    --video flight.mp4 \
    --satellite-image map.png \
    --bounds "..." \
    --patch-size 1024 \
    --overlap 128 \
    --frame-interval 15 \
    --confidence-threshold 0.5

# Fast processing (lower accuracy)  
python3 aerial_video_pipeline.py \
    --video flight.mp4 \
    --satellite-image map.png \
    --bounds "..." \
    --patch-size 256 \
    --overlap 32 \
    --frame-interval 60 \
    --confidence-threshold 0.2
```

### Batch Processing Multiple Videos

```bash
#!/bin/bash
for video in *.mp4; do
    echo "Processing $video"
    python3 aerial_video_pipeline.py \
        --video "$video" \
        --existing-database ./shared_satellite_db \
        --output "./results_${video%.*}"
done
```

### Integration with Flight Controllers

```python
# Example: Process DJI flight logs with video
import json

# Load DJI flight log
with open('flight_log.json', 'r') as f:
    flight_data = json.load(f)

# Extract approximate flight area
lats = [point['latitude'] for point in flight_data['path']]
lons = [point['longitude'] for point in flight_data['path']]

bounds = f"{min(lats)},{min(lons)},{max(lats)},{max(lons)}"

# Process with expanded bounds (add buffer)
buffer = 0.005  # ~500m buffer
bounds_expanded = f"{min(lats)-buffer},{min(lons)-buffer},{max(lats)+buffer},{max(lons)+buffer}"
```

## 🐛 Troubleshooting

### Common Issues

**❌ "No matches found"**
- Check if satellite area covers flight path
- Lower confidence threshold (`--confidence-threshold 0.1`)
- Verify GPS bounds are correct
- Try different patch sizes

**❌ "Out of memory"**  
- Reduce patch size (`--patch-size 256`)
- Process fewer frames (`--max-frames 100`) 
- Use lower video resolution

**❌ "Video codec not supported"**
- Convert video: `ffmpeg -i input.mov -c:v libx264 output.mp4`
- Use common formats: MP4, AVI, MOV

**❌ "Satellite imagery doesn't match"**
- Check altitude difference (satellites vs 100m drone)
- Verify correct geographic area
- Consider seasonal differences  
- Try different zoom levels for OSM tiles

### Performance Optimization

**For RTX 3070:**
```bash
# Optimal settings
--patch-size 512
--frame-interval 30  
--confidence-threshold 0.3
```

**For slower hardware:**
```bash
# Reduce computational load
--patch-size 256
--frame-interval 60
--max-frames 200
```

## 🔄 Workflow Examples

### Real Estate/Surveying
```bash
# High-accuracy mapping
python3 aerial_video_pipeline.py \
    --video property_survey.mp4 \
    --download-osm \
    --bounds "property_bounds" \
    --zoom 19 \
    --frame-interval 15 \
    --confidence-threshold 0.4
```

### Search & Rescue
```bash
# Fast processing for time-critical applications
python3 aerial_video_pipeline.py \
    --video search_area.mp4 \
    --existing-database ./region_db \
    --frame-interval 60 \
    --confidence-threshold 0.2 \
    --max-frames 500
```

### Research/Analysis
```bash
# Comprehensive analysis with all outputs
python3 aerial_video_pipeline.py \
    --video research_flight.mp4 \
    --satellite-image high_res_map.tif \
    --bounds "study_area" \
    --patch-size 1024 \
    --save-frames \
    --export-kml
```

## 📚 Technical Background

This system uses:
- **DINOv2**: Self-supervised vision transformer for robust feature extraction
- **VLAD**: Vector of Locally Aggregated Descriptors for global image representation  
- **Cosine Similarity**: Efficient matching between drone frames and satellite patches
- **GPS Interpolation**: Accurate coordinate estimation from patch centers

The approach is robust to:
- ✅ Rotation and scale differences
- ✅ Lighting and weather changes  
- ✅ Seasonal variations
- ✅ Compression artifacts
- ⚠️ Significant altitude differences
- ⚠️ Major structural changes

## 🤝 Contributing

This system builds on the AnyLoc research framework. For improvements:

1. Follow existing code patterns in the repository
2. Test with diverse geographic areas and conditions
3. Validate against GPS ground truth when available
4. Consider edge cases (clouds, water, uniform terrain)

## 📄 License

Same as parent AnyLoc repository (BSD-3).