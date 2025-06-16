# Drone Navigation System

Real-time drone localization using satellite image matching with DINOv2 + VLAD.

## 🚁 Complete Workflow

### 1. Offline: Build Satellite Database

```bash
# Prepare satellite images with GPS coordinates
python3 drone_navigation_system.py \
    --mode build-database \
    --satellite-dir ./satellite_images \
    --gps-file ./gps_metadata.json \
    --database-path ./my_area_db
```

**Required:**
- Satellite images (PNG/JPG) covering your flight area
- GPS metadata JSON file with coordinates for each image

### 2. Runtime: Live Drone Navigation

```bash
# Run real-time drone matching
python3 drone_navigation_system.py \
    --mode drone-navigation \
    --camera 0 \
    --database-path ./my_area_db
```

**Controls:**
- `q`: Quit navigation
- `s`: Save current frame
- `g`: Print current GPS estimate

## 📊 Performance Expectations

| Hardware | Expected FPS | Use Case |
|----------|-------------|----------|
| RTX 3070 Laptop | 35-40 FPS | Development/Testing |
| Jetson Orin Nano | 10-15 FPS | Production Drone |
| Jetson Xavier NX | 15-20 FPS | High-end Drone |

## 🛰️ GPS Metadata Format

Create `gps_metadata.json`:

```json
{
  "image_name_001": {
    "lat": 40.7589,
    "lon": -73.9851,
    "altitude": 100,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## 🔧 System Requirements

**For Database Building:**
- Satellite images covering flight area
- GPS coordinates for each image
- GPU with 4GB+ VRAM

**For Real-Time Navigation:**
- Camera/video feed from drone
- Pre-built satellite database
- GPU acceleration (recommended)

## 📈 Optimization Tips

### For Jetson Devices:
1. **FP16 Precision**: Automatically enabled for Jetson
2. **Frame Skipping**: Process every 3rd frame (10+ FPS effective)
3. **Lower Resolution**: Use 196x196 for maximum speed
4. **TensorRT**: Convert model for 2-3x speedup

### For Maximum Accuracy:
1. **High-Resolution Satellites**: Use 18+ zoom level
2. **Dense Coverage**: Overlap satellite images by 50%
3. **Recent Images**: Use up-to-date satellite data
4. **Consistent Altitude**: Match drone height to satellite perspective

## 🎯 Use Cases

### 1. Autonomous Navigation
- GPS-denied environments
- Urban canyon navigation
- Emergency landing site detection

### 2. Precision Agriculture
- Field boundary detection
- Crop monitoring alignment
- Irrigation system navigation

### 3. Search and Rescue
- Terrain-relative navigation
- Lost drone recovery
- Disaster area mapping

### 4. Military/Security
- GPS-jammed environments
- Covert operations
- Border monitoring

## 📊 Accuracy Expectations

| Environment | Accuracy | Notes |
|-------------|----------|-------|
| Rural/Agricultural | 85-95% | Best performance |
| Urban Areas | 70-85% | Good with recent satellites |
| Dense Urban | 60-75% | Challenging due to shadows |
| Changing Terrain | 50-70% | Requires frequent updates |

## 🔄 Real-World Deployment

### Step 1: Area Preparation
```bash
# Download satellite tiles for your area
# Convert to standard format (PNG/JPG)
# Create GPS metadata file
```

### Step 2: Database Creation
```bash
python3 drone_navigation_system.py --mode build-database \
    --satellite-dir ./flight_area_satellites \
    --gps-file ./area_gps.json
```

### Step 3: Drone Integration
```python
# In your drone control code
system = DroneNavigationSystem("./my_area_db")
system.load_satellite_database()

# For each drone frame
result = system.match_drone_frame(drone_image)
if result['is_valid_match']:
    gps_estimate = result['gps_estimate']
    # Use GPS estimate for navigation
```

## 🚀 Advanced Features

### Multi-Area Databases
Handle multiple flight areas by switching databases:

```python
area_databases = {
    "area_a": "./database_area_a",
    "area_b": "./database_area_b"
}

# Switch based on rough GPS or mission plan
current_db = area_databases[current_area]
system = DroneNavigationSystem(current_db)
```

### Confidence-Based Navigation
```python
result = system.match_drone_frame(frame)

if result['confidence'] == 'HIGH':
    # Use GPS estimate directly
    navigate_to_gps(result['gps_estimate'])
elif result['confidence'] == 'MEDIUM':
    # Use with caution, combine with other sensors
    weighted_gps = combine_with_imu(result['gps_estimate'])
else:
    # Fall back to dead reckoning
    use_last_known_position()
```

This system provides a complete solution for drone navigation using visual place recognition!