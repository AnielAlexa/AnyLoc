DINOv2 ViT-S Layer Testing Pipeline
====================================

This pipeline tests different DINOv2 ViT-S layers and attention facets to find
the optimal configuration for drone navigation with aerial imagery.


QUICK START
-----------

1. Configure your area in config.yaml (coordinates, zoom level, etc.)

2. Run the pipeline in order:

   Step 1: Download satellite imagery
   $ python 1_download_satellite.py

   Step 2: Compute descriptors for all configurations
   $ python 2_compute_descriptors.py

   Step 3: Match video with GUI (interactive)
   $ python 3_match_video.py

   Step 4: Analyze and compare results
   $ python 4_analyze_results.py


PIPELINE OVERVIEW
-----------------

Step 1: Download Satellite Patches
   - Downloads satellite imagery from Esri World Imagery
   - Splits into 640x640 patches with 50% overlap
   - Generates GPS metadata for each patch
   - Output: data/patches/

Step 2: Compute Descriptors
   - Tests layers 8, 9, 10, 11
   - Tests facets: key, query, value
   - Total: 12 configurations
   - Fits VLAD clusters on satellite patches
   - Output: data/descriptors/layer_X_facet/

Step 3: Match Video
   - Interactive GUI for real-time matching
   - Processes every 5th frame from flight_100m.mp4
   - Shows top-5 matches with confidence scores
   - Displays best match satellite image
   - Output: data/results/flight_100m_results.json

Step 4: Analyze Results
   - Compares all 12 configurations
   - Generates 4 comparison plots
   - Prints summary table with recommendations
   - Output: data/visualizations/*.png


CONFIGURATION
-------------

Edit config.yaml to customize:

- area: Your GPS bounding box (min/max lat/lon)
- satellite.zoom_level: 19 for 100m altitude (0.30 m/pixel)
- patches.patch_size: 640x640 pixels (default)
- patches.overlap: 0.5 (50% overlap)
- dinov2.layers_to_test: [8, 9, 10, 11]
- dinov2.facets_to_test: ["key", "query", "value"]
- vlad.num_clusters: 16 (standard)
- video.frame_sampling: 5 (process every 5th frame)


OUTPUT STRUCTURE
----------------

data/
├── patches/                      # Satellite image patches
│   ├── patch_0_0.png
│   ├── patch_0_1.png
│   └── gps_metadata.json
├── descriptors/                  # VLAD descriptors
│   ├── layer_8_key/
│   │   ├── vlad_descriptors.pt
│   │   ├── c_centers.pt
│   │   ├── patch_names.txt
│   │   └── metadata.json
│   └── ... (12 configurations)
├── results/                      # Matching results
│   └── flight_100m_results.json
└── visualizations/               # Analysis plots
    ├── layer_comparison.png
    ├── performance_metrics.png
    ├── confidence_trajectory.png
    └── facet_comparison.png


EXPECTED RESULTS
----------------

Based on AnyLoc research:
- Layers 9-10 typically perform best
- "key" facet often outperforms "query" and "value"
- Expected FPS: 18-20 on RTX 3070
- Confidence scores: 0.6-0.8 for good matches

The pipeline will identify the best configuration for your specific area.


TROUBLESHOOTING
---------------

1. "Video not found" error:
   - Make sure flight_100m.mp4 is in the parent directory
   - Or update video.path in config.yaml

2. CUDA out of memory:
   - Reduce patches.patch_size to 512 or 448
   - Process fewer frames (increase video.frame_sampling)

3. Slow downloads:
   - Increase satellite.delay_between_requests (e.g., 0.2)
   - Check internet connection

4. Import errors:
   - Make sure you're running from the correct directory
   - Parent utilities.py must be accessible
   - Install requirements: pip install -r requirements.txt


DEPENDENCIES
------------

See requirements.txt for full list.

Main dependencies:
- PyTorch 2.0+
- torchvision
- opencv-python
- PIL/Pillow
- matplotlib
- faiss-gpu
- einops
- fast-pytorch-kmeans


ESTIMATED RUNTIME
-----------------

For 12 patches, 12 configurations, 60 video frames:

1. Download satellite: ~10 seconds
2. Compute descriptors: ~2-3 minutes
3. Match video: ~3-4 seconds per configuration (GUI)
4. Analyze results: <5 seconds

Total: ~5-10 minutes for complete pipeline


NOTES
-----

- Pipeline reuses utilities from parent AnyLoc directory
- All paths are relative to this directory
- Results are cached for reuse
- GPU (CUDA) strongly recommended
- Zoom level 19 optimal for 100m altitude drone flights


For questions or issues, refer to the main AnyLoc documentation.
