# Level-2 EP1 episode-local localization

EP0 and EP1 GPS are converted into **episode-local** frames (origin at each
episode start, yaw 0 along that episode's start heading). The estimator never
reads Habitat world coordinates. World poses are used only to build GT
`T_ep1→ep0`.

## Method

**VLN lock (default):** DINO top-5 retrieval, SuperPoint+LightGlue matching,
depth 3D-3D SE(2) RANSAC. Lock if inliers ≥ 80 and unique vs the second
candidate (second ≥ 20 must lose by 1.5×). Then occupancy CSM refines T in
±0.5 m / ±15° only (no 90/180 search). See `geom_lock.py`.

**Occupancy CSM** remains for one-shot L2 eval and frontier exploration:

1. Build an EP0-local occupancy map from M0 RGB-D.
2. Take EP1's in-place spin at the episode start (or the first frame).
3. Match RGB thumbnails to M0 keyframes to seed `SE(2)`.
4. Correlative scan-match the EP1 depth occupancy onto the EP0 map.

## Run

```bash
/home/shizk/miniconda3/bin/python /home/shizk/LifeLongHomeNav/location/tests/test_geometry.py
/home/shizk/miniconda3/bin/python /home/shizk/LifeLongHomeNav/location/run_level2.py \
  --session-id level2_00824_003
/home/shizk/miniconda3/bin/python /home/shizk/LifeLongHomeNav/location/run_level2.py
```

Success bins: 0.10 m / 5°, 0.25 m / 10°, 0.50 m / 15°.
