
# Environment Installation
conda create -n jaltol-geo -c conda-forge python=3.11 geopandas rioxarray rasterio shapely pyproj pandas tqdm xarray dask

# Test 1
Goal: compute village-wise area of a single LULC class for all 18 years for a single village

Script works by looping through all years' rasters and then clipping the raster for each village, following which the pixel count happens.

```bash
python compute_lulc_area.py
```
Create a config file `compute_lulc_area.config.json`

## Learnings
- re-projection of raster takes time, total computation took 395s

# Test 2
Manually re-projected all rasters to UTM

```bash
python compute_lulc_area.py 
```
for a single LULC class for all 18 years for a single village (with projected rasters)

## Learnings
- total computation took 180s, repeated it for 2 villages and total time was 347s.
- after logging we figured out that file IO time was minimal, clipping was taking ~10-15s each village and year.
- moreover the village clip is repeated every year, but this isn't strictly necessary, rasterization of the village is needed only once.

# Test 3
One-shot village rasterization + vectorized counting (replace per-polygon clip)
What: For each year, rasterize all villages once into a “label raster” (burn pc11_tv_id as pixel values) aligned to the LULC grid. Then compute mask = (lulc == class_value) and use a single np.bincount(label[mask]) to get per-village pixel counts in one pass.
Why it helps: Eliminates thousands of rasterio.clip calls; turns many small window reads into 1-2 large sequential reads and pure NumPy counting

Create a config file `compute_lulc_area_bincount.bincount.config.json`

```bash
python compute_lulc_area_bincount.py
```

## Learnings
- some years have different raster grids/transforms and the script isn't resilient to this.


# Test 4
Same as Test 3, but resilient to changes in raster grids/transforms

## Learnings
- two villages computation for 18 years takes 48s (24s per village)





# Future Optimizations
Precompute and reuse polygon masks across years.
What: For each village, rasterize its mask once on the common grid (store the minimal bounding window + boolean mask). For each year, slice the window from the year’s raster and apply the precomputed mask to count pixels.
Why it helps: You pay the expensive geometry→mask step once, then reuse it for all years. IO becomes window reads; no repeated clipping.

Make IO and masking cheaper (COG tiling + polygon simplification)
What: Convert rasters to Cloud Optimized GeoTIFFs with internal tiling (e.g., 512×512), overviews, and a fast compression (DEFLATE/ZSTD) to make window reads fast and cache-friendly.
Simplify village geometries to raster resolution (e.g., tolerance ~ half pixel size), preserving topology.

Why it helps: Window reads stop thrashing the decompressor; masking fewer vertices cuts mask generation time sharply.

Load full raster once per year into memory
- If each raster fits in RAM (y≈13.7k, x≈8.9k → ~121M px; uint8 ≈ 120 MB), reading once and doing all operations in-memory can remove most disk latency: 3–6×.

Use all_touched=false and match simplification tolerance to pixel size
- Keeps masks small and consistent; often 1.2–2× faster on complex polygons.

Enable Dask chunking if memory-bound
- Helps overlap IO/compute and reuse chunks; 1.5–3× if you cannot hold full rasters.
