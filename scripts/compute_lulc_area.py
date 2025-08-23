import argparse
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray  # noqa: F401  (ensures .rio accessor is registered)
import xarray as xr
from shapely.geometry import mapping
from tqdm import tqdm


def parse_hydro_year_from_name(path: Path) -> str:
    # expects "proj32643_lulc250k_[YEAR]_XXXXX.tif", e.g., proj32643_lulc250k_0506_21475.tif
    m = re.search(r"proj32643_lulc250k_(\d{4})_", path.name, flags=re.IGNORECASE)
    return m.group(1) if m else ""


def open_raster(path: Path) -> xr.DataArray:
    da = xr.open_dataarray(path, engine="rasterio", masked=True)
    # Drop band dimension if present
    if "band" in da.dims and da.sizes.get("band", 1) == 1:
        da = da.isel(band=0, drop=True)
    return da


def ensure_projected_crs(da: xr.DataArray) -> bool:
    crs = da.rio.crs
    try:
        return crs is not None and getattr(crs, "is_projected", False)
    except Exception:
        return False


def count_class_pixels_in_polygon(da: xr.DataArray, geom, class_value: int, all_touched: bool) -> int:
    clipped = da.rio.clip([mapping(geom)], da.rio.crs, drop=True, all_touched=all_touched)
    # Use masked array-aware counting
    arr = np.asarray(clipped.data)
    if np.ma.isMaskedArray(arr):
        valid_mask = ~arr.mask
        data = arr.data
        return int(((data == class_value) & valid_mask).sum())
    return int((arr == class_value).sum())


def timed_count_class_pixels_in_polygon(
    da: xr.DataArray, geom, class_value: int, all_touched: bool
) -> tuple[int, float, float, int]:
    """Return (count, clip_seconds, count_seconds, num_elements_considered)."""
    t0 = time.perf_counter()
    clipped = da.rio.clip([mapping(geom)], da.rio.crs, drop=True, all_touched=all_touched)
    t1 = time.perf_counter()
    arr = np.asarray(clipped.data)
    if np.ma.isMaskedArray(arr):
        valid_mask = ~arr.mask
        data = arr.data
        c = int(((data == class_value) & valid_mask).sum())
    else:
        c = int((arr == class_value).sum())
    t2 = time.perf_counter()
    return c, (t1 - t0), (t2 - t1), int(arr.size)


def get_default_config_path() -> Path:
    return Path(__file__).with_suffix(".config.json")


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_log_level(level_name: str) -> int:
    mapping = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    return mapping.get(str(level_name).upper(), logging.INFO)


def setup_logging(level: int, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("compute_lulc_area")
    logger.setLevel(level)
    # Avoid duplicate handlers if re-run in the same interpreter
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger


def main():
    parser = argparse.ArgumentParser(description="Compute LULC class area per village for each raster year.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(get_default_config_path()),
        help="Path to JSON config file. Defaults to compute_lulc_area.config.json next to this script.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sample = {
            "raster_dir": "D:/Code/Jaltol/JaltolPreProcessing/data/raster/Karnataka",
            "vector": "D:/Code/Jaltol/JaltolPreProcessing/data/vector/Karnataka/villages.shp",
            "class_value": 3,
            "out_csv": "D:/Code/Jaltol/JaltolPreProcessing/outputs/lulc_area_by_village.csv",
            "all_touched": False,
            "target_epsg": None,
            "limit_polygons": None,
        }
        raise FileNotFoundError(
            f"Config file not found: {cfg_path}\nCreate it as JSON with keys like:\n{sample}"
        )
    cfg = load_config(cfg_path)

    log_level = parse_log_level(cfg.get("log_level", "INFO"))
    log_file = cfg.get("log_file")
    logger = setup_logging(log_level, log_file)
    logger.info("Starting LULC area computation")
    logger.info(f"Using config: {cfg_path}")

    raster_dir = Path(cfg["raster_dir"])
    shp_path = Path(cfg["vector"])
    out_csv = Path(cfg["out_csv"])
    class_value = int(cfg["class_value"])
    all_touched = bool(cfg.get("all_touched", False))
    target_epsg = cfg.get("target_epsg")
    limit_polygons = cfg.get("limit_polygons")
    slow_polygon_ms = float(cfg.get("slow_polygon_ms", 1500))  # default 1.5s
    log_every_n_polygons = int(cfg.get("log_every_n_polygons", 500))

    # Find target rasters
    tifs = sorted(raster_dir.glob("proj32643_lulc250k_*.tif"))
    if not tifs:
        logger.error(f"No rasters found in {raster_dir}")
        raise FileNotFoundError(f"No rasters found in {raster_dir}")
    logger.info(f"Found {len(tifs)} rasters in {raster_dir}")

    # Load villages
    logger.info(f"Loading villages from {shp_path}")
    villages = gpd.read_file(shp_path)
    if "pc11_tv_id" not in villages.columns:
        logger.error("Shapefile must contain 'pc11_tv_id' column.")
        raise KeyError("Shapefile must contain 'pc11_tv_id' column.")
    villages = villages[["pc11_tv_id", "geometry"]].copy()
    villages = villages[~villages.geometry.is_empty & villages.geometry.notna()].reset_index(drop=True)

    if limit_polygons:
        villages = villages.iloc[: limit_polygons].copy()
    logger.info(f"Prepared {len(villages)} village polygons for processing")

    results = []

    overall_start = time.perf_counter()
    for tif_path in tqdm(tifs, desc="Rasters"):
        hydro_year = parse_hydro_year_from_name(tif_path)
        if not hydro_year:
            # Skip files that do not match naming convention
            logger.warning(f"Skipping file not matching naming convention: {tif_path.name}")
            continue

        raster_start = time.perf_counter()
        logger.info(f"Opening raster {tif_path.name} (year {hydro_year})")
        try:
            t_open0 = time.perf_counter()
            da = open_raster(tif_path)
            t_open1 = time.perf_counter()
            logger.debug(f"Raster open time: {(t_open1 - t_open0):.3f}s | dims: {dict(da.sizes)} | dtype: {str(da.dtype)}")
        except Exception:
            logger.exception(f"Failed to open raster: {tif_path}")
            continue

        # Reproject raster if requested (e.g., to an equal-area CRS)
        if target_epsg:
            target = f"EPSG:{target_epsg}"
            logger.info(f"Reprojecting raster to {target}")
            try:
                t_reproj0 = time.perf_counter()
                da = da.rio.reproject(target)
                t_reproj1 = time.perf_counter()
                logger.debug(f"Raster reproject time: {(t_reproj1 - t_reproj0):.3f}s")
            except Exception:
                logger.exception(f"Failed to reproject raster: {tif_path} -> {target}")
                try:
                    da.close()
                except Exception:
                    pass
                continue

        # Warn if not projected (areas would be incorrect)
        if not ensure_projected_crs(da):
            logger.warning(f"{tif_path.name} has non-projected CRS. Set target_epsg in config for correct areas.")

        # Prepare vectors in raster CRS
        raster_crs = da.rio.crs
        try:
            t_vproj0 = time.perf_counter()
            v_in = villages.to_crs(raster_crs)
            t_vproj1 = time.perf_counter()
            logger.debug(f"Vector to CRS time: {(t_vproj1 - t_vproj0):.3f}s | CRS: {raster_crs.to_string() if raster_crs else 'None'}")
        except Exception:
            logger.exception("Failed to project villages to raster CRS")
            try:
                da.close()
            except Exception:
                pass
            continue

        # Pixel area in m²
        xres, yres = da.rio.resolution()
        pixel_area_m2 = abs(xres * yres)
        factor_to_ha = 1.0 / 10000.0
        logger.debug(f"Raster resolution: {xres} x {yres} (m), pixel area: {pixel_area_m2} m²")

        processed_polygons = 0
        clip_time_sum = 0.0
        count_time_sum = 0.0
        elems_sum = 0
        slow_polygons = 0
        for idx, (_, row) in enumerate(v_in.iterrows(), start=1):
            try:
                class_pixels, clip_s, count_s, elems = timed_count_class_pixels_in_polygon(
                    da=da,
                    geom=row.geometry,
                    class_value=class_value,
                    all_touched=all_touched,
                )
                clip_time_sum += clip_s
                count_time_sum += count_s
                elems_sum += elems
                area_ha = class_pixels * pixel_area_m2 * factor_to_ha
                results.append(
                    {
                        "pc11_tv_id": row["pc11_tv_id"],
                        "hydrological_year": hydro_year,
                        "area_ha": float(area_ha),
                    }
                )
                processed_polygons += 1
                total_ms = (clip_s + count_s) * 1000.0
                if total_ms >= slow_polygon_ms:
                    slow_polygons += 1
                    logger.debug(
                        f"Slow polygon pc11_tv_id={row['pc11_tv_id']} | {total_ms:.1f}ms (clip={clip_s*1000:.1f}ms, count={count_s*1000:.1f}ms, elems={elems})"
                    )
                if log_every_n_polygons and (idx % log_every_n_polygons == 0):
                    logger.debug(
                        f"Progress {idx}/{len(v_in)} polygons | avg clip={clip_time_sum/idx:.4f}s avg count={count_time_sum/idx:.4f}s"
                    )
            except Exception as exc:
                # Skip invalid geometries or other localized issues
                logging.getLogger("compute_lulc_area").exception(
                    f"Polygon {row['pc11_tv_id']} failed on {tif_path.name}"
                )

        # Free memory per raster
        try:
            da.close()
        except Exception:
            pass
        elapsed = time.perf_counter() - raster_start
        avg_clip = (clip_time_sum / processed_polygons) if processed_polygons else 0.0
        avg_count = (count_time_sum / processed_polygons) if processed_polygons else 0.0
        logger.info(
            f"Finished {tif_path.name} | polygons processed: {processed_polygons} | time: {elapsed:.2f}s | avg clip={avg_clip:.4f}s avg count={avg_count:.4f}s | slow_polygons={slow_polygons}"
        )

    if not results:
        logger.warning("No results produced. Check inputs and parameters.")
    df = pd.DataFrame(results)
    df.sort_values(["pc11_tv_id", "hydrological_year"], inplace=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    t_csv0 = time.perf_counter()
    df.to_csv(out_csv, index=False)
    t_csv1 = time.perf_counter()
    total_elapsed = time.perf_counter() - overall_start
    logger.info(f"Wrote results: {out_csv}")
    logger.info(f"Total time: {total_elapsed:.2f}s | csv write: {(t_csv1 - t_csv0):.3f}s | rows: {len(df)}")


if __name__ == "__main__":
    main()