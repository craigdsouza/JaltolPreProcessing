import argparse
import json
import logging
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from tqdm import tqdm


def parse_hydro_year_from_name(path: Path) -> str:
    # expects "proj32643_lulc250k_[YEAR]_XXXXX.tif", e.g., proj32643_lulc250k_0506_21475.tif
    m = re.search(r"proj32643_lulc250k_(\d{4})_", path.name, flags=re.IGNORECASE)
    return m.group(1) if m else ""


def get_default_config_path() -> Path:
    return Path(__file__).with_suffix(".bincount.config.json")


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
    logger = logging.getLogger("compute_lulc_area_bincount")
    logger.setLevel(level)
    if logger.handlers:
        return logger
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(level)
    logger.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger


def build_label_index(villages: gpd.GeoDataFrame) -> tuple[dict, list]:
    """
    Create mappings between village IDs and label indices for rasterization.

    This function takes a GeoDataFrame of villages, each with a 'pc11_tv_id',
    and builds two things:
      - id_to_index: a dictionary mapping each village ID (as a string) to a unique integer label (starting from 1).
      - index_to_id: a list where each index gives the village ID (index 0 is None, others are village IDs).

    These mappings help assign a unique label to each village when making a raster.

    Args:
        villages: GeoDataFrame with a 'pc11_tv_id' column.

    Returns:
        id_to_index: dict mapping village ID (str) to label index (int)
        index_to_id: list where index gives village ID (index 0 is None)
    """
    ids = list(villages["pc11_tv_id"].astype(str).values)
    # 1..N are village labels; 0 is background
    index_to_id = [None] + ids
    id_to_index = {vid: idx for idx, vid in enumerate(index_to_id) if idx != 0}
    return id_to_index, index_to_id


def rasterize_villages(
    villages: gpd.GeoDataFrame,
    transform: Affine,
    out_shape: tuple[int, int],
    id_to_index: dict,
    all_touched: bool,
    logger: logging.Logger,
) -> np.ndarray:
    t0 = time.perf_counter()
    shapes = []
    # Assemble (geometry, value) pairs
    for _, row in villages.iterrows():
        vid = str(row["pc11_tv_id"])
        value = id_to_index.get(vid)
        if value is None:
            continue
        shapes.append((row.geometry, int(value)))
    t1 = time.perf_counter()
    labels = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint32",
    )
    t2 = time.perf_counter()
    logger.debug(
        f"Rasterize prep: {(t1 - t0):.3f}s | rasterize: {(t2 - t1):.3f}s | shapes: {len(shapes)}"
    )
    return labels


def save_label_cache(
    path_tif: Path,
    label_array: np.ndarray,
    transform: Affine,
    crs,
    logger: logging.Logger,
):
    profile = {
        "driver": "GTiff",
        "height": label_array.shape[0],
        "width": label_array.shape[1],
        "count": 1,
        "dtype": "uint32",
        "transform": transform,
        "crs": crs,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "DEFLATE",
        "predictor": 2,
        "zlevel": 6,
        "BIGTIFF": "IF_SAFER",
    }
    path_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path_tif, "w", **profile) as dst:
        dst.write(label_array, 1)
    logger.info(f"Saved label cache: {path_tif}")


def build_per_year_label_path(base_path: Path, hydro_year: str) -> Path:
    base_path = Path(base_path)
    return base_path.with_name(f"{base_path.stem}_{hydro_year}{base_path.suffix}")


def main():
    parser = argparse.ArgumentParser(description="Bincount-optimized LULC area per village per year.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(get_default_config_path()),
        help="Path to JSON config file. Defaults to .bincount.config.json next to this script.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sample = {
            "raster_dir": "D:/Code/Jaltol/JaltolPreProcessing/data/raster/Karnataka/projected",
            "vector": "D:/Code/Jaltol/JaltolPreProcessing/data/vector/Karnataka/villages.shp",
            "class_value": 3,
            "out_csv": "D:/Code/Jaltol/JaltolPreProcessing/outputs/lulc_area_by_village.csv",
            "all_touched": False,
            "limit_polygons": None,
            "raster_glob": "proj32643_lulc250k_*.tif",
            "log_level": "INFO",
            "log_file": None,
            "label_cache_tif": None,
            "label_index_json": None,
        }
        raise FileNotFoundError(
            f"Config file not found: {cfg_path}\nCreate it as JSON with keys like:\n{sample}"
        )
    cfg = load_config(cfg_path)

    log_level = parse_log_level(cfg.get("log_level", "INFO"))
    log_file = cfg.get("log_file")
    logger = setup_logging(log_level, log_file)
    logger.info("Starting bincount LULC area computation")
    logger.info(f"Using config: {cfg_path}")

    raster_dir = Path(cfg["raster_dir"])
    shp_path = Path(cfg["vector"])
    out_csv = Path(cfg["out_csv"])
    # Backward compatible: support class_values (list) or class_value (single)
    cfg_class_values = cfg.get("class_values")
    if cfg_class_values is None:
        single_val = cfg.get("class_value")
        if isinstance(single_val, list):
            cfg_class_values = single_val
        elif single_val is not None:
            cfg_class_values = [single_val]
        else:
            raise KeyError("Provide 'class_values' list or 'class_value' in config.")
    class_values = [int(v) for v in cfg_class_values]
    all_touched = bool(cfg.get("all_touched", False))
    limit_polygons = cfg.get("limit_polygons")
    raster_glob = str(cfg.get("raster_glob", "proj32643_lulc250k_*.tif"))

    label_cache_tif = cfg.get("label_cache_tif")   # village_labels.tif
    label_index_json = cfg.get("label_index_json") # village_label_index.json

    # Find rasters
    tifs = sorted(raster_dir.glob(raster_glob))
    if not tifs:
        logger.error(f"No rasters found in {raster_dir} using glob '{raster_glob}'")
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
        villages = villages.iloc[: int(limit_polygons)].copy()
    logger.info(f"Prepared {len(villages)} village polygons for processing")

    # Prepare village ID index map once (consistent across years)
    id_to_index, index_to_id = build_label_index(villages)
    n_labels = len(index_to_id) - 1  # exclude background

    # Cache reprojected villages per CRS and previous label/grid to reuse when possible
    villages_by_crs: dict[str, gpd.GeoDataFrame] = {}
    prev_label_path: Path | None = None
    prev_grid_key: tuple | None = None

    results = []
    overall_start = time.perf_counter()
    for tif_path in tqdm(tifs, desc="Rasters"):
        hydro_year = parse_hydro_year_from_name(tif_path)
        if not hydro_year:
            logger.warning(f"Skipping file not matching naming convention: {tif_path.name}")
            continue

        logger.info(f"Processing raster {tif_path.name} (year {hydro_year})")
        t_r0 = time.perf_counter()
        with rasterio.open(tif_path) as ds:
            transform: Affine = ds.transform
            crs = ds.crs
            height, width = ds.height, ds.width
            xres = transform.a
            yres = -transform.e if transform.e < 0 else transform.e
            pixel_area_m2 = abs(xres * yres)

            # Unique key for this raster grid
            grid_key = (crs.to_string() if crs else None, transform.to_gdal(), height, width)

            # Prepare villages in this raster CRS (cache by CRS string)
            crs_key = crs.to_string() if crs else "None"
            if crs_key not in villages_by_crs:
                t_proj0 = time.perf_counter()
                villages_by_crs[crs_key] = villages.to_crs(crs)
                t_proj1 = time.perf_counter()
                logger.debug(f"Villages to CRS {crs_key} time: {(t_proj1 - t_proj0):.3f}s")
            v_in = villages_by_crs[crs_key]

            # Per-year label path (optional)
            per_year_label_path = None
            if label_cache_tif:
                per_year_label_path = build_per_year_label_path(Path(label_cache_tif), hydro_year)

            # Decide to copy previous labels or rasterize anew
            labels = None
            if prev_grid_key is not None and grid_key == prev_grid_key and prev_label_path and per_year_label_path:
                try:
                    shutil.copy2(prev_label_path, per_year_label_path)
                    logger.info(f"Reusing label grid -> copied {prev_label_path.name} to {per_year_label_path.name}")
                    with rasterio.open(per_year_label_path) as lb:
                        t_l0 = time.perf_counter()
                        labels = lb.read(1)
                        t_l1 = time.perf_counter()
                        logger.debug(f"Loaded copied label in {(t_l1 - t_l0):.3f}s")
                except Exception:
                    logger.exception("Failed to copy previous label cache; rasterizing labels instead.")

            if labels is None:
                t_rast0 = time.perf_counter()
                labels = rasterize_villages(
                    villages=v_in,
                    transform=transform,
                    out_shape=(height, width),
                    id_to_index=id_to_index,
                    all_touched=all_touched,
                    logger=logger,
                )
                t_rast1 = time.perf_counter()
                logger.debug(f"Rasterized labels for {hydro_year} in {(t_rast1 - t_rast0):.3f}s")
                if per_year_label_path:
                    save_label_cache(per_year_label_path, labels, transform, crs, logger)
                prev_label_path = per_year_label_path
                prev_grid_key = grid_key

            t_read0 = time.perf_counter()
            band = ds.read(1, masked=True)
            t_read1 = time.perf_counter()
        logger.debug(f"Raster read time: {(t_read1 - t_read0):.3f}s | dtype: {band.dtype}")

        # Vectorized bincount on labels masked by class
        t_mask0 = time.perf_counter()
        # Build a compact index for requested classes
        max_class_value = int(max(class_values))
        class_idx = np.full(max_class_value + 1, -1, dtype=np.int16)
        for i, cv in enumerate(class_values):
            if 0 <= cv <= max_class_value:
                class_idx[cv] = i
        # Valid pixels and selected classes only
        band_int = band.astype(np.int32, copy=False)
        valid = ~band_int.mask
        band_data = band_int.data  # plain ndarray
        in_range = (band_data >= 0) & (band_data <= max_class_value)
        idx_vals = np.full(band_data.shape, -1, dtype=np.int16)
        if in_range.any():
            idx_vals[in_range] = class_idx[band_data[in_range]]
        sel = valid & (idx_vals >= 0)
        t_mask1 = time.perf_counter()

        t_bin0 = time.perf_counter()
        # Flatten per selected pixels
        lab = labels[sel].ravel().astype(np.int64, copy=False)
        cls = idx_vals[sel].ravel().astype(np.int64, copy=False)
        K = len(class_values)
        flat = lab * K + cls
        counts_all = np.bincount(flat, minlength=(n_labels + 1) * K)
        counts_matrix = counts_all.reshape(n_labels + 1, K)
        t_bin1 = time.perf_counter()

        logger.debug(
            f"mask: {(t_mask1 - t_mask0):.3f}s | bincount: {(t_bin1 - t_bin0):.3f}s | positives: {int(sel.sum())}"
        )

        # Build results
        for idx in range(1, n_labels + 1):
            pc11 = index_to_id[idx]
            row = {"pc11_tv_id": pc11, "hydrological_year": hydro_year}
            for j, cv in enumerate(class_values):
                class_pixels = int(counts_matrix[idx, j])
                row[f"{cv}_area_ha"] = class_pixels * pixel_area_m2 / 10000.0
            results.append(row)
        t_r1 = time.perf_counter()
        logger.info(
            f"Finished {tif_path.name} | time: {(t_r1 - t_r0):.2f}s"
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
    logger.info(
        f"Total time: {total_elapsed:.2f}s | csv write: {(t_csv1 - t_csv0):.3f}s | rows: {len(df)}"
    )


if __name__ == "__main__":
    main()


