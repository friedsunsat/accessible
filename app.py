from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import folium
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from folium.features import GeoJson, GeoJsonTooltip
from matplotlib.colors import Normalize, PowerNorm
from shapely.geometry import Point, box
from streamlit_folium import st_folium
import streamlit.components.v1 as st_components


# =========================================================
# 경로 자동 탐색
# =========================================================
def pick_dropbox_base() -> Path:
    candidates = [
        Path(r"E:\Dropbox"),
        Path(r"C:\Users\82102\Dropbox"),
        Path.home() / "Dropbox",
    ]
    for p in candidates:
        if p.exists():
            return p
    # 마지막 fallback
    return Path(r"C:\Users\82102\Dropbox")


def pick_first_existing_path(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    # 하나도 없으면 첫 번째 반환만 해두고, 실제 실행 시 체크에서 에러 표시
    return paths[0]


DROPBOX_BASE = pick_dropbox_base()

ROOT_OUT = DROPBOX_BASE / r"01-대학원\02-Paper Work\01-개인연구\202603_격자 단위 다양한 시설 대중교통 접근성 결핍 진단\03-분석자료\01-기초자료\02-삽도자료\02_학회자료\01_교통학회"

CLASSIFIED_PATH = pick_first_existing_path(
    DROPBOX_BASE / r"01-대학원\02-Paper Work\01-개인연구\202603_격자 단위 다양한 시설 대중교통 접근성 결핍 진단\03-분석자료\01-기초자료\01-전처리\02_routing\01_intercity\02_500m\from_metrics_500m_intracity_classified.parquet",
    DROPBOX_BASE / r"01-대학원\02-Paper Work\01-개인연구\202603_격자 단위 다양한 시설 대중교통 접근성 결핍 진단\03-분석자료\01-기초자료\01-전처리\02_routing\01_intercity\02_500m\500m_classified.parquet",
)

GRID_PATH = DROPBOX_BASE / r"01-대학원\02-Paper Work\01-개인연구\202603_격자 단위 다양한 시설 대중교통 접근성 결핍 진단\03-분석자료\01-기초자료\01-전처리\02_routing\00_grid\500m.gpkg"
STATION_PATH = DROPBOX_BASE / r"01-대학원\02-Paper Work\01-개인연구\202603_격자 단위 다양한 시설 대중교통 접근성 결핍 진단\03-분석자료\01-기초자료\02-삽도자료\01_재료\station.gpkg"
SUBWAY_PATH = DROPBOX_BASE / r"01-대학원\02-Paper Work\01-개인연구\202603_격자 단위 다양한 시설 대중교통 접근성 결핍 진단\03-분석자료\01-기초자료\02-삽도자료\01_재료\subway.gpkg"
FAC_PATH = pick_first_existing_path(
    DROPBOX_BASE / r"01-대학원\02-Paper Work\01-개인연구\202603_격자 단위 다양한 시설 대중교통 접근성 결핍 진단\03-분석자료\01-기초자료\01-전처리\02_routing\01_intercity\02_500m\all_facilities.geoparquet",
    DROPBOX_BASE / r"01-대학원\02-Paper Work\01-개인연구\202603_격자 단위 다양한 시설 대중교통 접근성 결핍 진단\03-분석자료\01-기초자료\01-전처리\02_routing\01_intercity\02_500m\all_activities.geoparquet",
)

CACHE_DIR = ROOT_OUT / "dashboard_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_GRID = CACHE_DIR / "grid_dashboard_5179.geoparquet"
CACHE_GRID_SIMPLE = CACHE_DIR / "grid_dashboard_simple_5179.geoparquet"
CACHE_SGG = CACHE_DIR / "sgg_dashboard_5179.geoparquet"
CACHE_STATION = CACHE_DIR / "station_5179.geoparquet"
CACHE_SUBWAY = CACHE_DIR / "subway_5179.geoparquet"
CACHE_FAC = CACHE_DIR / "facilities_5179.geoparquet"
CACHE_TS = CACHE_DIR / "grid_timeseries.parquet"
CACHE_IDX       = CACHE_DIR / "grid_index_points_5179.parquet"
CACHE_CELL_DATA = CACHE_DIR / "cell_detail_data.parquet"

# 셀 클릭 상세 소스 (시계열 + car 지표)
INTRACITY_PATH = pick_first_existing_path(
    DROPBOX_BASE / r"01-대학원\02-Paper Work\01-개인연구\202603_격자 단위 다양한 시설 대중교통 접근성 결핍 진단\03-분석자료\01-기초자료\01-전처리\02_routing\01_intercity\02_500m\from_metrics_500m_intracity.parquet",
    DROPBOX_BASE / r"01-대학원\02-Paper Work\01-개인연구\202603_격자 단위 다양한 시설 대중교통 접근성 결핍 진단\03-분석자료\01-기초자료\01-전처리\02_routing\01_intercity\02_500m\from_metrics_500m_intracity_classified.parquet",
)

# 시군구별 GeoJSON + 색상 정보 미리 직렬화해두는 캐시 폴더
CACHE_GEOJSON_DIR = CACHE_DIR / "geojson_tiles"
# 정적 레이어 GeoJSON 캐시
CACHE_STATION_JSON = CACHE_GEOJSON_DIR / "station.json"
CACHE_SUBWAY_JSON  = CACHE_GEOJSON_DIR / "subway.json"
# 시설은 타입별로 분리 (문자열 템플릿)
CACHE_FAC_JSON_TPL = str(CACHE_GEOJSON_DIR / "fac_{ftype}.json")


# =========================================================
# 상수 / 스타일
# =========================================================
PLOT_CRS = "EPSG:5179"
WEB_CRS = "EPSG:4326"

GRID_ID_COL = "GRID_500M_"
GRID_JOIN_COL = "from_id"
SGG_CODE_COL = "from_sgg_key"
SGG_NAME_COL = "from_sgg"

FAC_KIND_COL = "facility_kind"
FAC_TYPE_COL = "facility_type"
FAC_DEPT_COL = "department"

TIME_SLOTS = ["08", "10", "12", "14", "16", "18", "20", "22"]
COV_COLS = [f"{t}_coverage" for t in TIME_SLOTS]
MAI_COLS = [f"{t}_mai" for t in TIME_SLOTS]

LAYER_LABEL_TO_KEY = {
    "F(s)": "fs",
    "F(d)": "fd",
    "T(c)": "tc",
    "T(f)": "tf",
    "Population": "pop",
}
LAYER_KEY_TO_LABEL = {v: k for k, v in LAYER_LABEL_TO_KEY.items()}
LAYER_HELP = {
    "F(s)": "Facility siting / sub-optimal location problem",
    "F(d)": "Facility dispersion problem",
    "T(c)": "Transit connection problem",
    "T(f)": "Transit frequency problem",
    "Population": "Population share or density",
}

CMAPS = {
    "fs": "BuPu",
    "fd": "GnBu",
    "tc": "RdPu",
    "tf": "YlOrBr",
    "pop": "Reds",
}

FACILITY_COLORS = {
    "park": "#66BB6A",
    "library": "#42A5F5",
    "m1": "#EF5350",
    "m2": "#AB47BC",
    "grocery": "#FFA726",
    "public": "#26C6DA",
    "pharmacy": "#EC407A",
}
FACILITY_LABELS_EN = {
    "park": "Park",
    "library": "Library",
    "m1": "Primary care",
    "m2": "Specialist outpatient",
    "grocery": "Grocery",
    "public": "Public service",
    "pharmacy": "Pharmacy",
}
FACILITY_ORDER = ["park", "library", "m1", "m2", "grocery", "public", "pharmacy"]

MED_GROUP_MAP_RAW = {
    "가정의학과": "m1",
    "내과": "m1",
    "소아청소년과": "m1",

    "정형외과": "m2",
    "재활의학과": "m2",
    "마취통증의학과": "m2",

    "안과": "m3",
    "이비인후과": "m3",
    "피부과": "m3",
    "비뇨의학과": "m3",
    "신경과": "m3",
    "산부인과": "m3",

    "정신건강의학과": "m4",

    "치과": "m5",
    "통합치의학과": "m5",
    "소아치과": "m5",
    "치과교정과": "m5",
    "치과보존과": "m5",
    "치과보철과": "m5",
    "치주과": "m5",
    "구강내과": "m5",

    "예방치과": "m8",
    "영상치의학과": "m8",
    "구강병리과": "m8",
    "구강악안면외과": "m8",

    "사상체질과": "m6",
    "침구과": "m6",
    "한방내과": "m6",
    "한방부인과": "m6",
    "한방소아과": "m6",
    "한방신경정신과": "m6",
    "한방안·이비인후·피부과": "m6",
    "한방재활의학과": "m6",

    "한방응급": "m7",
    "외과": "m7",
    "신경외과": "m7",
    "심장혈관흉부외과": "m7",
    "응급의학과": "m7",
    "결핵과": "m7",
    "방사선종양학과": "m7",
    "핵의학과": "m7",
    "병리과": "m7",
    "영상의학과": "m7",
    "진단검사의학과": "m7",
    "예방의학과": "m7",
    "직업환경의학과": "m7",
    "성형외과": "m7",
}
MED_ALLOWED_RAW = {"m1", "m2", "m3", "m4", "m5", "m6"}
MED_SPECIALIZED_RAW = {"m2", "m3", "m4", "m5", "m6"}

COV_LINE_COLOR = "#7B2CBF"
MAI_LINE_COLOR = "#9CCC65"

POP_MAP_MODE = "share"
GRID_ZOOM_THRESHOLD = 11
MAP_HEIGHT = 560


# =========================================================
# 유틸
# =========================================================
def normalize_code_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return out


def normalize_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def normalize_sgg_code(code) -> str:
    """sgg_code → 정수 문자열 ("23520.0" → "23520")."""
    try:
        return str(int(float(str(code).strip())))
    except Exception:
        return str(code).strip()


def parse_deficit_tokens(val) -> Set[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return set()

    if isinstance(val, (list, tuple, set)):
        raw = list(val)
    else:
        s = str(val).strip()
        if s in ["", "[]", "{}"]:
            return set()
        try:
            obj = json.loads(s)
            raw = obj if isinstance(obj, list) else [obj]
        except Exception:
            raw = re.findall(r"[FfTt]\([sScCdDfF]\)", s)

    out = set()
    for x in raw:
        t = str(x).strip().lower()
        if t == "f(s)":
            out.add("F(s)")
        elif t == "f(d)":
            out.add("F(d)")
        elif t == "t(c)":
            out.add("T(c)")
        elif t == "t(f)":
            out.add("T(f)")
    return out


def parse_department_list(val) -> List[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []

    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]

    s = str(val).strip()
    if s in ["", "[]", "nan", "None"]:
        return []

    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass

    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass

    s2 = s.strip("[]")
    parts = [p.strip().strip('"').strip("'") for p in s2.split(",")]
    return [p for p in parts if p]


def normalize_facility_type_from_row(row: pd.Series) -> str:
    kind = str(row.get(FAC_KIND_COL, "")).strip()
    ftype = str(row.get(FAC_TYPE_COL, "")).strip()
    kind_l = kind.lower()
    ftype_l = ftype.lower()

    if "공원" in kind or "park" in kind_l or "공원" in ftype or "park" in ftype_l:
        return "park"
    if "도서관" in kind or "library" in kind_l or "도서관" in ftype or "library" in ftype_l:
        return "library"
    if "약국" in kind or "pharmacy" in kind_l or "약국" in ftype or "pharmacy" in ftype_l:
        return "pharmacy"
    if (
        "식료품" in kind or "마트" in kind or "시장" in kind or "편의점" in kind or
        "grocery" in kind_l or "market" in kind_l or "mart" in kind_l or "supermarket" in kind_l or
        "식료품" in ftype or "마트" in ftype or "시장" in ftype or "편의점" in ftype
    ):
        return "grocery"
    if (
        "행정" in kind or "공공" in kind or "주민센터" in kind or "행정서비스" in kind or
        "행정" in ftype or "공공" in ftype or "주민센터" in ftype or "행정서비스" in ftype
    ):
        return "public"

    dept_list = parse_department_list(row.get(FAC_DEPT_COL))
    raw_groups = [MED_GROUP_MAP_RAW[d] for d in dept_list if d in MED_GROUP_MAP_RAW]
    raw_groups = [g for g in raw_groups if g in MED_ALLOWED_RAW]

    if "m1" in raw_groups:
        return "m1"
    if any(g in MED_SPECIALIZED_RAW for g in raw_groups):
        return "m2"

    if (
        "의료" in kind or "병원" in kind or "의원" in kind or "치과" in kind or "한의원" in kind or
        "의료" in ftype or "병원" in ftype or "의원" in ftype or "치과" in ftype or "한의원" in ftype
    ):
        if any(x in ftype for x in ["보건소", "보건지소", "보건진료소", "보건의료원"]):
            return "m1"
        return "m2"

    return "exclude"


class PowerNormSafe(PowerNorm):
    pass


def compute_group_norm_from_series(values: pd.Series, gamma: float = 0.55, force_zero_min: bool = True):
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    vals = vals[vals > 0]
    if len(vals) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = 0.0 if force_zero_min else float(vals.min())
        vmax = float(vals.max())
        if vmax <= vmin:
            vmax = vmin + 1e-9
    norm = PowerNormSafe(gamma=gamma, vmin=vmin, vmax=vmax)
    return vmin, vmax, norm


def compute_group_pop_norm(values: pd.Series, share_mode: bool = True):
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = 0.0 if share_mode else float(vals.min())
        vmax = float(vals.max())
        if vmax <= vmin:
            vmax = vmin + 1e-9
    norm = PowerNormSafe(gamma=0.6, vmin=vmin, vmax=vmax) if share_mode else Normalize(vmin=vmin, vmax=vmax)
    return vmin, vmax, norm


def get_grid_fill_color(value: float, cmap_name: str, norm) -> Optional[str]:
    if pd.isna(value) or value <= 0:
        return None
    cmap = matplotlib.colormaps[cmap_name]
    rgba = cmap(norm(value))
    return mcolors.to_hex(rgba, keep_alpha=False)


def gradient_css_from_cmap(cmap_name: str) -> str:
    cmap = matplotlib.colormaps[cmap_name]
    stops = []
    for p in np.linspace(0, 1, 7):
        stops.append(f"{mcolors.to_hex(cmap(p))} {int(p * 100)}%")
    return ", ".join(stops)


def make_pretty_ticks(vmin: float, vmax: float, n: int = 5) -> np.ndarray:
    if vmax <= vmin:
        return np.array([vmin, vmax + 1e-9])
    return np.linspace(vmin, vmax, n)


def choose_tick_decimals(vmin: float, vmax: float) -> int:
    span = abs(vmax - vmin)
    if span < 0.05:
        return 3
    if span < 0.5:
        return 2
    if span < 5:
        return 1
    return 0


def make_streamlit_colorbar_html(cmap_name: str, norm, title: str, percent: bool = True) -> str:
    vmin = float(norm.vmin)
    vmax = float(norm.vmax)
    ticks = make_pretty_ticks(vmin, vmax, n=5)
    dec = choose_tick_decimals(vmin, vmax)

    labels = []
    for t in ticks:
        labels.append(f"<div style='text-align:center;font-size:12px;color:#333;'>{t:.{dec}f}{'%' if percent else ''}</div>")

    return f"""
    <div style='width:100%;margin-top:4px;'>
      <div style='font-size:13px;font-weight:600;margin-bottom:4px;'>{title}</div>
      <div style='height:10px;border:1px solid #333;background: linear-gradient(to right, {gradient_css_from_cmap(cmap_name)});'></div>
      <div style='display:grid;grid-template-columns: repeat(5, 1fr);margin-top:3px;'>{''.join(labels)}</div>
    </div>
    """


def build_feature_legend_html() -> str:
    items = []
    for key in FACILITY_ORDER:
        color = FACILITY_COLORS[key]
        label = FACILITY_LABELS_EN[key]
        items.append(
            f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:4px;'>"
            f"<span style='width:10px;height:10px;border-radius:50%;display:inline-block;background:{color};border:1px solid white;'></span>"
            f"<span style='font-size:12px;'>{label}</span></div>"
        )
    return "".join(items)


def build_metric_help_html() -> str:
    rows = []
    for label in ["F(s)", "F(d)", "T(c)", "T(f)"]:
        rows.append(f"<li><b>{label}</b>: {LAYER_HELP[label]}</li>")
    return (
        '<ul style="margin-top:0; margin-bottom:0; padding-left:18px; font-size:12px;">'
        + "".join(rows)
        + "</ul>"
    )


def to_json_safe_value(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, tuple)):
        return list(x)
    if isinstance(x, np.generic):
        return x.item()
    if pd.isna(x):
        return None
    return x


def make_json_safe_geojson_gdf(gdf: gpd.GeoDataFrame, cols: List[str]) -> gpd.GeoDataFrame:
    keep_cols = [c for c in cols if c in gdf.columns] + ["geometry"]
    out = gdf[keep_cols].copy()
    for c in keep_cols:
        if c != "geometry":
            out[c] = out[c].apply(to_json_safe_value)
    return out


@st.cache_data(show_spinner=False, max_entries=128)
def _cached_geojson(gdf_hash: str, geojson_str: str) -> str:
    """GeoJSON 문자열을 캐시. gdf_hash는 캐시 키 역할."""
    return geojson_str


def gdf_to_cached_json(gdf: gpd.GeoDataFrame, cols: List[str]) -> str:
    """컬럼을 필터링하고 WGS84로 변환 후 JSON 직렬화. 결과를 캐시."""
    safe = make_json_safe_geojson_gdf(gdf, cols).to_crs(WEB_CRS)
    key = f"{len(safe)}_{list(safe.columns)}_{cols}"
    return _cached_geojson(key, safe.to_json())


def bbox_subset(gdf: gpd.GeoDataFrame, bounds_5179):
    if gdf.empty or bounds_5179 is None:
        return gdf
    minx, miny, maxx, maxy = bounds_5179
    try:
        return gdf.cx[minx:maxx, miny:maxy]
    except Exception:
        return gdf[gdf.intersects(box(minx, miny, maxx, maxy))]


def get_bbox_from_bounds(bounds_dict) -> Optional[Tuple[float, float, float, float]]:
    if not bounds_dict:
        return None
    try:
        south = bounds_dict["_southWest"]["lat"]
        west = bounds_dict["_southWest"]["lng"]
        north = bounds_dict["_northEast"]["lat"]
        east = bounds_dict["_northEast"]["lng"]
        bbox4326 = box(west, south, east, north)
        bbox5179 = gpd.GeoSeries([bbox4326], crs=WEB_CRS).to_crs(PLOT_CRS).iloc[0].bounds
        return bbox5179
    except Exception:
        return None


def find_clicked_cell(lat: float, lon: float, grid_bbox: gpd.GeoDataFrame) -> Optional[str]:
    if lat is None or lon is None or grid_bbox.empty:
        return None
    pt = gpd.GeoSeries([Point(lon, lat)], crs=WEB_CRS).to_crs(PLOT_CRS).iloc[0]
    hit = grid_bbox[grid_bbox.contains(pt)]
    if hit.empty:
        return None
    return hit.iloc[0][GRID_JOIN_COL]


# =========================================================
# 캐시 빌드 / 로드
# =========================================================
def build_dashboard_cache(progress_cb=None) -> Dict[str, str]:
    """progress_cb(step, total, msg) 콜백."""
    metrics = pd.read_parquet(CLASSIFIED_PATH).copy()
    metrics[GRID_JOIN_COL] = normalize_str_series(metrics[GRID_JOIN_COL])
    metrics[SGG_CODE_COL] = normalize_code_series(metrics[SGG_CODE_COL])
    metrics[SGG_NAME_COL] = normalize_str_series(metrics[SGG_NAME_COL])
    metrics["pop"] = pd.to_numeric(metrics["pop"], errors="coerce").fillna(0)

    for c in COV_COLS + MAI_COLS:
        metrics[c] = pd.to_numeric(metrics[c], errors="coerce")

    metrics["_nat_tokens"] = metrics["nat_deficit"].apply(parse_deficit_tokens)
    metrics["_sgg_tokens"] = metrics["sgg_deficit"].apply(parse_deficit_tokens)

    for tok, key in {"F(s)": "fs", "F(d)": "fd", "T(c)": "tc", "T(f)": "tf"}.items():
        metrics[f"nat_has_{key}"] = metrics["_nat_tokens"].apply(lambda s: tok in s)
        metrics[f"sgg_has_{key}"] = metrics["_sgg_tokens"].apply(lambda s: tok in s)

    metrics["sgg_pop_total"] = metrics.groupby(SGG_CODE_COL)["pop"].transform("sum")
    metrics["national_pop_total"] = float(metrics["pop"].sum())

    metrics["nat_pop_map"] = np.where(metrics["national_pop_total"] > 0, metrics["pop"] / metrics["national_pop_total"] * 100.0, 0.0)
    metrics["sgg_pop_map"] = np.where(metrics["sgg_pop_total"] > 0, metrics["pop"] / metrics["sgg_pop_total"] * 100.0, 0.0)
    metrics["local_pop_map"] = metrics["sgg_pop_map"]

    for key in ["fs", "fd", "tc", "tf"]:
        metrics[f"nat_{key}_ratio"] = np.where(
            metrics[f"nat_has_{key}"] & (metrics["sgg_pop_total"] > 0),
            metrics["pop"] / metrics["sgg_pop_total"] * 100.0,
            0.0,
        )
        metrics[f"sgg_{key}_ratio"] = np.where(
            metrics[f"sgg_has_{key}"] & (metrics["sgg_pop_total"] > 0),
            metrics["pop"] / metrics["sgg_pop_total"] * 100.0,
            0.0,
        )

    grid = gpd.read_file(GRID_PATH)
    if grid.crs is None:
        grid = grid.set_crs(epsg=4326)
    if str(grid.crs) != PLOT_CRS:
        grid = grid.to_crs(PLOT_CRS)
    grid[GRID_ID_COL] = normalize_str_series(grid[GRID_ID_COL])
    grid = grid[[GRID_ID_COL, "geometry"]].copy()

    gdf = grid.merge(metrics, left_on=GRID_ID_COL, right_on=GRID_JOIN_COL, how="inner")
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=PLOT_CRS)
    gdf["area_km2"] = gdf.geometry.area / 1_000_000.0
    gdf["pop_density_km2"] = np.where(gdf["area_km2"] > 0, gdf["pop"] / gdf["area_km2"], np.nan)

    centroid = gdf.geometry.centroid
    idx_df = pd.DataFrame({
        GRID_JOIN_COL: gdf[GRID_JOIN_COL],
        SGG_CODE_COL: gdf[SGG_CODE_COL],
        SGG_NAME_COL: gdf[SGG_NAME_COL],
        "x": centroid.x,
        "y": centroid.y,
        "pop": gdf["pop"],
        "avg_coverage": gdf.get("avg_coverage"),
        "avg_mai": gdf.get("avg_mai"),
        "cv_coverage": gdf.get("cv_coverage"),
        "cv_mai": gdf.get("cv_mai"),
        "car_coverage": gdf.get("car_coverage"),
        "car_mai": gdf.get("car_mai"),
    })
    for c in COV_COLS:
        idx_df[c] = gdf[c]
    for c in MAI_COLS:
        idx_df[c] = gdf[c]
    idx_df.to_parquet(CACHE_IDX, index=False)

    safe_grid_cols = [
        GRID_JOIN_COL, SGG_CODE_COL, SGG_NAME_COL, "pop",
        "nat_deficit", "sgg_deficit",
        "nat_pop_map", "sgg_pop_map", "local_pop_map",
        "nat_fs_ratio", "nat_fd_ratio", "nat_tc_ratio", "nat_tf_ratio",
        "sgg_fs_ratio", "sgg_fd_ratio", "sgg_tc_ratio", "sgg_tf_ratio",
        "avg_coverage", "avg_mai", "cv_coverage", "cv_mai",
        "car_coverage", "car_mai",
        "area_km2", "pop_density_km2",
        *COV_COLS, *MAI_COLS,
        "geometry",
    ]
    safe_grid_cols = [c for c in safe_grid_cols if c in gdf.columns]
    gdf_safe = gdf[safe_grid_cols].copy()

    gdf_simple = gdf_safe.copy()
    gdf_simple["geometry"] = gdf_simple.geometry.simplify(25, preserve_topology=True)

    gdf_safe.to_parquet(CACHE_GRID, index=False)
    gdf_simple.to_parquet(CACHE_GRID_SIMPLE, index=False)

    agg_rows = []
    for code, gg in gdf_safe.groupby(SGG_CODE_COL, dropna=True):
        row = {
            SGG_CODE_COL: code,
            SGG_NAME_COL: gg[SGG_NAME_COL].iloc[0],
            "sgg_pop_total": float(gg["pop"].sum()),
            "nat_pop_map": float(gg["nat_pop_map"].sum()),
            "sgg_pop_map": float(gg["sgg_pop_map"].sum()),
        }
        for key in ["fs", "fd", "tc", "tf"]:
            row[f"nat_{key}_ratio"] = float(gg[f"nat_{key}_ratio"].sum())
            row[f"sgg_{key}_ratio"] = float(gg[f"sgg_{key}_ratio"].sum())
        agg_rows.append(row)

    sgg_attr = pd.DataFrame(agg_rows)
    sgg_poly = gdf_safe[[SGG_CODE_COL, "geometry"]].dissolve(by=SGG_CODE_COL).reset_index()
    sgg_poly = sgg_poly.merge(sgg_attr, on=SGG_CODE_COL, how="left")
    sgg_poly = gpd.GeoDataFrame(sgg_poly, geometry="geometry", crs=PLOT_CRS)
    sgg_poly.to_parquet(CACHE_SGG, index=False)

    station = gpd.read_file(STATION_PATH)
    subway = gpd.read_file(SUBWAY_PATH)
    fac = gpd.read_parquet(FAC_PATH)

    for layer in [station, subway, fac]:
        if layer.crs is None:
            layer.set_crs(epsg=4326, inplace=True)

    if str(station.crs) != PLOT_CRS:
        station = station.to_crs(PLOT_CRS)
    if str(subway.crs) != PLOT_CRS:
        subway = subway.to_crs(PLOT_CRS)
    if str(fac.crs) != PLOT_CRS:
        fac = fac.to_crs(PLOT_CRS)

    fac["fac_type_norm"] = fac.apply(normalize_facility_type_from_row, axis=1)
    fac = fac[fac["fac_type_norm"].isin(FACILITY_ORDER)].copy()

    station.to_parquet(CACHE_STATION, index=False)
    subway.to_parquet(CACHE_SUBWAY, index=False)
    fac.to_parquet(CACHE_FAC, index=False)

    ts_keep = [
        GRID_JOIN_COL, SGG_CODE_COL, SGG_NAME_COL,
        "avg_coverage", "avg_mai", "cv_coverage", "cv_mai",
        "car_coverage", "car_mai",
        *COV_COLS, *MAI_COLS,
    ]
    ts_keep = [c for c in ts_keep if c in metrics.columns]
    metrics[ts_keep].to_parquet(CACHE_TS, index=False)

    # ── 정적 GeoJSON 캐시 ──────────────────────────────────
    CACHE_GEOJSON_DIR.mkdir(parents=True, exist_ok=True)

    def _write_json_safe(path: Path, data: dict) -> None:
        """pyogrio/Dropbox 파일 잠금 우회: json 모듈로 직접 쓰기."""
        import tempfile, os, time
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            # Windows atomic replace (같은 드라이브 내 rename)
            for attempt in range(5):
                try:
                    if path.exists():
                        path.unlink()
                    os.rename(tmp_path, str(path))
                    break
                except PermissionError:
                    time.sleep(0.3)
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise

    def _gdf_to_geojson_dict(gdf: gpd.GeoDataFrame) -> dict:
        """GeoDataFrame → GeoJSON dict (pyogrio 미사용)."""
        features = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            props = {}
            for col in gdf.columns:
                if col == "geometry":
                    continue
                val = row[col]
                if isinstance(val, float) and (val != val):  # NaN
                    props[col] = None
                elif hasattr(val, "item"):  # numpy scalar
                    props[col] = val.item()
                else:
                    props[col] = val
            features.append({
                "type": "Feature",
                "geometry": geom.__geo_interface__,
                "properties": props,
            })
        return {"type": "FeatureCollection", "features": features}

    # 역/노선/시설 GeoJSON (WGS84) 미리 직렬화
    station_w = station.to_crs(WEB_CRS)
    _write_json_safe(CACHE_STATION_JSON, _gdf_to_geojson_dict(station_w[["geometry"]]))

    subway_w = subway.to_crs(WEB_CRS)
    _write_json_safe(CACHE_SUBWAY_JSON, _gdf_to_geojson_dict(subway_w[["geometry"]]))

    fac_w = fac.to_crs(WEB_CRS)
    for ftype in FACILITY_ORDER:
        sub = fac_w[fac_w["fac_type_norm"] == ftype]
        if not sub.empty:
            path = Path(CACHE_FAC_JSON_TPL.format(ftype=ftype))
            _write_json_safe(path, _gdf_to_geojson_dict(sub[["geometry", "fac_type_norm"]]))

    # ── 시군구별 그리드 GeoJSON (색상 포함) 캐시 ──────────
    metric_cols = [
        "sgg_fs_ratio", "sgg_fd_ratio", "sgg_tc_ratio", "sgg_tf_ratio",
        "nat_fs_ratio", "nat_fd_ratio", "nat_tc_ratio", "nat_tf_ratio",
        "sgg_pop_map", "nat_pop_map", "local_pop_map",
        GRID_JOIN_COL, SGG_NAME_COL, SGG_CODE_COL,
    ]
    metric_cols = [c for c in metric_cols if c in gdf_safe.columns]

    gdf_web = gdf_simple.to_crs(WEB_CRS).copy()
    gdf_web[SGG_CODE_COL] = gdf_web[SGG_CODE_COL].apply(normalize_sgg_code)
    sgg_groups = list(gdf_web.groupby(SGG_CODE_COL))
    n_tiles = len(sgg_groups)
    for i, (sgg_code, grp) in enumerate(sgg_groups):
        key = normalize_sgg_code(sgg_code)
        keep = [c for c in metric_cols if c in grp.columns] + ["geometry"]
        sub = grp[keep].copy()
        _write_json_safe(CACHE_GEOJSON_DIR / f"grid_{key}.json", _gdf_to_geojson_dict(sub))
        if progress_cb:
            progress_cb(i + 1, n_tiles, f"GeoJSON 타일: {key} ({i+1}/{n_tiles})")

    # ── 셀 클릭 상세 데이터 캐시 ──────────────────────────
    if INTRACITY_PATH.exists():
        try:
            ic = pd.read_parquet(INTRACITY_PATH)
            ic[GRID_JOIN_COL] = ic[GRID_JOIN_COL].astype(str).str.strip()
            keep_c = [GRID_JOIN_COL, "pop", "car_coverage", "car_mai",
                      "avg_coverage", "avg_mai", "cv_coverage", "cv_mai",
                      *COV_COLS, *MAI_COLS]
            ic[[c for c in keep_c if c in ic.columns]].to_parquet(CACHE_CELL_DATA, index=False)
        except Exception:
            pass

    return {"ok": "1"}


@st.cache_data(show_spinner=False)
def load_cached_data():
    grid = gpd.read_parquet(CACHE_GRID)
    grid_simple = gpd.read_parquet(CACHE_GRID_SIMPLE)
    sgg = gpd.read_parquet(CACHE_SGG)
    station = gpd.read_parquet(CACHE_STATION)
    subway = gpd.read_parquet(CACHE_SUBWAY)
    fac = gpd.read_parquet(CACHE_FAC)
    ts = pd.read_parquet(CACHE_TS)
    idx = pd.read_parquet(CACHE_IDX)
    return grid, grid_simple, sgg, station, subway, fac, ts, idx


@st.cache_data(show_spinner=False)
def load_cell_detail_data() -> pd.DataFrame:
    """셀 클릭 패널용 상세 데이터. CACHE_CELL_DATA 우선, 없으면 CACHE_IDX fallback."""
    src = CACHE_CELL_DATA if CACHE_CELL_DATA.exists() else (CACHE_IDX if CACHE_IDX.exists() else None)
    if src is None:
        return pd.DataFrame()
    df = pd.read_parquet(src)
    df[GRID_JOIN_COL] = df[GRID_JOIN_COL].astype(str).str.strip()
    return df


# =========================================================
# 지도 레이어
# =========================================================


def _read_json_safe(path) -> str:
    p = Path(path)
    if not p.exists():
        return "null"
    with open(p, "r", encoding="utf-8") as fh:
        return fh.read()


@st.cache_data(show_spinner=False, max_entries=128)
def build_multi_map_html(
    sgg_code: str,
    metrics_json: str,
    center_lat: float,
    center_lng: float,
    zoom: int,
    height_px: int,
    fac_visible_json: str = "[]",
) -> str:
    """모든 지표 지도를 단일 HTML에 배치 — 같은 window에서 직접 Leaflet sync."""
    import copy as _copy
    sgg_key = normalize_sgg_code(sgg_code)
    metrics = json.loads(metrics_json)
    n = len(metrics)

    subway_js  = _read_json_safe(CACHE_SUBWAY_JSON)
    station_js = _read_json_safe(CACHE_STATION_JSON)

    fac_parts = []
    for ftype in FACILITY_ORDER:
        color = FACILITY_COLORS.get(ftype, "#999")
        label = FACILITY_LABELS_EN.get(ftype, ftype)
        fj = _read_json_safe(Path(CACHE_FAC_JSON_TPL.format(ftype=ftype)))
        if fj != "null":
            fac_parts.append(
                '{"id":' + json.dumps(ftype) + ',"label":' + json.dumps(label) +
                ',"c":' + json.dumps(color) + ',"d":' + fj + '}'
            )
    fac_js = "[" + ",".join(fac_parts) + "]"

    # 지표별 그리드 (색상 사전계산)
    grid_path = CACHE_GEOJSON_DIR / f"grid_{sgg_key}.json"
    if grid_path.exists():
        with open(grid_path, "r", encoding="utf-8") as fh:
            base_gj = json.load(fh)
    else:
        base_gj = None

    grid_js_list = []
    for m in metrics:
        if base_gj is None:
            grid_js_list.append("null")
            continue
        gj = _copy.deepcopy(base_gj)
        norm_obj = PowerNormSafe(gamma=m["gamma"], vmin=m["vmin"], vmax=m["vmax"])
        cmap_obj = matplotlib.colormaps[m["cmap"]]
        for feat in gj.get("features", []):
            val = feat["properties"].get(m["value_col"])
            try:
                fval = float(val) if val is not None else 0.0
            except Exception:
                fval = 0.0
            if fval > 0:
                try:
                    rgba = cmap_obj(norm_obj(fval))
                    feat["properties"]["_f"] = mcolors.to_hex(rgba, keep_alpha=False)
                    feat["properties"]["_o"] = 0.88
                except Exception:
                    feat["properties"]["_f"] = None
                    feat["properties"]["_o"] = 0.0
            else:
                feat["properties"]["_f"] = None
                feat["properties"]["_o"] = 0.0
        grid_js_list.append(json.dumps(gj))

    sgg_col = SGG_NAME_COL
    gj_col  = GRID_JOIN_COL
    grid_css = "display:block;" if n == 1 else "display:grid;grid-template-columns:1fr 1fr;gap:4px;"

    map_divs_parts = []
    for i, m in enumerate(metrics):
        map_divs_parts.append(
            '<div class="map-wrap">'
            '<div class="map-title">' + m["title"] + '</div>'
            '<div id="map' + str(i) + '" class="map-box"></div>'
            '</div>'
        )
    map_divs = "\n".join(map_divs_parts)

    maps_init_parts = []
    for i, m in enumerate(metrics):
        vcol  = json.dumps(m["value_col"])
        sc    = json.dumps(sgg_col)
        jc    = json.dumps(gj_col)
        mi    = str(i)
        maps_init_parts.append(
            "var map" + mi + "=L.map('map" + mi + "',{"
            "center:[" + str(center_lat) + "," + str(center_lng) + "],zoom:" + str(zoom) + ","
            "zoomControl:true,preferCanvas:true,renderer:L.canvas({tolerance:3})});\n"
            "L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',"
            "{attribution:'&copy; OpenStreetMap &copy; CARTO',subdomains:'abcd',maxZoom:19}).addTo(map" + mi + ");\n"
            # grid  ── Canvas 렌더러는 onEachFeature click 불가 → 지도 click으로 대체
            "(function(map,gd,vc,sc,jc){"
            "if(!gd)return;"
            # GeoJSON 레이어 (Canvas, 성능용)
            "L.geoJSON(gd,{style:function(f){var p=f.properties;"
            "if(!p._f)return{fillOpacity:0,weight:.3,color:'#aaa',opacity:.2};"
            "return{fillColor:p._f,fillOpacity:p._o||.88,weight:.5,color:'#555',opacity:.5};},"
            "onEachFeature:function(f,layer){var p=f.properties,t='';"
            "if(p[sc])t+='<b>'+p[sc]+'</b><br/>';"
            "var v=p[vc];if(v!=null)t+='Value: '+(+v).toFixed(3);"
            "if(t)layer.bindTooltip(t,{sticky:true,opacity:.95});},"
            "renderer:L.canvas({tolerance:2})}).addTo(map);"
            # 지도 click → features 배열에서 point-in-bbox 후 from_id 추출
            "var feats=gd.features||[];"
            "map.on('click',function(e){"
            "var lat=e.latlng.lat,lng=e.latlng.lng;"
            "var found=null;"
            "for(var i=0;i<feats.length;i++){"
            "var f=feats[i];"
            "if(!f.geometry)continue;"
            "var bb=getBBox(f.geometry);"  # bbox 먼저 체크 (빠름)
            "if(lat<bb[1]||lat>bb[3]||lng<bb[0]||lng>bb[2])continue;"
            "if(pointInPolygon(lat,lng,f.geometry)){found=f;break;}"
            "}"
            "if(!found)return;"
            "var fid=found.properties[jc];if(!fid)return;"
            "if(window._hlLayer){try{window._hlLayer.setStyle({weight:.5,color:'#555',opacity:.5});}catch(ex){}}"
            # Canvas에선 setStyle 불가 → 하이라이트 레이어를 별도 SVG GeoJSON으로
            "if(window._hlGeoJSON){map.removeLayer(window._hlGeoJSON);}"
            "window._hlGeoJSON=L.geoJSON(found,{style:function(){"
            "return{fill:false,weight:3,color:'#2458FF',opacity:1};"
            "}}).addTo(map);"
            "showCellPanel(String(fid));"
            "});"
            "})(map" + mi + ",gridData[" + mi + "]," + vcol + "," + sc + "," + jc + ");\n"
            # subway (그리드 위)
            "if(subwayData)L.geoJSON(subwayData,{style:function(){return{color:'#333',weight:2.2,opacity:.75,fill:false}},renderer:L.canvas()}).addTo(map" + mi + ");\n"
            # station (노선 위)
            "if(stationData)L.geoJSON(stationData,{pointToLayer:function(f,ll){"
            "return L.circleMarker(ll,{radius:5,color:'#222',weight:1.8,fillColor:'#fff',fillOpacity:1,opacity:1});},"
            "renderer:L.canvas()}).addTo(map" + mi + ");\n"
            # facilities (최상위)
            "facData.forEach(function(x){if(!x.d)return;"
            "if(!facLayers[x.id])facLayers[x.id]={maps:[],color:x.c,label:x.label};"
            "var lyr=L.geoJSON(x.d,{pointToLayer:function(f,ll){"
            "return L.circleMarker(ll,{radius:4.5,color:'#fff',weight:1.2,fillColor:x.c,fillOpacity:.9,opacity:.9});},"
            "renderer:L.canvas()});"
            "facLayers[x.id].maps.push({map:map" + mi + ",layer:lyr});"
            "if(facVisible.indexOf(x.id)>=0)lyr.addTo(map" + mi + ");});\n"
            "allMaps.push(map" + mi + ");"
        )
    maps_init_js = "\n".join(maps_init_parts)

    sync_js = (
        "allMaps.forEach(function(src){\n"
        "  src.on('moveend zoomend',function(){\n"
        "    if(src._syncBusy)return;\n"
        "    var c=src.getCenter(),z=src.getZoom();\n"
        "    allMaps.forEach(function(dst){\n"
        "      if(dst===src)return;\n"
        "      dst._syncBusy=true;\n"
        "      dst.setView(c,z,{animate:false,noMoveStart:true});\n"
        "      dst._syncBusy=false;\n"
        "    });\n"
        "  });\n"
        "});"
    )

    fac_toggle_js = (
        "var ctrl=document.getElementById('fc');\n"
        "Object.keys(facLayers).forEach(function(id){\n"
        "  var info=facLayers[id];\n"
        "  var lbl=document.createElement('label');\n"
        "  var cb=document.createElement('input'); cb.type='checkbox';\n"
        "  cb.checked=(facVisible.indexOf(id)>=0);\n"
        "  cb.addEventListener('change',function(){\n"
        "    var on=this.checked;\n"
        "    info.maps.forEach(function(x){\n"
        "      if(on)x.layer.addTo(x.map); else x.map.removeLayer(x.layer);\n"
        "    });\n"
        "  });\n"
        "  var dot=document.createElement('span'); dot.className='dot'; dot.style.background=info.color;\n"
        "  lbl.appendChild(cb); lbl.appendChild(dot);\n"
        "  lbl.appendChild(document.createTextNode(info.label));\n"
        "  ctrl.appendChild(lbl);\n"
        "});"
    )

    rows = 1 if n <= 2 else 2
    total_h = (height_px + 26) * rows + 16

    panel_css = (
        # 전체 레이아웃: 지도 + 패널을 가로로 배치
        '.outer-wrap{display:flex;flex-direction:row;gap:0;width:100%;}'
        '.maps-section{flex:1 1 0;min-width:0;}'
        # 클릭 패널
        '.cell-panel{'
        'flex:0 0 260px;width:260px;min-width:260px;'
        'background:#fafafa;border-left:1px solid #e0e0e0;'
        'padding:12px 14px 10px 14px;font-family:sans-serif;'
        'display:none;overflow-y:auto;'
        'max-height:' + str(height_px) + 'px;'
        '}'
        '.cell-panel.visible{display:block;}'
        '.cp-section-lbl{'
        'font-size:8.5px;font-weight:700;letter-spacing:.8px;'
        'text-transform:uppercase;margin-top:10px;margin-bottom:3px;}'
        '.cp-metrics{display:grid;grid-template-columns:1fr 1fr;gap:4px 8px;margin-bottom:2px;}'
        '.cp-m{background:#fff;border:1px solid #eee;border-radius:4px;'
        'padding:5px 7px;}'
        '.cp-m-label{font-size:8px;color:#999;margin-bottom:1px;}'
        '.cp-m-val{font-size:14px;font-weight:700;color:#222;line-height:1.2;}'
        '.cp-ref{display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px 6px;}'
        '.cp-r{}'
        '.cp-r-label{font-size:8px;color:#bbb;}'
        '.cp-r-val{font-size:11px;font-weight:600;color:#555;}'
        '.cp-close{float:right;cursor:pointer;font-size:16px;color:#bbb;'
        'line-height:1;margin-left:6px;}'
        '.cp-close:hover{color:#555;}'
        '.cp-id{font-size:12px;font-weight:700;color:#1a1a1a;'
        'border-bottom:2px solid #7B2CBF;padding-bottom:4px;margin-bottom:0;}'
        '.cp-id-sub{font-size:9px;color:#999;letter-spacing:.4px;margin-bottom:6px;}'
        # Chart.js canvas
        '#cp-chart-wrap{margin-top:4px;}'
        '#cp-chart{display:block;width:100%!important;height:150px!important;}'
    )

    chart_cdn = '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>'

    panel_js = (
        "var _cpChart=null;\n"
        "var TIME_SLOTS=['08:00','10:00','12:00','14:00','16:00','18:00','20:00','22:00'];\n"
        "var COV_COLS=['08_coverage','10_coverage','12_coverage','14_coverage','16_coverage','18_coverage','20_coverage','22_coverage'];\n"
        "var MAI_COLS=['08_mai','10_mai','12_mai','14_mai','16_mai','18_mai','20_mai','22_mai'];\n"
        "function fv(v,suf,dec){\n"
        "  if(v===null||v===undefined)return 'N/A';\n"
        "  var f=parseFloat(v);\n"
        "  if(isNaN(f))return 'N/A';\n"
        "  return f.toFixed(dec!==undefined?dec:2)+(suf!==undefined?suf:'%');\n"
        "}\n"
        "function showCellPanel(fid){\n"
        "  var d=(window.cellData||{})[fid];\n"
        "  var panel=document.getElementById('cell-panel');\n"
        "  document.getElementById('cp-fid').textContent=fid;\n"
        "  if(!d){panel.classList.add('visible');return;}\n"
        # PT metrics
        "  document.getElementById('cp-avg-cov').textContent=fv(d.avg_coverage);\n"
        "  document.getElementById('cp-avg-mai').textContent=fv(d.avg_mai);\n"
        "  document.getElementById('cp-cv-cov').textContent=fv(d.cv_coverage,'');\n"
        "  document.getElementById('cp-cv-mai').textContent=fv(d.cv_mai,'');\n"
        # Reference
        "  document.getElementById('cp-pop').textContent=fv(d.pop,'',0);\n"
        "  document.getElementById('cp-car-cov').textContent=fv(d.car_coverage);\n"
        "  document.getElementById('cp-car-mai').textContent=fv(d.car_mai);\n"
        # Chart
        "  var covVals=COV_COLS.map(function(c){var v=d[c];return(v!=null&&!isNaN(parseFloat(v)))?parseFloat(v):null;});\n"
        "  var maiVals=MAI_COLS.map(function(c){var v=d[c];return(v!=null&&!isNaN(parseFloat(v)))?parseFloat(v):null;});\n"
        "  if(_cpChart){_cpChart.destroy();_cpChart=null;}\n"
        "  var ctx=document.getElementById('cp-chart').getContext('2d');\n"
        "  _cpChart=new Chart(ctx,{type:'line',\n"
        "    data:{labels:TIME_SLOTS,datasets:[\n"
        "      {label:'Coverage (%)',data:covVals,borderColor:'#7B2CBF',backgroundColor:'#7B2CBF18',\n"
        "       pointBackgroundColor:'#7B2CBF',pointBorderColor:'#fff',pointBorderWidth:1.2,\n"
        "       pointRadius:3,borderWidth:1.8,tension:0.15,fill:false},\n"
        "      {label:'MAI (%)',data:maiVals,borderColor:'#9CCC65',backgroundColor:'#9CCC6518',\n"
        "       pointBackgroundColor:'#9CCC65',pointBorderColor:'#fff',pointBorderWidth:1.2,\n"
        "       pointRadius:3,borderWidth:1.8,tension:0.15,fill:false}\n"
        "    ]},\n"
        "    options:{responsive:true,maintainAspectRatio:false,\n"
        "      interaction:{mode:'index',intersect:false},\n"
        "      plugins:{\n"
        "        tooltip:{callbacks:{label:function(c){return c.dataset.label+': '+(c.parsed.y!=null?c.parsed.y.toFixed(1)+'%':'N/A');}}},\n"
        "        legend:{position:'top',labels:{font:{size:9},boxWidth:9,padding:4}}\n"
        "      },\n"
        "      scales:{\n"
        "        x:{grid:{color:'#f0f0f0'},ticks:{font:{size:8}}},\n"
        "        y:{min:0,max:100,grid:{color:'#f0f0f0'},ticks:{font:{size:8},callback:function(v){return v+'%';}}}\n"
        "      }\n"
        "    }});\n"
        "  panel.classList.add('visible');\n"
        "  allMaps.forEach(function(m){m.invalidateSize();});\n"
        "}\n"
        "document.getElementById('cp-close').addEventListener('click',function(){\n"
        "  document.getElementById('cell-panel').classList.remove('visible');\n"
        "  if(_cpChart){_cpChart.destroy();_cpChart=null;}\n"
        "  if(window._hlLayer){try{window._hlLayer.setStyle({weight:.5,color:'#555',opacity:.5});}catch(ex){}window._hlLayer=null;}\n"
        "  if(window._hlGeoJSON){allMaps.forEach(function(m){try{m.removeLayer(window._hlGeoJSON);}catch(ex){}});window._hlGeoJSON=null;}\n"
        "  allMaps.forEach(function(m){m.invalidateSize();});\n"
        "});\n"
    )

    panel_html = (
        '<div class="cell-panel" id="cell-panel">'
        '<div style="display:flex;align-items:flex-start;justify-content:space-between;">'
        '<div>'
        '<div class="cp-id-sub">SELECTED CELL</div>'
        '<div class="cp-id" id="cp-fid"></div>'
        '</div>'
        '<span class="cp-close" id="cp-close" title="닫기">&#x2715;</span>'
        '</div>'
        '<div class="cp-section-lbl" style="color:#7B2CBF;">PUBLIC TRANSIT</div>'
        '<div class="cp-metrics">'
        '<div class="cp-m"><div class="cp-m-label">avg. Coverage</div><div class="cp-m-val" id="cp-avg-cov">—</div></div>'
        '<div class="cp-m"><div class="cp-m-label">avg. MAI</div><div class="cp-m-val" id="cp-avg-mai">—</div></div>'
        '<div class="cp-m"><div class="cp-m-label">CV Coverage</div><div class="cp-m-val" id="cp-cv-cov">—</div></div>'
        '<div class="cp-m"><div class="cp-m-label">CV MAI</div><div class="cp-m-val" id="cp-cv-mai">—</div></div>'
        '</div>'
        '<div class="cp-section-lbl" style="color:#9CCC65;">REFERENCE</div>'
        '<div class="cp-ref">'
        '<div class="cp-r"><div class="cp-r-label">Population</div><div class="cp-r-val" id="cp-pop">—</div></div>'
        '<div class="cp-r"><div class="cp-r-label">Car Cov.</div><div class="cp-r-val" id="cp-car-cov">—</div></div>'
        '<div class="cp-r"><div class="cp-r-label">Car MAI</div><div class="cp-r-val" id="cp-car-mai">—</div></div>'
        '</div>'
        '<div class="cp-section-lbl" style="color:#555;">TIME-OF-DAY</div>'
        '<div id="cp-chart-wrap"><canvas id="cp-chart"></canvas></div>'
        '</div>'
    )

    parts = [
        '<!DOCTYPE html><html><head><meta charset="utf-8"/>',
        '<style>',
        '*{box-sizing:border-box;margin:0;padding:0;}',
        'body{background:#fff;font-family:sans-serif;}',
        '.maps-grid{' + grid_css + 'width:100%;}',
        '.map-wrap{position:relative;}',
        '.map-title{font-size:12px;font-weight:600;color:#333;padding:2px 6px;background:#f5f5f5;}',
        '.map-box{width:100%;height:' + str(height_px) + 'px;}',
        '.fac-ctrl{position:absolute;bottom:32px;right:8px;z-index:1000;',
        'background:rgba(255,255,255,.97);border:1px solid #bbb;border-radius:6px;',
        'padding:6px 10px;font-size:11px;line-height:1.9;',
        'box-shadow:0 2px 6px rgba(0,0,0,.12);min-width:150px;}',
        '.fac-hdr{font-weight:700;font-size:11px;color:#333;margin-bottom:3px;}',
        '.fac-ctrl label{display:flex;align-items:center;gap:6px;cursor:pointer;white-space:nowrap;}',
        '.fac-ctrl input{cursor:pointer;margin:0;width:13px;height:13px;}',
        '.dot{width:9px;height:9px;border-radius:50%;flex-shrink:0;border:1px solid rgba(0,0,0,.15);}',
        '.leaflet-tooltip{font-size:12px;background:rgba(255,255,255,.96);',
        'border:1px solid #ccc;padding:4px 8px;box-shadow:0 1px 4px rgba(0,0,0,.15);}',
        panel_css,
        '</style>',
        '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>',
        '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>',
        chart_cdn,
        '</head><body>',
        '<div class="outer-wrap">',
        '<div class="maps-section">',
        '<div class="maps-grid">',
        map_divs,
        '</div>',
        '<div class="fac-ctrl" id="fc"><div class="fac-hdr">Facilities</div></div>',
        '</div>',
        panel_html,
        '</div>',
        '<script>',
        'var gridData=[' + ",".join(grid_js_list) + '];',
        'var subwayData='  + subway_js  + ';',
        'var stationData=' + station_js + ';',
        'var facData='     + fac_js     + ';',
        'var facVisible='  + fac_visible_json + ';',
        'var facLayers={};',
        'var allMaps=[];',
        'window._hlLayer=null;',
        'window._hlGeoJSON=null;',
        # getBBox: GeoJSON geometry → [minLng, minLat, maxLng, maxLat]
        'function getBBox(geom){'
        'var mn=[Infinity,Infinity],mx=[-Infinity,-Infinity];'
        'function pt(c){if(c[0]<mn[0])mn[0]=c[0];if(c[1]<mn[1])mn[1]=c[1];'
        'if(c[0]>mx[0])mx[0]=c[0];if(c[1]>mx[1])mx[1]=c[1];}'
        'function ring(r){for(var i=0;i<r.length;i++)pt(r[i]);}'
        'var t=geom.type,c=geom.coordinates;'
        'if(t==="Polygon")for(var i=0;i<c.length;i++)ring(c[i]);'
        'else if(t==="MultiPolygon")for(var i=0;i<c.length;i++)for(var j=0;j<c[i].length;j++)ring(c[i][j]);'
        'return[mn[0],mn[1],mx[0],mx[1]];}',
        # pointInPolygon: ray-casting, lat/lng 순서 (Leaflet)
        'function pointInRing(lat,lng,ring){'
        'var inside=false;'
        'for(var i=0,j=ring.length-1;i<ring.length;j=i++){'
        'var xi=ring[i][0],yi=ring[i][1],xj=ring[j][0],yj=ring[j][1];'
        'if(((yi>lat)!=(yj>lat))&&(lng<(xj-xi)*(lat-yi)/(yj-yi)+xi))inside=!inside;}'
        'return inside;}'
        'function pointInPolygon(lat,lng,geom){'
        'var t=geom.type,c=geom.coordinates;'
        'if(t==="Polygon"){return pointInRing(lat,lng,c[0]);}'
        'if(t==="MultiPolygon"){'
        'for(var i=0;i<c.length;i++){if(pointInRing(lat,lng,c[i][0]))return true;}'
        'return false;}return false;}',
        maps_init_js,
        sync_js,
        fac_toggle_js,
        panel_js,
        '</script></body></html>',
    ]
    return "\n".join(parts)


def style_grid_feature(feature, cmap_name: str, norm, value_key: str):
    value = feature["properties"].get(value_key)
    color = get_grid_fill_color(value, cmap_name, norm)
    if color is None:
        return {"fillOpacity": 0.0, "weight": 0.15, "color": "#666666", "opacity": 0.25}
    return {"fillColor": color, "fillOpacity": 0.85, "weight": 0.15, "color": "#666666", "opacity": 0.30}



@st.cache_data(show_spinner=False, max_entries=256)
def load_colored_geojson(sgg_code: str, value_col: str, cmap_name: str,
                          norm_vmin: float, norm_vmax: float, norm_gamma: float) -> str:
    """시군구 GeoJSON을 디스크에서 읽고 각 feature에 fill_color를 미리 계산해서 반환.
    결과는 st.cache_data가 메모리에 캐시 → rerun마다 재계산 없음."""
    path = CACHE_GEOJSON_DIR / f"grid_{normalize_sgg_code(sgg_code)}.json"
    if not path.exists():
        return "{}"
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    norm = PowerNormSafe(gamma=norm_gamma, vmin=norm_vmin, vmax=norm_vmax)
    cmap = matplotlib.colormaps[cmap_name]

    for feat in gj.get("features", []):
        val = feat["properties"].get(value_col)
        if val is not None and val > 0:
            try:
                rgba = cmap(norm(float(val)))
                feat["properties"]["_fill"] = mcolors.to_hex(rgba, keep_alpha=False)
                feat["properties"]["_opacity"] = 0.85
            except Exception:
                feat["properties"]["_fill"] = None
                feat["properties"]["_opacity"] = 0.0
        else:
            feat["properties"]["_fill"] = None
            feat["properties"]["_opacity"] = 0.0
    return json.dumps(gj)


@st.cache_data(show_spinner=False)
def load_static_layer_json(path_str: str) -> str:
    """정적 GeoJSON 파일 읽기 캐시."""
    p = Path(path_str)
    if not p.exists():
        return "{}"
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def add_colored_grid_layer(m: folium.Map, sgg_code: str, value_col: str,
                            cmap_name: str, norm, tooltip_value_col: str):
    """색상이 properties에 미리 담긴 GeoJSON을 로드해서 style_function 없이 렌더링."""
    geojson_str = load_colored_geojson(
        sgg_code, value_col, cmap_name,
        float(norm.vmin), float(norm.vmax), float(norm.gamma),
    )
    if geojson_str == "{}":
        return

    fields = [GRID_JOIN_COL, SGG_NAME_COL, tooltip_value_col]
    aliases = ["Cell", "Municipality", "Value"]
    fields = [f for f in fields if f in [GRID_JOIN_COL, SGG_NAME_COL, tooltip_value_col]]

    GeoJson(
        data=geojson_str,
        style_function=lambda feat: {
            "fillColor": feat["properties"].get("_fill") or "#cccccc",
            "fillOpacity": feat["properties"].get("_opacity", 0.0),
            "weight": 0.15,
            "color": "#666666",
            "opacity": 0.30,
        },
        tooltip=GeoJsonTooltip(fields=fields, aliases=aliases[:len(fields)], sticky=False),
        control=False,
        smooth_factor=0,
        zoom_on_click=False,
    ).add_to(m)


def add_static_layers_from_cache(m: folium.Map, sgg_code: str,
                                  station_gdf, subway_gdf, fac_gdf,
                                  bbox_5179=None):
    """정적 레이어(역/노선/시설)를 디스크 캐시 GeoJSON에서 로드.
    캐시 파일이 없으면 메모리 GeoDataFrame fallback."""

    # 지하철 노선
    if CACHE_SUBWAY_JSON.exists():
        subway_json = load_static_layer_json(str(CACHE_SUBWAY_JSON))
    else:
        sub = bbox_subset(subway_gdf, bbox_5179) if bbox_5179 else subway_gdf
        subway_json = sub.to_crs(WEB_CRS).to_json() if len(sub) else None

    if subway_json and subway_json != "{}":
        folium.GeoJson(
            subway_json,
            style_function=lambda x: {"color": "#888888", "weight": 1.0, "opacity": 0.28},
            control=False,
            smooth_factor=2,
        ).add_to(m)

    # 역
    if CACHE_STATION_JSON.exists():
        station_json = load_static_layer_json(str(CACHE_STATION_JSON))
    else:
        sub = bbox_subset(station_gdf, bbox_5179) if bbox_5179 else station_gdf
        station_json = sub.to_crs(WEB_CRS).to_json() if len(sub) else None

    if station_json and station_json != "{}":
        folium.GeoJson(
            station_json,
            marker=folium.CircleMarker(radius=3, color="#666666", weight=0.9,
                                        fill=True, fill_color="white", fill_opacity=0.9),
            control=False,
        ).add_to(m)

    # 시설
    for ftype in FACILITY_ORDER:
        color = FACILITY_COLORS.get(ftype, "#999999")
        fac_path = Path(CACHE_FAC_JSON_TPL.format(ftype=ftype))
        if fac_path.exists():
            fac_json = load_static_layer_json(str(fac_path))
        else:
            sub = fac_gdf[fac_gdf["fac_type_norm"] == ftype]
            if bbox_5179:
                sub = bbox_subset(sub, bbox_5179)
            fac_json = sub.to_crs(WEB_CRS).to_json() if len(sub) else None

        if fac_json and fac_json != "{}":
            folium.GeoJson(
                fac_json,
                marker=folium.CircleMarker(radius=3, color="white", weight=0.6,
                                            fill=True, fill_color=color, fill_opacity=0.75),
                control=False,
            ).add_to(m)


def build_base_map(center: Tuple[float, float], zoom: int = 11) -> folium.Map:
    return folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="CartoDB positron",
        control_scale=False,
        prefer_canvas=True,
    )


def add_subway_station_facilities(m: folium.Map, station, subway, fac, bbox=None):
    if bbox is not None:
        station = bbox_subset(station, bbox) if not station.empty else station
        subway = bbox_subset(subway, bbox) if not subway.empty else subway
        fac = bbox_subset(fac, bbox) if not fac.empty else fac

    if len(subway):
        folium.GeoJson(
            subway.to_crs(WEB_CRS).to_json(),
            style_function=lambda x: {"color": "#666666", "weight": 1.0, "opacity": 0.30},
            control=False,
            smooth_factor=1,
        ).add_to(m)

    if len(station):
        station_w = station.to_crs(WEB_CRS)
        # CircleMarker 루프 대신 GeoJson으로 일괄 렌더링
        folium.GeoJson(
            station_w.to_json(),
            style_function=lambda x: {
                "radius": 4,
                "color": "#666666",
                "weight": 0.9,
                "fillColor": "white",
                "fillOpacity": 0.9,
                "opacity": 0.65,
            },
            marker=folium.CircleMarker(
                radius=3,
                color="#666666",
                weight=0.9,
                fill=True,
                fill_color="white",
                fill_opacity=0.9,
            ),
            control=False,
        ).add_to(m)

    if len(fac):
        fac_w = fac.to_crs(WEB_CRS)
        for ftype, gg in fac_w.groupby("fac_type_norm"):
            color = FACILITY_COLORS.get(ftype, "#999999")
            folium.GeoJson(
                gg.to_json(),
                style_function=lambda x, c=color: {
                    "radius": 5,
                    "color": "white",
                    "weight": 0.6,
                    "fillColor": c,
                    "fillOpacity": 0.72,
                    "opacity": 0.72,
                },
                marker=folium.CircleMarker(
                    radius=3,
                    color="white",
                    weight=0.6,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.72,
                ),
                control=False,
            ).add_to(m)


def add_grid_layer(m: folium.Map, cells: gpd.GeoDataFrame, value_col: str, cmap_name: str, norm):
    cells_safe = make_json_safe_geojson_gdf(cells, [GRID_JOIN_COL, SGG_NAME_COL, value_col]).to_crs(WEB_CRS)

    # GeoJSON 문자열로 변환해서 컬럼명 보존 (__geo_interface__는 properties 키가 달라질 수 있음)
    geojson_str = cells_safe.to_json()

    # GeoJSON properties 키를 직접 확인해서 tooltip fields 결정
    available_props = set(cells_safe.columns) - {"geometry"}
    fields = [c for c in [GRID_JOIN_COL, SGG_NAME_COL, value_col] if c in available_props]
    aliases_map = {
        GRID_JOIN_COL: "Cell",
        SGG_NAME_COL: "Municipality",
        value_col: "Value",
    }
    aliases = [aliases_map[c] for c in fields]

    tooltip = GeoJsonTooltip(fields=fields, aliases=aliases, sticky=False) if fields else None

    GeoJson(
        data=geojson_str,
        style_function=lambda feat: style_grid_feature(feat, cmap_name, norm, value_col),
        tooltip=tooltip,
        control=False,
        smooth_factor=0,
        zoom_on_click=False,
    ).add_to(m)


def add_sgg_layer(m: folium.Map, sgg: gpd.GeoDataFrame, value_col: str, cmap_name: str, norm):
    sgg_safe = make_json_safe_geojson_gdf(sgg, [SGG_NAME_COL, value_col]).to_crs(WEB_CRS)

    geojson_str = sgg_safe.to_json()

    available_props = set(sgg_safe.columns) - {"geometry"}
    fields = [c for c in [SGG_NAME_COL, value_col] if c in available_props]
    aliases_map = {
        SGG_NAME_COL: "Municipality",
        value_col: "Value",
    }
    aliases = [aliases_map[c] for c in fields]

    tooltip = GeoJsonTooltip(fields=fields, aliases=aliases, sticky=False) if fields else None

    GeoJson(
        data=geojson_str,
        style_function=lambda feat: style_grid_feature(feat, cmap_name, norm, value_col),
        tooltip=tooltip,
        control=False,
        smooth_factor=0,
        zoom_on_click=False,
    ).add_to(m)


# =========================================================
# 상세 패널
# =========================================================
def _fv(val, suffix: str = "%", dec: int = 2) -> str:
    if val is None:
        return "N/A"
    try:
        f = float(val)
        return "N/A" if f != f else f"{f:.{dec}f}{suffix}"
    except Exception:
        return "N/A"


@st.cache_data(show_spinner=False, max_entries=4096)
def _cell_chart_html(cov_json: str, mai_json: str, slots_json: str) -> str:
    """Chart.js 컴팩트 시계열 그래프 HTML (height ~160px)."""
    C, M = "#7B2CBF", "#9CCC65"
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"/>'
        '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>'
        '<style>*{margin:0;padding:0;box-sizing:border-box;}'
        'body{background:#fff;padding:0 2px;}'
        'canvas{display:block;width:100%!important;height:155px!important;}'
        '</style></head><body><canvas id="c"></canvas><script>'
        'new Chart(document.getElementById("c"),{type:"line",'
        'data:{labels:' + slots_json + ',datasets:['
        '{"label":"Coverage (%)","data":' + cov_json + ',"borderColor":"' + C + '",'
        '"backgroundColor":"' + C + '18","pointBackgroundColor":"' + C + '",'
        '"pointBorderColor":"#fff","pointBorderWidth":1.2,"pointRadius":3,'
        '"borderWidth":1.8,"tension":0.15,"fill":false},'
        '{"label":"MAI (%)","data":' + mai_json + ',"borderColor":"' + M + '",'
        '"backgroundColor":"' + M + '18","pointBackgroundColor":"' + M + '",'
        '"pointBorderColor":"#fff","pointBorderWidth":1.2,"pointRadius":3,'
        '"borderWidth":1.8,"tension":0.15,"fill":false}'
        ']},'
        'options:{responsive:true,maintainAspectRatio:false,'
        'interaction:{mode:"index",intersect:false},'
        'plugins:{'
        'tooltip:{callbacks:{label:function(c){'
        'return c.dataset.label+": "+(c.parsed.y!=null?c.parsed.y.toFixed(1)+"%":"N/A");}}},'
        'legend:{position:"top",labels:{font:{size:10},boxWidth:10,padding:5}}},'
        'scales:{'
        'x:{grid:{color:"#f0f0f0"},ticks:{font:{size:9}}},'
        'y:{min:0,max:100,grid:{color:"#f0f0f0"},'
        'ticks:{font:{size:9},callback:function(v){return v+"%";}}}}'
        '}});'
        '</script></body></html>'
    )


def render_cell_info_panel(cell_id: str, cell_df: pd.DataFrame):
    """학술발표용 컴팩트 셀 상세 패널 — 지도 우측 열에 배치."""
    row_df = cell_df[cell_df[GRID_JOIN_COL] == cell_id]
    if row_df.empty:
        st.caption(f"데이터 없음: {cell_id}")
        return
    row = row_df.iloc[0]

    # 셀 ID 헤더
    st.markdown(
        f'<div style="font-size:10px;color:#999;letter-spacing:.5px;margin-bottom:1px;">SELECTED CELL</div>'
        f'<div style="font-size:13px;font-weight:700;color:#1a1a1a;'
        f'border-bottom:2px solid #7B2CBF;padding-bottom:4px;margin-bottom:8px;">'
        f'{cell_id}</div>',
        unsafe_allow_html=True,
    )

    # ── PT 지표 (2×2 metric) ──────────────────────────────
    st.markdown(
        '<div style="font-size:9px;font-weight:700;letter-spacing:.8px;'
        'color:#7B2CBF;margin-bottom:4px;">PUBLIC TRANSIT</div>',
        unsafe_allow_html=True,
    )
    ca, cb = st.columns(2)
    ca.metric("avg. Coverage", _fv(row.get("avg_coverage")))
    cb.metric("avg. MAI",      _fv(row.get("avg_mai")))
    cc, cd = st.columns(2)
    cc.metric("CV Coverage",   _fv(row.get("cv_coverage"), suffix=""))
    cd.metric("CV MAI",        _fv(row.get("cv_mai"),      suffix=""))

    # ── 참고 지표 (소형 3열) ─────────────────────────────
    st.markdown(
        '<div style="font-size:9px;font-weight:700;letter-spacing:.8px;'
        'color:#9CCC65;margin-top:6px;margin-bottom:3px;">REFERENCE</div>',
        unsafe_allow_html=True,
    )

    def _ref_cell(label: str, val: str) -> str:
        return (
            f'<div style="line-height:1.35;">'
            f'<div style="font-size:9px;color:#aaa;">{label}</div>'
            f'<div style="font-size:12px;font-weight:600;color:#444;">{val}</div>'
            f'</div>'
        )

    r1, r2, r3 = st.columns(3)
    r1.markdown(_ref_cell("Population", _fv(row.get("pop"), suffix="", dec=0)), unsafe_allow_html=True)
    r2.markdown(_ref_cell("Car Cov.",   _fv(row.get("car_coverage"))),          unsafe_allow_html=True)
    r3.markdown(_ref_cell("Car MAI",    _fv(row.get("car_mai"))),               unsafe_allow_html=True)

    # ── 시계열 그래프 ─────────────────────────────────────
    st.markdown(
        '<div style="font-size:9px;font-weight:700;letter-spacing:.8px;'
        'color:#555;margin-top:8px;margin-bottom:2px;">TIME-OF-DAY PROFILE</div>',
        unsafe_allow_html=True,
    )
    cov_vals = [round(float(row[c]), 2) if pd.notna(row.get(c)) else None for c in COV_COLS]
    mai_vals = [round(float(row[c]), 2) if pd.notna(row.get(c)) else None for c in MAI_COLS]
    slots    = [t + ":00" for t in TIME_SLOTS]
    st_components.html(
        _cell_chart_html(json.dumps(cov_vals), json.dumps(mai_vals), json.dumps(slots)),
        height=172, scrolling=False,
    )


def render_selected_cell_panel(
    selected_from_id: str,
    grid_gdf: gpd.GeoDataFrame,
    idx_df: pd.DataFrame,
    metric_key: str,
    basis_key: str,
    station_gdf, subway_gdf, fac_gdf,
    cell_df: Optional[pd.DataFrame] = None,
):
    """cell_df 있으면 Chart.js 패널, 없으면 idx_df fallback."""
    data = cell_df if (cell_df is not None and not cell_df.empty) else idx_df
    render_cell_info_panel(selected_from_id, data)



# =========================================================
# norm / 값 컬럼
# =========================================================
def get_value_col(metric_key: str, basis_key: str) -> str:
    if metric_key == "pop":
        return "local_pop_map" if basis_key == "sgg" else "nat_pop_map"
    return f"{basis_key}_{metric_key}_ratio"


def get_norm_for_group(group: gpd.GeoDataFrame, metric_key: str, basis_key: str, scale_mode: str):
    global global_ratio_norm, global_pop_norm

    if scale_mode == "Global":
        return global_pop_norm if metric_key == "pop" else global_ratio_norm

    if metric_key == "pop":
        series = group["local_pop_map" if basis_key == "sgg" else "nat_pop_map"]
        return compute_group_pop_norm(series, share_mode=(POP_MAP_MODE == "share"))[2]

    vals = pd.concat(
        [
            pd.to_numeric(group[f"{basis_key}_fs_ratio"], errors="coerce"),
            pd.to_numeric(group[f"{basis_key}_fd_ratio"], errors="coerce"),
            pd.to_numeric(group[f"{basis_key}_tc_ratio"], errors="coerce"),
            pd.to_numeric(group[f"{basis_key}_tf_ratio"], errors="coerce"),
        ],
        axis=0,
    )
    return compute_group_norm_from_series(vals, gamma=0.55, force_zero_min=True)[2]


def subset_layers_by_bbox(
    bbox_5179: Optional[Tuple[float, float, float, float]],
    base_gdf: Optional[gpd.GeoDataFrame],
    station_gdf: gpd.GeoDataFrame,
    subway_gdf: gpd.GeoDataFrame,
    fac_gdf: gpd.GeoDataFrame,
):
    if bbox_5179 is None:
        if base_gdf is None or base_gdf.empty:
            return station_gdf.iloc[0:0], subway_gdf.iloc[0:0], fac_gdf.iloc[0:0]
        bbox_5179 = tuple(base_gdf.total_bounds)

    station_sub = bbox_subset(station_gdf, bbox_5179)
    subway_sub = bbox_subset(subway_gdf, bbox_5179)
    fac_sub = bbox_subset(fac_gdf, bbox_5179)
    return station_sub, subway_sub, fac_sub


def render_metric_maps(
    map_prefix: str,
    group_gdf: gpd.GeoDataFrame,
    aggregate_gdf: gpd.GeoDataFrame,
    selected_metrics: List[str],
    basis_key: str,
    scale_mode: str,
    station_gdf, subway_gdf, fac_gdf,
    initial_center: Tuple[float, float],
    initial_zoom: int = 11,
    click_source_gdf: Optional[gpd.GeoDataFrame] = None,
    selected_sgg_code: str = "",
    compare_partner_gdf: Optional[gpd.GeoDataFrame] = None,
    cell_df: Optional[pd.DataFrame] = None,
):
    if "fac_visible" not in st.session_state:
        st.session_state["fac_visible"] = list(FACILITY_ORDER)

    def get_norm_unified(mk):
        if compare_partner_gdf is not None and not compare_partner_gdf.empty:
            combined = pd.concat([group_gdf, compare_partner_gdf], ignore_index=True)
        else:
            combined = group_gdf
        return get_norm_for_group(combined, mk, basis_key, scale_mode)

    n = len(selected_metrics)
    metrics_list = []
    for mk in selected_metrics:
        vc   = get_value_col(mk, basis_key)
        norm = get_norm_unified(mk)
        metrics_list.append({
            "metric_key": mk, "value_col": vc,
            "cmap": CMAPS[mk],
            "vmin": float(norm.vmin), "vmax": float(norm.vmax), "gamma": float(norm.gamma),
            "title": LAYER_KEY_TO_LABEL[mk],
        })
    metrics_json = json.dumps(metrics_list)

    # ── 현재 시군구 셀 데이터를 JSON으로 직렬화해 iframe에 전달 ──
    data_src = cell_df if (cell_df is not None and not cell_df.empty) else None
    if data_src is not None:
        # 해당 시군구 셀만 필터링
        sgg_code_str = str(selected_sgg_code)
        if SGG_CODE_COL in data_src.columns:
            sub = data_src[data_src[SGG_CODE_COL].apply(normalize_sgg_code) == normalize_sgg_code(sgg_code_str)]
        else:
            # cell_df에 sgg 컬럼 없으면 group_gdf의 from_id 목록으로 필터
            valid_ids = set(group_gdf[GRID_JOIN_COL].astype(str).str.strip())
            sub = data_src[data_src[GRID_JOIN_COL].isin(valid_ids)]
        if sub.empty:
            cell_data_json = "{}"
        else:
            wanted_cols = ([GRID_JOIN_COL, "pop", "avg_coverage", "avg_mai",
                            "cv_coverage", "cv_mai", "car_coverage", "car_mai"]
                           + COV_COLS + MAI_COLS)
            sub = sub[[c for c in wanted_cols if c in sub.columns]].copy()

            def _safe(v):
                """float NaN/Inf → None, 나머지는 Python 기본형으로."""
                if v is None:
                    return None
                try:
                    f = float(v)
                    return None if (f != f or f == float('inf') or f == float('-inf')) else f
                except (TypeError, ValueError):
                    return str(v) if v else None

            cell_dict = {}
            for _, row in sub.iterrows():
                fid = str(row[GRID_JOIN_COL])
                cell_dict[fid] = {c: _safe(row[c]) for c in sub.columns if c != GRID_JOIN_COL}
            cell_data_json = json.dumps(cell_dict)
    else:
        cell_data_json = "{}"

    rows      = 1 if n <= 2 else 2
    iframe_h  = (MAP_HEIGHT + 26) * rows + 16

    html = build_multi_map_html(
        sgg_code         = str(selected_sgg_code),
        metrics_json     = metrics_json,
        center_lat       = initial_center[0],
        center_lng       = initial_center[1],
        zoom             = initial_zoom,
        height_px        = MAP_HEIGHT,
        fac_visible_json = json.dumps(st.session_state["fac_visible"]),
    )
    # cell_data_json은 캐시 밖에서 </body> 직전에 주입 (캐시 무력화 방지)
    html_final = html.replace(
        '</body></html>',
        f'<script>window.cellData={cell_data_json};</script></body></html>',
    )
    st_components.html(html_final, height=iframe_h, scrolling=False)

    if n == 1:
        norm = get_norm_unified(selected_metrics[0])
        st.markdown(make_streamlit_colorbar_html(
            cmap_name=CMAPS[selected_metrics[0]], norm=norm,
            title=LAYER_KEY_TO_LABEL[selected_metrics[0]] + " (%)", percent=True,
        ), unsafe_allow_html=True)
    else:
        cols = st.columns(2)
        for i, mk in enumerate(selected_metrics):
            with cols[i % 2]:
                norm = get_norm_unified(mk)
                st.markdown(make_streamlit_colorbar_html(
                    cmap_name=CMAPS[mk], norm=norm,
                    title=LAYER_KEY_TO_LABEL[mk] + " (%)", percent=True,
                ), unsafe_allow_html=True)
    return None



# =========================================================
# 앱 시작
# =========================================================
st.set_page_config(page_title="PT Deficit Dashboard", layout="wide")
st.title("PT accessibility deficit dashboard")

required_paths = [CLASSIFIED_PATH, GRID_PATH, STATION_PATH, SUBWAY_PATH, FAC_PATH]
missing_paths = [str(p) for p in required_paths if not p.exists()]
if missing_paths:
    st.error("필수 입력 파일이 없습니다.\n\n" + "\n".join(missing_paths))
    st.stop()

def _dropbox_safe_clear_geojson_dir():
    """Dropbox 동기화 잠금을 우회하며 geojson_tiles 폴더를 비운다.
    shutil.rmtree 대신 파일별 개별 삭제 + 재시도로 PermissionError 방지."""
    import os, time
    if not CACHE_GEOJSON_DIR.exists():
        CACHE_GEOJSON_DIR.mkdir(parents=True, exist_ok=True)
        return
    # 파일 개별 삭제 (최대 5회 재시도)
    for f in list(CACHE_GEOJSON_DIR.iterdir()):
        if f.is_file():
            for attempt in range(6):
                try:
                    f.unlink(missing_ok=True)
                    break
                except PermissionError:
                    time.sleep(0.4 * (attempt + 1))
    # 빈 하위 폴더 제거
    for d in list(CACHE_GEOJSON_DIR.iterdir()):
        if d.is_dir():
            for attempt in range(6):
                try:
                    d.rmdir()
                    break
                except (PermissionError, OSError):
                    time.sleep(0.4 * (attempt + 1))
    # 폴더 자체는 유지 (rmdir 대신)
    CACHE_GEOJSON_DIR.mkdir(parents=True, exist_ok=True)


def _run_build_with_progress():
    _dropbox_safe_clear_geojson_dir()
    load_cached_data.clear()
    load_colored_geojson.clear()
    load_static_layer_json.clear()
    try: build_multi_map_html.clear()
    except Exception: pass

    status = st.empty()
    pbar   = st.progress(0)
    stxt   = st.empty()
    status.info("⚙️ 캐시 빌드 시작...")
    pbar.progress(5)

    def _pcb(step, total, msg):
        pbar.progress(min(int(10 + 85 * step / max(total, 1)), 95))
        stxt.caption("🗺️ " + msg)

    build_dashboard_cache(progress_cb=_pcb)
    pbar.progress(100)
    stxt.empty()
    status.success("✅ 캐시 빌드 완료!")


with st.sidebar:
    st.header("Setup")
    if st.button("Build / refresh cached data", use_container_width=True, key="btn_refresh_cache"):
        _run_build_with_progress()
        st.rerun()

_tiles_ok   = CACHE_GEOJSON_DIR.exists() and any(CACHE_GEOJSON_DIR.glob("grid_*.json"))
cache_ready = _tiles_ok and all(
    p.exists() for p in [CACHE_GRID, CACHE_SGG, CACHE_STATION, CACHE_SUBWAY, CACHE_FAC, CACHE_TS, CACHE_IDX]
)
if not cache_ready:
    st.info("⚙️ 처음 실행입니다. 캐시를 빌드합니다 (수 분 소요)...")
    _pbar = st.progress(0)
    _stxt = st.empty()
    def _auto_pcb(step, total, msg):
        _pbar.progress(int(100 * step / max(total, 1)))
        _stxt.caption("🗺️ " + msg)
    build_dashboard_cache(progress_cb=_auto_pcb)
    _pbar.progress(100)
    _stxt.empty()
    st.rerun()

with st.spinner("데이터 로드 중..."):
    grid_gdf, grid_simple_gdf, sgg_gdf, station_gdf, subway_gdf, fac_gdf, ts_df, idx_df = load_cached_data()
    cell_df = load_cell_detail_data()

global_ratio_norm = compute_group_norm_from_series(
    pd.concat(
        [
            grid_gdf["nat_fs_ratio"],
            grid_gdf["nat_fd_ratio"],
            grid_gdf["nat_tc_ratio"],
            grid_gdf["nat_tf_ratio"],
            grid_gdf["sgg_fs_ratio"],
            grid_gdf["sgg_fd_ratio"],
            grid_gdf["sgg_tc_ratio"],
            grid_gdf["sgg_tf_ratio"],
        ],
        axis=0,
    ),
    gamma=0.55,
    force_zero_min=True,
)[2]
global_pop_norm = compute_group_pop_norm(grid_gdf["nat_pop_map"], share_mode=(POP_MAP_MODE == "share"))[2]

sgg_options = sorted(
    grid_gdf[[SGG_CODE_COL, SGG_NAME_COL]].drop_duplicates().itertuples(index=False, name=None),
    key=lambda x: x[1]
)
sgg_name_to_code = {name: code for code, name in sgg_options}

def _sido(name): return name.split("_")[0] if "_" in name else name
def _sgg(name):  return name.split("_", 1)[1] if "_" in name else name

sido_list = sorted(set(_sido(n) for n in sgg_name_to_code))
sido_to_names = {}
for name in sgg_name_to_code:
    sido_to_names.setdefault(_sido(name), []).append(name)
for k in sido_to_names:
    sido_to_names[k] = sorted(sido_to_names[k], key=_sgg)


def sgg_selector(prefix, la="Province", lb="Municipality"):
    sido_sel = st.selectbox(la, sido_list, key=prefix + "_sido")
    names_in = sido_to_names.get(sido_sel, list(sgg_name_to_code))
    disp     = [_sgg(n) for n in names_in]
    sgg_disp = st.selectbox(lb, disp, key=prefix + "_sgg")
    full     = sido_sel + "_" + sgg_disp
    return full, sgg_name_to_code.get(full)


with st.sidebar:
    st.header("Display")
    compare_mode = st.toggle("Compare two municipalities", value=False, key="toggle_compare")

    if not compare_mode:
        st.markdown("**Municipality**")
        selected_full, selected_code = sgg_selector("single")
    else:
        st.markdown("**Municipality A**")
        full1, code1 = sgg_selector("cmp_a", "Province A", "City/County A")
        st.markdown("**Municipality B**")
        full2, code2 = sgg_selector("cmp_b", "Province B", "City/County B")

    basis_label = st.selectbox(
        "Benchmark basis",
        ["Municipality-based benchmark", "National benchmark"],
        index=0, key="benchmark_basis",
    )
    basis = "sgg" if basis_label.startswith("Municipality") else "nat"

    selected_metric_labels = st.multiselect(
        "Indicators to display",
        ["F(s)", "F(d)", "T(c)", "T(f)", "Population"],
        default=["F(s)"], key="metric_multiselect",
    )
    if not selected_metric_labels:
        selected_metric_labels = ["F(s)"]

    scale_label = st.selectbox(
        "Color scale",
        ["Local scale by municipality", "Global scale"],
        index=0, key="scale_mode_select",
    )
    scale_mode = "Local by SGG" if scale_label.startswith("Local") else "Global"

    st.caption("Indicator definitions")
    st.markdown(build_metric_help_html(), unsafe_allow_html=True)
    st.caption("Stations, subway lines, and facility points shown on each map.")

selected_metric_keys = [LAYER_LABEL_TO_KEY[x] for x in selected_metric_labels]

if not compare_mode:
    if not selected_code:
        st.warning("선택된 시군구를 찾을 수 없습니다.")
        st.stop()
    group        = grid_simple_gdf[grid_simple_gdf[SGG_CODE_COL] == selected_code].copy()
    group_detail = grid_gdf[grid_gdf[SGG_CODE_COL] == selected_code].copy()
    sgg_group    = sgg_gdf[sgg_gdf[SGG_CODE_COL] == selected_code].copy()
    center_web   = group_detail.to_crs(WEB_CRS).geometry.centroid.unary_union.centroid

    render_metric_maps(
        map_prefix="single_city",
        group_gdf=group, aggregate_gdf=sgg_group,
        selected_metrics=selected_metric_keys,
        basis_key=basis, scale_mode=scale_mode,
        station_gdf=station_gdf, subway_gdf=subway_gdf, fac_gdf=fac_gdf,
        initial_center=(center_web.y, center_web.x), initial_zoom=11,
        click_source_gdf=group_detail,
        selected_sgg_code=str(selected_code),
        compare_partner_gdf=None,
        cell_df=cell_df,
    )

else:
    if not code1 or not code2:
        st.warning("선택된 시군구를 찾을 수 없습니다.")
        st.stop()
    group1        = grid_simple_gdf[grid_simple_gdf[SGG_CODE_COL] == code1].copy()
    group2        = grid_simple_gdf[grid_simple_gdf[SGG_CODE_COL] == code2].copy()
    group1_detail = grid_gdf[grid_gdf[SGG_CODE_COL] == code1].copy()
    group2_detail = grid_gdf[grid_gdf[SGG_CODE_COL] == code2].copy()
    sgg1_poly     = sgg_gdf[sgg_gdf[SGG_CODE_COL] == code1].copy()
    sgg2_poly     = sgg_gdf[sgg_gdf[SGG_CODE_COL] == code2].copy()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(full1)
        center1 = group1_detail.to_crs(WEB_CRS).geometry.centroid.unary_union.centroid
        render_metric_maps(
            map_prefix="compare_a",
            group_gdf=group1, aggregate_gdf=sgg1_poly,
            selected_metrics=selected_metric_keys,
            basis_key=basis, scale_mode=scale_mode,
            station_gdf=station_gdf, subway_gdf=subway_gdf, fac_gdf=fac_gdf,
            initial_center=(center1.y, center1.x), initial_zoom=11,
            click_source_gdf=group1_detail,
            selected_sgg_code=str(code1), compare_partner_gdf=group2,
            cell_df=cell_df,
        )
    with c2:
        st.subheader(full2)
        center2 = group2_detail.to_crs(WEB_CRS).geometry.centroid.unary_union.centroid
        render_metric_maps(
            map_prefix="compare_b",
            group_gdf=group2, aggregate_gdf=sgg2_poly,
            selected_metrics=selected_metric_keys,
            basis_key=basis, scale_mode=scale_mode,
            station_gdf=station_gdf, subway_gdf=subway_gdf, fac_gdf=fac_gdf,
            initial_center=(center2.y, center2.x), initial_zoom=11,
            click_source_gdf=group2_detail,
            selected_sgg_code=str(code2), compare_partner_gdf=group1,
            cell_df=cell_df,
        )