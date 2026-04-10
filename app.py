from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import Normalize, PowerNorm
from shapely.geometry import box
import streamlit.components.v1 as st_components


# =========================================================
# 경로 자동 탐색
# =========================================================
def pick_first_existing_path(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


import os as _os

_DATA_ROOT_ENV = _os.environ.get("DATA_ROOT", "")

if _DATA_ROOT_ENV:
    # ── 서버 모드: DATA_ROOT=/app/data (docker-compose volume) ──────────────
    # data/ 폴더에 아래 파일들을 flat하게 넣어두면 됩니다:
    #   from_metrics_500m_intracity_classified.parquet
    #   500m.gpkg, station.gpkg, subway.gpkg
    #   all_facilities.geoparquet
    #   from_metrics_500m_intracity.parquet
    #   od_500m_intracity.parquet
    #   deficit_ref_sgg.csv  (optional)
    DATA_ROOT = Path(_DATA_ROOT_ENV)
    ROOT_OUT  = DATA_ROOT

    CLASSIFIED_PATH = pick_first_existing_path(
        DATA_ROOT / "from_metrics_500m_intracity_oh_classified.parquet",
        DATA_ROOT / "from_metrics_500m_intracity_classified.parquet",
        DATA_ROOT / "500m_classified.parquet",
    )
    GRID_PATH    = DATA_ROOT / "500m.gpkg"
    STATION_PATH = DATA_ROOT / "station.gpkg"
    SUBWAY_PATH  = DATA_ROOT / "subway.gpkg"
    FAC_PATH     = pick_first_existing_path(
        DATA_ROOT / "all_activities.geoparquet",
        DATA_ROOT / "all_facilities.geoparquet",
    )
    INTRACITY_PATH = pick_first_existing_path(
        DATA_ROOT / "from_metrics_500m_intracity_oh.parquet",
        DATA_ROOT / "from_metrics_500m_intracity.parquet",
        DATA_ROOT / "from_metrics_500m_intracity_oh_classified.parquet",
    )
    OD_PATH = pick_first_existing_path(
        DATA_ROOT / "od_500m_intracity_oh.parquet",
        DATA_ROOT / "od_500m_intracity.parquet",
        DATA_ROOT / "od.parquet",
    )
    DEFICIT_REF_NAT_PATH = pick_first_existing_path(
        DATA_ROOT / "deficit_ref_sgg.csv",
        DATA_ROOT / "deficit_ref_nat.csv",
    )
    SPATIAL_ALL_PATH = pick_first_existing_path(
        DATA_ROOT / "from_metrics_500m_intracity_oh_spatial_all.parquet",
        DATA_ROOT / "spatial_all.parquet",
    )

else:
    # ── 로컬 모드: Dropbox 절대경로 ─────────────────────────────────────────
    def _pick_dropbox_base() -> Path:
        for p in [Path(r"E:\Dropbox"), Path(r"C:\Users\82102\Dropbox"), Path.home() / "Dropbox"]:
            if p.exists():
                return p
        return Path(r"C:\Users\82102\Dropbox")

    _DB    = _pick_dropbox_base()
    _PAPER = _DB / r"01-대학원\02-Paper Work\01-개인연구\202603_격자 단위 다양한 시설 대중교통 접근성 결핍 진단"
    _RAW   = _PAPER / r"03-분석자료\01-기초자료\01-전처리\02_routing\01_intercity\02_500m"
    _RAW2  = _PAPER / r"03-분석자료\01-기초자료\01-전처리\02_routing\02_intracity\02_500m"
    _CODE  = _DB / "06-Code"
    _VIZ   = _PAPER / r"03-분석자료\01-기초자료\02-삽도자료"

    ROOT_OUT = _PAPER / r"03-분석자료\01-기초자료\02-삽도자료\02_학회자료\01_교통학회"

    CLASSIFIED_PATH = pick_first_existing_path(
        _RAW / "from_metrics_500m_intracity_oh_classified.parquet",
        _RAW / "from_metrics_500m_intracity_classified.parquet",
        _RAW2 / "from_metrics_500m_intracity_classified.parquet",
        _RAW / "500m_classified.parquet",
    )
    GRID_PATH    = _PAPER / r"03-분석자료\01-기초자료\01-전처리\02_routing\00_grid\500m.gpkg"
    STATION_PATH = pick_first_existing_path(
        _CODE / "station.gpkg",
        _VIZ / r"01_재료\station.gpkg",
    )
    SUBWAY_PATH  = pick_first_existing_path(
        _CODE / "subway.gpkg",
        _VIZ / r"01_재료\subway.gpkg",
    )
    FAC_PATH     = pick_first_existing_path(
        _CODE / "all_activities.geoparquet",
        _RAW / "all_activities.geoparquet",
        _RAW / "all_facilities.geoparquet",
        _RAW2 / "all_activities.geoparquet",
    )
    INTRACITY_PATH = pick_first_existing_path(
        _RAW / "from_metrics_500m_intracity_oh.parquet",
        _RAW / "from_metrics_500m_intracity.parquet",
        _RAW / "from_metrics_500m_intracity_oh_classified.parquet",
        _RAW2 / "from_metrics_500m_intracity.parquet",
    )
    OD_PATH = pick_first_existing_path(
        _RAW / "od_500m_intracity_oh.parquet",
        _RAW / "od_500m_intracity.parquet",
        _RAW2 / "od_500m_intracity.parquet",
        _RAW / "od.parquet",
    )
    DEFICIT_REF_NAT_PATH = pick_first_existing_path(
        _PAPER / r"01-학회자료\01_교통학회\03_분석결과\deficit_ref_sgg.csv",
        _PAPER / r"03-분석자료\01-기초자료\deficit_ref_sgg.csv",
    )
    SPATIAL_ALL_PATH = pick_first_existing_path(
        _RAW / "from_metrics_500m_intracity_oh_spatial_all.parquet",
        _RAW / "spatial_all.parquet",
        _RAW2 / "spatial_all.parquet",
        _PAPER / r"03-분석자료\02-산출물\spatial_all.parquet",
    )

CACHE_DIR = ROOT_OUT / "dashboard_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_GRID        = CACHE_DIR / "grid_dashboard_5179.geoparquet"
CACHE_GRID_SIMPLE = CACHE_DIR / "grid_dashboard_simple_5179.geoparquet"
CACHE_SGG         = CACHE_DIR / "sgg_dashboard_5179.geoparquet"
CACHE_STATION     = CACHE_DIR / "station_5179.geoparquet"
CACHE_SUBWAY      = CACHE_DIR / "subway_5179.geoparquet"
CACHE_FAC         = CACHE_DIR / "facilities_5179.geoparquet"
CACHE_TS          = CACHE_DIR / "grid_timeseries.parquet"
CACHE_IDX         = CACHE_DIR / "grid_index_points_5179.parquet"
CACHE_CELL_DATA   = CACHE_DIR / "cell_detail_data.parquet"

CACHE_GEOJSON_DIR    = CACHE_DIR / "geojson_tiles"
CACHE_STATION_JSON   = CACHE_GEOJSON_DIR / "station.json"
CACHE_SUBWAY_JSON    = CACHE_GEOJSON_DIR / "subway.json"
CACHE_FAC_JSON_TPL   = str(CACHE_GEOJSON_DIR / "fac_{ftype}.json")
CACHE_FACILITY_ACCESS = CACHE_DIR / "grid_facility_access.parquet"
CACHE_NEIGHBORS       = CACHE_DIR / "neighbors.json"

# =========================================================
# 상수 / 스타일
# =========================================================
PLOT_CRS = "EPSG:5179"
WEB_CRS  = "EPSG:4326"

GRID_ID_COL   = "GRID_500M_"
GRID_JOIN_COL = "from_id"
SGG_CODE_COL  = "from_sgg_key"
SGG_NAME_COL  = "from_sgg"

FAC_KIND_COL = "facility_kind"
FAC_TYPE_COL = "facility_type"
FAC_DEPT_COL = "department"

TIME_SLOTS = ["06", "08", "10", "12", "14", "16", "18", "20", "22", "24"]
COV_COLS   = [f"pt{t}_coverage" for t in TIME_SLOTS]
MAI_COLS   = [f"pt{t}_mai"      for t in TIME_SLOTS]
COV_ALLOPEN_COLS = [f"pt{t}_coverage_allopen" for t in TIME_SLOTS]
MAI_ALLOPEN_COLS = [f"pt{t}_mai_allopen"      for t in TIME_SLOTS]
# Legacy fallback: old format without 'pt' prefix
_LEGACY_COV = [f"{t}_coverage" for t in TIME_SLOTS]
_LEGACY_MAI = [f"{t}_mai"      for t in TIME_SLOTS]

OD_FACILITY_COLS = ["pharmacy", "grocery", "library", "park", "public", "m1", "m2", "m3", "m4", "m5", "m6"]
# 5종 추가 시설 (Step-1 스크립트로 OD parquet에 추가 후 캐시 재빌드 필요)
NEW_FAC_COLS = ["nursery", "primary", "junior", "high", "elderly"]
ALL_FAC_COLS = OD_FACILITY_COLS + NEW_FAC_COLS  # 16종 (OD에 없으면 자동 스킵)

OD_FACILITY_LABELS = {
    "pharmacy": "Pharmacy",
    "grocery":  "Grocery",
    "library":  "Library",
    "park":     "Park",
    "public":   "Public service",
    "m1":       "Primary care",       # 가정의학과, 내과, 소아청소년과
    "m2":       "Rehab & ortho",      # 정형외과, 재활의학과, 마취통증의학과
    "m3":       "Specialty clinic",   # 안과, 이비인후과, 피부과, 비뇨기, 신경과, 산부인과
    "m4":       "Mental health",      # 정신건강의학과
    "m5":       "Dental",             # 치과 계열
    "m6":       "Korean medicine",    # 한방 계열
    "nursery":  "Childcare",          # 어린이집, 유치원
    "primary":  "Primary school",     # 초등학교
    "junior":   "Middle school",      # 중학교
    "high":     "High school",        # 고등학교
    "elderly":  "Senior welfare",     # 노인여가시설
}

# m2~m6을 "Specialist care"로 묶어 표시 (논문 기준: 하나라도 접근 가능하면 accessible)
# 세부 과목은 inaccessible 시 토글로 확인
SPECIALIST_COLS  = ["m2", "m3", "m4", "m5", "m6"]
SPECIALIST_LABEL = "Specialist care"
SPECIALIST_DETAIL_LABELS = {
    "m2": "Rehab & ortho",
    "m3": "Specialty clinic",
    "m4": "Mental health",
    "m5": "Dental",
    "m6": "Korean medicine",
}
# 패널/툴팁에서 실제로 표시할 시설 순서 (m2~m6 → specialist 하나로)
DISPLAY_FAC_COLS   = ["park", "library", "m1", "specialist", "grocery", "public", "pharmacy",
                      "nursery", "primary", "junior", "high", "elderly"]
DISPLAY_FAC_LABELS = {
    "park":       "Park",
    "library":    "Library",
    "m1":         "Primary care",
    "specialist": SPECIALIST_LABEL,
    "grocery":    "Grocery",
    "public":     "Public service",
    "pharmacy":   "Pharmacy",
    "nursery":    "Childcare",
    "primary":    "Primary school",
    "junior":     "Middle school",
    "high":       "High school",
    "elderly":    "Senior welfare",
}

# 사이드바/필터 패널용 12종 선택기 정의 (id, label, 실제 fac_cols 매핑)
FAC_SELECTOR_DEFS = [
    {"id": "park",       "label": "Park",            "fac_cols": ["park"]},
    {"id": "library",    "label": "Library",          "fac_cols": ["library"]},
    {"id": "m1",         "label": "Primary care",     "fac_cols": ["m1"]},
    {"id": "specialist", "label": "Specialist care",  "fac_cols": ["m2","m3","m4","m5","m6"]},
    {"id": "grocery",    "label": "Grocery",          "fac_cols": ["grocery"]},
    {"id": "public",     "label": "Public service",   "fac_cols": ["public"]},
    {"id": "pharmacy",   "label": "Pharmacy",         "fac_cols": ["pharmacy"]},
    {"id": "nursery",    "label": "Childcare",        "fac_cols": ["nursery"]},
    {"id": "primary",    "label": "Primary school",   "fac_cols": ["primary"]},
    {"id": "junior",     "label": "Middle school",    "fac_cols": ["junior"]},
    {"id": "high",       "label": "High school",      "fac_cols": ["high"]},
    {"id": "elderly",    "label": "Senior welfare",   "fac_cols": ["elderly"]},
]
# 기본 선택: 원래 7종 (Coverage/MAI 파이프라인 기준)
FAC_DEFAULT_SEL = ["park", "library", "m1", "specialist", "grocery", "public", "pharmacy"]

# 시설별 Coverage 접근 기준시간 (분) — 국토부 제2차 국가도시재생기본방침 기준
FAC_COV_THRESH = {
    "park":     15, "library":  15,
    "m1":       10, "m2":       15, "m3":       15,
    "m4":       15, "m5":       15, "m6":       15,
    "grocery":  10, "public":   15, "pharmacy": 10,
    "nursery":  15, "primary":  10, "junior":   15,
    "high":     15, "elderly":  15,
}
# MAI는 항상 15분 기준 (출발지에서 15분 내 도달 가능한 to_id 중 최다 시설 그리드)
MAI_THRESH = 15

LAYER_LABEL_TO_KEY = {"F(s)": "fs", "F(d)": "fd", "F(o)": "fo", "T(c)": "tc", "T(f)": "tf", "Population": "pop"}
LAYER_KEY_TO_LABEL = {v: k for k, v in LAYER_LABEL_TO_KEY.items()}
LAYER_HELP = {
    "F(s)": "Facility siting / sub-optimal location problem",
    "F(d)": "Facility dispersion problem",
    "F(o)": "Facility operating hours constraint",
    "T(c)": "Transit connection problem",
    "T(f)": "Transit frequency problem",
    "Population": "Population share or density",
}

DEFICIT_KEYS = ["fs", "fd", "fo", "tc", "tf"]

BASE_MAP_KEYS   = ["coverage", "mai", "mvg", "pop"]
ALL_MAP_KEYS    = BASE_MAP_KEYS  # 4 maps only (JCL via Cluster View overlay)
BASE_MAP_LABELS = {
    "pop": "Population",
    "coverage": "Coverage (avg.)",
    "mai": "MAI (avg.)",
    "mvg": "MV Geary",
}

# ── 컬러맵 (요청 반영) ────────────────────────────
CMAPS = {
    "fs":       "RdPu",
    "fd":       "BuPu",
    "fo":       "OrRd",
    "tc":       "PuRd",
    "tf":       "YlOrBr",
    "pop":      "Reds",
    "coverage": "YlGnBu",
    "mai":      "BuPu",
    "jcl":      "PuBuGn",     # 시원한 청록 계열
    "mvg":      "Set2",       # categorical placeholder (실제론 직접 색상)
}

# JCL 클러스터 오버레이 색상 (deficit type별)
JCL_COLORS = {
    "fs": "#D32F2F",  "fd": "#E65100",  "fo": "#FF5722",
    "tc": "#6A1B9A",  "tf": "#00838F",
}
# MVG 프로파일별 색상
MVG_PROFILE_COLORS = {
    "None":                "#A8D8A8",   # soft mint green
    "T(c)":                "#7BA7CC",   # dusty blue
    "T(c)+T(f)":           "#B5A4C8",   # soft lavender
    "F(o)":                "#F5A882",   # peach
    "F(o)+T(c)":           "#E8878E",   # dusty rose
    "F(d)+T(c)":           "#91CEC2",   # soft teal
    "F(d)+F(o)+T(c)":      "#F7D794",   # warm vanilla
    "F(s)+F(d)+T(c)":      "#D4A5A5",   # muted coral
    "F(s)+T(c)+T(f)":      "#C9B8D9",   # light mauve
    "F(s)+F(d)+T(c)+T(f)": "#E07070",   # soft red
}
MVG_HETERO_COLOR = "#D5DDE0"    # very light blue-grey
MVG_NOTSIG_COLOR = "#F5F5F5"

# 결핍 격자 테두리 색상 (유형별 구분, fill 없음)
DEFICIT_COLORS = {
    "fs": "#E53935",   # 빨강
    "fd": "#F4A100",   # 골드/주황
    "fo": "#FF7043",   # 따뜻한 오렌지
    "tc": "#7B1FA2",   # 보라
    "tf": "#2E7D32",   # 진한 틸 (가시성 향상)
}
DEFICIT_LABELS = {"fs": "F(s)", "fd": "F(d)", "fo": "F(o)", "tc": "T(c)", "tf": "T(f)"}

FACILITY_COLORS = {
    "park":     "#43A047", "library": "#1E88E5",
    "m1":       "#E53935", "m2":      "#8E24AA",
    "grocery":  "#FB8C00", "public":  "#2E7D32",
    "pharmacy": "#D81B60",
    "nursery":  "#00ACC1", "primary": "#5E35B1",
    "junior":   "#3949AB", "high":    "#546E7A",
    "elderly":  "#795548",
}
FACILITY_LABELS_EN = {
    "park":     "Park",
    "library":  "Library",
    "m1":       "Primary care",
    "m2":       "Specialist care",
    "grocery":  "Grocery",
    "public":   "Public service",
    "pharmacy": "Pharmacy",
    "nursery":  "Childcare",
    "primary":  "Primary school",
    "junior":   "Middle school",
    "high":     "High school",
    "elderly":  "Senior welfare",
}
FACILITY_ORDER = ["park", "library", "m1", "m2", "grocery", "public", "pharmacy",
                  "nursery", "primary", "junior", "high", "elderly"]

MED_GROUP_MAP_RAW = {
    "가정의학과": "m1", "내과": "m1", "소아청소년과": "m1",
    "정형외과": "m2", "재활의학과": "m2", "마취통증의학과": "m2",
    "안과": "m3", "이비인후과": "m3", "피부과": "m3",
    "비뇨의학과": "m3", "신경과": "m3", "산부인과": "m3",
    "정신건강의학과": "m4",
    "치과": "m5", "통합치의학과": "m5", "소아치과": "m5",
    "치과교정과": "m5", "치과보존과": "m5", "치과보철과": "m5",
    "치주과": "m5", "구강내과": "m5",
    "예방치과": "m8", "영상치의학과": "m8", "구강병리과": "m8", "구강악안면외과": "m8",
    "사상체질과": "m6", "침구과": "m6", "한방내과": "m6",
    "한방부인과": "m6", "한방소아과": "m6", "한방신경정신과": "m6",
    "한방안·이비인후·피부과": "m6", "한방재활의학과": "m6",
    "한방응급": "m7", "외과": "m7", "신경외과": "m7",
    "심장혈관흉부외과": "m7", "응급의학과": "m7", "결핵과": "m7",
    "방사선종양학과": "m7", "핵의학과": "m7", "병리과": "m7",
    "영상의학과": "m7", "진단검사의학과": "m7", "예방의학과": "m7",
    "직업환경의학과": "m7", "성형외과": "m7",
}
MED_ALLOWED_RAW    = {"m1", "m2", "m3", "m4", "m5", "m6"}
MED_SPECIALIZED_RAW = {"m2", "m3", "m4", "m5", "m6"}

COV_LINE_COLOR = "#5C6BC0"
MAI_LINE_COLOR = "#26A69A"

POP_MAP_MODE = "share"
MAP_HEIGHT   = 560


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
    try:    return str(int(float(str(code).strip())))
    except: return str(code).strip()

def parse_deficit_tokens(val) -> Set[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return set()
    if isinstance(val, (list, tuple, set)):
        raw = list(val)
    else:
        s = str(val).strip()
        if s in ["", "[]", "{}"]: return set()
        try:
            obj = json.loads(s)
            raw = obj if isinstance(obj, list) else [obj]
        except:
            raw = re.findall(r"[FfTt]\([sScCdDfF]\)", s)
    out = set()
    for x in raw:
        t = str(x).strip().lower()
        if   t == "f(s)": out.add("F(s)")
        elif t == "f(d)": out.add("F(d)")
        elif t == "f(o)": out.add("F(o)")
        elif t == "t(c)": out.add("T(c)")
        elif t == "t(f)": out.add("T(f)")
    return out

def parse_department_list(val) -> List[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)): return []
    if isinstance(val, list): return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if s in ["", "[]", "nan", "None"]: return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list): return [str(x).strip() for x in obj if str(x).strip()]
    except: pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list): return [str(x).strip() for x in obj if str(x).strip()]
    except: pass
    s2 = s.strip("[]")
    parts = [p.strip().strip('"').strip("'") for p in s2.split(",")]
    return [p for p in parts if p]

def normalize_facility_type_from_row(row: pd.Series) -> str:
    kind = str(row.get(FAC_KIND_COL, "")).strip(); ftype = str(row.get(FAC_TYPE_COL, "")).strip()
    kl = kind.lower(); fl = ftype.lower()
    if "공원" in kind or "park" in kl or "공원" in ftype or "park" in fl: return "park"
    if "도서관" in kind or "library" in kl or "도서관" in ftype or "library" in fl: return "library"
    if "약국" in kind or "pharmacy" in kl or "약국" in ftype or "pharmacy" in fl: return "pharmacy"
    if ("식료품" in kind or "마트" in kind or "시장" in kind or "편의점" in kind or
        "grocery" in kl or "market" in kl or "mart" in kl or "supermarket" in kl or
        "식료품" in ftype or "마트" in ftype or "시장" in ftype or "편의점" in ftype): return "grocery"
    if ("행정" in kind or "공공" in kind or "주민센터" in kind or "행정서비스" in kind or
        "행정" in ftype or "공공" in ftype or "주민센터" in ftype or "행정서비스" in ftype): return "public"
    dept_list  = parse_department_list(row.get(FAC_DEPT_COL))
    raw_groups = [MED_GROUP_MAP_RAW[d] for d in dept_list if d in MED_GROUP_MAP_RAW]
    raw_groups = [g for g in raw_groups if g in MED_ALLOWED_RAW]
    if "m1" in raw_groups: return "m1"
    if any(g in MED_SPECIALIZED_RAW for g in raw_groups): return "m2"
    if ("의료" in kind or "병원" in kind or "의원" in kind or "치과" in kind or "한의원" in kind or
        "의료" in ftype or "병원" in ftype or "의원" in ftype or "치과" in ftype or "한의원" in ftype):
        if any(x in ftype for x in ["보건소", "보건지소", "보건진료소", "보건의료원"]): return "m1"
        return "m2"
    # ── 추가 5종 시설 ──
    if any(x in kind or x in ftype for x in ["어린이집", "유치원", "nursery", "childcare", "kindergarten"]): return "nursery"
    if any(x in kind or x in ftype for x in ["초등학교", "primary school"]): return "primary"
    if any(x in kind or x in ftype for x in ["중학교", "middle school"]): return "junior"
    if any(x in kind or x in ftype for x in ["고등학교", "high school"]): return "high"
    if any(x in kind or x in ftype for x in ["노인", "경로", "senior", "elderly"]): return "elderly"
    return "exclude"


class PowerNormSafe(PowerNorm):
    pass

def compute_group_norm_from_series(values: pd.Series, gamma: float = 0.55, force_zero_min: bool = True):
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    vals = vals[vals > 0]
    vmin, vmax = (0.0, 1.0) if len(vals) == 0 else (0.0 if force_zero_min else float(vals.min()), float(vals.max()))
    if vmax <= vmin: vmax = vmin + 1e-9
    return vmin, vmax, PowerNormSafe(gamma=gamma, vmin=vmin, vmax=vmax)

def compute_group_pop_norm(values: pd.Series, share_mode: bool = True):
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    vmin, vmax = (0.0, 1.0) if len(vals) == 0 else (0.0 if share_mode else float(vals.min()), float(vals.max()))
    if vmax <= vmin: vmax = vmin + 1e-9
    norm = PowerNormSafe(gamma=0.6, vmin=vmin, vmax=vmax) if share_mode else Normalize(vmin=vmin, vmax=vmax)
    return vmin, vmax, norm

def compute_continuous_norm(values: pd.Series, gamma: float = 0.55):
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    vals = vals[vals > 0]
    vmin = 0.0; vmax = float(vals.max()) if len(vals) else 100.0
    if vmax <= vmin: vmax = 100.0
    return vmin, vmax, PowerNormSafe(gamma=gamma, vmin=vmin, vmax=vmax)

def gradient_css_from_cmap(cmap_name: str) -> str:
    cmap  = matplotlib.colormaps[cmap_name]
    stops = [f"{mcolors.to_hex(cmap(p))} {int(p*100)}%" for p in np.linspace(0, 1, 7)]
    return ", ".join(stops)

def make_pretty_ticks(vmin, vmax, n=5):
    if vmax <= vmin: return np.array([vmin, vmax + 1e-9])
    return np.linspace(vmin, vmax, n)

def choose_tick_decimals(vmin, vmax):
    span = abs(vmax - vmin)
    if span < 0.05: return 3
    if span < 0.5:  return 2
    if span < 5:    return 1
    return 0

def build_dashboard_cache(progress_cb=None) -> Dict[str, str]:
    metrics = pd.read_parquet(CLASSIFIED_PATH).copy()
    metrics[GRID_JOIN_COL] = normalize_str_series(metrics[GRID_JOIN_COL])
    metrics[SGG_CODE_COL]  = normalize_code_series(metrics[SGG_CODE_COL])
    metrics[SGG_NAME_COL]  = normalize_str_series(metrics[SGG_NAME_COL])
    metrics["pop"] = pd.to_numeric(metrics["pop"], errors="coerce").fillna(0)
    # Legacy column rename: 08_coverage → pt08_coverage (if needed)
    for old, new in zip(_LEGACY_COV + _LEGACY_MAI, COV_COLS + MAI_COLS):
        if old in metrics.columns and new not in metrics.columns:
            metrics.rename(columns={old: new}, inplace=True)
    for c in COV_COLS + MAI_COLS + COV_ALLOPEN_COLS + MAI_ALLOPEN_COLS:
        if c in metrics.columns:
            metrics[c] = pd.to_numeric(metrics[c], errors="coerce")
    for c in ["avg_coverage", "avg_mai", "cv_coverage", "cv_mai", "car_coverage", "car_mai"]:
        if c in metrics.columns:
            metrics[c] = pd.to_numeric(metrics[c], errors="coerce")

    metrics["_nat_tokens"] = metrics["nat_deficit"].apply(parse_deficit_tokens)
    metrics["_sgg_tokens"] = metrics["sgg_deficit"].apply(parse_deficit_tokens)
    for tok, key in {"F(s)": "fs", "F(d)": "fd", "F(o)": "fo", "T(c)": "tc", "T(f)": "tf"}.items():
        metrics[f"nat_has_{key}"] = metrics["_nat_tokens"].apply(lambda s, _t=tok: _t in s)
        metrics[f"sgg_has_{key}"] = metrics["_sgg_tokens"].apply(lambda s, _t=tok: _t in s)

    # ── national 기준 결핍 CSV 보강 ──────────────────────────────────────────
    # deficit_ref_sgg.csv 가 있으면 nat_has_* 컬럼을 CSV 값으로 덮어씀
    # 예상 컬럼: from_id (or grid_id), nat_has_fs, nat_has_fd, nat_has_tc, nat_has_tf
    if DEFICIT_REF_NAT_PATH.exists():
        try:
            ref = pd.read_csv(DEFICIT_REF_NAT_PATH)
            # from_id 컬럼 탐색 (다양한 이름 허용)
            id_col = next((c for c in ref.columns if c.lower() in ("from_id","grid_id","id","from_sgg_key")), None)
            if id_col:
                ref[id_col] = ref[id_col].astype(str).str.strip()
                # nat_has_* 컬럼이 있으면 덮어쓰기, bool/int 통일
                nat_cols = [c for c in ref.columns if c.startswith("nat_has_") and c in [f"nat_has_{k}" for k in DEFICIT_KEYS]]
                if nat_cols:
                    ref_sub = ref[[id_col] + nat_cols].rename(columns={id_col: GRID_JOIN_COL})
                    for c in nat_cols:
                        ref_sub[c] = ref_sub[c].astype(bool)
                    # metrics에서 기존 nat_has_* 제거 후 merge
                    drop_cols = [c for c in nat_cols if c in metrics.columns]
                    metrics = metrics.drop(columns=drop_cols)
                    metrics = metrics.merge(ref_sub, on=GRID_JOIN_COL, how="left")
                    for c in nat_cols:
                        if c in metrics.columns:
                            metrics[c] = metrics[c].fillna(False).astype(bool)
                        else:
                            metrics[c] = False
        except Exception:
            import traceback; traceback.print_exc()

    metrics["sgg_pop_total"]      = metrics.groupby(SGG_CODE_COL)["pop"].transform("sum")
    metrics["national_pop_total"] = float(metrics["pop"].sum())
    metrics["nat_pop_map"]   = np.where(metrics["national_pop_total"] > 0, metrics["pop"] / metrics["national_pop_total"] * 100.0, 0.0)
    metrics["sgg_pop_map"]   = np.where(metrics["sgg_pop_total"] > 0,      metrics["pop"] / metrics["sgg_pop_total"] * 100.0,      0.0)
    metrics["local_pop_map"] = metrics["sgg_pop_map"]

    # ── T(f) / T(c) 정확한 정의로 재계산 (token parse 덮어씀) ────────────
    # T(f): cv_coverage 또는 cv_mai가 인구 가중평균보다 큰 격자
    #   sgg 기준 → 시군구 내 인구 가중평균
    #   nat 기준 → 전국 인구 가중평균
    # T(c): cmag 또는 mmag가 인구 가중평균 이하인 격자
    #   sgg 기준 → 시군구 내 인구 가중평균
    #   nat 기준 → 전국 인구 가중평균

    def _pop_wavg_transform(col):
        w = metrics["pop"].clip(lower=0)
        def _wavg(x):
            wi = w.loc[x.index]
            valid = x.notna() & (wi > 0)
            if not valid.any(): valid = x.notna()
            if not valid.any(): return np.nan
            return float(np.average(x[valid], weights=wi[valid].clip(lower=1)))
        return metrics.groupby(SGG_CODE_COL)[col].transform(_wavg)

    def _pop_wavg_national(col):
        mask = metrics[col].notna() & (metrics["pop"] > 0)
        if not mask.any(): return np.nan
        return float(np.average(metrics.loc[mask, col], weights=metrics.loc[mask, "pop"]))

    _cv_cols = [c for c in ["cv_coverage", "cv_mai"] if c in metrics.columns]
    if _cv_cols:
        _sgg_tf = pd.Series(False, index=metrics.index)
        _nat_tf = pd.Series(False, index=metrics.index)
        # cv_coverage AND cv_mai 둘 다 인구 가중평균 초과일 때만 T(f)
        _sgg_avgs = {_col: _pop_wavg_transform(_col) for _col in _cv_cols}
        _nat_avgs = {_col: _pop_wavg_national(_col)  for _col in _cv_cols}
        _sgg_tf = pd.Series(True, index=metrics.index)
        _nat_tf = pd.Series(True, index=metrics.index)
        for _col in _cv_cols:
            _sgg_avg = _sgg_avgs[_col]
            _nat_avg = _nat_avgs[_col]
            _sgg_tf &= metrics[_col].notna() & _sgg_avg.notna() & (metrics[_col] > _sgg_avg)
            if not (isinstance(_nat_avg, float) and np.isnan(_nat_avg)):
                _nat_tf &= metrics[_col].notna() & (metrics[_col] > _nat_avg)
            else:
                _nat_tf &= False
        metrics["sgg_has_tf"] = _sgg_tf
        metrics["nat_has_tf"] = _nat_tf

    _mag_cols = [c for c in ["cmag", "mmag"] if c in metrics.columns]
    if _mag_cols:
        _sgg_tc = pd.Series(False, index=metrics.index)
        _nat_tc = pd.Series(False, index=metrics.index)
        for _col in _mag_cols:
            _sgg_avg = _pop_wavg_transform(_col)
            _nat_avg = _pop_wavg_national(_col)
            _sgg_tc |= metrics[_col].notna() & _sgg_avg.notna() & (metrics[_col] <= _sgg_avg)
            if not (isinstance(_nat_avg, float) and np.isnan(_nat_avg)):
                _nat_tc |= metrics[_col].notna() & (metrics[_col] <= _nat_avg)
        metrics["sgg_has_tc"] = _sgg_tc
        metrics["nat_has_tc"] = _nat_tc


    for key in DEFICIT_KEYS:
        metrics[f"nat_{key}_ratio"] = np.where(metrics[f"nat_has_{key}"] & (metrics["sgg_pop_total"] > 0), metrics["pop"] / metrics["sgg_pop_total"] * 100.0, 0.0)
        metrics[f"sgg_{key}_ratio"] = np.where(metrics[f"sgg_has_{key}"] & (metrics["sgg_pop_total"] > 0), metrics["pop"] / metrics["sgg_pop_total"] * 100.0, 0.0)

    # 시군구 가중 평균
    def _sgg_wmean(col):
        return (
            metrics.assign(_w=metrics[col] * metrics["pop"])
            .groupby(SGG_CODE_COL).agg(_w=("_w","sum"), pop=("pop","sum"))
            .assign(v=lambda x: np.where(x["pop"] > 0, x["_w"] / x["pop"], np.nan))["v"]
        )
    if "avg_coverage" in metrics.columns:
        metrics["sgg_avg_coverage"] = metrics[SGG_CODE_COL].map(_sgg_wmean("avg_coverage"))
    if "avg_mai" in metrics.columns:
        metrics["sgg_avg_mai"]      = metrics[SGG_CODE_COL].map(_sgg_wmean("avg_mai"))

    grid = gpd.read_file(GRID_PATH)
    if grid.crs is None: grid = grid.set_crs(epsg=4326)
    if str(grid.crs) != PLOT_CRS: grid = grid.to_crs(PLOT_CRS)
    grid[GRID_ID_COL] = normalize_str_series(grid[GRID_ID_COL])
    grid = grid[[GRID_ID_COL, "geometry"]].copy()

    gdf = grid.merge(metrics, left_on=GRID_ID_COL, right_on=GRID_JOIN_COL, how="inner")
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=PLOT_CRS)
    gdf["area_km2"]        = gdf.geometry.area / 1_000_000.0
    gdf["pop_density_km2"] = np.where(gdf["area_km2"] > 0, gdf["pop"] / gdf["area_km2"], np.nan)

    # ── spatial_all.parquet 머지 (JCL/MVG 컬럼) ──────────────────────────
    if SPATIAL_ALL_PATH.exists():
        try:
            sp = pd.read_parquet(SPATIAL_ALL_PATH)
            sp["from_id"] = sp["from_id"].astype(str).str.strip()
            # JCL/MVG 관련 컬럼만 추출
            _sp_cols = [c for c in sp.columns if any(c.startswith(p) for p in
                        ["sgg_jcl_", "nat_jcl_", "sgg_mv_geary", "nat_mv_geary",
                         "sgg_mvg_profile", "nat_mvg_profile",
                         "sgg_deficit_profile", "nat_deficit_profile",
                         "sgg_has_fo", "nat_has_fo"])]
            # has_fo 가 spatial_all에 있으면 metrics의 F(o)를 덮어쓰기
            _fo_cols = [c for c in _sp_cols if "has_fo" in c]
            _jcl_mvg_cols = [c for c in _sp_cols if c not in _fo_cols]
            # from_id + selected cols
            sp_sub = sp[["from_id"] + _sp_cols].copy()
            # 기존 gdf에서 중복 컬럼 제거 후 머지
            _drop = [c for c in _sp_cols if c in gdf.columns]
            if _drop:
                gdf = gdf.drop(columns=_drop)
            gdf = gdf.merge(sp_sub, left_on=GRID_JOIN_COL, right_on="from_id",
                            how="left", suffixes=("", "_sp"))
            # 중복 from_id_sp 제거
            if "from_id_sp" in gdf.columns:
                gdf = gdf.drop(columns=["from_id_sp"])
            # JCL _cl 컬럼: NaN → 0 (비유의)
            for c in gdf.columns:
                if c.endswith("_cl"):
                    gdf[c] = gdf[c].fillna(0).astype(int)
            # MVG sig 컬럼: NaN → "not_sig"
            for c in ["sgg_mv_geary_sig", "nat_mv_geary_sig"]:
                if c in gdf.columns:
                    gdf[c] = gdf[c].fillna("not_sig")
            # MVG profile 컬럼: NaN → ""
            for c in ["sgg_mvg_profile", "nat_mvg_profile"]:
                if c in gdf.columns:
                    gdf[c] = gdf[c].fillna("")
            if progress_cb:
                progress_cb(0, 1, f"spatial_all.parquet merged ({len(_sp_cols)} cols)")
        except Exception:
            import traceback; traceback.print_exc()

    # idx_df: sgg_avg 포함해서 저장
    centroid = gdf.geometry.centroid
    idx_cols = {
        GRID_JOIN_COL: gdf[GRID_JOIN_COL], SGG_CODE_COL: gdf[SGG_CODE_COL], SGG_NAME_COL: gdf[SGG_NAME_COL],
        "x": centroid.x, "y": centroid.y, "pop": gdf["pop"],
        "avg_coverage": gdf["avg_coverage"] if "avg_coverage" in gdf.columns else np.nan,
        "avg_mai":      gdf["avg_mai"]      if "avg_mai"      in gdf.columns else np.nan,
        "cv_coverage":  gdf["cv_coverage"]  if "cv_coverage"  in gdf.columns else np.nan,
        "cv_mai":       gdf["cv_mai"]       if "cv_mai"       in gdf.columns else np.nan,
        "car_coverage": gdf["car_coverage"] if "car_coverage" in gdf.columns else np.nan,
        "car_mai":      gdf["car_mai"]      if "car_mai"      in gdf.columns else np.nan,
        "sgg_avg_coverage": gdf["sgg_avg_coverage"] if "sgg_avg_coverage" in gdf.columns else np.nan,
        "sgg_avg_mai":      gdf["sgg_avg_mai"]      if "sgg_avg_mai"      in gdf.columns else np.nan,
    }
    idx_df = pd.DataFrame(idx_cols)
    for c in COV_COLS + MAI_COLS + COV_ALLOPEN_COLS + MAI_ALLOPEN_COLS:
        if c in gdf.columns:
            idx_df[c] = gdf[c].values
    idx_df.to_parquet(CACHE_IDX, index=False)

    # JCL/MVG 동적 컬럼 수집
    _jcl_mvg_dyn = [c for c in gdf.columns if any(c.startswith(p) for p in
                    ["sgg_jcl_", "nat_jcl_", "sgg_mv_geary", "nat_mv_geary",
                     "sgg_mvg_profile", "nat_mvg_profile",
                     "sgg_deficit_profile", "nat_deficit_profile"])]
    safe_grid_cols = [
        GRID_JOIN_COL, SGG_CODE_COL, SGG_NAME_COL, "pop",
        "nat_deficit", "sgg_deficit",
        "nat_pop_map", "sgg_pop_map", "local_pop_map",
        *[f"nat_{k}_ratio" for k in DEFICIT_KEYS],
        *[f"sgg_{k}_ratio" for k in DEFICIT_KEYS],
        *[f"nat_has_{k}" for k in DEFICIT_KEYS],
        *[f"sgg_has_{k}" for k in DEFICIT_KEYS],
        "avg_coverage", "avg_mai", "cv_coverage", "cv_mai",
        "car_coverage", "car_mai",
        "sgg_avg_coverage", "sgg_avg_mai",
        "area_km2", "pop_density_km2",
        *_jcl_mvg_dyn,
        *COV_COLS, *MAI_COLS,
        *COV_ALLOPEN_COLS, *MAI_ALLOPEN_COLS,
        "geometry",
    ]
    safe_grid_cols = [c for c in safe_grid_cols if c in gdf.columns]
    gdf_safe   = gdf[safe_grid_cols].copy()
    gdf_simple = gdf_safe.copy()
    gdf_simple["geometry"] = gdf_simple.geometry.simplify(25, preserve_topology=True)
    gdf_safe.to_parquet(CACHE_GRID,        index=False)
    gdf_simple.to_parquet(CACHE_GRID_SIMPLE, index=False)

    agg_rows = []
    for code, gg in gdf_safe.groupby(SGG_CODE_COL, dropna=True):
        row = {SGG_CODE_COL: code, SGG_NAME_COL: gg[SGG_NAME_COL].iloc[0],
               "sgg_pop_total": float(gg["pop"].sum()),
               "nat_pop_map": float(gg["nat_pop_map"].sum()),
               "sgg_pop_map": float(gg["sgg_pop_map"].sum())}
        for k in DEFICIT_KEYS:
            row[f"nat_{k}_ratio"] = float(gg[f"nat_{k}_ratio"].sum())
            row[f"sgg_{k}_ratio"] = float(gg[f"sgg_{k}_ratio"].sum())
        agg_rows.append(row)
    sgg_attr = pd.DataFrame(agg_rows)
    sgg_poly = gdf_safe[[SGG_CODE_COL, "geometry"]].dissolve(by=SGG_CODE_COL).reset_index()
    sgg_poly = sgg_poly.merge(sgg_attr, on=SGG_CODE_COL, how="left")
    sgg_poly = gpd.GeoDataFrame(sgg_poly, geometry="geometry", crs=PLOT_CRS)
    sgg_poly.to_parquet(CACHE_SGG, index=False)

    # ── 인접 시군구 계산 (National 모드용) ──────────────────────────────
    try:
        _sgg_buf = sgg_poly.copy()
        _sgg_buf["geometry"] = _sgg_buf.geometry.buffer(150)  # 150m buffer
        _neighbors = {}
        for _code in _sgg_buf[SGG_CODE_COL].unique():
            _geom = _sgg_buf[_sgg_buf[SGG_CODE_COL] == _code].geometry.iloc[0]
            _touching = _sgg_buf[
                _sgg_buf.geometry.intersects(_geom) & (_sgg_buf[SGG_CODE_COL] != _code)
            ]
            _neighbors[normalize_sgg_code(_code)] = [
                normalize_sgg_code(c) for c in _touching[SGG_CODE_COL].tolist()
            ]
        with open(CACHE_NEIGHBORS, "w") as _nf:
            json.dump(_neighbors, _nf)
    except Exception:
        import traceback; traceback.print_exc()

    station = gpd.read_file(STATION_PATH)
    subway  = gpd.read_file(SUBWAY_PATH)
    fac     = gpd.read_parquet(FAC_PATH)
    for lyr in [station, subway, fac]:
        if lyr.crs is None: lyr.set_crs(epsg=4326, inplace=True)
    if str(station.crs) != PLOT_CRS: station = station.to_crs(PLOT_CRS)
    if str(subway.crs)  != PLOT_CRS: subway  = subway.to_crs(PLOT_CRS)
    if str(fac.crs)     != PLOT_CRS: fac     = fac.to_crs(PLOT_CRS)
    fac["fac_type_norm"] = fac.apply(normalize_facility_type_from_row, axis=1)
    fac = fac[fac["fac_type_norm"].isin(FACILITY_ORDER)].copy()
    station.to_parquet(CACHE_STATION, index=False)
    subway.to_parquet(CACHE_SUBWAY,  index=False)
    fac.to_parquet(CACHE_FAC,        index=False)

    # ── 추가 지표 컬럼: deficit 동적 재분류용 ──
    _EXTRA_DIAG = [
        "car_cov_cv", "car_mai_cv",           # F(o) 진단
        "cv_coverage_allopen", "cv_mai_allopen",  # T(f) 진단
        "avg_coverage_allopen", "avg_mai_allopen", # allopen avg
        "cmag_allopen", "mmag_allopen",           # T(c) 진단
        "cmag", "mmag",                           # 복합 modal gap
    ]
    _CAR_SLOT_COLS = [f"car_cov_{s}" for s in TIME_SLOTS] + [f"car_mai_{s}" for s in TIME_SLOTS]
    ts_keep = [GRID_JOIN_COL, SGG_CODE_COL, SGG_NAME_COL,
               "avg_coverage", "avg_mai", "cv_coverage", "cv_mai", "car_coverage", "car_mai",
               "sgg_avg_coverage", "sgg_avg_mai",
               *COV_COLS, *MAI_COLS, *COV_ALLOPEN_COLS, *MAI_ALLOPEN_COLS,
               *_EXTRA_DIAG, *_CAR_SLOT_COLS]
    ts_keep = [c for c in ts_keep if c in metrics.columns]
    metrics[ts_keep].to_parquet(CACHE_TS, index=False)

    # ── OD 기반 시설 접근 가능 여부 (시간대별 Coverage + MAI) ─────────────────
    if OD_PATH.exists():
        try:
            if progress_cb: progress_cb(0, 1, "Computing facility accessibility from OD file...")
            od = pd.read_parquet(OD_PATH)
            od["from_id"] = od["from_id"].astype(str).str.strip()
            od["to_id"]   = od["to_id"].astype(str).str.strip()

            PT_SLOT_COL = {s: f"pt{s}" for s in TIME_SLOTS}
            PT_SLOT_COL = {s: c for s, c in PT_SLOT_COL.items() if c in od.columns}
            FAC_COLS    = [c for c in ALL_FAC_COLS if c in od.columns]

            if not FAC_COLS or not PT_SLOT_COL:
                raise ValueError(f"Required columns missing. PT: {list(PT_SLOT_COL)}, fac: {FAC_COLS}")

            # ── float32로 통일 (메모리 절약) ──────────────────────
            for c in PT_SLOT_COL.values():
                od[c] = od[c].astype(np.float32)
            for c in FAC_COLS:
                od[c] = od[c].astype(np.int8)

            # ── Coverage: groupby + transform → from_id별 슬롯×시설 집계 ──
            # 각 (from_id, slot, fac) 조합에 대해 threshold 내에 시설 존재 여부
            cov_result_cols = {}
            for slot, pt_col in PT_SLOT_COL.items():
                pt_vals = od[pt_col].values
                for fc in FAC_COLS:
                    thresh    = FAC_COV_THRESH.get(fc, 15)
                    col_name  = f"cov_{slot}_{fc}"
                    # 해당 행이 조건 만족 여부 (0/1)
                    od[col_name] = ((pt_vals <= thresh) & (od[fc].values > 0)).astype(np.int8)
                    # from_id 그룹 내 any → max()
                    cov_result_cols[col_name] = "max"

            # ── 슬롯별 최솟값 PT 계산 ──────────────────────────
            pt_cols_list = list(PT_SLOT_COL.values())
            od["_min_pt"] = od[pt_cols_list].min(axis=1).astype(np.float32)

            # ── MAI: 15분 내 to_id 중 최다 시설 (벡터화) ──────
            # 기존: _min_pt 기반 (전체 슬롯 중 최소 PT)
            within_mask = od["_min_pt"] <= MAI_THRESH
            od_w = od[within_mask].copy()

            if not od_w.empty:
                od_w["_n_fac"] = od_w[FAC_COLS].gt(0).sum(axis=1).astype(np.int16)
                max_fac = od_w.groupby("from_id")["_n_fac"].transform("max")
                best_mask = od_w["_n_fac"] == max_fac
                od_best   = od_w[best_mask].copy()
                min_pt_in_best = od_best.groupby("from_id")["_min_pt"].transform("min")
                od_best = od_best[od_best["_min_pt"] == min_pt_in_best]
                od_best = od_best.groupby("from_id", as_index=False).first()
                od_best = od_best.set_index("from_id")

                tie_counts = od_w[best_mask].groupby("from_id")["_min_pt"].count()
                od_best["mai_is_tie"] = (tie_counts > 1).astype(np.int8)
                od_best["mai_best_to_id"] = od_best["to_id"]
                mai_fac_cols = {f"mai_{fc}": od_best[fc].gt(0).astype(np.int8) for fc in FAC_COLS}
                for col, series in mai_fac_cols.items():
                    od_best[col] = series
                mai_df = od_best[["mai_is_tie", "mai_best_to_id"] + [f"mai_{fc}" for fc in FAC_COLS]].reset_index()
            else:
                mai_df = pd.DataFrame(columns=["from_id", "mai_is_tie", "mai_best_to_id"] + [f"mai_{fc}" for fc in FAC_COLS])

            # ── Per-slot MAI: 슬롯별 best destination ──────────
            mai_slot_dfs = []
            for slot, pt_col in PT_SLOT_COL.items():
                slot_within = od[od[pt_col] <= MAI_THRESH].copy()
                if slot_within.empty:
                    continue
                slot_within["_n_fac_s"] = slot_within[FAC_COLS].gt(0).sum(axis=1).astype(np.int16)
                _max = slot_within.groupby("from_id")["_n_fac_s"].transform("max")
                _best = slot_within[slot_within["_n_fac_s"] == _max].copy()
                _min_pt = _best.groupby("from_id")[pt_col].transform("min")
                _best = _best[_best[pt_col] == _min_pt]
                _best = _best.groupby("from_id", as_index=False).first()
                _best = _best.set_index("from_id")
                for fc in FAC_COLS:
                    _best[f"mai_{slot}_{fc}"] = (_best[fc].values > 0).astype(np.int8)
                mai_slot_cols = [f"mai_{slot}_{fc}" for fc in FAC_COLS]
                mai_slot_dfs.append(_best[mai_slot_cols].reset_index())

            if mai_slot_dfs:
                mai_slot_merged = mai_slot_dfs[0]
                for extra in mai_slot_dfs[1:]:
                    mai_slot_merged = mai_slot_merged.merge(extra, on="from_id", how="outer")
                mai_df = mai_df.merge(mai_slot_merged, on="from_id", how="left")
                # NaN → 0
                for c in mai_slot_merged.columns:
                    if c != "from_id":
                        mai_df[c] = mai_df[c].fillna(0).astype(np.int8)

            # ── Coverage 집계: from_id 그룹별 max ──────────────
            cov_cols_list = list(cov_result_cols.keys())
            cov_df = od.groupby("from_id", as_index=False)[cov_cols_list].max()

            # ── 병합 ────────────────────────────────────────────
            result = cov_df.merge(mai_df, on="from_id", how="left")
            # MAI 없는 from_id → 0 채우기
            for fc in FAC_COLS:
                result[f"mai_{fc}"] = result[f"mai_{fc}"].fillna(0).astype(np.int8)
            result["mai_is_tie"]    = result["mai_is_tie"].fillna(0).astype(np.int8)

            result.rename(columns={"from_id": GRID_JOIN_COL}, inplace=True)
            result.to_parquet(CACHE_FACILITY_ACCESS, index=False)
            if progress_cb: progress_cb(1, 1, f"OD facility cache built ({len(result):,} grids).")
        except Exception:
            import traceback; traceback.print_exc()

    # ── 정적 GeoJSON 캐시 ──────────────────────────────────
    CACHE_GEOJSON_DIR.mkdir(parents=True, exist_ok=True)

    def _write_json_safe(path: Path, data: dict) -> None:
        import tempfile, os, time
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            for attempt in range(5):
                try:
                    if path.exists(): path.unlink()
                    os.rename(tmp_path, str(path)); break
                except PermissionError:
                    time.sleep(0.3)
        except:
            try: os.unlink(tmp_path)
            except: pass
            raise

    def _gdf_to_geojson_dict(gdf: gpd.GeoDataFrame) -> dict:
        """iterrows 없이 벡터화 변환. bool/numpy 타입 자동 직렬화."""
        prop_cols = [c for c in gdf.columns if c != "geometry"]
        # numpy 타입 → Python 기본 타입 일괄 변환
        rec = gdf[prop_cols].copy()
        for c in prop_cols:
            if rec[c].dtype.kind in ("b",):          rec[c] = rec[c].astype(bool)
            elif rec[c].dtype.kind in ("i", "u"):    rec[c] = rec[c].astype(int)
            elif rec[c].dtype.kind in ("f",):        rec[c] = rec[c].astype(float)
        records = rec.where(rec.notna(), other=None).to_dict(orient="records")
        geoms   = gdf.geometry.values
        feats   = [
            {"type": "Feature", "geometry": g.__geo_interface__, "properties": p}
            for g, p in zip(geoms, records)
            if g is not None and not g.is_empty
        ]
        return {"type": "FeatureCollection", "features": feats}

    _write_json_safe(CACHE_STATION_JSON, _gdf_to_geojson_dict(station.to_crs(WEB_CRS)[["geometry"]]))
    _write_json_safe(CACHE_SUBWAY_JSON,  _gdf_to_geojson_dict(subway.to_crs(WEB_CRS)[["geometry"]]))
    fac_w = fac.to_crs(WEB_CRS)
    for ftype in FACILITY_ORDER:
        sub = fac_w[fac_w["fac_type_norm"] == ftype]
        if not sub.empty:
            _write_json_safe(Path(CACHE_FAC_JSON_TPL.format(ftype=ftype)),
                             _gdf_to_geojson_dict(sub[["geometry", "fac_type_norm"]]))

    # ── 시군구별 그리드 GeoJSON ──────────────────────────
    # JCL _cl 컬럼 + MVG sig/profile 컬럼 동적 수집
    _jcl_cl_cols = [c for c in gdf_safe.columns if c.endswith("_cl")]
    _mvg_cols    = [c for c in gdf_safe.columns if any(c.startswith(p) for p in
                    ["sgg_mv_geary_sig", "nat_mv_geary_sig",
                     "sgg_mvg_profile", "nat_mvg_profile"])]
    metric_cols = [
        *[f"sgg_{k}_ratio" for k in DEFICIT_KEYS],
        *[f"nat_{k}_ratio" for k in DEFICIT_KEYS],
        "sgg_pop_map","nat_pop_map","local_pop_map",
        "avg_coverage","avg_mai",
        *[f"nat_has_{k}" for k in DEFICIT_KEYS],
        *[f"sgg_has_{k}" for k in DEFICIT_KEYS],
        *_jcl_cl_cols,
        *_mvg_cols,
        GRID_JOIN_COL, SGG_NAME_COL, SGG_CODE_COL,
    ]
    metric_cols = [c for c in metric_cols if c in gdf_safe.columns]
    gdf_web = gdf_simple.to_crs(WEB_CRS).copy()
    gdf_web[SGG_CODE_COL] = gdf_web[SGG_CODE_COL].apply(normalize_sgg_code)
    sgg_groups = list(gdf_web.groupby(SGG_CODE_COL))
    n_tiles = len(sgg_groups)
    for i, (sgg_code, grp) in enumerate(sgg_groups):
        key  = normalize_sgg_code(sgg_code)
        keep = [c for c in metric_cols if c in grp.columns] + ["geometry"]
        _write_json_safe(CACHE_GEOJSON_DIR / f"grid_{key}.json", _gdf_to_geojson_dict(grp[keep].copy()))
        if progress_cb:
            progress_cb(i + 1, n_tiles, f"GeoJSON tile: {key} ({i+1}/{n_tiles})")

    # ── 셀 상세 데이터 캐시 (sgg_avg 포함) ─────────────
    if INTRACITY_PATH.exists():
        try:
            ic = pd.read_parquet(INTRACITY_PATH)
            ic[GRID_JOIN_COL] = ic[GRID_JOIN_COL].astype(str).str.strip()
            keep_c = [GRID_JOIN_COL, SGG_CODE_COL, "pop", "car_coverage", "car_mai",
                      "avg_coverage", "avg_mai", "cv_coverage", "cv_mai",
                      "car_cov_cv", "car_mai_cv",
                      "cv_coverage_allopen", "cv_mai_allopen",
                      "avg_coverage_allopen", "avg_mai_allopen",
                      "cmag_allopen", "mmag_allopen", "cmag", "mmag",
                      *[f"car_cov_{s}" for s in TIME_SLOTS],
                      *[f"car_mai_{s}" for s in TIME_SLOTS],
                      *COV_COLS, *MAI_COLS, *COV_ALLOPEN_COLS, *MAI_ALLOPEN_COLS]
            ic_save = ic[[c for c in keep_c if c in ic.columns]].copy()
            # sgg_avg_coverage / sgg_avg_mai 를 metrics에서 merge
            sgg_ref = metrics[[GRID_JOIN_COL, "sgg_avg_coverage", "sgg_avg_mai"]].copy() if "sgg_avg_coverage" in metrics.columns else None
            if sgg_ref is not None:
                ic_save = ic_save.merge(sgg_ref, on=GRID_JOIN_COL, how="left")
            ic_save.to_parquet(CACHE_CELL_DATA, index=False)
        except Exception:
            pass

    return {"ok": "1"}


@st.cache_data(show_spinner=False)
def load_cached_data():
    grid        = gpd.read_parquet(CACHE_GRID)
    grid_simple = gpd.read_parquet(CACHE_GRID_SIMPLE)
    sgg         = gpd.read_parquet(CACHE_SGG)
    station     = gpd.read_parquet(CACHE_STATION)
    subway      = gpd.read_parquet(CACHE_SUBWAY)
    fac         = gpd.read_parquet(CACHE_FAC)
    ts          = pd.read_parquet(CACHE_TS)
    idx         = pd.read_parquet(CACHE_IDX)
    return grid, grid_simple, sgg, station, subway, fac, ts, idx

@st.cache_data(show_spinner=False)
def load_cell_detail_data() -> pd.DataFrame:
    src = CACHE_CELL_DATA if CACHE_CELL_DATA.exists() else (CACHE_IDX if CACHE_IDX.exists() else None)
    if src is None: return pd.DataFrame()
    df = pd.read_parquet(src)
    df[GRID_JOIN_COL] = df[GRID_JOIN_COL].astype(str).str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_facility_access_data() -> pd.DataFrame:
    if not CACHE_FACILITY_ACCESS.exists(): return pd.DataFrame()
    df = pd.read_parquet(CACHE_FACILITY_ACCESS)
    return df


@st.cache_data(show_spinner=False, max_entries=256)
def get_cell_data_json(sgg_code: str, _cell_df, _group_gdf) -> str:
    """시군구별 cell data JSON — sgg_code 기준 캐시"""
    import json as _json
    ncode = normalize_sgg_code(sgg_code)

    # 1) SGG_CODE_COL로 필터
    if SGG_CODE_COL in _cell_df.columns:
        sub = _cell_df[_cell_df[SGG_CODE_COL].apply(normalize_sgg_code) == ncode].copy()
    else:
        # fallback: group_gdf GRID_JOIN_COL 기준
        valid_ids = set(_group_gdf[GRID_JOIN_COL].astype(str).str.strip())
        sub = _cell_df[_cell_df[GRID_JOIN_COL].astype(str).str.strip().isin(valid_ids)].copy()

    if sub.empty: return "{}"

    # 2) deficit 플래그 merge from group_gdf
    def_cols = [c for c in _group_gdf.columns
                if c.startswith("sgg_has_") or c.startswith("nat_has_")]
    if def_cols and GRID_JOIN_COL in _group_gdf.columns:
        def_ref = _group_gdf[[GRID_JOIN_COL] + def_cols].copy()
        def_ref[GRID_JOIN_COL] = def_ref[GRID_JOIN_COL].astype(str).str.strip()
        sub[GRID_JOIN_COL] = sub[GRID_JOIN_COL].astype(str).str.strip()
        sub = sub.merge(def_ref, on=GRID_JOIN_COL, how="left")

    # 3) sgg 평균
    sgg_cov = sgg_mai = None
    for df_ in [_group_gdf, sub]:
        if sgg_cov is None and "sgg_avg_coverage" in df_.columns:
            v = df_["sgg_avg_coverage"].dropna()
            if len(v): sgg_cov = float(v.iloc[0])
        if sgg_mai is None and "sgg_avg_mai" in df_.columns:
            v = df_["sgg_avg_mai"].dropna()
            if len(v): sgg_mai = float(v.iloc[0])
    if sgg_cov is None and {"avg_coverage","pop"} <= set(sub.columns):
        ps = pd.to_numeric(sub["pop"], errors="coerce").fillna(0)
        cs = pd.to_numeric(sub["avg_coverage"], errors="coerce")
        t  = ps.sum()
        if t > 0: sgg_cov = float((cs * ps).sum() / t)
    if sgg_mai is None and {"avg_mai","pop"} <= set(sub.columns):
        ps = pd.to_numeric(sub["pop"], errors="coerce").fillna(0)
        ms = pd.to_numeric(sub["avg_mai"], errors="coerce")
        t  = ps.sum()
        if t > 0: sgg_mai = float((ms * ps).sum() / t)

    # 4) 원하는 컬럼만 추출
    _extra_diag = ["car_cov_cv", "car_mai_cv",
                  "cv_coverage_allopen", "cv_mai_allopen",
                  "avg_coverage_allopen", "avg_mai_allopen",
                  "cmag_allopen", "mmag_allopen", "cmag", "mmag"]
    _car_slot = [f"car_cov_{s}" for s in ['06','08','10','12','14','16','18','20','22','24']] + \
                [f"car_mai_{s}" for s in ['06','08','10','12','14','16','18','20','22','24']]
    wanted = [GRID_JOIN_COL, "pop", "avg_coverage", "avg_mai",
              "cv_coverage", "cv_mai", "car_coverage", "car_mai"] + COV_COLS + MAI_COLS + COV_ALLOPEN_COLS + MAI_ALLOPEN_COLS + _extra_diag + _car_slot + def_cols
    sub = sub[[c for c in dict.fromkeys(wanted) if c in sub.columns]].copy()
    for c in sub.columns:
        if c == GRID_JOIN_COL: continue
        if sub[c].dtype.kind in ("f","i","u","b"):
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan)
    sub[GRID_JOIN_COL] = sub[GRID_JOIN_COL].astype(str).str.strip()

    records = sub.where(sub.notna(), other=None).to_dict(orient="records")
    out = {r[GRID_JOIN_COL]: {k: v for k, v in r.items() if k != GRID_JOIN_COL}
           for r in records}
    # __sgg_avg_* → 모든 셀에 municipality avg 주입 (JS에서 d.sgg_avg_coverage로 접근)
    # + deficit 참조 기준값 (인구 가중평균): car_coverage, car_mai, car_cov_cv, car_mai_cv,
    #   cv_coverage_allopen, cv_mai_allopen, cmag_allopen, mmag_allopen
    _ref_cols = ["car_coverage", "car_mai", "car_cov_cv", "car_mai_cv",
                 "cv_coverage_allopen", "cv_mai_allopen",
                 "cmag_allopen", "mmag_allopen"]
    sgg_refs = {}
    if {"pop"} <= set(sub.columns):
        ps = pd.to_numeric(sub["pop"], errors="coerce").fillna(0)
        t = ps.sum()
        for rc in _ref_cols:
            if rc in sub.columns:
                vs = pd.to_numeric(sub[rc], errors="coerce")
                if t > 0:
                    sgg_refs[f"_ref_{rc}"] = float((vs * ps).sum() / t)
    if sgg_cov is not None or sgg_mai is not None or sgg_refs:
        for cell_dict in out.values():
            if isinstance(cell_dict, dict):
                if sgg_cov is not None: cell_dict["sgg_avg_coverage"] = sgg_cov
                if sgg_mai is not None: cell_dict["sgg_avg_mai"] = sgg_mai
                for k, v in sgg_refs.items():
                    cell_dict[k] = v
    return _json.dumps(out, default=lambda x: None if (isinstance(x, float) and (x != x or abs(x) == float("inf"))) else x)


# =========================================================
# 지도 HTML 빌드
# =========================================================
def _read_json_safe(path) -> str:
    p = Path(path)
    if not p.exists(): return "null"
    with open(p, "r", encoding="utf-8") as fh: return fh.read()


def _cmap_to_js_stops(cmap_name: str, n: int = 24) -> list:
    """matplotlib colormap → hex 정지점 리스트 (JS interpolation용)."""
    cmap = matplotlib.colormaps[cmap_name]
    return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]



def _hex_colors_for(base_feats, vcol, norm_obj, cmap_obj, fallback_cols=None):
    """vcol 없으면 fallback_cols 순서로 시도. coverage/mai는 0도 색상 부여."""
    def _get_val(props, col):
        v = props.get(col)
        return float(v) if v is not None else None
    cols_to_try = [vcol] + (fallback_cols or [])
    raw = []
    for f in base_feats:
        p = f["properties"]
        val = None
        for c in cols_to_try:
            v = _get_val(p, c)
            if v is not None:
                val = v; break
        raw.append(val if val is not None else np.nan)
    raw = np.array(raw, dtype=np.float32)
    # has_data: 실제 값이 있는 셀 (nan 아님) — 0도 색상 부여
    has_data = ~np.isnan(raw)
    raw = np.nan_to_num(raw, nan=0.0)
    out = [None] * len(base_feats)
    if has_data.any():
        rgba = cmap_obj(norm_obj(raw[has_data]))
        for j, hi in enumerate(np.where(has_data)[0]):
            out[hi] = mcolors.to_hex(rgba[j])
    return out


def _make_colorbar_html(cmap_name, vmin, vmax):
    ticks  = make_pretty_ticks(vmin, vmax, n=5)
    dec    = choose_tick_decimals(vmin, vmax)
    grad   = gradient_css_from_cmap(cmap_name)
    labels = "".join(
        f"<div style='text-align:center;font-size:10px;color:#666;'>{t:.{dec}f}%</div>"
        for t in ticks)
    return (
        f"<div style='padding:4px 8px 6px;background:#fff;border-top:1px solid #eee;'>"
        f"<div style='height:7px;border-radius:2px;margin-bottom:2px;"
        f"background:linear-gradient(to right,{grad});'></div>"
        f"<div style='display:grid;grid-template-columns:repeat(5,1fr);'>{labels}</div>"
        f"</div>")


@st.cache_data(show_spinner=False, max_entries=128)
def build_multi_map_html(
    sgg_code: str,
    height_px: int,
    deficit_colors_json: str = "{}",
    _cache_ver: str = "",
) -> str:
    """Iframe HTML. sgg_code만으로 캐시. Layer show/hide는 JS+localStorage로."""
    sgg_key = normalize_sgg_code(sgg_code)
    metric_keys = ALL_MAP_KEYS  # 4 base + 5 JCL = 9
    n           = len(metric_keys)

    subway_js  = _read_json_safe(CACHE_SUBWAY_JSON)
    station_js = _read_json_safe(CACHE_STATION_JSON)

    fac_parts = []
    for ftype in FACILITY_ORDER:
        fj = _read_json_safe(Path(CACHE_FAC_JSON_TPL.format(ftype=ftype)))
        if fj != "null":
            fac_parts.append(
                '{"id":' + json.dumps(ftype) +
                ',"label":' + json.dumps(FACILITY_LABELS_EN.get(ftype, ftype)) +
                ',"c":' + json.dumps(FACILITY_COLORS.get(ftype, "#999")) +
                ',"d":' + fj + '}')
    fac_js = "[" + ",".join(fac_parts) + "]"

    sgg_key_norm = normalize_sgg_code(sgg_code)
    grid_path = CACHE_GEOJSON_DIR / f"grid_{sgg_key_norm}.json"
    base_gj   = None
    if grid_path.exists():
        with open(grid_path, "r", encoding="utf-8") as fh:
            base_gj = json.load(fh)
    else:
        # 파일 없으면 전체 파일 목록에서 가장 유사한 key 탐색
        import glob as _glob
        existing = {p.stem.replace("grid_",""): p
                    for p in CACHE_GEOJSON_DIR.glob("grid_*.json")}
        # 시도 1: 변형된 형태
        for _alt in [
            str(int(float(sgg_key_norm))) if sgg_key_norm.replace(".","",1).isdigit() else None,
            sgg_key_norm.zfill(5),
            sgg_key_norm.lstrip("0") or sgg_key_norm,
        ]:
            if _alt and _alt in existing:
                with open(existing[_alt], "r", encoding="utf-8") as fh:
                    base_gj = json.load(fh)
                break
        # 시도 2: 앞 5자리 prefix 매칭
        if base_gj is None:
            prefix = sgg_key_norm[:5]
            for k, p in existing.items():
                if k.startswith(prefix) or k.zfill(5).startswith(prefix.zfill(5)):
                    with open(p, "r", encoding="utf-8") as fh:
                        base_gj = json.load(fh)
                    break
    base_feats = base_gj.get("features", []) if base_gj else []

    # ── 인접 시군구 그리드 로드 (National 모드: 1-hop, 최대 15k) ──
    for feat in base_feats:
        feat["properties"]["_sel"] = 1
    _MAX_NB = 15000
    if CACHE_NEIGHBORS.exists():
        try:
            with open(CACHE_NEIGHBORS, "r") as _nf:
                _all_nbrs = json.load(_nf)
            _nbr = _all_nbrs.get(sgg_key, []) or _all_nbrs.get(sgg_key.zfill(5), [])
            _added = 0
            for _nc in _nbr:
                if _added >= _MAX_NB: break
                _np = CACHE_GEOJSON_DIR / f"grid_{_nc}.json"
                if not _np.exists():
                    _np = CACHE_GEOJSON_DIR / f"grid_{_nc.zfill(5)}.json"
                if not _np.exists(): continue
                with open(_np, "r", encoding="utf-8") as _nfh:
                    _ngj = json.load(_nfh)
                for _nf2 in _ngj.get("features", []):
                    _nf2["properties"]["_sel"] = 0
                    base_feats.append(_nf2)
                    _added += 1
        except Exception:
            pass

    # ── 모든 metric × sgg/nat 색상+colorbar 미리 계산 ────────────────────────
    colors_by_mk = {}
    cbars_by_mk  = {}

    for mk in metric_keys:
        cmap_name = CMAPS.get(mk, "Blues")
        cmap_obj  = matplotlib.colormaps[cmap_name]
        colors_by_mk[mk] = {}
        cbars_by_mk[mk]  = {}

        if mk == "mvg":
            # ── MVG: 프로파일별 카테고리 색상 ──────────────────────
            for basis in ("sgg", "nat"):
                hex_out = []
                _profiles_seen = set()
                for f in base_feats:
                    p = f["properties"]
                    sig = p.get(f"{basis}_mv_geary_sig", "not_sig")
                    if not sig or sig == "not_sig":
                        hex_out.append(None)
                    elif sig == "heterogeneous":
                        hex_out.append(MVG_HETERO_COLOR)
                        _profiles_seen.add("Heterogeneous")
                    else:
                        prof = p.get(f"{basis}_mvg_profile", "")
                        if prof == "None" or prof == "":
                            hex_out.append("#FFFFFF")
                            _profiles_seen.add("None")
                        else:
                            c = MVG_PROFILE_COLORS.get(prof, "#9E9E9E")
                            hex_out.append(c)
                            _profiles_seen.add(prof)
                colors_by_mk[mk][basis] = hex_out
                # categorical legend
                _leg_items = []
                if "None" in _profiles_seen:
                    _leg_items.append(
                        f"<span style='display:inline-flex;align-items:center;gap:3px;margin:1px 3px;'>"
                        f"<span style='width:9px;height:9px;border-radius:2px;background:#fff;border:1.5px solid #bbb;flex-shrink:0;'></span>"
                        f"<span style='font-size:8px;color:#555;'>None</span></span>")
                for pname, pcol in MVG_PROFILE_COLORS.items():
                    if pname != "None" and pname in _profiles_seen:
                        _leg_items.append(
                            f"<span style='display:inline-flex;align-items:center;gap:3px;margin:1px 3px;'>"
                            f"<span style='width:9px;height:9px;border-radius:2px;background:{pcol};flex-shrink:0;'></span>"
                            f"<span style='font-size:8px;color:#555;'>{pname}</span></span>")
                if "Heterogeneous" in _profiles_seen:
                    _leg_items.append(
                        f"<span style='display:inline-flex;align-items:center;gap:3px;margin:1px 3px;'>"
                        f"<span style='width:9px;height:9px;border-radius:2px;background:{MVG_HETERO_COLOR};flex-shrink:0;'></span>"
                        f"<span style='font-size:8px;color:#555;'>Heterogeneous</span></span>")
                cbars_by_mk[mk][basis] = (
                    f"<div style='padding:4px 6px 4px;background:#fff;border-top:1px solid #eee;"
                    f"display:flex;flex-wrap:wrap;gap:0;'>"
                    + "".join(_leg_items) + "</div>")
            continue

        # ── 기존 continuous metrics (coverage/mai/pop 등) ──────────
        for basis in ("sgg", "nat"):
            vcol = get_value_col(mk, basis)
            if mk in ("coverage", "mai"):
                _raw_v = [f["properties"].get(vcol) for f in base_feats]
                vals_s = pd.Series(
                    [float(v) if v is not None else np.nan for v in _raw_v],
                    dtype=np.float32)
                _vals_nz = vals_s.dropna()
                if _vals_nz.empty: _vals_nz = pd.Series([0.0], dtype=np.float32)
                _, vmax, norm_obj = compute_continuous_norm(_vals_nz, gamma=0.6)
                vmin = 0.0
            elif mk == "pop":
                _pop_candidates = [vcol, "nat_pop_map", "local_pop_map", "pop"]
                vcol = next((c for c in _pop_candidates
                             if any(f["properties"].get(c) for f in base_feats[:5])), vcol)
                vals_s = pd.Series(
                    [float(f["properties"].get(vcol) or 0) for f in base_feats],
                    dtype=np.float32)
                vmin, vmax, norm_obj = compute_group_pop_norm(vals_s, share_mode=True)
            else:
                all_vals = []
                for k2 in DEFICIT_KEYS:
                    vc2 = f"{basis}_{k2}_ratio"
                    all_vals.extend(
                        [float(f["properties"].get(vc2) or 0) for f in base_feats])
                vmin, vmax, norm_obj = compute_group_norm_from_series(
                    pd.Series(all_vals, dtype=np.float32), gamma=0.55, force_zero_min=True)
            _fb_cols = (["nat_pop_map","local_pop_map","pop"] if mk=="pop" else
                        ["avg_coverage"] if mk=="coverage" else
                        ["avg_mai"] if mk=="mai" else [])
            colors_by_mk[mk][basis] = _hex_colors_for(base_feats, vcol, norm_obj, cmap_obj, fallback_cols=_fb_cols)
            cbars_by_mk[mk][basis]  = _make_colorbar_html(cmap_name, vmin, vmax)

    # ── GeoJSON: per-map gridData (기존 방식 복원) ──────────────────────────
    for fi, feat in enumerate(base_feats):
        feat["properties"]["_idx"] = fi

    grid_js_list = []
    if not base_feats:
        grid_js_list = ["null"] * n
    else:
        for mk in metric_keys:
            sc_ = colors_by_mk[mk]["sgg"]
            nc_ = colors_by_mk[mk]["nat"]
            new_feats = []
            for fi, feat in enumerate(base_feats):
                p = dict(feat["properties"])
                p["_fs"] = sc_[fi]
                p["_fn"] = nc_[fi]
                new_feats.append({"type":"Feature",
                                  "geometry":feat["geometry"],
                                  "properties":p})
            grid_js_list.append(
                json.dumps({"type":"FeatureCollection","features":new_feats}))

    cbars_js_arr = '[' + ','.join(
        json.dumps({"sgg": cbars_by_mk[mk]["sgg"], "nat": cbars_by_mk[mk]["nat"]})
        for mk in metric_keys) + ']'

    sgg_col = SGG_NAME_COL
    gj_col  = GRID_JOIN_COL

    cols_css = "display:block;" if n == 1 else "display:grid;grid-template-columns:1fr 1fr;gap:4px;"
    map_divs = ""
    for i, mk in enumerate(metric_keys):
        map_divs += (
            f'<div class="map-wrap" id="mwrap{i}">'
            f'<div class="map-title">{BASE_MAP_LABELS.get(mk, LAYER_KEY_TO_LABEL.get(mk, mk))}</div>'
            f'<div id="map{i}" class="map-box"></div>'
            f'<div id="cb{i}" class="colorbar-wrap">{cbars_by_mk[mk]["sgg"]}</div>'
            f'</div>')

    deficit_colors = json.loads(deficit_colors_json)

    # ── JS: 지도 초기화 ──────────────────────────────────────────────────────
    maps_init = ""
    for mi, mk in enumerate(metric_keys):
        sc_js  = json.dumps(sgg_col)
        jc_js  = json.dumps(gj_col)
        maps_init += (
            f"try{{\n"
            f"var map{mi}=L.map('map{mi}',{{"
            f"center:[36.5,127.9],zoom:10,"
            f"zoomControl:true,preferCanvas:true,renderer:L.canvas({{tolerance:3}})}});\n"
            f"L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',"
            f"{{attribution:'&copy; OpenStreetMap &copy; CARTO',subdomains:'abcd',maxZoom:19}}).addTo(map{mi});\n"
            f"(function(map,gd,sc,jc){{"
            f"if(!gd)return;"
            f"var glyr=L.geoJSON(gd,{{style:function(f){{"
            f"  var p=f.properties;"
            f"  var basis=window.deficitBasis||'sgg';"
            f"  var fc=basis==='nat'?p._fn:p._fs;"
            f"  if({json.dumps(mk)}==='coverage'&&p._dyn_cov!==undefined)fc=p._dyn_cov;"
            f"  else if({json.dumps(mk)}==='mai'&&p._dyn_mai!==undefined)fc=p._dyn_mai;"
            f"  var isSgg=basis!=='nat';"
            f"  if(isSgg&&p._sel===0)return{{fillOpacity:0,weight:0,opacity:0}};"
            f"  if(!fc)return{{fillOpacity:0,weight:0.5,color:'#ccc',opacity:0.25}};"
            f"  return{{fillColor:fc,fillOpacity:0.80,weight:0.8,color:'#888',opacity:0.5}};}},\n"
            f"onEachFeature:function(f,layer){{"
            + (  # MVG: dynamic tooltip with profile
                f"layer.bindTooltip(function(){{"
                f"var p=layer.feature.properties,t='';"
                f"if(p[{sc_js}])t+='<b>'+p[{sc_js}]+'</b><br/>';"
                f"var _b=window.deficitBasis||'sgg';"
                f"var _sig=p[_b+'_mv_geary_sig'];"
                f"if(_sig&&_sig!=='not_sig'){{"
                f"  var _prof=p[_b+'_mvg_profile']||'';"
                f"  if(_sig==='homogeneous'&&_prof)t+='<span style=\"font-size:11px;font-weight:700;color:#1a1a1a;\">'+_prof+'</span>';"
                f"  else if(_sig==='heterogeneous')t+='<span style=\"font-size:10px;color:#888;\">Heterogeneous</span>';"
                f"}}"
                f"return t||null;}},{{sticky:true,opacity:.95}});"
                if mk == "mvg" else
                f"var p=f.properties,t='';"
                f"if(p[{sc_js}])t+='<b>'+p[{sc_js}]+'</b><br/>';"
                f"if(t)layer.bindTooltip(t,{{sticky:true,opacity:.95}});"
            ) + f"}},\n"
            f"renderer:L.canvas({{tolerance:2}})}}).addTo(map);\n"
            f"gridLayers.push(glyr);\n"
            f"var feats=gd.features||[];"
            f"map.on('click',function(e){{"
            f"var lat=e.latlng.lat,lng=e.latlng.lng,found=null;"
            f"for(var i=0;i<feats.length;i++){{"
            f"var f=feats[i];if(!f.geometry)continue;"
            f"var bb=getBBox(f.geometry);"
            f"if(lat<bb[1]||lat>bb[3]||lng<bb[0]||lng>bb[2])continue;"
            f"if(pointInPolygon(lat,lng,f.geometry)){{found=f;break;}}"
            f"}}"
            f"if(!found)return;"
            f"var fid=found.properties[jc];if(!fid)return;"
            f"if(window._hlGeoJSON){{allMaps.forEach(function(m){{try{{m.removeLayer(window._hlGeoJSON);}}catch(e){{}}}});window._hlGeoJSON=null;}}"
            f"window._hlGeoJSON=L.geoJSON(found,{{style:function(){{"
            f"return{{fill:false,weight:3.5,color:'#1565C0',opacity:1}};}},"
            f"renderer:L.svg()}}).addTo(map);"
            f"showCellPanel(String(fid));"
            f"}});"
            f"}})(map{mi},gridData[{mi}],{sc_js},{jc_js});\n"
            f"if(subwayData)L.geoJSON(subwayData,{{style:function(){{return{{color:'#424242',weight:2,opacity:0.6,fill:false}}}},renderer:L.canvas()}}).addTo(map{mi});\n"
            f"if(stationData)L.geoJSON(stationData,{{pointToLayer:function(f,ll){{"
            f"return L.circleMarker(ll,{{radius:4.5,color:'#333',weight:1.5,fillColor:'#fff',fillOpacity:1,opacity:1}});}},renderer:L.canvas()}}).addTo(map{mi});\n"
            f"facData.forEach(function(x){{if(!x.d)return;"
            f"if(!facLayers[x.id])facLayers[x.id]={{maps:[],color:x.c,label:x.label}};"
            f"var lyr=L.geoJSON(x.d,{{pointToLayer:function(f,ll){{"
            f"return L.circleMarker(ll,{{radius:4,color:'#fff',weight:1,fillColor:x.c,fillOpacity:.88,opacity:.88}});}},renderer:L.canvas()}});"
            f"facLayers[x.id].maps.push({{map:map{mi},layer:lyr}});"
            f"if(facVisible.indexOf(x.id)>=0)lyr.addTo(map{mi});}});\n"
            f"Object.keys(deficitInfo).forEach(function(dk){{"
            f"  if(!deficitLayers[dk])deficitLayers[dk]={{maps:[],color:deficitInfo[dk].c,label:deficitInfo[dk].label}};"
            f"  (function(dkLocal,dcolLocal){{"
            f"    function makeStyleFns(basis){{"
            f"      var col=basis+'_has_'+dkLocal;"
            f"      var isNat=basis==='nat';"
            f"      var wFn=function(f){{var p=f.properties;"
            f"        if(!isNat&&p._sel===0)return{{fill:false,weight:0,opacity:0}};"
            f"        var _dd=window._defFlags,_ff=_dd&&_dd[p.from_id];var isD=_ff?!!_ff[dkLocal]:(p[col]===true||p[col]==='true'||p[col]===1);"
            f"        if(!isD)return{{fill:false,weight:0,opacity:0}};"
            f"        return{{fill:false,weight:5,color:'#ffffff',opacity:1}};}};"
            f"      var cFn=function(f){{var p=f.properties;"
            f"        if(!isNat&&p._sel===0)return{{fill:false,weight:0,opacity:0}};"
            f"        var _dd=window._defFlags,_ff=_dd&&_dd[p.from_id];var isD=_ff?!!_ff[dkLocal]:(p[col]===true||p[col]==='true'||p[col]===1);"
            f"        if(!isD)return{{fill:false,weight:0,opacity:0}};"
            f"        return{{fill:false,weight:2.5,color:dcolLocal,opacity:1}};}};"
            f"      return{{white:wFn,color:cFn}};"
            f"    }}"
            f"    var fns=makeStyleFns(window.deficitBasis||'sgg');"
            f"    var dlyrW=L.geoJSON(gridData[{mi}],{{style:fns.white,renderer:L.canvas({{tolerance:0}})}});"
            f"    var dlyr=L.geoJSON(gridData[{mi}],{{style:fns.color,renderer:L.canvas({{tolerance:0}})}});"
            f"    deficitLayers[dkLocal].maps.push({{map:map{mi},layer:dlyr,whiteLayer:dlyrW,makeStyleFns:makeStyleFns}});"
            f"    if(deficitVisible.indexOf(dkLocal)>=0){{dlyrW.addTo(map{mi});dlyr.addTo(map{mi});}}"
            f"  }})(dk,deficitInfo[dk].c);"
            f"}});\n"
        )
        maps_init += f"allMaps.push(map{mi});\n}}catch(_me){{}}\n"

    sync_js = (
        "allMaps.forEach(function(src){"
        "allMaps.forEach(function(dst){"
        "if(src===dst)return;"
        "src.on('move',function(){dst.setView(src.getCenter(),src.getZoom(),{animate:false});});"
        "});});"
    )

    fac_toggle_js = (
        "var ctrl=document.getElementById('fc');\n"
        "(function(){"
        "  var btn=document.createElement('button');"
        "  var _fvs=localStorage.getItem('facVisible');"
        "  var _allOff=_fvs&&JSON.parse(_fvs).length===0;"
        "  btn.textContent=_allOff?'All On':'All Off';"
        "  btn.style.cssText='font-size:9px;padding:1px 7px;margin-bottom:4px;cursor:pointer;"
        "border:1px solid #ccc;border-radius:3px;background:#f5f5f5;color:#555;display:block;width:100%;';"
        "  try{if(_fvs){var _fva=JSON.parse(_fvs);"
        "    if(_fva.length===0)btn.textContent='All On';"
        "  }}catch(e){}"
        "  btn.addEventListener('click',function(){"
        "    var cbs=ctrl.querySelectorAll('input[type=checkbox]');"
        "    var allOff=btn.textContent==='All Off';"
        "    if(allOff){"
        "      cbs.forEach(function(cb){if(cb.checked){cb.checked=false;cb.dispatchEvent(new Event('change'));}});"
        "      try{localStorage.setItem('facVisible',JSON.stringify([]));}catch(e){}"
        "      btn.textContent='All On';"
        "    }else{"
        "      var allIds=Object.keys(facLayers);"
        "      cbs.forEach(function(cb){if(!cb.checked){cb.checked=true;cb.dispatchEvent(new Event('change'));}});"
        "      try{localStorage.setItem('facVisible',JSON.stringify(allIds));}catch(e){}"
        "      btn.textContent='All Off';"
        "    }"
        "  });"
        "  ctrl.appendChild(btn);"
        "})();\n"
        "Object.keys(facLayers).forEach(function(id){"
        "  var info=facLayers[id];"
        "  var lbl=document.createElement('label');"
        "  var cb=document.createElement('input');cb.type='checkbox';cb.setAttribute('data-fid',id);"
        "  cb.checked=(facVisible.indexOf(id)>=0);"
        "  cb.addEventListener('change',function(){"
        "    var on=this.checked;"
        "    info.maps.forEach(function(x){if(on)x.layer.addTo(x.map);else x.map.removeLayer(x.layer);});"
        "    var vis=Object.keys(facLayers).filter(function(k){"
        "      var c=ctrl.querySelector('input[data-fid=\"'+k+'\"]');return c&&c.checked;});"
        "    try{localStorage.setItem('facVisible',JSON.stringify(vis));}catch(e){}"
        "  });"
        "  var dot=document.createElement('span');dot.className='dot';dot.style.background=info.color;"
        "  lbl.appendChild(cb);lbl.appendChild(dot);"
        "  lbl.appendChild(document.createTextNode(info.label));"
        "  ctrl.appendChild(lbl);"
        "});"
    )

    rows    = 1 if n <= 2 else (2 if n <= 4 else 5)
    panel_h = (height_px + 26 + 42) * rows

    panel_css = (
        ".outer-wrap{display:flex;flex-direction:row;gap:0;width:100%;min-height:0;}"
        ".maps-section{flex:1 1 0;min-width:0;display:flex;flex-direction:column;}"
        ".cell-panel{"
        "display:none;flex:0 0 280px;width:280px;min-width:280px;"
        "background:#fefefe;border-left:1px solid #e8e8e8;"
        "padding:12px 14px 10px 12px;font-family:'Inter',system-ui,sans-serif;"
        "overflow-y:auto;max-height:" + str(panel_h) + "px;"
        "}"
        ".cell-panel.visible{"
        "display:block;"
        "}"
        ".cp-sec{font-size:7.5px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;"
        "margin-top:10px;margin-bottom:3px;}"
        ".cp-metrics{display:grid;grid-template-columns:1fr 1fr;gap:4px 8px;margin-bottom:2px;}"
        ".cp-m{background:#f9f9f9;border:1px solid #eee;border-radius:4px;padding:5px 7px;}"
        ".cp-m-label{font-size:7.5px;color:#aaa;margin-bottom:1px;font-weight:500;}"
        ".cp-m-val{font-size:14px;font-weight:700;color:#1a1a1a;line-height:1.2;}"
        ".cp-ref{display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px 6px;}"
        ".cp-r-label{font-size:7.5px;color:#bbb;font-weight:500;}"
        ".cp-r-val{font-size:11px;font-weight:600;color:#555;}"
        ".cp-close{float:right;cursor:pointer;font-size:14px;color:#ccc;margin-left:5px;padding:2px;}"
        ".cp-close:hover{color:#555;}"
        ".cp-id{font-size:11px;font-weight:700;color:#1a1a1a;border-bottom:2px solid #3F51B5;padding-bottom:3px;}"
        ".cp-id-sub{font-size:7px;color:#bbb;letter-spacing:.6px;margin-bottom:4px;text-transform:uppercase;}"
        ".cp-sgg-ref{display:grid;grid-template-columns:1fr 1fr;gap:4px 8px;}"
        ".cp-sgg-m{background:#f0f4ff;border:1px solid #dde5f5;border-radius:4px;padding:4px 7px;}"
        ".cp-sgg-label{font-size:7.5px;color:#8a9ac0;margin-bottom:1px;font-weight:500;}"
        ".cp-sgg-val{font-size:12px;font-weight:700;color:#3F51B5;}"
        ".cp-inacc-tags{display:flex;flex-wrap:wrap;gap:3px;margin-top:2px;}"
        ".cp-deficit-tags{display:flex;flex-wrap:wrap;gap:4px;margin-top:2px;}"
        ".cp-inacc-tag{font-size:8.5px;padding:2px 5px;border-radius:10px;"
        "background:#FFF3E0;color:#E65100;border:1px solid #FFB74D;font-weight:600;}"
        ".cp-inacc-ok{font-size:8.5px;color:#43A047;font-weight:600;}"
        "#cp-chart{display:block;width:100%!important;height:145px!important;}"
        ".fac-ctrl{position:absolute;top:6px;right:6px;z-index:2000;"
        "background:rgba(255,255,255,.97);border:1px solid #ddd;border-radius:6px;"
        "padding:6px 10px;font-size:11px;line-height:1.9;"
        "box-shadow:0 2px 8px rgba(0,0,0,.10);min-width:145px;max-height:calc(100% - 20px);overflow-y:auto;"
        "user-select:none;touch-action:none;}"
        ".fac-hdr{font-weight:700;font-size:9.5px;color:#666;margin-bottom:2px;letter-spacing:.5px;text-transform:uppercase;cursor:grab;}"
        ".fac-ctrl label{display:flex;align-items:center;gap:5px;cursor:pointer;white-space:nowrap;color:#444;}"
        ".fac-ctrl input{cursor:pointer;margin:0;width:12px;height:12px;}"
        ".dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;border:1px solid rgba(0,0,0,.12);}"
        ".map-wrap{position:relative;}"
        ".map-title{font-size:11px;font-weight:600;color:#555;padding:3px 8px;background:#fafafa;"
        "letter-spacing:.3px;border-bottom:1px solid #eee;}"
        ".colorbar-wrap{background:#fff;border-top:none;}"
        ".map-box{width:100%;height:" + str(height_px) + "px;}"
        ".maps-grid{" + cols_css + "width:100%;position:relative;}"
        ".layer-bar{display:flex;align-items:center;gap:6px;padding:4px 10px;flex-wrap:wrap;"
        "background:#fff;border-bottom:1px solid #e8e8e8;min-height:30px;flex-shrink:0;}"
        ".layer-bar .lbar-sep{width:1px;height:16px;background:#ddd;margin:0 2px;flex-shrink:0;}"
        "#dlbl-fs input{accent-color:#E53935;}"
        "#dlbl-fd input{accent-color:#F4A100;}"
        "#dlbl-fo input{accent-color:#FF7043;}"
        "#dlbl-tc input{accent-color:#7B1FA2;}"
        "#dlbl-tf input{accent-color:#2E7D32;}"
        ".layer-bar label{display:flex;align-items:center;gap:3px;font-size:10.5px;"
        "color:#444;cursor:pointer;user-select:none;white-space:nowrap;}"
        ".layer-bar input{accent-color:#555;cursor:pointer;}"
        ".layer-bar .lbar-title{font-size:8px;font-weight:700;letter-spacing:1px;"
        "text-transform:uppercase;color:#aaa;margin-right:2px;}"
        ".leaflet-tooltip{font-size:11px;background:rgba(255,255,255,.97);"
        "border:1px solid #ddd;padding:4px 8px;box-shadow:0 2px 6px rgba(0,0,0,.10);color:#333;}"
        ".cluster-info-box{display:none;position:absolute;top:28px;right:0;width:320px;background:#fff;"
        "border:1px solid #ddd;border-radius:6px;padding:10px 12px;font-size:11px;line-height:1.5;"
        "color:#555;box-shadow:0 3px 12px rgba(0,0,0,0.12);z-index:999;}"
        ".cluster-info-box.visible{display:block;}"
    )

    chart_cdn = '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>'

    panel_js = (
        "var _cpChart=null;\n"
        "var TIME_SLOTS=['06:00','08:00','10:00','12:00','14:00','16:00','18:00','20:00','22:00','24:00'];\n"
        "var SLOT_KEYS=['06','08','10','12','14','16','18','20','22','24'];\n"
        "var COV_COLS=" + json.dumps(COV_COLS) + ";\n"
        "var MAI_COLS=" + json.dumps(MAI_COLS) + ";\n"
        "var COV_ALLOPEN_COLS=" + json.dumps(COV_ALLOPEN_COLS) + ";\n"
        "var MAI_ALLOPEN_COLS=" + json.dumps(MAI_ALLOPEN_COLS) + ";\n"
        "var FAC_LABEL_MAP=" + json.dumps(OD_FACILITY_LABELS) + ";\n"
        "var FAC_OD_COLS="   + json.dumps(OD_FACILITY_COLS)   + ";\n"
        "var FAC_COV_THRESH=" + json.dumps(FAC_COV_THRESH)    + ";\n"
        "var MAI_THRESH=" + str(MAI_THRESH) + ";\n"
        "var DISPLAY_FAC_COLS=" + json.dumps(DISPLAY_FAC_COLS) + ";\n"
        "var DISPLAY_FAC_LABELS=" + json.dumps(DISPLAY_FAC_LABELS) + ";\n"
        "var SPECIALIST_COLS=" + json.dumps(SPECIALIST_COLS) + ";\n"
        "var SPECIALIST_DETAIL_LABELS=" + json.dumps(SPECIALIST_DETAIL_LABELS) + ";\n"
        "var SPECIALIST_LABEL='" + SPECIALIST_LABEL + "';\n"
        "var DEFICIT_LABEL_MAP={fs:'F(s)',fd:'F(d)',fo:'F(o)',tc:'T(c)',tf:'T(f)'};\n"
        "var DEFICIT_COLOR_MAP=" + json.dumps(DEFICIT_COLORS) + ";\n"
        "var JCL_COLORS=" + json.dumps(JCL_COLORS) + ";\n"
        "var MVG_PROFILE_COLORS=" + json.dumps(MVG_PROFILE_COLORS) + ";\n"
        "var MVG_HETERO_COLOR='" + MVG_HETERO_COLOR + "';\n"
        "var MVG_NOTSIG_COLOR='" + MVG_NOTSIG_COLOR + "';\n"
        "var DEFICIT_KEYS=" + json.dumps(DEFICIT_KEYS) + ";\n"
        # DEFICIT_KEYS, JCL_COLORS, MVG_* are already defined in global script block
        # Only keep panel-specific constants here
        # DEFICIT_BASIS는 localStorage에서 동적으로 읽음 (basis 전환 시 뷰포트 유지)
        "function getDeficitBasis(){return window.deficitBasis||'sgg';}\n"
        # 선택된 시설만 포함하는 DISPLAY_FAC_COLS 반환
        "function getSelectedDisplayFacCols(){\n"
        "  var sel=window.selectedFacs||" + json.dumps(FAC_DEFAULT_SEL) + ";\n"
        "  return DISPLAY_FAC_COLS.filter(function(dc){return sel.indexOf(dc)>=0;});\n"
        "}\n"
        "function fv(v,suf,dec){"
        "  if(v===null||v===undefined)return 'N/A';"
        "  var f=parseFloat(v);if(isNaN(f))return 'N/A';"
        "  return f.toFixed(dec!==undefined?dec:2)+(suf!==undefined?suf:'%');}\n"
        # ── 시설 선택에 따른 동적 재계산 함수 ──
        "function _dynMetrics(d,acc,sel){\n"
        "  if(!sel||!sel.length||!acc)return null;\n"
        "  var odMap={};\n"
        "  FAC_SELECTOR_DEFS.forEach(function(def){odMap[def.id]=def.fac_cols;});\n"
        "  var nSel=sel.length;\n"
        "  var hSM=SLOT_KEYS.some(function(s){return FAC_OD_COLS.some(function(fc){return acc['mai_'+s+'_'+fc]!==undefined;});});\n"
        "  var sCA={},sMA={};\n"
        "  SLOT_KEYS.forEach(function(s){\n"
        "    var cA=0,mA=0;\n"
        "    sel.forEach(function(dc){\n"
        "      var cols=odMap[dc]||[dc];\n"
        "      if(cols.some(function(fc){return acc['cov_'+s+'_'+fc]===1;}))cA++;\n"
        "      if(hSM){if(cols.some(function(fc){return acc['mai_'+s+'_'+fc]===1;}))mA++;}\n"
        "      else{if(cols.some(function(fc){return acc['mai_'+fc]===1;}))mA++;}\n"
        "    });\n"
        "    sCA[s]=(cA/nSel)*100;sMA[s]=(mA/nSel)*100;\n"
        "  });\n"
        "  var sCO={},sMO={};\n"
        "  SLOT_KEYS.forEach(function(s,i){\n"
        "    var pC=d[COV_ALLOPEN_COLS[i]],pM=d[MAI_ALLOPEN_COLS[i]];\n"
        "    var oC=d[COV_COLS[i]],oM=d[MAI_COLS[i]];\n"
        "    var rC=(pC>0&&oC!=null)?oC/pC:1,rM=(pM>0&&oM!=null)?oM/pM:1;\n"
        "    if(rC>1)rC=1;if(rM>1)rM=1;\n"
        "    sCO[s]=sCA[s]*rC;sMO[s]=sMA[s]*rM;\n"
        "  });\n"
        "  function _st(o){\n"
        "    var v=SLOT_KEYS.map(function(s){return o[s]||0;}),n=v.length,sm=0;\n"
        "    v.forEach(function(x){sm+=x;});var a=sm/n,ss=0;\n"
        "    v.forEach(function(x){ss+=(x-a)*(x-a);});\n"
        "    return{avg:a,cv:a>0?Math.sqrt(ss/n)/a:0};\n"
        "  }\n"
        "  var aoC=_st(sCA),aoM=_st(sMA),ohC=_st(sCO),ohM=_st(sMO);\n"
        "  return{slotCovAO:sCA,slotMaiAO:sMA,slotCovOH:sCO,slotMaiOH:sMO,\n"
        "    avgCov:aoC.avg,cvCov:aoC.cv,avgMai:aoM.avg,cvMai:aoM.cv,\n"
        "    ohAvgCov:ohC.avg,ohCvCov:ohC.cv,ohAvgMai:ohM.avg,ohCvMai:ohM.cv};\n"
        "}\n"
        # ── sgg 평균: per-display-type pre-aggregate (O(cells) 1회 → O(types) per selection) ──
        "function _buildSggTypeAvg(){\n"
        "  if(window._sggTypeAvg)return;\n"
        "  var cd=window.cellData||{},fa=window.facAccessData||{};\n"
        "  var odMap={};FAC_SELECTOR_DEFS.forEach(function(d){odMap[d.id]=d.fac_cols;});\n"
        "  var nS=SLOT_KEYS.length,tPop=0;\n"
        "  var dcIds=FAC_SELECTOR_DEFS.map(function(d){return d.id;});\n"
        "  var covSum={},maiSum={};dcIds.forEach(function(dc){covSum[dc]=0;maiSum[dc]=0;});\n"
        "  Object.keys(cd).forEach(function(fid){\n"
        "    var dd=cd[fid],aa=fa[fid],pop=parseFloat(dd&&dd.pop)||0;\n"
        "    if(pop<=0||!aa)return;\n"
        "    tPop+=pop;\n"
        "    dcIds.forEach(function(dc){\n"
        "      var cols=odMap[dc]||[dc],cS=0,mS=0;\n"
        "      SLOT_KEYS.forEach(function(s){\n"
        "        if(cols.some(function(fc){return aa['cov_'+s+'_'+fc]===1;}))cS++;\n"
        "        if(cols.some(function(fc){return aa['mai_'+s+'_'+fc]===1||aa['mai_'+fc]===1;}))mS++;\n"
        "      });\n"
        "      covSum[dc]+=pop*(cS/nS*100);\n"
        "      maiSum[dc]+=pop*(mS/nS*100);\n"
        "    });\n"
        "  });\n"
        "  var r={};dcIds.forEach(function(dc){\n"
        "    r[dc]={cov:tPop>0?covSum[dc]/tPop:null,mai:tPop>0?maiSum[dc]/tPop:null};\n"
        "  });\n"
        "  r._tPop=tPop;\n"
        "  window._sggTypeAvg=r;\n"
        "}\n"
        "function _dynSggAvg(sel){\n"
        "  _buildSggTypeAvg();\n"
        "  var ta=window._sggTypeAvg;if(!ta||!sel||!sel.length)return{cov:null,mai:null};\n"
        "  var cS=0,mS=0,n=sel.length;\n"
        "  sel.forEach(function(dc){var t=ta[dc];if(t){cS+=t.cov||0;mS+=t.mai||0;}});\n"
        "  return{cov:cS/n,mai:mS/n};\n"
        "}\n"
        # ── deficit flags 동적 재계산 (classify_one 10-rule 완전 구현) ──
        "function _dynDeficit(d,acc,sel,basis){\n"
        "  if(!d)return {};\n"
        "  var dyn=acc&&sel&&sel.length?_dynMetrics(d,acc,sel):null;\n"
        # pre-computed car metrics (per-type 없으므로 고정)
        "  var carCov=d.car_coverage,carMai=d.car_mai;\n"
        "  var carCovCv=d.car_cov_cv,carMaiCv=d.car_mai_cv;\n"
        # dynamic allopen metrics
        "  var cvCovAo=dyn?dyn.cvCov:d.cv_coverage_allopen;\n"
        "  var cvMaiAo=dyn?dyn.cvMai:d.cv_mai_allopen;\n"
        "  var avgCovAo=dyn?dyn.avgCov:d.avg_coverage_allopen;\n"
        "  var avgMaiAo=dyn?dyn.avgMai:d.avg_mai_allopen;\n"
        # cmag_allopen = avg_coverage(allopen) - car_coverage
        "  var cmagAo=(avgCovAo!=null&&carCov!=null)?avgCovAo-carCov:d.cmag_allopen;\n"
        "  var mmagAo=(avgMaiAo!=null&&carMai!=null)?avgMaiAo-carMai:d.mmag_allopen;\n"
        # SGG reference thresholds (from cellData _ref_* fields)
        "  var ref={\n"
        "    car_coverage:d._ref_car_coverage,car_mai:d._ref_car_mai,\n"
        "    car_cov_cv:d._ref_car_cov_cv,car_mai_cv:d._ref_car_mai_cv,\n"
        "    cv_cov_ao:d._ref_cv_coverage_allopen,cv_mai_ao:d._ref_cv_mai_allopen,\n"
        "    cmag_ao:d._ref_cmag_allopen,mmag_ao:d._ref_mmag_allopen\n"
        "  };\n"
        "  function ok(a,b){return a!=null&&!isNaN(a)&&b!=null&&!isNaN(b);}\n"
        "  if(ref.cmag_ao==null&&ref.mmag_ao==null&&ref.cv_cov_ao==null)return null;\n"
        "  var def={fs:false,fd:false,fo:false,tc:false,tf:false};\n"
        # Rule 1: T(f) — cv_coverage_allopen & cv_mai_allopen both > ref
        "  if(ok(cvCovAo,ref.cv_cov_ao)&&ok(cvMaiAo,ref.cv_mai_ao)&&cvCovAo>ref.cv_cov_ao&&cvMaiAo>ref.cv_mai_ao)def.tf=true;\n"
        # Rule 2,3: T(c) — cmag_allopen < ref OR mmag_allopen < ref
        "  if(ok(cmagAo,ref.cmag_ao)&&cmagAo<ref.cmag_ao)def.tc=true;\n"
        "  if(ok(mmagAo,ref.mmag_ao)&&mmagAo<ref.mmag_ao)def.tc=true;\n"
        # Rule 4: null cmag or cv_cov → F(s),T(c),T(f)
        "  if(cmagAo==null||isNaN(cmagAo)||cvCovAo==null||isNaN(cvCovAo)){def.fs=true;def.tc=true;def.tf=true;}\n"
        # Rule 5: null mmag or cv_mai → F(d),T(c),T(f)
        "  if(mmagAo==null||isNaN(mmagAo)||cvMaiAo==null||isNaN(cvMaiAo)){def.fd=true;def.tc=true;def.tf=true;}\n"
        # Rule 6: cmag > ref AND car_cov < ref → T(c)
        "  if(ok(cmagAo,ref.cmag_ao)&&ok(carCov,ref.car_coverage)&&cmagAo>ref.cmag_ao&&carCov<ref.car_coverage)def.tc=true;\n"
        # Rule 7: cmag < ref AND car_cov < ref → F(s),T(c)
        "  if(ok(cmagAo,ref.cmag_ao)&&ok(carCov,ref.car_coverage)&&cmagAo<ref.cmag_ao&&carCov<ref.car_coverage){def.fs=true;def.tc=true;}\n"
        # Rule 8: mmag > ref AND car_mai < ref → F(d)
        "  if(ok(mmagAo,ref.mmag_ao)&&ok(carMai,ref.car_mai)&&mmagAo>ref.mmag_ao&&carMai<ref.car_mai)def.fd=true;\n"
        # Rule 9: mmag < ref AND car_mai < ref → F(d),T(c)
        "  if(ok(mmagAo,ref.mmag_ao)&&ok(carMai,ref.car_mai)&&mmagAo<ref.mmag_ao&&carMai<ref.car_mai){def.fd=true;def.tc=true;}\n"
        # Rule 10: F(o) — non-F(s) AND car_cov_cv > ref AND car_mai_cv > ref
        "  if(!def.fs&&ok(carCovCv,ref.car_cov_cv)&&ok(carMaiCv,ref.car_mai_cv)&&carCovCv>ref.car_cov_cv&&carMaiCv>ref.car_mai_cv)def.fo=true;\n"
        "  return def;\n"
        "}\n"
        "function showCellPanel(fid){\n"
        "  var d=(window.cellData||{})[fid];\n"
        "  var panel=document.getElementById('cell-panel');\n"
        "  document.getElementById('cp-fid').textContent=fid;\n"
        "  if(!d){panel.classList.add('visible');return;}\n"
        "  window._currentCellFid=fid;\n"
        "  var acc=(window.facAccessData||{})[fid];\n"
        "  var selDcs=getSelectedDisplayFacCols();\n"
        "  var dyn=_dynMetrics(d,acc,selDcs);\n"
        # PT metrics — dynamic if available, else fallback to pre-computed
        "  if(dyn){\n"
        "    document.getElementById('cp-avg-cov').textContent=fv(dyn.ohAvgCov);\n"
        "    document.getElementById('cp-avg-mai').textContent=fv(dyn.ohAvgMai);\n"
        "    document.getElementById('cp-cv-cov').textContent=fv(dyn.ohCvCov,'');\n"
        "    document.getElementById('cp-cv-mai').textContent=fv(dyn.ohCvMai,'');\n"
        "  } else {\n"
        "    document.getElementById('cp-avg-cov').textContent=fv(d.avg_coverage);\n"
        "    document.getElementById('cp-avg-mai').textContent=fv(d.avg_mai);\n"
        "    document.getElementById('cp-cv-cov').textContent=fv(d.cv_coverage,'');\n"
        "    document.getElementById('cp-cv-mai').textContent=fv(d.cv_mai,'');\n"
        "  }\n"
        # SGG avg — pre-aggregated (O(types) per call)
        "  var sggDyn=_dynSggAvg(selDcs);\n"
        "  document.getElementById('cp-sgg-cov').textContent=fv(sggDyn.cov!=null?sggDyn.cov:d.sgg_avg_coverage);\n"
        "  document.getElementById('cp-sgg-mai').textContent=fv(sggDyn.mai!=null?sggDyn.mai:d.sgg_avg_mai);\n"
        # Deficit types — dynamic
        "  var defEl=document.getElementById('cp-deficit-tags');\n"
        "  var _basis=getDeficitBasis();\n"
        "  var dynDef=_dynDeficit(d,acc,selDcs,_basis);\n"
        "  var defTypes=[];\n"
        "  DEFICIT_KEYS.forEach(function(dk){\n"
        "    if(dynDef[dk])defTypes.push(dk);\n"
        "  });\n"
        "  if(defTypes.length>0){\n"
        "    defEl.innerHTML=defTypes.map(function(dk){\n"
        "      var c=DEFICIT_COLOR_MAP[dk]||'#999';\n"
        "      return '<span style=\"display:inline-flex;align-items:center;gap:3px;font-size:9px;font-weight:700;padding:2px 7px;border-radius:10px;border:1.5px solid '+c+';color:'+c+';background:'+c+'18;\">'+DEFICIT_LABEL_MAP[dk]+'</span>';\n"
        "    }).join('');\n"
        "  } else {\n"
        "    defEl.innerHTML='<span style=\"font-size:9px;color:#aaa;\">None</span>';\n"
        "  }\n"
        # Reference
        "  document.getElementById('cp-pop').textContent=fv(d.pop,'',0);\n"
        "  document.getElementById('cp-car-cov').textContent=fv(d.car_coverage);\n"
        "  document.getElementById('cp-car-mai').textContent=fv(d.car_mai);\n"
        # Inaccessible facilities (static panel) — m2~m6 → "Specialist care" 묶음 표시
        "  var tagsEl=document.getElementById('cp-inacc-tags');\n"
        "  function buildInaccHTML(getAccFn){\n"
        "    var selDcs=getSelectedDisplayFacCols();\n"
        "    var rawAcc={};\n"
        "    FAC_OD_COLS.forEach(function(fc){rawAcc[fc]=getAccFn(fc);});\n"
        "    var specSelected=selDcs.indexOf('specialist')>=0;\n"
        "    var specOk=specSelected&&SPECIALIST_COLS.some(function(fc){return rawAcc[fc]===true;});\n"
        "    var det=specSelected?SPECIALIST_COLS.filter(function(fc){return !rawAcc[fc];}):[];\n"
        "    var inaccDcs=selDcs.filter(function(dc){return dc!=='specialist'&&!rawAcc[dc];});\n"
        "    var showSpec=specSelected&&!(specOk&&det.length===0);\n"
        "    var hasAnyInacc=inaccDcs.length>0||(!specOk&&specSelected);\n"
        "    if(!hasAnyInacc&&!showSpec)return '<span class=\"cp-inacc-ok\">✓ All accessible</span>';\n"
        "    var html='';\n"
        "    html+=inaccDcs.map(function(dc){\n"
        "      return '<span class=\"cp-inacc-tag\">'+(DISPLAY_FAC_LABELS[dc]||dc)+'</span>';\n"
        "    }).join('');\n"
        # specialist tag: accessible→초록, inaccessible→기존 태그색
        "    if(showSpec){\n"
        "      var tid='sp-detail-'+Math.random().toString(36).slice(2,7);\n"
        "      var dh=det.map(function(fc){\n"
        "        return '<span style=\"font-size:8px;margin-right:3px;\">· '+SPECIALIST_DETAIL_LABELS[fc]+'</span>';\n"
        "      }).join('');\n"
        "      if(specOk){\n"
        # accessible이지만 세부 inaccessible 있음 → 초록 태그 + ℹ
        "        html+='<span class=\"cp-inacc-tag\" style=\"background:#E8F5E9;color:#2E7D32;border-color:#A5D6A7;cursor:default;\">'\n"
        "          +'✓ '+SPECIALIST_LABEL\n"
        "          +' <span onclick=\"(function(){var e=document.getElementById(\\''+tid+'\\');e.style.display=e.style.display===\\'none\\'?\\'block\\':\\'none\\';})()\" '\n"
        "          +'style=\"cursor:pointer;font-size:8px;color:#2E7D32;border:1px solid #A5D6A7;border-radius:3px;padding:0 3px;margin-left:2px;\" title=\"Show inaccessible sub-types\">ℹ</span>'\n"
        "          +'</span>'\n"
        "          +(det.length>0\n"
        "            ?'<div id=\"'+tid+'\" style=\"display:none;width:100%;margin-top:2px;background:#E8F5E9;border-radius:4px;padding:3px 6px;font-size:8px;color:#2E7D32;\">Inaccessible: '+dh+'</div>'\n"
        "            :'');\n"
        "      } else {\n"
        # inaccessible → 주황 태그 + ℹ
        "        html+='<span class=\"cp-inacc-tag\" style=\"cursor:default;\">'\n"
        "          +SPECIALIST_LABEL\n"
        "          +' <span onclick=\"(function(){var e=document.getElementById(\\''+tid+'\\');e.style.display=e.style.display===\\'none\\'?\\'block\\':\\'none\\';})()\" '\n"
        "          +'style=\"cursor:pointer;font-size:8px;color:#8E24AA;border:1px solid #CE93D8;border-radius:3px;padding:0 3px;margin-left:2px;\" title=\"Show inaccessible sub-types\">ℹ</span>'\n"
        "          +'</span>'\n"
        "          +(det.length>0\n"
        "            ?'<div id=\"'+tid+'\" style=\"display:none;width:100%;margin-top:2px;background:#F3E5F5;border-radius:4px;padding:3px 6px;font-size:8px;color:#6A1B9A;\">Inaccessible: '+dh+'</div>'\n"
        "            :'');\n"
        "      }\n"
        "    }\n"
        "    if(!html)return '<span class=\"cp-inacc-ok\">✓ All accessible</span>';\n"
        "    return html;\n"
        "  }\n"
        "  (function(){\n"
        "    if(!acc){\n"
        "      tagsEl.innerHTML='<span style=\"font-size:8.5px;color:#bbb;\">OD data not available</span>';\n"
        "      return;\n"
        "    }\n"
        "    var hasCovSlot=FAC_OD_COLS.some(function(fc){return acc['cov_08_'+fc]!==undefined;});\n"
        "    tagsEl.innerHTML=buildInaccHTML(function(fc){\n"
        "      if(hasCovSlot){\n"
        "        return SLOT_KEYS.some(function(s){return acc['cov_'+s+'_'+fc]===1;});\n"
        "      } else {\n"
        "        return !(acc['pt_'+fc]===0||acc['pt_'+fc]===false);\n"
        "      }\n"
        "    });\n"
        "  })();\n"
        # Chart
        "  var covVals=COV_COLS.map(function(c,i){if(dyn){return dyn.slotCovOH[SLOT_KEYS[i]];}var v=d[c];return(v!=null&&!isNaN(parseFloat(v)))?parseFloat(v):null;});\n"
        "  var maiVals=MAI_COLS.map(function(c,i){if(dyn){return dyn.slotMaiOH[SLOT_KEYS[i]];}var v=d[c];return(v!=null&&!isNaN(parseFloat(v)))?parseFloat(v):null;});\n"
        "  var covAllopenVals=COV_ALLOPEN_COLS.map(function(c,i){if(dyn){return dyn.slotCovAO[SLOT_KEYS[i]];}var v=d[c];return(v!=null&&!isNaN(parseFloat(v)))?parseFloat(v):null;});\n"
        "  var maiAllopenVals=MAI_ALLOPEN_COLS.map(function(c,i){if(dyn){return dyn.slotMaiAO[SLOT_KEYS[i]];}var v=d[c];return(v!=null&&!isNaN(parseFloat(v)))?parseFloat(v):null;});\n"
        "  var hasAllopen=!!dyn||covAllopenVals.some(function(v){return v!==null;})||maiAllopenVals.some(function(v){return v!==null;});\n"
        "  if(_cpChart){_cpChart.destroy();_cpChart=null;}\n"
        "  var ctx=document.getElementById('cp-chart').getContext('2d');\n"
        "  var covBySlot={},maiBySlot={};\n"
        "  SLOT_KEYS.forEach(function(s,i){\n"
        "    var cv=d[COV_COLS[i]],mv=d[MAI_COLS[i]];\n"
        "    covBySlot[s]=(cv!=null&&!isNaN(parseFloat(cv)))?parseFloat(cv):null;\n"
        "    maiBySlot[s]=(mv!=null&&!isNaN(parseFloat(mv)))?parseFloat(mv):null;\n"
        "  });\n"
        # ── getCovRawAcc / getCovInacc / getMaiInacc / renderInaccItems (specialist 묶음) ──
        # specialist는 항상 items에 포함. specOk=true여도 세부 inaccessible 과목 표시
        "  function getCovRawAcc(slot){\n"
        "    var hasCovSlot=acc&&FAC_OD_COLS.some(function(fc){return acc['cov_'+slot+'_'+fc]!==undefined;});\n"
        "    var hasPtFac=acc&&FAC_OD_COLS.some(function(fc){return acc['pt_'+fc]!==undefined;});\n"
        "    var raw={};\n"
        "    FAC_OD_COLS.forEach(function(fc){\n"
        "      if(!acc){raw[fc]=false;return;}\n"
        "      if(hasCovSlot){raw[fc]=acc['cov_'+slot+'_'+fc]===1;}\n"
        "      else if(hasPtFac){raw[fc]=!(acc['pt_'+fc]===0||acc['pt_'+fc]===false);}\n"
        "      else{raw[fc]=false;}\n"
        "    });\n"
        "    return raw;\n"
        "  }\n"
        "  function getCovInacc(slot){\n"
        "    var covV=covBySlot[slot];\n"
        "    var selDcs=getSelectedDisplayFacCols();\n"
        "    if(!acc){\n"
        "      if(covV===null||covV===0)return {items:selDcs.map(function(dc){return {dc:dc,label:DISPLAY_FAC_LABELS[dc]||dc,accessible:false,specDetail:[]};}).filter(function(){return true;}),unknown:false};\n"
        "      if(covV>=99.99)return {items:[],unknown:false};\n"
        "      return {items:[],unknown:true};\n"
        "    }\n"
        "    var raw=getCovRawAcc(slot);\n"
        "    var items=[];\n"
        "    selDcs.forEach(function(dc){\n"
        "      if(dc==='specialist'){\n"
        "        var specOk=SPECIALIST_COLS.some(function(fc){return raw[fc]===true;});\n"
        "        var det=SPECIALIST_COLS.filter(function(fc){return !raw[fc];});\n"
        "        if(specOk&&det.length===0)return;\n"
        "        items.push({dc:'specialist',label:SPECIALIST_LABEL,accessible:specOk,specDetail:det});\n"
        "      } else {\n"
        "        if(!raw[dc])items.push({dc:dc,label:DISPLAY_FAC_LABELS[dc]||dc,accessible:false,specDetail:[]});\n"
        "      }\n"
        "    });\n"
        "    return {items:items,unknown:false};\n"
        "  }\n"
        "  function getMaiInacc(){\n"
        "    var maiV=d.avg_mai;\n"
        "    var selDcs=getSelectedDisplayFacCols();\n"
        "    if(!acc){\n"
        "      if(maiV===null||maiV===0)return {items:selDcs.map(function(dc){return {dc:dc,label:DISPLAY_FAC_LABELS[dc]||dc,accessible:false,specDetail:[]};}).filter(function(){return true;}),unknown:false,tie:false};\n"
        "      if(maiV>=99.99)return {items:[],unknown:false,tie:false};\n"
        "      return {items:[],unknown:true,tie:false};\n"
        "    }\n"
        "    var hasNewMai=FAC_OD_COLS.some(function(fc){return acc['mai_'+fc]!==undefined;});\n"
        "    var hasPtFacM=FAC_OD_COLS.some(function(fc){return acc['pt_'+fc]!==undefined;});\n"
        "    var rawM={};\n"
        "    FAC_OD_COLS.forEach(function(fc){\n"
        "      if(hasNewMai){rawM[fc]=acc['mai_'+fc]===1;}\n"
        "      else if(hasPtFacM){rawM[fc]=!(acc['pt_'+fc]===0||acc['pt_'+fc]===false);}\n"
        "      else{rawM[fc]=false;}\n"
        "    });\n"
        "    var items=[];\n"
        "    selDcs.forEach(function(dc){\n"
        "      if(dc==='specialist'){\n"
        "        var specOkM=SPECIALIST_COLS.some(function(fc){return rawM[fc]===true;});\n"
        "        var detM=SPECIALIST_COLS.filter(function(fc){return !rawM[fc];});\n"
        "        if(specOkM&&detM.length===0)return;\n"
        "        items.push({dc:'specialist',label:SPECIALIST_LABEL,accessible:specOkM,specDetail:detM});\n"
        "      } else {\n"
        "        if(!rawM[dc])items.push({dc:dc,label:DISPLAY_FAC_LABELS[dc]||dc,accessible:false,specDetail:[]});\n"
        "      }\n"
        "    });\n"
        "    return {items:items,unknown:false,tie:acc['mai_is_tie']===1};\n"
        "  }\n"
        # renderInaccItems: accessible specialist → 초록 태그 + ℹ, inaccessible → 빨강/청록 태그 + ℹ
        "  function renderInaccItems(items,pillBg,pillColor,pillBorder,detBg,detColor){\n"
        "    if(!items||items.length===0)return '';\n"
        "    return items.map(function(item){\n"
        "      if(item.dc==='specialist'){\n"
        "        var tid='sd'+Math.random().toString(36).slice(2,7);\n"
        # accessible 여부에 따라 태그 색상 결정
        "        var bg=item.accessible?'#E8F5E9':pillBg;\n"
        "        var fg=item.accessible?'#2E7D32':pillColor;\n"
        "        var bd=item.accessible?'#A5D6A7':pillBorder;\n"
        "        var ps='font-size:8.5px;background:'+bg+';color:'+fg+';border:1px solid '+bd+';border-radius:8px;padding:1px 5px;display:inline-block;';\n"
        "        var prefix=item.accessible?'✓ ':'';\n"
        "        var dh=item.specDetail.map(function(fc){return'<span style=\"font-size:8px;margin-right:3px;\">· '+SPECIALIST_DETAIL_LABELS[fc]+'</span>';}).join('');\n"
        "        return '<span style=\"'+ps+'cursor:default;\">'+prefix+item.label\n"
        "          +' <span onclick=\"(function(){var e=document.getElementById(\\''+tid+'\\');e.style.display=e.style.display===\\'none\\'?\\'block\\':\\'none\\';})()\" '\n"
        "          +'style=\"cursor:pointer;font-size:7.5px;color:'+fg+';border:1px solid '+bd+';border-radius:3px;padding:0 3px;margin-left:2px;\" title=\"Show inaccessible sub-types\">ℹ</span>'\n"
        "          +'</span>'\n"
        "          +(item.specDetail.length>0\n"
        "            ?'<div id=\"'+tid+'\" style=\"display:none;width:100%;margin-top:2px;background:'+detBg+';border-radius:4px;padding:2px 6px;font-size:8px;color:'+detColor+';\">'+'Inaccessible: '+dh+'</div>'\n"
        "            :'<div id=\"'+tid+'\" style=\"display:none;width:100%;margin-top:2px;background:#E8F5E9;border-radius:4px;padding:2px 6px;font-size:8px;color:#2E7D32;\">All sub-types accessible</div>'\n"
        "          );\n"
        "      }\n"
        "      var ps='font-size:8.5px;background:'+pillBg+';color:'+pillColor+';border:1px solid '+pillBorder+';border-radius:8px;padding:1px 5px;display:inline-block;';\n"
        "      return '<span style=\"'+ps+'\">'+item.label+'</span>';\n"
        "    }).join('');\n"
        "  }\n"
        "  var _datasets=["
        "      {label:'Coverage (Opening Hours)',data:covVals,borderColor:'" + COV_LINE_COLOR + "',"
        "backgroundColor:'" + COV_LINE_COLOR + "15',"
        "pointBackgroundColor:'" + COV_LINE_COLOR + "',pointBorderColor:'#fff',pointBorderWidth:1.2,"
        "pointRadius:4,pointHoverRadius:6,borderWidth:2,tension:0.15,fill:false},"
        "      {label:'MAI (Opening Hours)',data:maiVals,borderColor:'" + MAI_LINE_COLOR + "',"
        "backgroundColor:'" + MAI_LINE_COLOR + "15',"
        "pointBackgroundColor:'" + MAI_LINE_COLOR + "',pointBorderColor:'#fff',pointBorderWidth:1.2,"
        "pointRadius:4,pointHoverRadius:6,borderWidth:2,tension:0.15,fill:false}"
        "    ];\n"
        "  if(hasAllopen){\n"
        "    _datasets.push({label:'Coverage (Always Open)',data:covAllopenVals,borderColor:'" + COV_LINE_COLOR + "88',"
        "borderDash:[5,3],pointRadius:2,pointHoverRadius:4,borderWidth:1.5,tension:0.15,fill:false});\n"
        "    _datasets.push({label:'MAI (Always Open)',data:maiAllopenVals,borderColor:'" + MAI_LINE_COLOR + "88',"
        "borderDash:[5,3],pointRadius:2,pointHoverRadius:4,borderWidth:1.5,tension:0.15,fill:false});\n"
        "  }\n"
        "  _cpChart=new Chart(ctx,{type:'line',"
        "    data:{labels:TIME_SLOTS,datasets:_datasets},"
        "    options:{responsive:true,maintainAspectRatio:false,"
        # mode:'index' + intersect:false → x축 어디서든 해당 시간대 감지
        "      interaction:{mode:'index',intersect:false,axis:'x'},"
        # top padding 넉넉히 줘서 100% 점 잘림 방지
        "      layout:{padding:{top:12,bottom:2,left:0,right:0}},"
        "      plugins:{"
        "        tooltip:{"
        "          enabled:false,"
        "          external:function(context){"
        "            var wrap=document.getElementById('cp-chart-wrap');"
        "            var tip=document.getElementById('cp-tooltip');"
        "            if(!tip){"
        "              tip=document.createElement('div');tip.id='cp-tooltip';"
        "              tip.style.cssText='position:absolute;background:#fff;border:1px solid #e0e0e0;"
        "border-radius:7px;padding:9px 11px;font-size:10px;line-height:1.55;"
        "box-shadow:0 3px 12px rgba(0,0,0,0.13);z-index:9999;width:230px;"
        "word-break:keep-all;white-space:normal;pointer-events:none;';"
        "              wrap.appendChild(tip);"
        "            }"
        "            var model=context.tooltip;"
        # opacity===0이면 숨기되, 짧은 지연 후 숨겨서 깜빡임 방지
        "            if(model.opacity===0){"
        "              if(tip._hideTimer)clearTimeout(tip._hideTimer);"
        "              tip._hideTimer=setTimeout(function(){tip.style.display='none';},120);"
        "              return;"
        "            }"
        "            if(tip._hideTimer){clearTimeout(tip._hideTimer);tip._hideTimer=null;}"
        "            var idx=model.dataPoints&&model.dataPoints[0]?model.dataPoints[0].dataIndex:null;"
        "            if(idx===null){tip.style.display='none';return;}"
        "            var slot=SLOT_KEYS[idx];"
        "            var covV=covBySlot[slot],maiV=maiBySlot[slot];"
        "            var inaccCov=getCovInacc(slot);"
        "            var inaccMai=getMaiInacc();"
        "            var h='';"
        "            h+='<div style=\"font-weight:700;font-size:11px;color:#222;margin-bottom:6px;"
        "border-bottom:1.5px solid #f0f0f0;padding-bottom:4px;\">'+TIME_SLOTS[idx]+'</div>';"
        "            h+='<div style=\"display:flex;align-items:center;gap:5px;margin-bottom:1px;\">';"
        "            h+='<span style=\"width:9px;height:9px;border-radius:50%;background:" + COV_LINE_COLOR + ";flex-shrink:0;\"></span>';"
        "            h+='<span style=\"font-weight:700;color:" + COV_LINE_COLOR + ";font-size:10px;\">Coverage: '+(covV!=null?covV.toFixed(1)+'%':'N/A')+'</span></div>';"
        "            h+='<div style=\"font-size:8px;color:#888;margin-left:14px;margin-bottom:3px;\">Inaccessible within threshold</div>';"
        "            if(inaccCov.unknown){h+='<div style=\"font-size:9px;color:#B0651A;margin-left:14px;margin-bottom:5px;\">⚠ Details unavailable</div>';}"
        "            else if(inaccCov.items.length===0){h+='<div style=\"font-size:9px;color:#43A047;margin-left:14px;margin-bottom:5px;\">✓ All accessible</div>';}"
        "            else{h+='<div style=\"margin-left:14px;margin-bottom:5px;display:flex;flex-wrap:wrap;gap:2px;\">'+renderInaccItems(inaccCov.items,'#FFEBEE','#b71c1c','#ef9a9a','#FFEBEE','#b71c1c')+'</div>';}"
        "            h+='<div style=\"border-top:1px solid #f5f5f5;padding-top:5px;margin-top:2px;\">';"
        "            h+='<div style=\"display:flex;align-items:center;gap:5px;margin-bottom:1px;\">';"
        "            h+='<span style=\"width:9px;height:9px;border-radius:50%;background:" + MAI_LINE_COLOR + ";flex-shrink:0;\"></span>';"
        "            h+='<span style=\"font-weight:700;color:" + MAI_LINE_COLOR + ";font-size:10px;\">MAI: '+(maiV!=null?maiV.toFixed(1)+'%':'N/A')+'</span></div>';"
        "            var maiSub='Best reachable grid (within 15 min)'+(inaccMai.tie?' *':'');"
        "            h+='<div style=\"font-size:8px;color:#888;margin-left:14px;margin-bottom:3px;\">'+maiSub+'</div>';"
        "            if(inaccMai.tie){h+='<div style=\"font-size:7.5px;color:#aaa;margin-left:14px;margin-bottom:2px;\">* Nearest of tied grids</div>';}"
        "            if(inaccMai.unknown){h+='<div style=\"font-size:9px;color:#B0651A;margin-left:14px;\">⚠ Details unavailable</div>';}"
        "            else if(inaccMai.items.length===0){h+='<div style=\"font-size:9px;color:#43A047;margin-left:14px;\">✓ All facility types present</div>';}"
        "            else{h+='<div style=\"margin-left:14px;display:flex;flex-wrap:wrap;gap:2px;\">'+renderInaccItems(inaccMai.items,'#E0F2F1','#00695C','#80CBC4','#E0F2F1','#00695C')+'</div>';}"
        "            h+='</div>';"
        "            tip.innerHTML=h;"
        "            var wrapW=wrap.offsetWidth||240;"
        "            var posX=model.caretX+14;"
        "            if(posX+235>wrapW)posX=model.caretX-244;"
        "            if(posX<0)posX=2;"
        "            var posY=model.caretY-20;"
        "            if(posY<0)posY=2;"
        "            tip.style.left=posX+'px';"
        "            tip.style.top=posY+'px';"
        "            tip.style.display='block';"
        "          }"
        "        },"
        "        legend:{position:'top',align:'start',labels:{font:{size:9},boxWidth:9,padding:6}}"
        "      },"
        "      scales:{"
        "        x:{grid:{color:'#f4f4f4'},ticks:{font:{size:8}}},"
        # y 최대 103으로 → 100% 점 상단 잘림 방지, suggestedMax로 여유 확보
        "        y:{min:0,max:103,grid:{color:'#f4f4f4'},ticks:{font:{size:8},callback:function(v){return v<=100?v+'%':'';},"
        "stepSize:20}}"
        "      }"
        "    }});\n"
        "  panel.classList.add('visible');\n"
        "  allMaps.forEach(function(m){m.invalidateSize();});\n"
        "}\n"
        "document.getElementById('cp-close').addEventListener('click',function(){\n"
        "  document.getElementById('cell-panel').classList.remove('visible');\n"
        "  if(_cpChart){_cpChart.destroy();_cpChart=null;}\n"
        "  if(window._hlGeoJSON){allMaps.forEach(function(m){try{m.removeLayer(window._hlGeoJSON);}catch(ex){}});window._hlGeoJSON=null;}\n"
        "  allMaps.forEach(function(m){m.invalidateSize();});\n"
        "});\n"
    )


    panel_html = (
        '<div class="cell-panel" id="cell-panel">'
        '<div style="display:flex;align-items:flex-start;justify-content:space-between;">'
        '<div style="flex:1;min-width:0;"><div class="cp-id-sub">Selected Cell</div>'
        '<div class="cp-id" id="cp-fid"></div></div>'
        '<span class="cp-close" id="cp-close">&#x2715;</span></div>'
        # PT
        '<div class="cp-sec" style="color:#3F51B5;">Public Transit</div>'
        '<div class="cp-metrics">'
        '<div class="cp-m"><div class="cp-m-label">avg. Coverage</div><div class="cp-m-val" id="cp-avg-cov">—</div></div>'
        '<div class="cp-m"><div class="cp-m-label">avg. MAI</div><div class="cp-m-val" id="cp-avg-mai">—</div></div>'
        '<div class="cp-m"><div class="cp-m-label">CV Coverage</div><div class="cp-m-val" id="cp-cv-cov">—</div></div>'
        '<div class="cp-m"><div class="cp-m-label">CV MAI</div><div class="cp-m-val" id="cp-cv-mai">—</div></div>'
        '</div>'
        # SGG avg
        '<div class="cp-sec" style="color:#3F51B5;">Municipality Avg.</div>'
        '<div class="cp-sgg-ref">'
        '<div class="cp-sgg-m"><div class="cp-sgg-label">City Coverage</div><div class="cp-sgg-val" id="cp-sgg-cov">—</div></div>'
        '<div class="cp-sgg-m"><div class="cp-sgg-label">City MAI</div><div class="cp-sgg-val" id="cp-sgg-mai">—</div></div>'
        '</div>'
        # Deficit types
        '<div class="cp-sec" style="color:#555;margin-top:10px;">Deficit Type</div>'
        '<div class="cp-deficit-tags" id="cp-deficit-tags"></div>'
        # Reference
        '<div class="cp-sec" style="color:#555;">Reference</div>'
        '<div class="cp-ref">'
        '<div class="cp-r"><div class="cp-r-label">Population</div><div class="cp-r-val" id="cp-pop">—</div></div>'
        '<div class="cp-r"><div class="cp-r-label">Car Cov.</div><div class="cp-r-val" id="cp-car-cov">—</div></div>'
        '<div class="cp-r"><div class="cp-r-label">Car MAI</div><div class="cp-r-val" id="cp-car-mai">—</div></div>'
        '</div>'
        # Inaccessible
        '<div class="cp-sec" style="color:#E65100;">Inaccessible Facilities</div>'
        '<div class="cp-inacc-tags" id="cp-inacc-tags"></div>'
        # Chart
        '<div class="cp-sec" style="color:#555;margin-top:10px;">Time-of-Day Profile</div>'
        '<div id="cp-chart-wrap" style="position:relative;margin-top:2px;"><canvas id="cp-chart"></canvas></div>'
        '</div>'
    )



    html_parts = [
        '<!DOCTYPE html><html><head><meta charset="utf-8"/>',
        '<style>',
        '*{box-sizing:border-box;margin:0;padding:0;}',
        'body{background:#fff;font-family:"Inter",system-ui,sans-serif;}',
        panel_css,
        '</style>',
        '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>',
        '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>',
        chart_cdn,
        '</head><body>',
        '<div class="outer-wrap">',
        '<div class="maps-section">',
        # Layer bar: maps-section 안, maps-grid 밖에 배치 → grid item 간섭 없음
        '<div class="layer-bar">'
        '<span class="lbar-title">Layers</span>'
        '<label><input type="checkbox" id="lbl-coverage" checked onchange="_layerToggle()"> Coverage</label>'
        '<label><input type="checkbox" id="lbl-mai" checked onchange="_layerToggle()"> MAI</label>'
        '<label><input type="checkbox" id="lbl-mvg" checked onchange="_layerToggle()"> MV Geary</label>'
        '<span style="position:relative;display:inline-flex;align-items:center;">'
        '<span style="cursor:pointer;font-size:10px;color:#5C6BC0;border:1px solid #C5CAE9;'
        'border-radius:3px;padding:0 4px;font-weight:700;" '
        'onmouseenter="document.getElementById(\'mvg-info-box\').classList.add(\'visible\')" '
        'onmouseleave="document.getElementById(\'mvg-info-box\').classList.remove(\'visible\')">i</span>'
        '<div id="mvg-info-box" class="cluster-info-box" style="left:0;right:auto;">'
        '<b style="color:#333;">MV Local Geary</b><br/>'
        'Multivariate Local Geary statistic (Anselin, 2019) identifies grids where '
        'the combination of five deficit indicators is <b>homogeneous</b> (similar to neighbors) '
        'or <b>heterogeneous</b> (dissimilar). Homogeneous grids are colored by their deficit profile.</div>'
        '</span>'
        '<label><input type="checkbox" id="lbl-pop" checked onchange="_layerToggle()"> Pop</label>'
        '<span class="lbar-sep"></span>'
        '<span class="lbar-title">Deficit</span>'
        '<label id="dlbl-fs"><input type="checkbox" id="dcb-fs" onchange="_deficitToggle()"> F(s)</label>'
        '<label id="dlbl-fd"><input type="checkbox" id="dcb-fd" onchange="_deficitToggle()"> F(d)</label>'
        '<label id="dlbl-fo"><input type="checkbox" id="dcb-fo" onchange="_deficitToggle()"> F(o)</label>'
        '<label id="dlbl-tc"><input type="checkbox" id="dcb-tc" onchange="_deficitToggle()"> T(c)</label>'
        '<label id="dlbl-tf"><input type="checkbox" id="dcb-tf" onchange="_deficitToggle()"> T(f)</label>'
        '<span class="lbar-sep"></span>'
        '<span class="lbar-title">Basis</span>'
        '<label><input type="radio" name="basis" id="rb-sgg" value="sgg" checked onchange="_basisToggle(this.value)"> Municipality</label>'
        '<label><input type="radio" name="basis" id="rb-nat" value="nat" onchange="_basisToggle(this.value)"> National</label>'
        '<span class="lbar-sep"></span>'
        '<span style="position:relative;display:inline-flex;align-items:center;">'
        '<label style="font-weight:600;"><input type="checkbox" id="cb-cluster" onchange="_clusterToggle()"> Cluster View</label>'
        '<span id="cluster-info-btn" style="cursor:pointer;font-size:10px;color:#5C6BC0;border:1px solid #C5CAE9;'
        'border-radius:3px;padding:0 4px;margin-left:4px;font-weight:700;" '
        'onmouseenter="document.getElementById(\'cluster-info-box\').classList.add(\'visible\')" '
        'onmouseleave="document.getElementById(\'cluster-info-box\').classList.remove(\'visible\')">i</span>'
        '<div id="cluster-info-box" class="cluster-info-box">'
        '<b style="color:#333;">Cluster View — Local Join Count</b><br/>'
        'When enabled, significant spatial clusters (&#945; &lt; 0.05) are highlighted '
        'with diagonal hatching using the Local Join Count statistic '
        '(Anselin &amp; Li, 2019) with 999 conditional permutations.<br/>'
        '<b>Hatched grids</b> = significant co-location cluster.<br/>'
        '<b>Border only</b> = deficit present but not significant.<br/>'
        '<span style="color:#777;font-size:9px;margin-top:3px;display:block;">'
        'Cluster View and Multivariate Geary are computed on the default facility set '
        '(pharmacy, park, library, primary care, specialist care, grocery, public service).</span></div>'
        '</span>'
        '</div>',
        # Cluster View legend (hidden by default)
        '<div id="cluster-legend" style="display:none;padding:4px 12px;background:#fafafa;'
        'border-bottom:1px solid #eee;font-size:10px;color:#555;'
        'align-items:center;gap:10px;flex-wrap:wrap;">'
        '<span style="font-weight:700;color:#333;margin-right:4px;">Cluster View</span>'
        '<span style="display:inline-flex;align-items:center;gap:3px;"><span style="width:12px;height:10px;background:#E5393566;border:2px solid #E53935;border-radius:2px;"></span><span style="font-size:9px;">F(s)</span></span><span style="display:inline-flex;align-items:center;gap:3px;"><span style="width:12px;height:10px;background:#F4A10066;border:2px solid #F4A100;border-radius:2px;"></span><span style="font-size:9px;">F(d)</span></span><span style="display:inline-flex;align-items:center;gap:3px;"><span style="width:12px;height:10px;background:#FF704366;border:2px solid #FF7043;border-radius:2px;"></span><span style="font-size:9px;">F(o)</span></span><span style="display:inline-flex;align-items:center;gap:3px;"><span style="width:12px;height:10px;background:#7B1FA266;border:2px solid #7B1FA2;border-radius:2px;"></span><span style="font-size:9px;">T(c)</span></span><span style="display:inline-flex;align-items:center;gap:3px;"><span style="width:12px;height:10px;background:#2E7D3266;border:2px solid #2E7D32;border-radius:2px;"></span><span style="font-size:9px;">T(f)</span></span>'
        '<span style="color:#aaa;font-size:9px;margin-left:4px;">= sig. cluster (p &lt; 0.05)</span>'
        '</div>',
        '<div class="maps-grid" style="position:relative;">',
        map_divs,
        '<div class="fac-ctrl" id="fc"><div class="fac-hdr">Facilities</div></div>',
        '</div>',
        '</div>',
        panel_html,
        '</div>',
        '<script>',
        'var gridData=[' + ",".join(grid_js_list) + '];',
        'var subwayData='  + subway_js  + ';',
        'var stationData=' + station_js + ';',
        'var facData='     + fac_js + ';',
        'var colorbarsData=' + cbars_js_arr + ';',
        # ── Colormap stops (JS interpolation용) ──
        'var COV_CMAP_STOPS=' + json.dumps(_cmap_to_js_stops(CMAPS["coverage"])) + ';',
        'var MAI_CMAP_STOPS=' + json.dumps(_cmap_to_js_stops(CMAPS["mai"])) + ';',
        # ── Facility selector definitions ──
        'var FAC_SELECTOR_DEFS=' + json.dumps(FAC_SELECTOR_DEFS) + ';',
        # fac_visible: localStorage 우선, 없으면 전부 표시
        '(function(){try{'
        '  var s=localStorage.getItem("facVisible");'
        '  window._initFacVisible=s?JSON.parse(s):null;'
        '}catch(e){window._initFacVisible=null;}})();',
        'var facVisible=window._initFacVisible||' + json.dumps(list(FACILITY_ORDER)) + ';',
        'var deficitInfo=' + json.dumps({
            k: {"c": v, "label": DEFICIT_LABELS[k]}
            for k, v in deficit_colors.items()
        }) + ';',
        'var deficitVisible=[];',
        'var facLayers={};',
        'var deficitLayers={};',
        'var gridLayers=[];',
        'var allMaps=[];',
        'var GRID_ID_KEY=' + json.dumps(GRID_JOIN_COL) + ';',
        # ── JS 상수: maps_init보다 먼저 정의해야 함 ──
        'var DEFICIT_KEYS=' + json.dumps(DEFICIT_KEYS) + ';',
        'var JCL_COLORS=' + json.dumps(JCL_COLORS) + ';',
        'var MVG_PROFILE_COLORS=' + json.dumps(MVG_PROFILE_COLORS) + ';',
        'var MVG_HETERO_COLOR="' + MVG_HETERO_COLOR + '";',
        'var MVG_NOTSIG_COLOR="' + MVG_NOTSIG_COLOR + '";',
        'var _GRID_ID_KEY=' + json.dumps(GRID_JOIN_COL) + ';',
        'var _SGG_KEY=' + json.dumps(SGG_CODE_COL) + ';',
        # selectedFacs 초기값: localStorage 우선, 없으면 기본 7종
        '(function(){'
        '  var _def=' + json.dumps(FAC_DEFAULT_SEL) + ';'
        '  try{'
        '    var s=localStorage.getItem("ffSelectedFacs");'
        '    window.selectedFacs=s?JSON.parse(s):_def;'
        '  }catch(e){window.selectedFacs=_def;}'
        '})();',
        'window._hlGeoJSON=null;',
        'function getBBox(geom){'
        'var mn=[Infinity,Infinity],mx=[-Infinity,-Infinity];'
        'function pt(c){if(c[0]<mn[0])mn[0]=c[0];if(c[1]<mn[1])mn[1]=c[1];'
        'if(c[0]>mx[0])mx[0]=c[0];if(c[1]>mx[1])mx[1]=c[1];}'
        'function ring(r){for(var i=0;i<r.length;i++)pt(r[i]);}'
        'var t=geom.type,c=geom.coordinates;'
        'if(t==="Polygon")for(var i=0;i<c.length;i++)ring(c[i]);'
        'else if(t==="MultiPolygon")for(var i=0;i<c.length;i++)for(var j=0;j<c[i].length;j++)ring(c[i][j]);'
        'return[mn[0],mn[1],mx[0],mx[1]];}',
        'function pointInRing(lat,lng,ring){'
        'var inside=false;'
        'for(var i=0,j=ring.length-1;i<ring.length;j=i++){'
        'var xi=ring[i][0],yi=ring[i][1],xj=ring[j][0],yj=ring[j][1];'
        'if(((yi>lat)!=(yj>lat))&&(lng<(xj-xi)*(lat-yi)/(yj-yi)+xi))inside=!inside;}'
        'return inside;}',
        'function pointInPolygon(lat,lng,geom){'
        'var t=geom.type,c=geom.coordinates;'
        'if(t==="Polygon"){return pointInRing(lat,lng,c[0]);}'
        'if(t==="MultiPolygon"){for(var i=0;i<c.length;i++){if(pointInRing(lat,lng,c[i][0]))return true;}return false;}'
        'return false;}',
        maps_init,
        sync_js,
        fac_toggle_js,
        'window.toggleDeficit=function(dk,on){'
        '  if(!deficitLayers[dk])return;'
        '  deficitLayers[dk].maps.forEach(function(x){'
        '    if(on){if(x.whiteLayer)x.whiteLayer.addTo(x.map);x.layer.addTo(x.map);}'
        '    else{if(x.whiteLayer)x.map.removeLayer(x.whiteLayer);x.map.removeLayer(x.layer);}'
        '  });'
        '};'
        # basis 변경: grid recolor + colorbar 교체 + deficit recolor (뷰포트 유지)
        'window.updateDeficitBasis=function(newBasis){'
        '  window.deficitBasis=newBasis;'
        '  gridLayers.forEach(function(lyr){lyr.setStyle(lyr.options.style);});'
        '  colorbarsData.forEach(function(cb,i){'
        '    var el=document.getElementById("cb"+i);'
        '    if(el)el.innerHTML=cb[newBasis]||cb["sgg"];'
        '  });'
        '  Object.keys(deficitLayers).forEach(function(dk){'
        '    deficitLayers[dk].maps.forEach(function(x){'
        '      if(!x.makeStyleFns)return;'
        '      var fns=x.makeStyleFns(newBasis);'
        '      x.whiteLayer.setStyle(fns.white);x.layer.setStyle(fns.color);'
        '    });'
        '  });'
        '};'
        # base layer show/hide — allMaps는 maps_init에서 push됨
        'var _mapKeys=' + json.dumps(ALL_MAP_KEYS) + ';'
        'window.applyBaseLayerVis=function(vis){'
        '  var _snap=null;'
        '  for(var _i=0;_i<allMaps.length;_i++){try{_snap={c:allMaps[_i].getCenter(),z:allMaps[_i].getZoom()};break;}catch(e){}}'
        '  var shown=_mapKeys.filter(function(k){return vis[k]!==false;});'
        '  if(!shown.length)shown=[_mapKeys[0]];'
        '  var grid=document.querySelector(".maps-grid");'
        '  if(grid){if(shown.length===1){grid.style.display="block";grid.style.gridTemplateColumns="";}else{grid.style.display="grid";grid.style.gridTemplateColumns="1fr 1fr";}}'
        '  _mapKeys.forEach(function(k,i){'
        '    var wrap=document.getElementById("mwrap"+i);if(!wrap)return;'
        '    if(shown.indexOf(k)<0){wrap.style.display="none";}'
        '    else{wrap.style.display="";'
        '      if(allMaps[i])(function(m,s){try{m.invalidateSize({animate:false});if(s)m.setView(s.c,s.z,{animate:false});}catch(e){}})(allMaps[i],_snap);}'
        '  });'
        '  if(_snap)setTimeout(function(){allMaps.forEach(function(m){try{m.invalidateSize({animate:false});m.setView(_snap.c,_snap.z,{animate:false});}catch(e){}});},150);'
        '};'
        'window.setMapView=function(lat,lng,z){'
        '  allMaps.forEach(function(m){'
        '    try{m.setView([lat,lng],z,{animate:false});}catch(e){}'
        '  });'
        '};'
        # layer toggle: base 4 maps
        'function _getVisState(){'
        '  var vis={};'
        '  ["coverage","mai","mvg","pop"].forEach(function(k){'
        '    var el=document.getElementById("lbl-"+k);'
        '    vis[k]=el?el.checked:true;'
        '  });'
        '  if(!["coverage","mai","mvg","pop"].some(function(k){return vis[k];})){'
        '    vis.coverage=true;'
        '    var el=document.getElementById("lbl-coverage");if(el)el.checked=true;'
        '  }'
        '  return vis;'
        '}'
        'function _applyVis(){'
        '  var vis=_getVisState();'
        '  window.applyBaseLayerVis(vis);'
        '}'
        'function _layerToggle(){_applyVis();}'
        # basis 토글: grid recolor + deficit + JCL/MVG restyle
        'function _basisToggle(val){'
        '  window.updateDeficitBasis(val);'
        '  _restyleDeficitLayers();'
        '  _buildSigLayers();'
        '}'
        # deficit 토글: fo 포함 5개
        'function _deficitToggle(){'
        '  DEFICIT_KEYS.forEach(function(dk){'
        '    var el=document.getElementById("dcb-"+dk);'
        '    if(el)window.toggleDeficit(dk,el.checked);'
        '  });'
        '  _restyleDeficitLayers();'
        '  _buildSigLayers();'
        '  _applyVis();'
        '}'
        # cluster view 토글: deficit restyle + JCL map show/hide
        'window._clusterView=false;'
        'window._sigLyrs=[];'
        'function _destroySigLayers(){'
        '  window._sigLyrs.forEach(function(x){try{x.map.removeLayer(x.layer);}catch(e){}});'
        '  window._sigLyrs=[];'
        '}'
        'function _buildSigLayers(){'
        '  _destroySigLayers();'
        '  if(!window._clusterView)return;'
        '  var basis=window.deficitBasis||"sgg";'
        '  var baseCnt=' + str(len(BASE_MAP_KEYS)) + ';'
        '  DEFICIT_KEYS.forEach(function(dk){'
        '    var dEl=document.getElementById("dcb-"+dk);'
        '    if(!dEl||!dEl.checked)return;'
        '    var clCol=basis+"_jcl_"+dk+"_cl";'
        '    var hasCol=basis+"_has_"+dk;'
        '    var dc=DEFICIT_COLOR_MAP[dk]||"#999";'
        '    for(var mi=0;mi<baseCnt;mi++){'
        '      if(!allMaps[mi]||!gridData[mi])continue;'
        '      var lyr=L.geoJSON(gridData[mi],{'
        '        filter:function(f){var p=f.properties;'
        '          var _df2=window._defFlags,_fid2=p.from_id;'\
        '          var _isD2=_df2&&_df2[_fid2]?!!_df2[_fid2][dk]:(p[hasCol]===true||p[hasCol]===1);'\
        '          return _isD2&&(p[clCol]===1||p[clCol]===true);},'
        '        style:function(){return{fillColor:dc,fillOpacity:0.50,weight:4,color:dc,opacity:1};},'
        '        renderer:L.canvas({tolerance:0})'
        '      }).addTo(allMaps[mi]);'
        '      window._sigLyrs.push({map:allMaps[mi],layer:lyr});'
        '    }'
        '  });'
        '}'
        'function _restyleDeficitLayers(){'
        '  var basis=window.deficitBasis||"sgg";'
        '  Object.keys(deficitLayers).forEach(function(dk){'
        '    deficitLayers[dk].maps.forEach(function(x){'
        '      if(!x.makeStyleFns)return;'
        '      var fns=x.makeStyleFns(basis);'
        '      x.whiteLayer.setStyle(fns.white);x.layer.setStyle(fns.color);'
        '    });'
        '  });'
        '}'
        'function _clusterToggle(){'
        '  var el=document.getElementById("cb-cluster");'
        '  window._clusterView=el?el.checked:false;'
        '  var leg=document.getElementById("cluster-legend");'
        '  if(leg)leg.style.display=window._clusterView?"flex":"none";'
        '  _buildSigLayers();'
        '  _applyVis();'
        '}'
        # localStorage polling
        '(function(){'
        '  var _prev={basis:null,deficit:null,facs:null};'
        '  function _r(k){try{return localStorage.getItem(k);}catch(e){return null;}}'
        '  function _poll(){'
        '    var b=_r("deficitBasis")||"sgg";'
        '    if(b!==_prev.basis){_prev.basis=b;try{window.deficitBasis=b;window.updateDeficitBasis(b);}catch(e){}}'
        '    var s=_r("deficitState");'
        '    if(s!==_prev.deficit){_prev.deficit=s;'
        '      try{var st=JSON.parse(s||"{}")||{};'
        '        Object.keys(st).forEach(function(dk){window.toggleDeficit(dk,!!st[dk]);});'
        '      }catch(e){}'
        '    }'
        '    var f=_r("ffSelectedFacs");'
        '    if(f!==null&&f!==_prev.facs){'
        '      _prev.facs=f;'
        '      try{'
        '        var facs=JSON.parse(f);'
        '        if(Array.isArray(facs)){window.selectedFacs=facs;window.recomputeMapColors();if(window._currentCellFid)showCellPanel(window._currentCellFid);}'
        '      }catch(e){}'
        '    }'
        '  }'
        '  setTimeout(function(){'
        '    try{'
        '      var b0=_r("deficitBasis")||"sgg";_prev.basis=b0;window.deficitBasis=b0;'
        '      if(b0!=="sgg")window.updateDeficitBasis(b0);'
        '      var _initOff={fs:false,fd:false,fo:false,tc:false,tf:false};'
        '      try{localStorage.setItem("deficitState",JSON.stringify(_initOff));}catch(e){}'
        '      _prev.deficit=JSON.stringify(_initOff);'
        '      var f0=_r("ffSelectedFacs");'
        '      if(f0){_prev.facs=f0;try{var fs=JSON.parse(f0);if(Array.isArray(fs)){window.selectedFacs=fs;if(fs.length<FAC_SELECTOR_DEFS.length){window.recomputeMapColors();if(window._currentCellFid)showCellPanel(window._currentCellFid);}}}catch(e){}}'
        '      else{_prev.facs=null;}'
        '    }catch(e){}'
        '  },400);'
        '  setInterval(_poll,400);'
        '})();',
        # facility 패널 드래그
        '(function(){'
        '  var el=document.getElementById("fc");'
        '  if(!el)return;'
        '  var hdr=el.querySelector(".fac-hdr");'
        '  if(!hdr)return;'
        '  var dx=0,dy=0,sx=0,sy=0,dragging=false;'
        '  function onDown(e){'
        '    dragging=true;'
        '    var pt=e.touches?e.touches[0]:e;'
        '    sx=pt.clientX-el.offsetLeft;sy=pt.clientY-el.offsetTop;'
        '    hdr.style.cursor="grabbing";'
        '    e.preventDefault();'
        '  }'
        '  function onMove(e){'
        '    if(!dragging)return;'
        '    var pt=e.touches?e.touches[0]:e;'
        '    var nx=pt.clientX-sx,ny=pt.clientY-sy;'
        '    var par=el.offsetParent||document.body;'
        '    nx=Math.max(0,Math.min(nx,par.clientWidth-el.offsetWidth));'
        '    ny=Math.max(0,Math.min(ny,par.clientHeight-el.offsetHeight));'
        '    el.style.left=nx+"px";el.style.top=ny+"px";'
        '    el.style.right="auto";'
        '    e.preventDefault();'
        '  }'
        '  function onUp(){dragging=false;hdr.style.cursor="grab";}'
        '  hdr.addEventListener("mousedown",onDown);'
        '  hdr.addEventListener("touchstart",onDown,{passive:false});'
        '  document.addEventListener("mousemove",onMove);'
        '  document.addEventListener("touchmove",onMove,{passive:false});'
        '  document.addEventListener("mouseup",onUp);'
        '  document.addEventListener("touchend",onUp);'
        '})();',
        panel_js,
        # ── Facility Filter Dynamic Recompute ─────────────────────────────────
        r"""
(function(){
var GRID_ID_KEY='from_id';
var COV_GAMMA=0.6;

function hexToRgb(h){
  return[parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];
}
function rgbToHex(r,g,b){
  return'#'+[r,g,b].map(function(v){
    return Math.max(0,Math.min(255,Math.round(v))).toString(16).padStart(2,'0');
  }).join('');
}
function applyColormap(val,vmax,stops){
  if(!vmax||vmax<=0)return stops[0];
  var t=Math.pow(Math.min(Math.max(val,0)/vmax,1.0),COV_GAMMA);
  var idx=t*(stops.length-1);
  var lo=Math.floor(idx),hi=Math.min(Math.ceil(idx),stops.length-1);
  if(lo===hi)return stops[lo];
  var f=idx-lo,c1=hexToRgb(stops[lo]),c2=hexToRgb(stops[hi]);
  return rgbToHex(c1[0]+f*(c2[0]-c1[0]),c1[1]+f*(c2[1]-c1[1]),c1[2]+f*(c2[2]-c1[2]));
}

function getActiveFacCols(){
  var sel=window.selectedFacs||[];
  var active=[];
  FAC_SELECTOR_DEFS.forEach(function(def){
    if(sel.indexOf(def.id)>=0){
      def.fac_cols.forEach(function(fc){if(active.indexOf(fc)<0)active.push(fc);});
    }
  });
  return active;
}

function computeNewCov(fid,activeFacCols){
  var acc=(window.facAccessData||{})[fid];
  if(!acc||activeFacCols.length===0)return null;
  var sum=0,cnt=0;
  SLOT_KEYS.forEach(function(s){
    var hasCovSlot=activeFacCols.some(function(fc){return acc['cov_'+s+'_'+fc]!==undefined;});
    if(!hasCovSlot)return;
    var ok=0;
    activeFacCols.forEach(function(fc){if(acc['cov_'+s+'_'+fc]===1)ok++;});
    sum+=ok/activeFacCols.length*100; cnt++;
  });
  return cnt>0?sum/cnt:null;
}

function computeNewMai(fid,activeFacCols){
  var acc=(window.facAccessData||{})[fid];
  if(!acc||activeFacCols.length===0)return null;
  var hasAny=activeFacCols.some(function(fc){return acc['mai_'+fc]!==undefined;});
  if(!hasAny)return null;
  var ok=0;
  activeFacCols.forEach(function(fc){if(acc['mai_'+fc]===1)ok++;});
  return ok/activeFacCols.length*100;
}

window.recomputeMapColors=function(){
  var activeFacCols=getActiveFacCols();
  var hasFacData=window.facAccessData&&Object.keys(window.facAccessData).length>0;
  var baseFeats=(gridData[0]&&gridData[0].features)||[];

  if(!hasFacData||activeFacCols.length===0){
    for(var gi=0;gi<gridData.length;gi++){
      if(!gridData[gi])continue;
      (gridData[gi].features||[]).forEach(function(f){
        delete f.properties._dyn_cov;delete f.properties._dyn_mai;
      });
    }
    gridLayers.forEach(function(lyr){lyr.setStyle(lyr.options.style);});
    _buildSigLayers();
    return;
  }

  var newCovMap={},newMaiMap={};
  baseFeats.forEach(function(f){
    var fid=f.properties[GRID_ID_KEY];
    if(!fid)return;
    newCovMap[fid]=computeNewCov(String(fid),activeFacCols);
    newMaiMap[fid]=computeNewMai(String(fid),activeFacCols);
  });

  for(var gi=0;gi<gridData.length;gi++){
    if(!gridData[gi])continue;
    (gridData[gi].features||[]).forEach(function(f){
      var p=f.properties,fid=p[GRID_ID_KEY];
      var cov=newCovMap[fid],mai=newMaiMap[fid];
      p._dyn_cov=(cov!==null&&cov!==undefined)?applyColormap(cov,100,COV_CMAP_STOPS):undefined;
      p._dyn_mai=(mai!==null&&mai!==undefined)?applyColormap(mai,100,MAI_CMAP_STOPS):undefined;
    });
  }
  gridLayers.forEach(function(lyr){lyr.setStyle(lyr.options.style);});
  _buildSigLayers();
  setTimeout(function(){window._recomputeDefFlags();},50);
};

window._recomputeDefFlags=function(){
  var selDcs=getSelectedDisplayFacCols();
  if(!selDcs.length){window._defFlags=null;_restyleDeficitLayers();_buildSigLayers();return;}
  var _def=["park","library","m1","specialist","grocery","public","pharmacy"];
  var isDefault=selDcs.length===_def.length&&_def.every(function(d){return selDcs.indexOf(d)>=0;});
  if(isDefault){window._defFlags=null;_restyleDeficitLayers();_buildSigLayers();return;}
  var cd=window.cellData||{};
  var fa=window.facAccessData||{};
  var basis=window.deficitBasis||'sgg';
  var keys=Object.keys(cd);
  var flags={};
  var i=0,chunk=3000;
  function step(){
    try{
      var end=Math.min(i+chunk,keys.length);
      for(;i<end;i++){
        try{var _r=_dynDeficit(cd[keys[i]],fa[keys[i]],selDcs,basis);if(_r)flags[keys[i]]=_r;}
        catch(e){flags[keys[i]]={fs:false,fd:false,fo:false,tc:false,tf:false};}
      }
      if(i<keys.length){setTimeout(step,0);}
      else{window._defFlags=flags;_restyleDeficitLayers();_buildSigLayers();}
    }catch(e){
      window._defFlags=null;_restyleDeficitLayers();_buildSigLayers();
    }
  }
  step();
};

/* 초기 적용 (selectedFacs가 기본값과 다를 경우) */
setTimeout(function(){
  if(window.selectedFacs&&window.selectedFacs.length<FAC_SELECTOR_DEFS.length){
    window.recomputeMapColors();
  }
},700);

})();
""",
        '</script></body></html>',
    ]
    return "\n".join(html_parts)


# =========================================================
# norm / 값 컬럼
# =========================================================
def get_value_col(metric_key: str, basis_key: str) -> str:
    if metric_key == "pop":      return "local_pop_map" if basis_key == "sgg" else "nat_pop_map"
    if metric_key == "coverage": return "avg_coverage"
    if metric_key == "mai":      return "avg_mai"
    return f"{basis_key}_{metric_key}_ratio"

def get_norm_for_group(group: gpd.GeoDataFrame, metric_key: str, basis_key: str):
    """norm 계산 (Local 스케일 고정 — scale_mode 제거됨)."""
    if metric_key == "coverage":
        return compute_continuous_norm(group.get("avg_coverage", pd.Series([], dtype=float)), gamma=0.6)[2]
    if metric_key == "mai":
        return compute_continuous_norm(group.get("avg_mai", pd.Series([], dtype=float)), gamma=0.6)[2]
    if metric_key == "pop":
        return compute_group_pop_norm(group.get("local_pop_map" if basis_key == "sgg" else "nat_pop_map", pd.Series([], dtype=float)), share_mode=True)[2]
    vals = pd.concat([pd.to_numeric(group.get(f"{basis_key}_{k}_ratio", pd.Series()), errors="coerce") for k in DEFICIT_KEYS], axis=0)
    return compute_group_norm_from_series(vals, gamma=0.55, force_zero_min=True)[2]


def render_metric_maps(
    map_prefix: str,
    group_gdf: gpd.GeoDataFrame,
    aggregate_gdf: gpd.GeoDataFrame,
    selected_metrics: List[str],
    basis_key: str,
    station_gdf, subway_gdf, fac_gdf,
    initial_center: Tuple[float, float],
    initial_zoom: int = 11,
    click_source_gdf: Optional[gpd.GeoDataFrame] = None,
    selected_sgg_code: str = "",
    compare_partner_gdf: Optional[gpd.GeoDataFrame] = None,
    cell_df: Optional[pd.DataFrame] = None,
    fac_access_df: Optional[pd.DataFrame] = None,
):
    if "fac_visible" not in st.session_state:
        st.session_state["fac_visible"] = list(FACILITY_ORDER)

    # basis_key는 sidebar에서 localStorage로 JS에 전달되므로 여기선 cell_data 용도로만 사용

    # ── cell data JSON ────────────────────────────────────
    cell_data_json = "{}"
    if cell_df is not None and not cell_df.empty and selected_sgg_code:
        cell_data_json = get_cell_data_json(
            str(selected_sgg_code), cell_df, group_gdf
        )
        # 디버그: 셀 데이터 건수 확인
        try:
            _cd_count = len(json.loads(cell_data_json))
            if _cd_count == 0:
                _has_col = SGG_CODE_COL in cell_df.columns
                _sample_codes = list(cell_df[SGG_CODE_COL].apply(normalize_sgg_code).unique()[:3]) if _has_col else []
                st.warning(f"⚠️ cellData 비어있음: sgg_code={selected_sgg_code}, has_SGG_COL={_has_col}, sample_codes={_sample_codes}")
        except Exception: pass

    # ── facility access JSON ─────────────────────────
    fac_access_json = "{}"
    if fac_access_df is not None and not fac_access_df.empty:
        valid_ids = set(group_gdf[GRID_JOIN_COL].astype(str).str.strip())
        fac_sub   = fac_access_df[fac_access_df[GRID_JOIN_COL].isin(valid_ids)].copy()
        if not fac_sub.empty:
            new_cov_cols = [c for c in fac_sub.columns if c.startswith("cov_")]
            new_mai_cols = [c for c in fac_sub.columns if c.startswith("mai_") and c not in ("mai_is_tie","mai_best_to_id")]
            legacy_cols  = [c for c in fac_sub.columns if c.startswith("pt_")]
            keep_cols    = [GRID_JOIN_COL] + new_cov_cols + new_mai_cols + legacy_cols
            if "mai_is_tie"     in fac_sub.columns: keep_cols.append("mai_is_tie")
            if "mai_best_to_id" in fac_sub.columns: keep_cols.append("mai_best_to_id")
            fac_sub = fac_sub[[c for c in keep_cols if c in fac_sub.columns]].copy()
            # int8/int16 컬럼 → int, str 컬럼 유지
            for c in fac_sub.columns:
                if c in (GRID_JOIN_COL, "mai_best_to_id"): continue
                if fac_sub[c].dtype.kind in ("i", "u", "b"):
                    fac_sub[c] = fac_sub[c].astype(int)
            fac_sub[GRID_JOIN_COL] = fac_sub[GRID_JOIN_COL].astype(str)
            records = fac_sub.where(fac_sub.notna(), other=None).to_dict(orient="records")
            fac_dict = {r[GRID_JOIN_COL]: {k: v for k, v in r.items() if k != GRID_JOIN_COL} for r in records}
            fac_access_json = json.dumps(fac_dict)

    # 항상 3개 맵 embed → 2행 고정
    rows     = 2  # 4 maps: 2×2
    iframe_h = (MAP_HEIGHT + 42 + 26) * rows + 16 + 34  # 34 = layer bar

    # cache version: neighbors.json mtime 기반으로 캐시 무효화
    _cv = ""
    if CACHE_NEIGHBORS.exists():
        _cv = str(int(CACHE_NEIGHBORS.stat().st_mtime))

    html = build_multi_map_html(
        sgg_code=str(selected_sgg_code),
        height_px=MAP_HEIGHT,
        deficit_colors_json=json.dumps(DEFICIT_COLORS),
        _cache_ver=_cv,
    )
    _clat, _clng, _czoom = initial_center[0], initial_center[1], initial_zoom

    # html_out: cell_data + fac_access + initial setView만 포함
    # selected_metrics(layer 선택)는 html_out에 포함하지 않음
    # → layer 바뀌어도 html_out 동일 → iframe 재마운트 없음 → 뷰포트 유지
    _setview = (
        f'(function(){{try{{'
        f'allMaps.forEach(function(m){{'
        f'try{{m.setView([{_clat},{_clng}],{_czoom},{{animate:false}});}}catch(e){{}}'
        f'}})}}catch(e){{}}}})()'  
    )
    html_out = html.replace(
        '</body></html>',
        f'<script>'
        f'window.cellData={cell_data_json};'
        f'window.facAccessData={fac_access_json};'
        f'{_setview}'
        f'</script></body></html>'
    )
    st_components.html(html_out, height=iframe_h, scrolling=False)



# =========================================================
# 앱 시작
# =========================================================
st.set_page_config(page_title="PT Deficit Dashboard", layout="wide")
st.markdown(
    '<h1 style="font-size:20px;font-weight:700;color:#1a1a1a;margin-bottom:2px;">'
    'PT Accessibility Deficit Dashboard</h1>'
    '<p style="font-size:12px;color:#999;margin-bottom:14px;">Grid-level public transit accessibility analysis</p>',
    unsafe_allow_html=True,
)

required_paths = [CLASSIFIED_PATH, GRID_PATH, STATION_PATH, SUBWAY_PATH, FAC_PATH]
missing_paths  = [str(p) for p in required_paths if not p.exists()]
if missing_paths:
    st.error("Required input files missing:\n\n" + "\n".join(missing_paths))
    st.stop()


def _dropbox_safe_clear_geojson_dir():
    import os, time
    if not CACHE_GEOJSON_DIR.exists():
        CACHE_GEOJSON_DIR.mkdir(parents=True, exist_ok=True); return
    for f in list(CACHE_GEOJSON_DIR.iterdir()):
        if f.is_file():
            for attempt in range(6):
                try: f.unlink(missing_ok=True); break
                except PermissionError: time.sleep(0.4 * (attempt + 1))
    CACHE_GEOJSON_DIR.mkdir(parents=True, exist_ok=True)


def _run_build_with_progress():
    _dropbox_safe_clear_geojson_dir()
    for fn in [load_cached_data, load_cell_detail_data, load_facility_access_data]:
        fn.clear()
    try: build_multi_map_html.clear()
    except: pass
    status = st.empty(); pbar = st.progress(0); stxt = st.empty()
    status.info("⚙️ Building cache...")
    pbar.progress(5)
    def _pcb(step, total, msg):
        pbar.progress(min(int(10 + 85 * step / max(total, 1)), 95))
        stxt.caption("🗺️ " + msg)
    build_dashboard_cache(progress_cb=_pcb)
    pbar.progress(100); stxt.empty()
    status.success("✅ Cache build complete!")


with st.sidebar:
    st.header("Setup")
    if st.button("Build / refresh cached data", use_container_width=True, key="btn_refresh_cache"):
        _run_build_with_progress()
        st.rerun()

_tiles_ok   = CACHE_GEOJSON_DIR.exists() and any(CACHE_GEOJSON_DIR.glob("grid_*.json"))
cache_ready = _tiles_ok and all(p.exists() for p in [CACHE_GRID, CACHE_SGG, CACHE_STATION, CACHE_SUBWAY, CACHE_FAC, CACHE_TS, CACHE_IDX, CACHE_NEIGHBORS])
if not cache_ready:
    st.info("⚙️ First run — building cache (this may take a few minutes)...")
    _pbar = st.progress(0); _stxt = st.empty()
    def _auto_pcb(step, total, msg):
        _pbar.progress(int(100 * step / max(total, 1)))
        _stxt.caption("🗺️ " + msg)
    build_dashboard_cache(progress_cb=_auto_pcb)
    try: build_multi_map_html.clear()
    except: pass
    _pbar.progress(100); _stxt.empty()
    st.rerun()

with st.spinner("Loading data..."):
    grid_gdf, grid_simple_gdf, sgg_gdf, station_gdf, subway_gdf, fac_gdf, ts_df, idx_df = load_cached_data()
    cell_df      = load_cell_detail_data()
    fac_access_df = load_facility_access_data()

# sgg_avg merge fallback
if not cell_df.empty and "sgg_avg_coverage" not in cell_df.columns:
    if "sgg_avg_coverage" in grid_gdf.columns:
        sgg_ref  = grid_gdf[[GRID_JOIN_COL, "sgg_avg_coverage", "sgg_avg_mai"]].copy()
        cell_df  = cell_df.merge(sgg_ref, on=GRID_JOIN_COL, how="left")
if not cell_df.empty and SGG_CODE_COL not in cell_df.columns:
    if SGG_CODE_COL in grid_gdf.columns:
        sgg_code_ref = grid_gdf[[GRID_JOIN_COL, SGG_CODE_COL]].copy()
        cell_df      = cell_df.merge(sgg_code_ref, on=GRID_JOIN_COL, how="left")


sgg_options    = sorted(grid_gdf[[SGG_CODE_COL, SGG_NAME_COL]].drop_duplicates().itertuples(index=False, name=None), key=lambda x: x[1])
# code를 항상 정규화된 문자열로 저장
sgg_name_to_code = {name: normalize_sgg_code(code) for code, name in sgg_options}

def _sido(n): return n.split("_")[0] if "_" in n else n
def _sgg(n):  return n.split("_",1)[1] if "_" in n else n

sido_list    = sorted(set(_sido(n) for n in sgg_name_to_code))
sido_to_names = {}
for name in sgg_name_to_code:
    sido_to_names.setdefault(_sido(name), []).append(name)
for k in sido_to_names:
    sido_to_names[k] = sorted(sido_to_names[k], key=_sgg)

def sgg_selector(prefix, la="Province", lb="Municipality", default_sido=None, default_sgg=None):
    sido_idx = sido_list.index(default_sido) if default_sido and default_sido in sido_list else 0
    sido_sel = st.selectbox(la, sido_list, index=sido_idx, key=prefix + "_sido")
    names_in = sido_to_names.get(sido_sel, list(sgg_name_to_code))
    disp     = [_sgg(n) for n in names_in]
    sgg_idx  = disp.index(default_sgg) if default_sgg and default_sgg in disp else 0
    sgg_disp = st.selectbox(lb, disp, index=sgg_idx, key=prefix + "_sgg")
    full_key  = sido_sel + "_" + sgg_disp   # 데이터 조회용 (원본 키)
    full_disp = sido_sel + " " + sgg_disp   # 표시용
    return full_disp, sgg_name_to_code.get(full_key)


# ── 사이드바 ──────────────────────────────────────────
with st.sidebar:
    st.header("Display")
    compare_mode = st.toggle("Compare two municipalities", value=False, key="toggle_compare")

    if not compare_mode:
        st.markdown("**Municipality**")
        selected_full, selected_code = sgg_selector("single")
        # single 선택값을 session_state에 저장 (compare 모드 전환 시 A로 사용)
        st.session_state["_last_sido"] = selected_full.split(" ")[0] if selected_full else None
        st.session_state["_last_sgg"]  = " ".join(selected_full.split(" ")[1:]) if selected_full else None
    else:
        # compare 모드 첫 진입 시: A=이전 선택, B=서울 성동구
        # session_state에 cmp_a_sido 없을 때만 default 반영
        _prev_sido = st.session_state.get("_last_sido")
        _prev_sgg  = st.session_state.get("_last_sgg")
        _cmp_first = "cmp_a_sido" not in st.session_state
        st.markdown("**Municipality A**")
        full1, code1 = sgg_selector("cmp_a", "Province A", "City/County A",
                                    default_sido=_prev_sido if _cmp_first else None,
                                    default_sgg=_prev_sgg if _cmp_first else None)
        # B: 서울 성동구 기본값 (첫 진입 시만)
        _cmp_b_first = "cmp_b_sido" not in st.session_state
        # 서울 sido key 탐색 (데이터마다 "서울" or "서울특별시" 등 다를 수 있음)
        _seoul_sido = next((s for s in sido_list if "서울" in s), None)
        _seoul_sgg  = "성동구"
        st.markdown("**Municipality B**")
        full2, code2 = sgg_selector("cmp_b", "Province B", "City/County B",
                                    default_sido=_seoul_sido if _cmp_b_first else None,
                                    default_sgg=_seoul_sgg if _cmp_b_first else None)

    # ── 시설 종류 선택 ──────────────────────────────────
    st.markdown("---")
    st.markdown("**Multi-activity bundle**")
    _fac_opts  = {d["id"]: d["label"] for d in FAC_SELECTOR_DEFS}
    # 데이터에 실제 존재하는 시설만 선택 가능하게 (facAccessData 컬럼 확인)
    # → 12종 전부 표시 (데이터 없는 시설은 선택해도 효과 없음)
    _fac_avail = [d["id"] for d in FAC_SELECTOR_DEFS]

    _default_sel = [f for f in FAC_DEFAULT_SEL if f in _fac_avail]

    # ── HTML checkbox panel (모든 옵션 한눈에 보이고 토글) ──
    _fac_items_html = ""
    for fid in _fac_avail:
        _lbl = _fac_opts.get(fid, fid)
        _is_default = fid in _default_sel
        _def_attr = ' data-default="1"' if _is_default else ' data-default="0"'
        _chk = " checked" if _is_default else ""
        _fac_items_html += (
            f'<label class="fp-item">'
            f'<input type="checkbox" value="{fid}"{_def_attr}{_chk}>'
            f'<span class="fp-label">{_lbl}</span>'
            f'</label>'
        )
    _n_fac = len(_fac_avail)
    _fac_panel_h = 38 + 26 * ((_n_fac + 1) // 2) + 8  # 2-col grid height estimate
    st_components.html(
        '<style>'
        '.fp-wrap{font-family:Inter,system-ui,sans-serif;font-size:12px;}'
        '.fp-btns{display:flex;gap:4px;margin-bottom:6px;}'
        '.fp-btns button{font-size:10px;padding:2px 8px;border:1px solid #ddd;border-radius:4px;'
        'background:#fafafa;color:#555;cursor:pointer;transition:all .15s;}'
        '.fp-btns button:hover{background:#eee;border-color:#bbb;}'
        '.fp-grid{display:grid;grid-template-columns:1fr 1fr;gap:2px 8px;}'
        '.fp-item{display:flex;align-items:center;gap:4px;padding:3px 4px;border-radius:4px;cursor:pointer;'
        'transition:background .1s;}'
        '.fp-item:hover{background:#f5f5f5;}'
        '.fp-item input{accent-color:#5C6BC0;margin:0;cursor:pointer;}'
        '.fp-label{color:#444;font-size:11.5px;user-select:none;white-space:nowrap;}'
        '.fp-item input:checked+.fp-label{color:#1a1a1a;font-weight:600;}'
        '.fp-note{font-size:9.5px;color:#999;margin-top:6px;line-height:1.3;}'
        '</style>'
        '<div class="fp-wrap">'
        '<div class="fp-btns">'
        '<button id="fp-btn-all">All</button>'
        '<button id="fp-btn-def">Default</button>'
        '<button id="fp-btn-clr">Clear</button>'
        '</div>'
        '<div class="fp-grid">'
        + _fac_items_html +
        '</div>'
        '<div class="fp-note">Coverage &amp; MAI map colors only. Cluster View &amp; MV Geary use the default set.</div>'
        '</div>'
        '<script>'
        'function _fpSync(){'
        '  var cbs=document.querySelectorAll(".fp-item input[type=checkbox]");'
        '  var sel=[];cbs.forEach(function(c){if(c.checked)sel.push(c.value);});'
        '  if(!sel.length){cbs.forEach(function(c){if(c.dataset.default==="1"){c.checked=true;sel.push(c.value);}});}'
        '  var v=JSON.stringify(sel);'
        '  try{localStorage.setItem("ffSelectedFacs",v);}catch(e){}'
        '  try{window.parent.localStorage.setItem("ffSelectedFacs",v);}catch(e){}'
        '}'
        'function _fpAll(){document.querySelectorAll(".fp-item input").forEach(function(c){c.checked=true;});_fpSync();}'
        'function _fpDefault(){document.querySelectorAll(".fp-item input").forEach(function(c){c.checked=c.dataset.default==="1";});_fpSync();}'
        'function _fpNone(){document.querySelectorAll(".fp-item input").forEach(function(c){c.checked=false;});_fpSync();}'
        'document.getElementById("fp-btn-all").addEventListener("click",_fpAll);'
        'document.getElementById("fp-btn-def").addEventListener("click",_fpDefault);'
        'document.getElementById("fp-btn-clr").addEventListener("click",_fpNone);'
        'document.querySelectorAll(".fp-item input").forEach(function(c){c.addEventListener("change",_fpSync);});'
        '_fpSync();'
        '</script>',
        height=_fac_panel_h,
    )
    # Python 쪽 selected_facs는 default로 유지 (실제 선택은 JS→localStorage→iframe polling)
    selected_facs = _default_sel

    basis = "sgg"  # JS가 window.deficitBasis로 관리

    # Layer 선택은 지도 iframe 내부 layer bar 체크박스로만 처리
    # (Python rerun 없이 JS로 즉시 적용 → 뷰포트 유지 + 빠른 반응)
    selected_metric_keys = ALL_MAP_KEYS  # 9개 embed (JCL은 JS로 show/hide)

    st.markdown("---")
    st.caption("Stations, subway lines, and facility points shown on each map.")


# ── 메인 맵 렌더링 ────────────────────────────────────
def _render(code, full_name, compare_partner_gdf=None, metric_keys=None):
    # code 정규화: int/float/str 모두 처리
    code = normalize_sgg_code(code)
    # SGG_CODE_COL도 정규화해서 비교
    def _match(col_series): return col_series.apply(normalize_sgg_code) == code
    group        = grid_simple_gdf[_match(grid_simple_gdf[SGG_CODE_COL])].copy()
    group_detail = grid_gdf[_match(grid_gdf[SGG_CODE_COL])].copy()
    sgg_group    = sgg_gdf[_match(sgg_gdf[SGG_CODE_COL])].copy()
    # 디버그: 격자 없으면 경고 표시
    _gpath = CACHE_GEOJSON_DIR / f"grid_{code}.json"
    if not _gpath.exists():
        import glob as _glob
        _all = list(CACHE_GEOJSON_DIR.glob("grid_*.json"))
        _keys = [p.stem.replace("grid_","") for p in _all[:10]]
        st.warning(f"⚠️ grid_{code}.json 없음. group rows={len(group)}. 캐시 키 샘플: {_keys[:5]}")
    center       = group_detail.to_crs(WEB_CRS).geometry.centroid.unary_union.centroid
    _selected_metrics = metric_keys if metric_keys else BASE_MAP_KEYS
    render_metric_maps(
        map_prefix=f"map_{code}",
        group_gdf=group, aggregate_gdf=sgg_group,
        selected_metrics=_selected_metrics,
        basis_key=basis, station_gdf=station_gdf, subway_gdf=subway_gdf, fac_gdf=fac_gdf,
        initial_center=(center.y, center.x), initial_zoom=11,
        click_source_gdf=group_detail,
        selected_sgg_code=str(code),
        compare_partner_gdf=compare_partner_gdf,
        cell_df=cell_df,
        fac_access_df=fac_access_df,
    )

def _clean_title(t): return t.replace('_', ' ') if t else t

if not compare_mode:
    if not selected_code: st.warning("Municipality not found."); st.stop()
    _render(selected_code, selected_full, metric_keys=selected_metric_keys)
else:
    if not code1 or not code2: st.warning("Municipality not found."); st.stop()
    group1 = grid_simple_gdf[grid_simple_gdf[SGG_CODE_COL] == code1].copy()
    group2 = grid_simple_gdf[grid_simple_gdf[SGG_CODE_COL] == code2].copy()
    c1, c2 = st.columns(2)
    with c1:
        _render(code1, full1, compare_partner_gdf=group2, metric_keys=selected_metric_keys)
    with c2:
        _render(code2, full2, compare_partner_gdf=group1, metric_keys=selected_metric_keys)
