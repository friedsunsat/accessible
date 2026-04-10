"""
터미널 간 대중교통 경로 분석 (r5py detailed itineraries)
=========================================================
- Origin / Destination: 모든 버스터미널 + 철도역 (자기자신 제외)
- 경로 필터링:
  1) Origin 터미널에서 도보로 먼 다른 터미널까지 걸어간 뒤 대중교통을
     타는 경로는 제외 (origin 터미널을 실제로 이용하지 않는 OD)
  2) OD별 최적 경로 4개만 유지:
     a. 최소 환승  b. 최소 비용  c. 최소 대기시간 합  d. 최단 소요시간
- 최적화: vectorized pandas, 병렬 청크 처리
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import sys
import time
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

# ── r5py ────────────────────────────────────────────────────
try:
    import r5py
except ImportError:
    sys.exit("r5py 가 설치되어 있지 않습니다. `pip install r5py` 를 실행하세요.")


# =========================================================
# 0. 설정
# =========================================================
DEFAULT_CRS = "EPSG:4326"

# 첫 번째 대중교통 leg 출발지와 origin 간 최대 허용 직선거리 (m)
# 이 거리를 초과하면 "origin 터미널을 이용하지 않는 경로" 로 판정하여 제외
ORIGIN_WALK_THRESHOLD_M = 800

# 기본 요금 테이블 (원) ── 필요에 따라 수정
FARE_TABLE = {
    "WALK": 0,
    "BICYCLE": 0,
    "BUS": 1_400,
    "RAIL": 1_400,      # 일반 철도 / 지하철
    "SUBWAY": 1_350,
    "TRAM": 1_250,
    "FERRY": 1_000,
    "CABLE_CAR": 0,
    "GONDOLA": 0,
    "FUNICULAR": 0,
}
TRANSFER_DISCOUNT = 0  # 환승 할인 (동일 수단 간) ── 정책에 맞게 조정

# r5py 라우팅 파라미터
DEPARTURE_TIME = dt.datetime(2025, 4, 15, 8, 0)  # 출발 시각
MAX_TRIP_DURATION = dt.timedelta(hours=4)
MAX_WALK_DISTANCE = 2_000       # 최대 도보 거리 (m)
MAX_RIDES = 8                    # 최대 탑승 횟수

# 청크 크기 (한 번에 라우팅할 OD 쌍 수) ── 메모리/속도 트레이드오프
CHUNK_SIZE = 200


# =========================================================
# 1. 터미널 데이터 로드
# =========================================================
def load_terminals(
    terminal_path: str | Path,
    id_col: str = "id",
    type_col: Optional[str] = "type",
) -> gpd.GeoDataFrame:
    """
    터미널(버스/철도) GeoDataFrame 로드.
    필수: geometry (Point), id 컬럼.
    선택: type 컬럼 ('bus_terminal', 'rail_station' 등).

    지원 포맷: gpkg, geojson, shp, geoparquet
    """
    p = Path(terminal_path)
    if p.suffix in (".parquet", ".geoparquet"):
        gdf = gpd.read_parquet(p)
    else:
        gdf = gpd.read_file(p)

    # id 컬럼 표준화
    if id_col not in gdf.columns:
        # 자동 탐색
        candidates = [c for c in gdf.columns if c.lower() in ("id", "stop_id", "station_id", "terminal_id", "name")]
        if candidates:
            gdf = gdf.rename(columns={candidates[0]: "id"})
        else:
            gdf["id"] = gdf.index.astype(str)
    else:
        gdf = gdf.rename(columns={id_col: "id"})

    gdf["id"] = gdf["id"].astype(str)

    # CRS → WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs(DEFAULT_CRS)
    if str(gdf.crs) != DEFAULT_CRS:
        gdf = gdf.to_crs(DEFAULT_CRS)

    return gdf


# =========================================================
# 2. OD 쌍 생성 (자기자신 제외)
# =========================================================
def build_od_pairs(terminals: gpd.GeoDataFrame) -> pd.DataFrame:
    """모든 터미널 간 OD 쌍 (self-loop 제외)."""
    ids = terminals["id"].unique()
    pairs = [(o, d) for o, d in itertools.product(ids, ids) if o != d]
    return pd.DataFrame(pairs, columns=["from_id", "to_id"])


# =========================================================
# 3. r5py Transport Network 구축
# =========================================================
def build_network(
    osm_pbf: str | Path,
    gtfs_paths: list[str | Path],
) -> r5py.TransportNetwork:
    """OSM PBF + GTFS zip 으로 r5py TransportNetwork 구축."""
    print(f"[네트워크 구축] OSM: {osm_pbf}")
    for g in gtfs_paths:
        print(f"  GTFS: {g}")
    tn = r5py.TransportNetwork(
        osm_pbf=str(osm_pbf),
        gtfs=[str(g) for g in gtfs_paths],
    )
    print("[네트워크 구축 완료]")
    return tn


# =========================================================
# 4. Detailed Itineraries 계산 (청크 단위)
# =========================================================
def compute_itineraries_chunk(
    network: r5py.TransportNetwork,
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame,
    departure: dt.datetime,
) -> pd.DataFrame:
    """
    origins → destinations 에 대해 r5py DetailedItinerariesComputer 실행.
    반환: legs DataFrame (option, leg, mode, travel_time, wait_time, distance, geometry 등).
    """
    computer = r5py.DetailedItinerariesComputer(
        transport_network=network,
        origins=origins,
        destinations=destinations,
        departure=departure,
        transport_modes=[r5py.TransportMode.TRANSIT],
        access_modes=[r5py.TransportMode.WALK],
        egress_modes=[r5py.TransportMode.WALK],
        max_time=MAX_TRIP_DURATION,
        max_public_transport_rides=MAX_RIDES,
        max_time_walking=dt.timedelta(seconds=int(MAX_WALK_DISTANCE / 1.2)),
    )
    result = computer.request()
    return result


# =========================================================
# 5. 경로별 지표 집계
# =========================================================
def aggregate_itinerary_metrics(legs_df: pd.DataFrame) -> pd.DataFrame:
    """
    legs 단위 → itinerary(option) 단위 집계.
    컬럼: from_id, to_id, option, n_transfers, total_time_min,
          total_wait_min, total_cost, first_transit_distance_from_origin_m
    """
    if legs_df.empty:
        return pd.DataFrame(columns=[
            "from_id", "to_id", "option",
            "n_transfers", "total_time_min", "total_wait_min",
            "total_cost", "first_transit_geom",
        ])

    df = legs_df.copy()

    # ── 컬럼 이름 표준화 (r5py 버전별 차이 대응) ──
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "from_id" in cl or cl == "origin":
            col_map[c] = "from_id"
        elif "to_id" in cl or cl == "destination":
            col_map[c] = "to_id"
        elif cl in ("option", "itinerary", "alternative"):
            col_map[c] = "option"
        elif cl in ("segment", "leg"):
            col_map[c] = "leg"
        elif cl == "transport_mode" or cl == "mode":
            col_map[c] = "mode"
        elif "travel_time" in cl or cl == "duration":
            col_map[c] = "travel_time"
        elif "wait" in cl:
            col_map[c] = "wait_time"
        elif "distance" in cl:
            col_map[c] = "distance"

    df = df.rename(columns=col_map)

    # 필수 컬럼 확인
    for req in ["from_id", "to_id", "option"]:
        if req not in df.columns:
            raise KeyError(f"legs DataFrame 에 '{req}' 컬럼이 없습니다. 컬럼: {list(df.columns)}")

    # mode 문자열 표준화
    if "mode" in df.columns:
        df["mode"] = df["mode"].astype(str).str.upper().str.strip()
    else:
        df["mode"] = "UNKNOWN"

    # travel_time → 분 단위 숫자
    if "travel_time" in df.columns:
        tt = df["travel_time"]
        if pd.api.types.is_timedelta64_dtype(tt):
            df["travel_time_min"] = tt.dt.total_seconds() / 60
        else:
            df["travel_time_min"] = pd.to_numeric(tt, errors="coerce").fillna(0)
    else:
        df["travel_time_min"] = 0

    # wait_time → 분 단위 숫자
    if "wait_time" in df.columns:
        wt = df["wait_time"]
        if pd.api.types.is_timedelta64_dtype(wt):
            df["wait_time_min"] = wt.dt.total_seconds() / 60
        else:
            df["wait_time_min"] = pd.to_numeric(wt, errors="coerce").fillna(0)
    else:
        df["wait_time_min"] = 0

    # distance → 미터
    if "distance" not in df.columns:
        df["distance"] = 0.0

    # ── 그룹 키 ──
    grp = ["from_id", "to_id", "option"]

    # transit legs (WALK, BICYCLE 제외)
    walk_modes = {"WALK", "BICYCLE", "TRANSFERMODE.WALK", "TRANSITMODE.WALK",
                  "LEGMODE.WALK", "LEGMODE.BICYCLE", "ACCESSMODE.WALK",
                  "EGRESSMODE.WALK"}
    is_transit = ~df["mode"].isin(walk_modes)

    # n_transfers = (대중교통 leg 수 - 1), 최소 0
    n_transit_legs = df[is_transit].groupby(grp, sort=False).size().reset_index(name="_n_transit")
    n_transit_legs["n_transfers"] = (n_transit_legs["_n_transit"] - 1).clip(lower=0)

    # total_time_min (전체 여행 시간 = 모든 leg travel_time + wait_time 합)
    agg_time = df.groupby(grp, sort=False).agg(
        total_time_min=("travel_time_min", "sum"),
        total_wait_min=("wait_time_min", "sum"),
    ).reset_index()

    # 비용 계산
    transit_df = df[is_transit].copy()
    transit_df["leg_fare"] = transit_df["mode"].map(FARE_TABLE).fillna(0)
    fare_agg = transit_df.groupby(grp, sort=False)["leg_fare"].sum().reset_index(name="total_cost")

    # 첫 번째 대중교통 leg의 geometry (origin 필터용)
    first_transit = (
        df[is_transit]
        .sort_values(["from_id", "to_id", "option", "leg"] if "leg" in df.columns else grp)
        .groupby(grp, sort=False)
        .first()
        .reset_index()
    )
    first_transit_geom = first_transit[grp + (["geometry"] if "geometry" in first_transit.columns else [])].copy()
    if "geometry" in first_transit_geom.columns:
        first_transit_geom = first_transit_geom.rename(columns={"geometry": "first_transit_geom"})
    else:
        first_transit_geom["first_transit_geom"] = None

    # ── 병합 ──
    result = agg_time.merge(n_transit_legs[grp + ["n_transfers"]], on=grp, how="left")
    result = result.merge(fare_agg, on=grp, how="left")
    result = result.merge(first_transit_geom[grp + ["first_transit_geom"]], on=grp, how="left")

    result["n_transfers"] = result["n_transfers"].fillna(0).astype(int)
    result["total_cost"] = result["total_cost"].fillna(0)
    result["total_time_min"] = result["total_time_min"] + result["total_wait_min"]

    return result


# =========================================================
# 6. Origin 터미널 실제 이용 여부 필터
# =========================================================
def filter_origin_usage(
    metrics: pd.DataFrame,
    terminals: gpd.GeoDataFrame,
    threshold_m: float = ORIGIN_WALK_THRESHOLD_M,
) -> pd.DataFrame:
    """
    첫 번째 대중교통 leg 출발 geometry 가 origin 터미널에서
    threshold_m 이내인 itinerary 만 유지.

    즉, origin 터미널 근처에서 대중교통을 탑승하지 않으면 제외.
    """
    if metrics.empty:
        return metrics

    if "first_transit_geom" not in metrics.columns or metrics["first_transit_geom"].isna().all():
        print("[경고] first_transit_geom 이 없어 origin 필터를 건너뜁니다.")
        return metrics

    # 터미널 좌표 딕셔너리 (id → Point in projected CRS)
    term_proj = terminals.to_crs("EPSG:5179").copy()
    term_points = dict(zip(term_proj["id"], term_proj.geometry))

    # first_transit_geom 을 projected CRS 로 변환
    from shapely.geometry import Point as ShapelyPoint
    from shapely.ops import transform as shapely_transform
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)

    def project_geom(geom):
        if geom is None or (hasattr(geom, "is_empty") and geom.is_empty):
            return None
        try:
            # line/point 의 첫 좌표
            if geom.geom_type == "Point":
                x, y = transformer.transform(geom.x, geom.y)
                return ShapelyPoint(x, y)
            else:
                # LineString 등 → 첫 좌표
                coords = list(geom.coords)
                x, y = transformer.transform(coords[0][0], coords[0][1])
                return ShapelyPoint(x, y)
        except Exception:
            return None

    metrics = metrics.copy()
    metrics["_first_pt"] = metrics["first_transit_geom"].apply(project_geom)

    def calc_dist(row):
        origin_pt = term_points.get(str(row["from_id"]))
        first_pt = row["_first_pt"]
        if origin_pt is None or first_pt is None:
            return np.inf
        return origin_pt.distance(first_pt)

    metrics["_dist_to_origin"] = metrics.apply(calc_dist, axis=1)
    filtered = metrics[metrics["_dist_to_origin"] <= threshold_m].copy()

    n_before = len(metrics)
    n_after = len(filtered)
    print(f"[Origin 필터] {n_before} → {n_after} itineraries "
          f"(제거: {n_before - n_after}, threshold={threshold_m}m)")

    filtered = filtered.drop(columns=["_first_pt", "_dist_to_origin"], errors="ignore")
    return filtered


# =========================================================
# 7. OD별 최적 경로 4개 선택
# =========================================================
def select_best_routes(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    OD 쌍별로 4개 최적 경로 선택:
      1. 최소 환승 (n_transfers → total_time_min tiebreak)
      2. 최소 비용 (total_cost → total_time_min tiebreak)
      3. 최소 대기시간 합 (total_wait_min → total_time_min tiebreak)
      4. 최단 시간 (total_time_min)
    중복 option 은 제거.
    """
    if metrics.empty:
        return metrics

    grp = ["from_id", "to_id"]
    results = []

    criteria = [
        ("min_transfer", ["n_transfers", "total_time_min"]),
        ("min_cost",     ["total_cost",    "total_time_min"]),
        ("min_wait",     ["total_wait_min","total_time_min"]),
        ("min_time",     ["total_time_min"]),
    ]

    for label, sort_cols in criteria:
        sorted_df = metrics.sort_values(grp + sort_cols)
        best = sorted_df.groupby(grp, sort=False).first().reset_index()
        best["route_type"] = label
        results.append(best)

    combined = pd.concat(results, ignore_index=True)
    # OD 내에서 같은 option 이 여러 기준으로 선택되면 → route_type 을 합쳐서 하나만 유지
    combined = (
        combined
        .groupby(grp + ["option"], sort=False)
        .agg({
            **{c: "first" for c in combined.columns if c not in grp + ["option", "route_type"]},
            "route_type": lambda x: "|".join(sorted(set(x))),
        })
        .reset_index()
    )

    print(f"[최적 경로 선택] {len(metrics)} → {len(combined)} itineraries")
    return combined


# =========================================================
# 8. 메인 파이프라인
# =========================================================
def run(
    terminal_path: str | Path,
    osm_pbf: str | Path,
    gtfs_paths: list[str | Path],
    output_path: str | Path = "terminal_od_routes.parquet",
    departure: dt.datetime = DEPARTURE_TIME,
    id_col: str = "id",
    chunk_size: int = CHUNK_SIZE,
    walk_threshold_m: float = ORIGIN_WALK_THRESHOLD_M,
):
    t0 = time.time()

    # 1) 터미널 로드
    print("=" * 60)
    print("[1/5] 터미널 데이터 로드")
    terminals = load_terminals(terminal_path, id_col=id_col)
    print(f"  터미널 수: {len(terminals)}")

    # 2) 네트워크 구축
    print("=" * 60)
    print("[2/5] 네트워크 구축")
    network = build_network(osm_pbf, gtfs_paths)

    # 3) OD 쌍 → 청크 단위 라우팅
    print("=" * 60)
    print("[3/5] 경로 계산 (Detailed Itineraries)")
    od_pairs = build_od_pairs(terminals)
    n_od = len(od_pairs)
    print(f"  총 OD 쌍: {n_od}")

    # unique origin 기준으로 청크 분할 (같은 origin 은 한 청크에)
    unique_origins = od_pairs["from_id"].unique()
    all_metrics = []

    # origin 단위로 분할하여 처리
    origin_chunks = [
        unique_origins[i : i + chunk_size]
        for i in range(0, len(unique_origins), chunk_size)
    ]

    for ci, origin_batch in enumerate(origin_chunks, 1):
        od_sub = od_pairs[od_pairs["from_id"].isin(origin_batch)]
        o_ids = od_sub["from_id"].unique()
        d_ids = od_sub["to_id"].unique()

        origins_gdf = terminals[terminals["id"].isin(o_ids)].copy()
        dests_gdf = terminals[terminals["id"].isin(d_ids)].copy()

        print(f"  청크 {ci}/{len(origin_chunks)}: "
              f"origins={len(origins_gdf)}, dests={len(dests_gdf)}, "
              f"OD={len(od_sub)}")

        try:
            legs = compute_itineraries_chunk(network, origins_gdf, dests_gdf, departure)
        except Exception as e:
            print(f"  [오류] 청크 {ci} 라우팅 실패: {e}")
            continue

        if legs.empty:
            print(f"  [정보] 청크 {ci} 결과 없음")
            continue

        chunk_metrics = aggregate_itinerary_metrics(legs)
        all_metrics.append(chunk_metrics)
        print(f"  → {len(chunk_metrics)} itineraries 집계 완료")

    if not all_metrics:
        print("[완료] 유효한 경로가 없습니다.")
        return pd.DataFrame()

    metrics = pd.concat(all_metrics, ignore_index=True)
    print(f"\n  전체 집계: {len(metrics)} itineraries")

    # 4) Origin 터미널 실제 이용 필터
    print("=" * 60)
    print("[4/5] Origin 터미널 이용 여부 필터")
    metrics = filter_origin_usage(metrics, terminals, threshold_m=walk_threshold_m)

    # 5) OD별 최적 경로 선택
    print("=" * 60)
    print("[5/5] OD별 최적 경로 선택 (4가지 기준)")
    best = select_best_routes(metrics)

    # geometry 컬럼 제거 후 저장
    drop_cols = [c for c in best.columns if "geom" in c.lower()]
    best = best.drop(columns=drop_cols, errors="ignore")

    output_path = Path(output_path)
    best.to_parquet(output_path, index=False)
    elapsed = time.time() - t0
    print("=" * 60)
    print(f"[완료] {len(best)} 경로 → {output_path}")
    print(f"  소요시간: {elapsed:.1f}초")

    return best


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="터미널 간 대중교통 경로 분석 (r5py detailed itineraries)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python terminal_routing.py \\
    --terminals terminals.gpkg \\
    --osm south-korea.osm.pbf \\
    --gtfs gtfs_bus.zip gtfs_rail.zip \\
    --output terminal_routes.parquet \\
    --departure "2025-04-15 08:00" \\
    --id-col stop_id \\
    --walk-threshold 800 \\
    --chunk-size 200
        """,
    )
    parser.add_argument("--terminals", required=True, help="터미널 GeoDataFrame 파일 (gpkg/geojson/shp/geoparquet)")
    parser.add_argument("--osm", required=True, help="OSM PBF 파일 경로")
    parser.add_argument("--gtfs", required=True, nargs="+", help="GTFS zip 파일 경로 (복수 가능)")
    parser.add_argument("--output", default="terminal_od_routes.parquet", help="출력 parquet 경로")
    parser.add_argument("--departure", default="2025-04-15 08:00", help="출발 시각 (YYYY-MM-DD HH:MM)")
    parser.add_argument("--id-col", default="id", help="터미널 ID 컬럼명")
    parser.add_argument("--walk-threshold", type=float, default=ORIGIN_WALK_THRESHOLD_M,
                        help=f"Origin 터미널 이용 판정 최대 거리 (m, 기본 {ORIGIN_WALK_THRESHOLD_M})")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"한 번에 라우팅할 origin 수 (기본 {CHUNK_SIZE})")

    args = parser.parse_args()

    departure = dt.datetime.strptime(args.departure, "%Y-%m-%d %H:%M")

    result = run(
        terminal_path=args.terminals,
        osm_pbf=args.osm,
        gtfs_paths=args.gtfs,
        output_path=args.output,
        departure=departure,
        id_col=args.id_col,
        chunk_size=args.chunk_size,
        walk_threshold_m=args.walk_threshold,
    )

    if not result.empty:
        print("\n[결과 미리보기]")
        print(result.head(10).to_string(index=False))
        print(f"\n[route_type 분포]")
        for rt, cnt in result["route_type"].value_counts().items():
            print(f"  {rt}: {cnt}")


if __name__ == "__main__":
    main()
