"""坐标系转换核心：WGS-84 ⇄ GCJ-02 ⇄ BD-09，纯函数，不含界面、不联网。

算法：
  WGS-84 → GCJ-02  开源社区流传的公开加偏公式（克拉索夫斯基椭球参数 + 正弦扰动项），
                   与第 2 天 scripts/station_district.py、本包 scripts/jingan_food_heat_folium.py 同源；
  GCJ-02 → BD-09   百度公开的 x_pi 公式；
  两个反算         公式没有解析逆，用迭代法：从目标点出发，正算一次、把残差（正算结果 − 目标）扣回去，
                   固定修正 5 轮。外滩实测往返误差逐轮 479 m → 2.2 m → 7 mm → 0.025 mm → 0.0001 mm → 0；
                   网上常见的「一步反算」就是只修正 1 轮，误差约 2 米。
约定：所有函数参数顺序都是 (经度 lng, 纬度 lat)，单位度。

对外入口 convert_point（单点）、convert_table（整表）都先校验、正常的才转换：
零值、经纬度写反、国外、海上、缺失这些问题点不转换，返回原因；整表里问题行照样保留、打上标记，不丢、不自动改。
"""
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape

A = 6378245.0                      # 克拉索夫斯基椭球长半轴（GCJ-02 公式用）
EE = 0.00669342162296594323         # 同椭球第一偏心率平方
X_PI = math.pi * 3000.0 / 180.0     # BD-09 公式常数
EARTH_R = 6371008.8                 # 地球平均半径（米），算偏移距离用

CRS = ("WGS-84", "GCJ-02", "BD-09")


# ---------- 基础工具 ----------
def out_of_china(lng, lat):
    """粗略矩形框；框外 GCJ-02 不加偏（与主流开源实现一致）。"""
    return not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271)


def normalize_crs(name):
    """'wgs84' / 'GCJ_02' / 'bd-09' 等写法统一成 CRS 里的标准名。"""
    key = str(name).upper().replace("-", "").replace("_", "").replace(" ", "")
    for crs in CRS:
        if key == crs.replace("-", ""):
            return crs
    raise ValueError(f"不认识的坐标系：{name!r}，可选 {', '.join(CRS)}")


def check_lnglat(lng, lat):
    if not (math.isfinite(lng) and math.isfinite(lat)):
        raise ValueError("经纬度必须是有限数字")
    if not -180.0 <= lng <= 180.0:
        raise ValueError(f"经度 {lng} 超出 [-180, 180]，是不是把经纬度填反了？")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"纬度 {lat} 超出 [-90, 90]，是不是把经纬度填反了？")


def haversine_m(lng1, lat1, lng2, lat2):
    """两点球面距离（米）。"""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(h))


INVERT_ROUNDS = 5                   # 反算的残差修正轮数


def _invert(forward, lng, lat, rounds=INVERT_ROUNDS):
    """迭代反算：找 p 使 forward(p) ≈ (lng, lat)。从 p = 目标点起步，每轮 p -= forward(p) − 目标。"""
    x, y = lng, lat
    for _ in range(rounds):
        fx, fy = forward(x, y)
        x, y = x - (fx - lng), y - (fy - lat)
    return x, y


# ---------- WGS-84 ⇄ GCJ-02 ----------
def _transform_lat(x, y):
    r = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    r += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    r += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    r += (160.0 * math.sin(y / 12.0 * math.pi) + 320.0 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return r


def _transform_lng(x, y):
    r = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    r += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    r += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    r += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return r


def wgs84_to_gcj02(lng, lat):
    if out_of_china(lng, lat):
        return lng, lat
    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    rad = lat / 180.0 * math.pi
    magic = 1 - EE * math.sin(rad) ** 2
    sqrt_magic = math.sqrt(magic)
    dlat = dlat * 180.0 / ((A * (1 - EE)) / (magic * sqrt_magic) * math.pi)
    dlng = dlng * 180.0 / (A / sqrt_magic * math.cos(rad) * math.pi)
    return lng + dlng, lat + dlat


def gcj02_to_wgs84(lng, lat):
    return _invert(wgs84_to_gcj02, lng, lat)


# ---------- GCJ-02 ⇄ BD-09 ----------
def gcj02_to_bd09(lng, lat):
    z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * X_PI)
    theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * X_PI)
    return z * math.cos(theta) + 0.0065, z * math.sin(theta) + 0.006


def bd09_to_gcj02(lng, lat):
    return _invert(gcj02_to_bd09, lng, lat)


# ---------- 经 GCJ-02 中转 ----------
def wgs84_to_bd09(lng, lat):
    return gcj02_to_bd09(*wgs84_to_gcj02(lng, lat))


def bd09_to_wgs84(lng, lat):
    return gcj02_to_wgs84(*bd09_to_gcj02(lng, lat))


_FUNCS = {
    ("WGS-84", "GCJ-02"): wgs84_to_gcj02,
    ("GCJ-02", "WGS-84"): gcj02_to_wgs84,
    ("GCJ-02", "BD-09"): gcj02_to_bd09,
    ("BD-09", "GCJ-02"): bd09_to_gcj02,
    ("WGS-84", "BD-09"): wgs84_to_bd09,
    ("BD-09", "WGS-84"): bd09_to_wgs84,
}


# ---------- 公式分派 ----------
def convert(lng, lat, src, dst):
    """任意两个坐标系之间套公式转换一个点，返回 (lng, lat)。
    只拦非数字和超出 ±180 / ±90，不判断零值、写反、国外；给用户的结果请走 convert_point / convert_table。"""
    lng, lat = float(lng), float(lat)
    check_lnglat(lng, lat)
    src, dst = normalize_crs(src), normalize_crs(dst)
    if src == dst:
        return lng, lat
    return _FUNCS[(src, dst)](lng, lat)


# ---------- 批量：转换 + 校验 ----------
# 省级边界：阿里云 DataV 数据（GCJ-02 坐标，从第 2 天数据包 data/geo 复制进来，应用可单独拷走）；
# 判断前先把点转成 GCJ-02
BOUNDARY_PATH = Path(__file__).resolve().parent / "data/中国省级行政区_DataV.geojson"

OK = "正常"
ISSUE_SWAPPED = "经纬度写反"
ISSUE_RANGE = "超出范围或零值"
ISSUE_ABROAD = "落在国外"
ISSUE_OFFLAND = "框内但不在省级区"
ISSUE_MISSING = "缺失或非数字"
ISSUES = (ISSUE_SWAPPED, ISSUE_RANGE, ISSUE_ABROAD, ISSUE_OFFLAND, ISSUE_MISSING)

OUT_COLS = ["wgs84_lng", "wgs84_lat", "gcj02_lng", "gcj02_lat", "bd09_lng", "bd09_lat",
            "gcj02_offset_m", "bd09_offset_m", "province", "check", "issue", "check_note"]


@lru_cache(maxsize=4)
def load_provinces(path=BOUNDARY_PATH):
    """读省级边界，去掉没有名字的九段线要素（adcode 100000_JD），返回 (省名, 几何, 空间索引)。"""
    with open(path, encoding="utf-8") as f:
        feats = [ft for ft in json.load(f)["features"] if ft["properties"].get("name")]
    names = np.array([ft["properties"]["name"] for ft in feats], dtype=object)
    geoms = np.array([shape(ft["geometry"]) for ft in feats])
    return names, geoms, shapely.STRtree(geoms)


def locate_province(gcj_lng, gcj_lat, boundary_path=BOUNDARY_PATH):
    """GCJ-02 坐标数组 → 所在省级区名数组，不在任何省级区内为 None。"""
    names, _, tree = load_provinces(boundary_path)
    out = np.full(len(gcj_lng), None, dtype=object)
    if len(out):
        i_pt, i_geom = tree.query(shapely.points(gcj_lng, gcj_lat), predicate="intersects")
        out[i_pt] = names[i_geom]
    return out


def nearest_province(gcj_lng, gcj_lat, boundary_path=BOUNDARY_PATH):
    """离一个 GCJ-02 点最近的省级区，返回 (省名, 距离米)。"""
    names, geoms, tree = load_provinces(boundary_path)
    pt = shapely.Point(gcj_lng, gcj_lat)
    i = tree.nearest(pt)
    q = shapely.shortest_line(geoms[i], pt).coords[0]
    return names[i], haversine_m(gcj_lng, gcj_lat, q[0], q[1])


def _to_all(xs, ys, src):
    """一批合法点 → (n, 6) 数组：WGS-84、GCJ-02、BD-09 各一对经纬度。"""
    out = np.empty((len(xs), 6))
    for k, (x, y) in enumerate(zip(xs, ys)):
        for j, crs in enumerate(CRS):
            out[k, 2 * j:2 * j + 2] = convert(x, y, src, crs)
    return out


def _show_raw(v):
    """原值写进说明：空的写「（空）」，其余原样。"""
    return "（空）" if pd.isna(v) or not str(v).strip() else str(v)


def check_and_convert(lng, lat, src, boundary_path=BOUNDARY_PATH):
    """一列经度、一列纬度逐行校验并转换，返回 OUT_COLS 那些列的表；行数、顺序与输入一致。

    每行只标一个最能说明问题的类型，判断顺序：
      缺失或非数字 → (0,0) → 在省级区内 = 正常 → 交换经纬度后在省级区内 = 写反
      → 超出 ±180/±90 → 在中国范围框外 = 国外 → 其余 = 框内但不在省级区（海上或邻国）。
    只有正常行转换；问题行坐标、偏移列留空，issue / check_note 说明原因。
    """
    src = normalize_crs(src)
    raw_x = pd.Series(lng).reset_index(drop=True)
    raw_y = pd.Series(lat).reset_index(drop=True)
    x = pd.to_numeric(raw_x, errors="coerce").to_numpy(float)
    y = pd.to_numeric(raw_y, errors="coerce").to_numpy(float)
    n = len(x)

    finite = np.isfinite(x) & np.isfinite(y)
    zero = finite & (x == 0) & (y == 0)
    in_range = finite & (np.abs(x) <= 180) & (np.abs(y) <= 90)
    usable = in_range & ~zero

    coords = np.full((n, 6), np.nan)
    prov = np.full(n, None, dtype=object)
    idx = np.flatnonzero(usable)
    coords[idx] = _to_all(x[idx], y[idx], src)
    prov[idx] = locate_province(coords[idx, 2], coords[idx, 3], boundary_path)

    # 不在省级区内的点，试一下交换经纬度能不能落进省级区
    swap_prov = np.full(n, None, dtype=object)
    idx = np.flatnonzero(finite & ~zero & pd.isna(prov) & (np.abs(y) <= 180) & (np.abs(x) <= 90))
    if idx.size:
        g = _to_all(y[idx], x[idx], src)
        swap_prov[idx] = locate_province(g[:, 2], g[:, 3], boundary_path)

    rows = []
    for i in range(n):
        keep, issue, note = False, None, ""
        if not finite[i]:
            issue, note = ISSUE_MISSING, f"原值 经度={_show_raw(raw_x[i])} 纬度={_show_raw(raw_y[i])}"
        elif zero[i]:
            issue, note = ISSUE_RANGE, "(0, 0)：多半是缺失值被填成了 0"
        elif prov[i]:
            keep = True
        elif swap_prov[i]:
            issue = ISSUE_SWAPPED
            note = f"交换后 ({y[i]}, {x[i]}) 落在{swap_prov[i]}，请核对后交换"
        elif not in_range[i]:
            issue, note = ISSUE_RANGE, "经度须在 ±180、纬度须在 ±90 以内"
        elif out_of_china(coords[i, 0], coords[i, 1]):
            issue = ISSUE_ABROAD
            note = "在中国范围框外：GCJ-02 / BD-09 只在国内地图上用，框外换算没有意义"
        else:
            issue = ISSUE_OFFLAND
            name, dist = nearest_province(coords[i, 2], coords[i, 3], boundary_path)
            note = f"在中国范围框内但不在任何省级区内（海上或邻国），最近的是{name}，约 {dist / 1000:.1f} km"

        rec = dict.fromkeys(OUT_COLS)
        if keep:
            c = coords[i]
            rec.update(zip(OUT_COLS[:6], np.round(c, 7)))
            rec["gcj02_offset_m"] = round(haversine_m(c[0], c[1], c[2], c[3]), 2)
            rec["bd09_offset_m"] = round(haversine_m(c[0], c[1], c[4], c[5]), 2)
        rec.update(province=prov[i], check="问题" if issue else OK, issue=issue, check_note=note)
        rows.append(rec)
    return pd.DataFrame(rows, columns=OUT_COLS)


# ---------- 对外入口：单点、整表（都先校验，正常的才转换） ----------
def convert_point(lng, lat, src, dst, boundary_path=BOUNDARY_PATH):
    """单点转换，规则和整表一样。返回 dict：
      src、dst、lng、lat、offset_m（转换前后两组数字画在同一张图上相差多少米）、
      check（正常 / 问题）、issue（问题类型）、note（原因）、province。
    问题点不转换：lng、lat、offset_m 为 None，看 issue 和 note。"""
    src, dst = normalize_crs(src), normalize_crs(dst)
    r = check_and_convert([lng], [lat], src, boundary_path).iloc[0]
    out = {"src": src, "dst": dst, "lng": None, "lat": None, "offset_m": None, "check": r["check"],
           "issue": r["issue"], "note": r["check_note"], "province": r["province"]}
    if r["check"] == OK:
        x, y = float(lng), float(lat)
        out["lng"], out["lat"] = convert(x, y, src, dst)
        out["offset_m"] = haversine_m(x, y, out["lng"], out["lat"])
    return out


def convert_table(df, lng_col, lat_col, src, boundary_path=BOUNDARY_PATH):
    """原表 + 三套坐标 + 偏移米数 + 校验列。原始列、行数、行序都不动。"""
    clash = [c for c in OUT_COLS if c in df.columns]
    if clash:
        raise ValueError(f"原表已有同名列 {clash}，先改名再转换")
    res = check_and_convert(df[lng_col], df[lat_col], src, boundary_path)
    res.index = df.index
    return pd.concat([df, res], axis=1)


def read_csv(path_or_buffer):
    """所有列按文本读（原样保留，比如前导 0、空值不变 NaN）；先试 UTF-8，不行再试 GBK。"""
    for enc in ("utf-8-sig", "gbk"):
        try:
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
            return pd.read_csv(path_or_buffer, dtype=str, encoding=enc, keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("读不出来：既不是 UTF-8 也不是 GBK 编码")


def guess_columns(columns):
    """按列名猜 (经度列, 纬度列, 源坐标系)，猜不到的为 None。"""
    def find(keys):
        return next((c for c in columns if any(k in str(c).lower() for k in keys)), None)

    lng_col, lat_col = find(("lng", "lon", "经度")), find(("lat", "纬度"))
    hint = f"{lng_col}{lat_col}".lower()
    src = next((crs for crs in CRS if crs.replace("-", "").lower()[:3] in hint), None)
    return lng_col, lat_col, src


def summarize(result, label_col=None):
    """总点数、正常点数、每类问题对应哪些点（用 label_col 的值标点，缺省用行号）。"""
    labels = (result[label_col] if label_col else pd.Series(result.index, index=result.index)).astype(str)
    return {
        "total": len(result),
        "ok": int((result["check"] == OK).sum()),
        "issues": {issue: labels[result["issue"] == issue].tolist() for issue in ISSUES},
    }
