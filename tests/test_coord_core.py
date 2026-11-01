"""coord_core 单元测试。在 apps/coord-tool 目录下运行：python -m pytest -q"""
import io
import itertools
from pathlib import Path

import pandas as pd
import pytest

import coord_core
from coord_core import (CRS, ISSUE_ABROAD, ISSUE_MISSING, ISSUE_OFFLAND, ISSUE_RANGE, ISSUE_SWAPPED, OK, OUT_COLS,
                        _invert, convert, convert_point, convert_table, gcj02_to_wgs84, haversine_m, read_csv,
                        summarize, wgs84_to_gcj02)

SAMPLE_CSV = Path(__file__).resolve().parents[3] / "data/raw/坐标点位_待转换.csv"
need_sample = pytest.mark.skipif(not SAMPLE_CSV.exists(), reason="课程数据 data/raw 不在（应用被单独拷走时跳过）")

TIANANMEN_WGS = (116.397128, 39.908722)
BUND_WGS = (121.490317, 31.236305)
VALUE_COLS = OUT_COLS[:8]   # 三套坐标 + 两个偏移米数

# 往返测试用的五个城市（WGS-84）
CITIES = {"北京天安门": TIANANMEN_WGS, "上海外滩": BUND_WGS, "广州": (113.2644, 23.1291),
          "乌鲁木齐": (87.6168, 43.8256), "哈尔滨": (126.5350, 45.8038)}
# 六向往返再多两个：拉萨、三亚
SAMPLES = list(CITIES.values()) + [(91.1409, 29.6456), (109.5120, 18.2528)]

# 问题点：名称 → ((经度, 纬度), 应标的问题类型)
PROBLEMS = {
    "零值": ((0, 0), ISSUE_RANGE),
    "经纬度写反": ((39.908722, 116.397128), ISSUE_SWAPPED),
    "巴黎": ((2.352222, 48.856613), ISSUE_ABROAD),
    "空值": ((None, 39.908722), ISSUE_MISSING),
}


# ---------- 1）天安门 ----------
def test_tiananmen_wgs84_to_gcj02():
    r = convert_point(*TIANANMEN_WGS, "WGS-84", "GCJ-02")
    assert (round(r["lng"], 6), round(r["lat"], 6)) == (116.403371, 39.910125)
    assert round(r["offset_m"]) == 555
    assert r["check"] == OK and r["province"] == "北京市"


def test_tiananmen_wgs84_to_bd09():
    r = convert_point(*TIANANMEN_WGS, "WGS-84", "BD-09")
    assert (round(r["lng"], 6), round(r["lat"], 6)) == (116.409744, 39.916466)


# ---------- 2）3）反算与往返误差 ----------
@pytest.mark.parametrize("wgs", CITIES.values(), ids=CITIES.keys())
def test_round_trip_wgs_gcj_wgs_under_1mm(wgs):
    gcj = wgs84_to_gcj02(*wgs)
    assert haversine_m(*wgs, *gcj) > 100              # 先确认真加了偏，不是原样返回
    assert haversine_m(*wgs, *gcj02_to_wgs84(*gcj)) < 1e-3


def test_gcj02_to_wgs84_corrects_5_rounds(monkeypatch):
    """固定 5 轮，不看收敛提前停（旧版按 1e-10 度收敛就停，各城市 4～5 次、框外 1 次）。"""
    forward = coord_core.wgs84_to_gcj02
    for p in [*CITIES.values(), (2.352222, 48.856613)]:
        calls = []
        monkeypatch.setattr(coord_core, "wgs84_to_gcj02", lambda x, y: calls.append(1) or forward(x, y))
        coord_core.gcj02_to_wgs84(*forward(*p))
        assert len(calls) == 5, p                     # 正算 5 次 = 残差修正 5 轮


def test_one_step_inverse_is_meters_off():
    """网上常见的一步反算 = 只修正 1 轮，外滩差 2 米多；所以要多轮。"""
    gcj = wgs84_to_gcj02(*BUND_WGS)
    assert 1 < haversine_m(*BUND_WGS, *_invert(wgs84_to_gcj02, *gcj, rounds=1)) < 5


@pytest.mark.parametrize("src,dst", list(itertools.permutations(CRS, 2)))
def test_round_trip_all_six_directions(src, dst):
    for p in SAMPLES:
        back = convert(*convert(*p, src, dst), dst, src)
        assert haversine_m(*p, *back) < 1e-3


# ---------- 4）问题点不转换，返回原因 ----------
@pytest.mark.parametrize("point,issue", PROBLEMS.values(), ids=PROBLEMS.keys())
def test_problem_point_not_converted(point, issue):
    for src, dst in itertools.product(CRS, CRS):
        r = convert_point(*point, src, dst)
        assert r["check"] != OK and r["issue"] == issue and r["note"], (src, dst)
        assert r["lng"] is None and r["lat"] is None and r["offset_m"] is None, (src, dst)


def test_problem_rows_not_converted_in_table():
    names = list(PROBLEMS) + ["天安门"]
    points = [p for p, _ in PROBLEMS.values()] + [TIANANMEN_WGS]
    df = pd.DataFrame({"name": names, "x": [p[0] for p in points], "y": [p[1] for p in points]})
    for src in CRS:
        r = convert_table(df, "x", "y", src)
        assert r["issue"].tolist() == [issue for _, issue in PROBLEMS.values()] + [None]
        assert (r.loc[:3, "check_note"].str.len() > 0).all()
        assert r.loc[:3, VALUE_COLS].isna().all(axis=None)
        assert r.loc[4, "check"] == OK and r.loc[4, VALUE_COLS].notna().all()


def test_zero_to_bd09_not_shifted():
    """(0, 0) 多半是缺失值被填成 0；直接套 BD-09 公式会得到 (0.0065, 0.006)，看着像真坐标。"""
    for src in CRS:
        r = convert_point(0, 0, src, "BD-09")
        assert (r["lng"], r["lat"]) != (0.0065, 0.006)
        assert r["lng"] is None and r["lat"] is None
        assert r["issue"] == ISSUE_RANGE and "(0, 0)" in r["note"]
    t = convert_table(pd.DataFrame({"x": ["0", "0.0"], "y": ["0", "0"]}), "x", "y", "WGS-84")
    assert t[["bd09_lng", "bd09_lat"]].isna().all(axis=None) and (t["issue"] == ISSUE_RANGE).all()


# ---------- 其他 ----------
def test_out_of_china_no_gcj_offset():
    tokyo = (139.6917, 35.6895)
    assert wgs84_to_gcj02(*tokyo) == tokyo


def test_same_crs_and_aliases():
    assert convert(*TIANANMEN_WGS, "wgs84", "WGS_84") == TIANANMEN_WGS
    assert convert(*TIANANMEN_WGS, "wgs84", "gcj02") == convert(*TIANANMEN_WGS, "WGS-84", "GCJ-02")


@pytest.mark.parametrize("lng,lat", [(39.9, 116.4), (200, 30), (float("nan"), 30)])
def test_convert_rejects_bad_numbers(lng, lat):
    with pytest.raises(ValueError):
        convert(lng, lat, "WGS-84", "GCJ-02")


def test_bad_crs():
    with pytest.raises(ValueError):
        convert(*TIANANMEN_WGS, "WGS-84", "CGCS2000")
    with pytest.raises(ValueError):
        convert_point(*TIANANMEN_WGS, "WGS-84", "CGCS2000")


@need_sample
def test_sample_csv():
    df = read_csv(SAMPLE_CSV)
    r = convert_table(df, "lon_wgs84", "lat_wgs84", "WGS-84")
    s = summarize(r, "name")
    assert (s["total"], s["ok"]) == (14, 10)
    assert s["issues"] == {ISSUE_SWAPPED: ["错点_经纬度写反"], ISSUE_RANGE: ["错点_零值"],
                           ISSUE_ABROAD: ["错点_落在巴黎"], ISSUE_OFFLAND: ["错点_落在东海"], ISSUE_MISSING: []}
    assert r.loc[r["check"] != OK, VALUE_COLS].isna().all(axis=None)


def test_edge_cases_kept_and_flagged():
    df = pd.DataFrame({
        "id": ["001", "002", "003", "004", "005", "006", "007", "008"],
        "x": ["43.8256", "200", "-74.006", "126.978", "", "abc", "116.403371", "121.4737"],
        "y": ["87.6168", "30", "40.7128", "37.5665", "39.9", "39.9", "39.910125", "31.2304"],
    })
    r = convert_table(df, "x", "y", "GCJ-02")
    assert r["id"].tolist() == df["id"].tolist()  # 行数、行序、前导 0 都不动
    assert r["issue"].tolist() == [ISSUE_SWAPPED,   # 乌鲁木齐写反：数值不越界也要认出来
                                   ISSUE_RANGE, ISSUE_ABROAD,
                                   ISSUE_OFFLAND,   # 首尔：在中国范围框内，但不在省级区
                                   ISSUE_MISSING, ISSUE_MISSING, None, None]
    assert r.loc[6, "province"] == "北京市" and r.loc[7, "check"] == OK
    assert haversine_m(r.loc[6, "wgs84_lng"], r.loc[6, "wgs84_lat"], *TIANANMEN_WGS) < 0.1
    assert r.loc[:5, VALUE_COLS].isna().all(axis=None)   # 问题行一律不转换
    assert r.loc[6:, VALUE_COLS].notna().all(axis=None)


def test_column_clash():
    with pytest.raises(ValueError):
        convert_table(pd.DataFrame({"lng": [116.4], "lat": [39.9], "check": ["x"]}), "lng", "lat", "WGS-84")


def test_read_gbk():
    raw = "名称,经度,纬度\n天安门,116.397128,39.908722\n".encode("gbk")
    assert read_csv(io.BytesIO(raw)).columns.tolist() == ["名称", "经度", "纬度"]
