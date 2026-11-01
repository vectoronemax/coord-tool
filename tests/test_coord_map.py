"""coord_map 测试：图层数量、问题点上图。在 apps/coord-tool 目录下运行：python -m pytest -q"""
from pathlib import Path

import pandas as pd
import pytest

from coord_core import convert_table, read_csv
from coord_map import build_map

SAMPLE_CSV = Path(__file__).resolve().parents[3] / "data/raw/坐标点位_待转换.csv"


@pytest.mark.skipif(not SAMPLE_CSV.exists(), reason="课程数据 data/raw 不在（应用被单独拷走时跳过）")
def test_map_layers():
    df = read_csv(SAMPLE_CSV)
    page = build_map(convert_table(df, "lon_wgs84", "lat_wgs84", "WGS-84"),
                     "lon_wgs84", "lat_wgs84", "WGS-84", "name").get_root().render()
    assert page.count("L.circleMarker(") == 20   # 10 个正常点 × 蓝、橙各一
    assert page.count("L.polyline(") == 10
    assert page.count("L.marker(") == 4          # 4 个问题点都能上图（写反的按交换后位置）
    assert "autonavi.com" in page and "tile.openstreetmap.org" in page and "问题点 4 个" in page


def test_undrawable_points_listed():
    page = build_map(convert_table(pd.DataFrame({"x": ["", "200"], "y": ["39.9", "30"]}), "x", "y", "WGS-84"),
                     "x", "y", "WGS-84").get_root().render()
    assert page.count("L.marker(") == 0 and page.count("（无法上图）") == 2   # 画不了的列进说明框，不漏
