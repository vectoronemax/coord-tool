"""地图预览：Folium，高德底图（GCJ-02）/ OSM 底图（WGS-84）可切换。只负责画，不做转换和校验。

输入是 coord_core.convert_table 的结果表：
  正常点    蓝点 = 转换前 WGS-84 原值，橙点 = 转换后 GCJ-02，中间虚线；悬停看点名和偏移米数；
  问题点    红色叉号。问题点不转换，叉号按原值换到 WGS-84 大致画出（只作显示）；写反的画在交换后的位置，
           (0,0) 就画在 (0,0)；缺失、越界没法上图，列在右下角说明框里，不会漏掉。
注意：坐标转换不联网，但底图瓦片要联网加载（高德、OSM 瓦片服务器）。
"""
import html

import folium
import pandas as pd

from coord_core import ISSUE_SWAPPED, OK, convert, out_of_china

C_WGS = "#2b6cb0"    # 蓝：转换前 WGS-84
C_GCJ = "#dd6b20"    # 橙：转换后 GCJ-02
C_BAD = "#d62728"    # 红叉：问题点
INK = "#0b0b0b"
MUTED = "#52514e"

AMAP = {"tiles": "https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}",
        "subdomains": "1234", "name": "高德底图（GCJ-02）", "attr": "&copy; 高德地图"}
OSM = {"tiles": "https://tile.openstreetmap.org/{z}/{x}/{y}.png", "subdomains": "abc", "name": "OSM 底图（WGS-84）",
       "attr": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}

X_ICON = (f'<svg width="18" height="18" viewBox="0 0 18 18"><path d="M3 3L15 15M15 3L3 15" '
          f'stroke="#fff" stroke-width="6" stroke-linecap="round"/><path d="M3 3L15 15M15 3L3 15" '
          f'stroke="{C_BAD}" stroke-width="3" stroke-linecap="round"/></svg>')


def _dot(color):
    return (f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
            f'background:{color};border:1.5px solid #fff;box-shadow:0 0 0 1px {color};margin-right:6px"></span>')


def _problem_position(row, lng_col, lat_col, src):
    """问题点画在哪 (lat, lng)；画不了返回 None。问题点结果里没有坐标，这里按原值换到 WGS-84，只作显示。"""
    try:
        x, y = float(row[lng_col]), float(row[lat_col])
    except (TypeError, ValueError):
        return None
    if x == 0 and y == 0:
        return 0.0, 0.0
    if row["issue"] == ISSUE_SWAPPED:
        x, y = y, x
    try:
        lng, lat = convert(x, y, src, "WGS-84")
    except ValueError:   # NaN、超出 ±180 / ±90
        return None
    return lat, lng


def build_map(result, lng_col, lat_col, src, label_col=None):
    """结果表 → folium.Map。默认高德底图，右上角可切 OSM、可单独开关各图层。"""
    m = folium.Map(tiles=None, control_scale=True, max_zoom=18)
    for t in (AMAP, OSM):   # 两个都是底图，只显示高德，OSM 留给右上角切换
        folium.TileLayer(t["tiles"], attr=t["attr"], name=t["name"], subdomains=t["subdomains"],
                         max_zoom=18, show=t is AMAP).add_to(m)
    g_line = folium.FeatureGroup("连线").add_to(m)
    g_wgs = folium.FeatureGroup(f"{_dot(C_WGS)}转换前 WGS-84 原值").add_to(m)
    g_gcj = folium.FeatureGroup(f"{_dot(C_GCJ)}转换后 GCJ-02").add_to(m)
    g_bad = folium.FeatureGroup(f'<span style="color:{C_BAD};font-weight:700;margin:0 6px 0 1px">✕</span>问题点').add_to(m)

    labels = result[label_col].astype(str) if label_col else pd.Series(
        [f"第 {i + 1} 行" for i in range(len(result))], index=result.index)
    bounds, bad = [], []
    for i, row in result.iterrows():
        name = html.escape(labels[i])
        if row["check"] == OK:
            w, g = (row["wgs84_lat"], row["wgs84_lng"]), (row["gcj02_lat"], row["gcj02_lng"])
            tip = f"<b>{name}</b><br>WGS-84 → GCJ-02 偏移 {row['gcj02_offset_m']:.1f} 米"
            folium.PolyLine([w, g], color=MUTED, weight=2, dash_array="5 5", tooltip=tip).add_to(g_line)
            for pos, color, group, what in ((w, C_WGS, g_wgs, "转换前 WGS-84 原值"), (g, C_GCJ, g_gcj, "转换后 GCJ-02")):
                folium.CircleMarker(pos, radius=6, color="#fff", weight=1.5, fill=True, fill_color=color,
                                    fill_opacity=1, tooltip=f"{tip}<br>{_dot(color)}{what}").add_to(group)
            bounds += [w, g]
            continue
        pos = _problem_position(row, lng_col, lat_col, src)
        bad.append(f"{name}：{html.escape(row['issue'])}" + ("" if pos else "（无法上图）"))
        if pos is None:
            continue
        folium.Marker(pos, icon=folium.DivIcon(html=X_ICON, icon_size=(18, 18), icon_anchor=(9, 9)),
                      tooltip=f"<b>{name}</b><br>{html.escape(row['issue'])}<br>{html.escape(row['check_note'])}"
                      ).add_to(g_bad)
        if not out_of_china(pos[1], pos[0]):   # 国外、(0,0) 不参与初始视野，免得缩成全世界
            bounds.append(pos)

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(bounds or [(18.0, 73.0), (54.0, 135.0)], max_zoom=16)

    lines = [f'<b style="color:{INK}">怎么看</b>',
             f"{_dot(C_GCJ)}橙点（GCJ-02）对高德底图",
             f"{_dot(C_WGS)}蓝点（WGS-84）对 OSM 底图",
             "右上角切换底图，悬停看点名和偏移米数",
             "若高德上反而是蓝点压在路口：源数据其实是 GCJ-02，源坐标系选错了"]
    if bad:
        lines += [f'<b style="color:{C_BAD}">✕ 问题点 {len(bad)} 个</b>'] + bad
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed; right:10px; bottom:26px; z-index:1000; max-width:300px;'
        f' padding:8px 10px; background:rgba(255,255,255,0.92); border:1px solid #c9c8c2; border-radius:4px;'
        f' font:12px/1.6 \'PingFang SC\',\'Microsoft YaHei\',sans-serif; color:{MUTED};">{"<br>".join(lines)}</div>'))
    return m
