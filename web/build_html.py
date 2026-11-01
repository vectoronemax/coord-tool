"""把 web/template.html 和省级边界拼成一个独立的离线网页 坐标转换器.html（单文件、不联网，可以直接发给别人）。

运行：python web/build_html.py    （改了 template.html 或 data/ 里的边界后重跑）
边界只留省名和坐标，环压成扁平数组 [x0, y0, x1, y1, ...]；和 coord_core.load_provinces 一样去掉没有名字的九段线要素。
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent
BOUNDARY = APP / "data/中国省级行政区_DataV.geojson"
OUT = APP / "坐标转换器.html"


def provinces():
    out = []
    for f in json.loads(BOUNDARY.read_text(encoding="utf-8"))["features"]:
        name = f["properties"].get("name")
        if not name:
            continue
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        out.append({"n": name, "p": [[[v for pt in ring for v in pt[:2]] for ring in poly] for poly in polys]})
    return out


def main():
    html = (HERE / "template.html").read_text(encoding="utf-8")
    assert html.count("__PROVINCES__") == 1
    data = json.dumps(provinces(), ensure_ascii=False, separators=(",", ":"))
    OUT.write_text(html.replace("__PROVINCES__", data), encoding="utf-8")
    print(f"写出 {OUT}（{OUT.stat().st_size / 1024:.0f} KB）")


if __name__ == "__main__":
    main()
