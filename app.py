"""Vector的坐标转换器（Streamlit 界面）。只管输入输出：算法和校验在 coord_core.py，画图在 coord_map.py。

启动：cd apps/coord-tool && streamlit run app.py
（要在本目录启动，.streamlit/config.toml 里的主题色和隐私设置才会生效）
"""
import io

import streamlit as st
from streamlit_folium import st_folium

from coord_core import CRS, ISSUES, OK, convert_point, convert_table, guess_columns, read_csv, summarize
from coord_map import build_map

APP_NAME = "Vector的坐标转换器"
BAD_ROW = "background-color: #fde2e1; color: #9b1c1c"   # 问题行标红
PREVIEW_ROWS = 5000                                      # 表格预览上限，完整结果走下载
COORD_COLS = ["wgs84_lng", "wgs84_lat", "gcj02_lng", "gcj02_lat", "bd09_lng", "bd09_lat"]


@st.cache_data(show_spinner=False)
def load_csv(data: bytes):
    return read_csv(io.BytesIO(data))


@st.cache_data(show_spinner="正在转换…")
def run_convert(df, lng_col, lat_col, src):
    return convert_table(df, lng_col, lat_col, src)


def preview_frame(df):
    """表格预览用：数字按精度转成文本、空值留白（st.dataframe 会把 NaN / None 一律显示成 None）。下载不受影响。"""
    out = df.astype(object).where(df.notna(), "")
    for cols, fmt in ((COORD_COLS, "{:.6f}"), (["gcj02_offset_m", "bd09_offset_m"], "{:.1f}")):
        for c in cols:
            out[c] = [fmt.format(v) if v == v else "" for v in df[c]]   # v == v 为假即 NaN
    return out


def single_tab():
    c1, c2 = st.columns(2)
    lng = c1.number_input("经度 lng", min_value=-180.0, max_value=180.0, value=116.397128,
                          step=0.000001, format="%.6f")
    lat = c2.number_input("纬度 lat", min_value=-90.0, max_value=90.0, value=39.908722,
                          step=0.000001, format="%.6f")
    c3, c4 = st.columns(2)
    src = c3.selectbox("源坐标系", CRS, index=0)
    dst = c4.selectbox("目标坐标系", CRS, index=1)

    r = convert_point(lng, lat, src, dst)
    st.subheader(f"{r['src']} → {r['dst']}")
    if r["check"] != OK:
        st.error(f"**{r['issue']}**，不转换：{r['note']}")
        return
    m1, m2, m3 = st.columns(3)
    m1.metric("经度 lng", f"{r['lng']:.6f}")
    m2.metric("纬度 lat", f"{r['lat']:.6f}")
    m3.metric("偏移距离", f"{r['offset_m']:.1f} 米")
    st.code(f"{r['lng']:.6f},{r['lat']:.6f}", language=None)
    st.caption(f"落在{r['province']}")
    if src == dst:
        st.info("源和目标是同一个坐标系，原样输出。")


def batch_tab():
    up = st.file_uploader("① 上传 CSV（UTF-8 或 GBK 编码）", type=["csv"])
    if up is None:
        st.caption("文件只在运行本程序的那台机器上处理（在自己电脑上运行就是你的电脑），只进内存、不存盘、不转发。")
        return
    try:
        df = load_csv(up.getvalue())
    except ValueError as e:
        st.error(str(e))
        return
    if df.empty:
        st.warning("文件里没有数据行。")
        return
    st.caption(f"读入 {len(df)} 行 × {len(df.columns)} 列")

    st.markdown("**② 选择经纬度列和源坐标系**")
    cols = list(df.columns)
    g_lng, g_lat, g_src = guess_columns(cols)
    others = [c for c in cols if c not in (g_lng, g_lat)]
    b1, b2, b3, b4 = st.columns(4)
    lng_col = b1.selectbox("经度列", cols, index=cols.index(g_lng) if g_lng else 0)
    lat_col = b2.selectbox("纬度列", cols, index=cols.index(g_lat) if g_lat else min(1, len(cols) - 1))
    src = b3.selectbox("源坐标系", CRS, index=CRS.index(g_src) if g_src else 0, key="batch_src",
                       help="按列名猜的，请核对：列名写着 WGS-84 的数据，实际可能是从国内地图上取的 GCJ-02")
    label = b4.selectbox("点名列（预览和地图用）", ["（行号）"] + cols,
                         index=cols.index(others[0]) + 1 if others else 0)
    label_col = None if label == "（行号）" else label

    # 选项一变，结果就隐藏，等再点一次「转换」
    params = (up.name, up.size, lng_col, lat_col, src, label_col)
    if st.button("③ 转换", type="primary"):
        st.session_state["batch_params"] = params
    if st.session_state.get("batch_params") != params:
        return
    if lng_col == lat_col:
        st.error("经度列和纬度列选成了同一列。")
        return
    try:
        res = run_convert(df, lng_col, lat_col, src)
    except ValueError as e:
        st.error(str(e))
        return
    s = summarize(res, label_col)

    st.subheader("④ 表格预览")
    k1, k2, k3 = st.columns(3)
    k1.metric("总点数", s["total"])
    k2.metric(OK, s["ok"])
    k3.metric("有问题（标红）", s["total"] - s["ok"])
    found = [(i, s["issues"][i]) for i in ISSUES if s["issues"][i]]
    if found:
        st.markdown("\n".join(f"- **{i}** {len(names)} 个：{'、'.join(names)}" for i, names in found))
    view = res[res["check"] != OK] if st.toggle("只看问题行") else res
    if len(view) > PREVIEW_ROWS:
        st.caption(f"只预览前 {PREVIEW_ROWS} 行，完整结果请下载。")
        view = view.head(PREVIEW_ROWS)
    st.dataframe(preview_frame(view).style
                 .apply(lambda row: [BAD_ROW if row["check"] != OK else ""] * len(row), axis=1))

    st.subheader("⑤ 地图预览")
    stem = up.name.rsplit(".", 1)[0]
    if st.toggle("显示地图（底图瓦片需联网加载）", value=True):
        fmap = build_map(res, lng_col, lat_col, src, label_col)
        st_folium(fmap, height=560, use_container_width=True, returned_objects=[])
        st.download_button("下载地图 HTML", fmap.get_root().render().encode("utf-8"),
                           file_name=f"{stem}_地图.html", mime="text/html", on_click="ignore")
    else:
        st.caption("地图已关闭：整个流程完全离线。")

    st.subheader("⑥ 下载结果")
    st.download_button("下载结果 CSV（utf-8-sig，Excel 直接打开不乱码）",
                       res.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{stem}_已转换.csv", mime="text/csv", type="primary", on_click="ignore")
    st.caption("原始列原样保留，后面追加三套坐标、偏移米数和校验列；问题行不删不改、不转换（坐标列留空），只打标记。")


def footer():
    st.divider()
    st.markdown("#### 算法说明")
    st.markdown(
        "- **WGS-84 → GCJ-02**：开源社区流传的公开加偏公式（克拉索夫斯基椭球参数 + 正弦扰动项）。"
        "非官方发布，主流开源库都用这一版；中国范围框外不加偏。\n"
        "- **GCJ-02 → BD-09**：百度公开的 x_pi 公式。\n"
        "- **反算**（GCJ-02 → WGS-84、BD-09 → GCJ-02）：公式没有解析逆，从目标点出发，用正算的残差修正 5 轮，"
        "往返误差 < 1 毫米（外滩实测逐轮 479 m → 2.2 m → 7 mm → 0.025 mm → 0）；WGS-84 ⇄ BD-09 经 GCJ-02 中转。\n"
        "- **偏移距离**：转换前后两组数字画在同一张图上的球面距离（半正矢公式）。\n"
        "- **校验**：单点和批量同一套规则，只有正常的点才转换，问题点不转换、给出原因。"
        "每个点只标一个最能说明问题的类型，依次判断：缺失或非数字 → (0, 0) → 落在省级区内即正常 → "
        "交换经纬度后落进省级区＝写反 → 超出 ±180 / ±90 → 中国范围框外＝国外 → 其余＝框内但不在省级区（海上或邻国）。"
        "省级边界为阿里云 DataV 数据（GCJ-02），边界经过简化，紧贴海岸的点可能被误标，说明里会给出最近省份和距离。")
    st.success("🔒 **离线计算、不上传数据**：所有换算和校验都在运行本程序的那台机器上完成，不调用任何在线 API；"
               "上传的 CSV 只读进内存，不存盘、不转发，本程序也不发送使用统计。唯一联网的是地图预览的底图瓦片（高德 / OSM）"
               "和地图脚本，关掉地图预览即完全离线。")
    st.caption("⚠️ 如果你是通过网址（比如 xxx.streamlit.app）打开的本页面，程序跑在那台云端服务器上，CSV 会先传到服务器内存里处理。"
               "敏感数据请在自己电脑上运行，或用不联网的离线版「坐标转换器.html」。")


st.set_page_config(page_title=APP_NAME, page_icon="🧭", layout="wide")
st.title(APP_NAME)
st.markdown("WGS-84 ⇄ GCJ-02 ⇄ BD-09 六向互转：单点即时换算，或整表批量转换，自动揪出写反、越界、国外、海上的坏点。")

tab_one, tab_batch = st.tabs(["单点转换", "批量转换"])
with tab_one:
    single_tab()
with tab_batch:
    batch_tab()
footer()
