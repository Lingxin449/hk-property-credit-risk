"""港股地产信用风险预警系统 · Streamlit 只读看板

性质：**纯只读**。仅读取 `preprocessed/*_v2.csv`（二期产物）与 `reports/figures/*_v2.png`；
不重训、不拟合任何统计量、不改动任何口径（`口径纪律`/§8）。看板内所有图表与派生指标
均由上述既有产物现场算出，不含任何硬编码读数。

启动：`streamlit run app.py`
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
PRE = ROOT / "preprocessed"
FIG = ROOT / "reports" / "figures"

NAV_COLS = ["基准净值", "v1净值", "v2净值"]
NAV_LABEL = {"基准净值": "基准 · 全池等权", "v1净值": "v1 · 绝对分数阈值", "v2净值": "v2 · 排名阈值"}

st.set_page_config(page_title="港股地产信用风险预警 · 看板", page_icon="📉", layout="wide")


# --------------------------------------------------------------------------- #
# 数据加载（只读，带缓存）
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_pre(name: str) -> pd.DataFrame:
    """读取 `preprocessed/` 下的二期 `_v2` 产物（只读，不写任何统计量）。"""
    df = pd.read_csv(PRE / name)
    if "交易日期" in df.columns:
        df["交易日期"] = pd.to_datetime(df["交易日期"])
    return df


@st.cache_data(show_spinner=False)
def code_to_name() -> dict:
    """股票代码 → 公司简称（仅用于展示）。

    优先读 `preprocessed/issuers.csv`（只含代码与简称两列）；该文件不存在时
    回退到特征矩阵，本地全量环境与对外精简环境都能跑。
    """
    slim = PRE / "issuers.csv"
    src = slim if slim.exists() else PRE / "feature_matrix_v2.csv"
    fm = pd.read_csv(src, usecols=["股票代码", "公司简称"])
    return dict(fm.drop_duplicates("股票代码")[["股票代码", "公司简称"]].values)


def drawdown(nav: pd.Series) -> pd.Series:
    """由净值序列算回撤（仅展示用，不拟合统计量）。"""
    return nav / nav.cummax() - 1.0


_BIN_RE = re.compile(r"^\s*[\(\[]\s*([^,]+?)\s*,\s*([^\]\)]+?)\s*[\)\]]\s*$")


def _to_edge(text: str) -> float:
    """把分档边界字符串转 float（支持 ±inf / ±∞）。"""
    t = text.strip().replace("−", "-").replace("∞", "inf").lower()
    return float("-inf") if t.startswith("-inf") else (float("inf") if t == "inf" else float(t))


@st.cache_data(show_spinner=False)
def scorecard_bins(points: pd.DataFrame) -> pd.DataFrame:
    """把评分卡分档字符串解析成 (变量, lo, hi, 分档, 本档得分) 便于逐样本匹配。"""
    rows = []
    for _, r in points.iterrows():
        m = _BIN_RE.match(str(r["分档"]))
        if not m:
            continue
        rows.append({"变量": r["变量"], "lo": _to_edge(m.group(1)), "hi": _to_edge(m.group(2)),
                     "分档": r["分档"], "本档得分": r["本档得分"]})
    return pd.DataFrame(rows)


def main() -> None:
    st.title("港股地产信用风险预警系统 · 二期看板")
    st.caption(
        "只读展示二期（方案 E：重训 + 2025 新考卷）的 `_v2` 产物。"
        "不重训、不拟合统计量、不改口径；切分与口径见 `口径纪律`，数值见 `内部数值表`。"
    )

    with st.sidebar:
        st.markdown("### 数据来源")
        st.markdown(
            "- 回测：`backtest_nav_v2.csv` / `backtest_membership_v2.csv` / `backtest_positions_v2.csv`\n"
            "- 评估：`backtest_metrics_v2.csv` / `backtest_v2_placebo_v2.csv`\n"
            "- 可解释：`shap_*_v2.csv` / `scorecard_*_v2.csv`\n"
            "- 区间：训练 ≤2023-12-31 / 留出 2024 / 测试 2025（测试期只评估、不参与 fit）"
        )
        st.caption("测试期（2025）为一次性考卷，仅展示、不据此回改模型。")

    tab_nav, tab_black, tab_shap, tab_score = st.tabs(
        ["① 净值 · 回撤", "② 黑名单 · 持仓", "③ SHAP 可解释", "④ 评分卡"]
    )

    # ===================================================================== #
    # ① 净值 · 回撤
    # ===================================================================== #
    with tab_nav:
        nav = load_pre("backtest_nav_v2.csv").set_index("交易日期")
        metrics = load_pre("backtest_metrics_v2.csv")

        st.subheader("回测净值与回撤（测试期 2025）")
        test_metrics = metrics[metrics["分段"] == "测试期"]
        cols = st.columns(3)
        for c, name in zip(cols, NAV_LABEL.values()):
            row = test_metrics[test_metrics["口径"].str.startswith(name.split(" · ")[0])]
            if not row.empty:
                c.metric(name, f"{row['累计收益'].iloc[0] * 100:.2f}%",
                         f"最大回撤 {row['最大回撤'].iloc[0] * 100:.2f}%")

        st.line_chart(nav[NAV_COLS].rename(columns=NAV_LABEL), height=320)
        dd = pd.DataFrame({c: drawdown(nav[c]) for c in NAV_COLS}).rename(columns=NAV_LABEL)
        st.caption("回撤（由净值序列现场算得）")
        st.area_chart(dd, height=260)

        st.divider()
        st.subheader("分段指标（训练 / 留出验证 / 测试）")
        st.dataframe(metrics, hide_index=True)

        st.subheader("安慰剂对照（同规模随机剔除，展示分布）")
        placebo = load_pre("backtest_v2_placebo_v2.csv")
        st.caption(f"安慰剂样本数：{len(placebo)}（自变量仅为随机，非模型）——用于判断策略改善是否来自模型排序能力。")
        st.dataframe(placebo.describe().T, hide_index=False)

    # ===================================================================== #
    # ② 黑名单 · 持仓
    # ===================================================================== #
    with tab_black:
        mem = load_pre("backtest_membership_v2.csv").set_index("交易日期")
        pos = load_pre("backtest_positions_v2.csv")
        trades = load_pre("backtest_v2_trades_v2.csv")
        c2n = code_to_name()

        st.subheader("逐周黑名单规模与持仓家数")
        st.line_chart(mem[["v1黑名单家数", "v2黑名单家数"]], height=260)
        st.line_chart(mem[["v1白名单持仓数", "v2白名单持仓数"]], height=260)

        st.divider()
        st.subheader("成员进出（v2 · HELD ↔ 非 HELD）")
        order = pos.sort_values(["股票代码", "交易日期"]).copy()
        order["前一状态"] = order.groupby("股票代码")["v2状态"].shift()
        enters = order[(order["v2状态"] == "HELD") & (order["前一状态"].notna()) & (order["前一状态"] != "HELD")]
        exits = order[(order["v2状态"] != "HELD") & (order["前一状态"] == "HELD")]

        weeks = sorted(pos["交易日期"].dt.date.unique())
        pick = st.select_slider("选择周", options=weeks, value=weeks[-1])
        pick_ts = pd.Timestamp(pick)

        col_e, col_x = st.columns(2)
        with col_e:
            st.markdown("**当周买入/进入**")
            e = enters[enters["交易日期"] == pick_ts][["股票代码", "分数"]].copy()
            e["公司简称"] = e["股票代码"].map(c2n)
            st.dataframe(e, hide_index=True)
        with col_x:
            st.markdown("**当周卖出/退出**")
            x = exits[exits["交易日期"] == pick_ts][["股票代码", "分数"]].copy()
            x["公司简称"] = x["股票代码"].map(c2n)
            st.dataframe(x, hide_index=True)

        with st.expander("当周全部公司的 v1 / v2 状态与分数"):
            w = pos[pos["交易日期"] == pick_ts][["股票代码", "可交易", "分数", "v1状态", "v2状态"]].copy()
            w["公司简称"] = w["股票代码"].map(c2n)
            st.dataframe(w.sort_values("分数"), hide_index=True)

        st.divider()
        st.subheader("换手明细（v2）")
        st.dataframe(trades, hide_index=True)

    # ===================================================================== #
    # ③ SHAP 可解释
    # ===================================================================== #
    with tab_shap:
        imp = load_pre("shap_importance_v2.csv")
        direc = load_pre("shap_direction_v2.csv")
        vs_iv = load_pre("shap_vs_iv_v2.csv")

        st.subheader("特征重要度（mean |SHAP|）")
        st.bar_chart(imp.set_index("Feature")["mean_abs_SHAP"], height=300)

        st.subheader("取值方向（低值组 vs 高值组 SHAP）")
        st.bar_chart(direc.set_index("Feature")[["低值组SHAP", "高值组SHAP"]], height=300)
        st.dataframe(direc, hide_index=True)

        st.divider()
        st.subheader("SHAP × WOE/IV 双轨互补对照")
        st.dataframe(vs_iv, hide_index=True)

        st.divider()
        st.subheader("既有 SHAP 图（二期 `_v2`）")
        figure_files = [
            ("蜂群图（正窗口）", "shap_蜂群图_正窗口_v2.png"),
            ("重要度（正窗口）", "shap_重要度_正窗口_v2.png"),
            ("个案 waterfall · 合景泰富", "shap_waterfall_合景泰富_v2.png"),
            ("个案 waterfall · 富力地产", "shap_waterfall_富力地产_v2.png"),
        ]
        fcols = st.columns(2)
        for i, (title, fname) in enumerate(figure_files):
            path = FIG / fname
            with fcols[i % 2]:
                st.markdown(f"**{title}**")
                if path.exists():
                    st.image(str(path))
                else:
                    st.info(f"未找到图：{fname}")

    # ===================================================================== #
    # ④ 评分卡
    # ===================================================================== #
    with tab_score:
        iv = load_pre("scorecard_iv_v2.csv")
        points = load_pre("scorecard_points_v2.csv")
        scores = load_pre("scorecard_scores_v2.csv")
        # 特征矩阵体积较大且含供应商派生数据，对外精简环境可能不带；缺失时本页其余内容照常展示
        try:
            fm = load_pre("feature_matrix_v2.csv")
        except FileNotFoundError:
            fm = None
        c2n = code_to_name()

        st.subheader("特征 IV / PSI 与稳定性")
        st.dataframe(iv, hide_index=True)

        st.divider()
        st.subheader("公司评分卡得分（risk_score）时序")
        names = sorted(scores["公司简称"].unique())
        pick_name = st.selectbox("选择公司", names, index=names.index("合景泰富") if "合景泰富" in names else 0)
        sub = scores[scores["公司简称"] == pick_name].sort_values("交易日期")
        seg_pick = st.multiselect("分段", ["训练段", "留出验证段", "测试期"],
                                  default=["训练段", "留出验证段", "测试期"])
        sub = sub[sub["分段"].isin(seg_pick)]
        st.line_chart(sub.set_index("交易日期")["risk_score"], height=280)
        st.dataframe(sub[["交易日期", "分段", "y", "logit", "risk_score", "proba_lr"]].tail(12), hide_index=True)

        st.divider()
        st.subheader("分档贡献（点选公司 + 周，逐变量命中分档 → 本档得分）")
        if fm is None:
            st.info("特征矩阵 `feature_matrix_v2.csv` 未随本仓库分发，分档贡献不可用；"
                    "IV 表与 risk_score 时序不受影响。")
            return
        sub_code = sub["股票代码"].iloc[0] if not sub.empty else scores["股票代码"].iloc[0]
        fm_sub = fm[fm["股票代码"] == sub_code].sort_values("交易日期")
        dates = list(fm_sub["交易日期"].dt.date)
        if dates:
            pick_date = st.select_slider("选择周", options=dates, value=dates[-1], key="sc_date")
            row = fm_sub[fm_sub["交易日期"] == pd.Timestamp(pick_date)]
            bins = scorecard_bins(points)
            contrib = []
            for var in bins["变量"].unique():
                if var not in row.columns:
                    continue
                val = row[var].iloc[0]
                hit = bins[(bins["变量"] == var) & (val > bins["lo"]) & (val <= bins["hi"])]
                contrib.append({
                    "变量": var,
                    "取值": round(float(val), 4),
                    "命中分档": hit["分档"].iloc[0] if not hit.empty else "—（未命中）",
                    "本档得分": hit["本档得分"].iloc[0] if not hit.empty else np.nan,
                })
            cf = pd.DataFrame(contrib)
            st.dataframe(cf, hide_index=True)
            total = cf["本档得分"].sum()
            ref = scores[(scores["股票代码"] == sub_code) & (scores["交易日期"] == pd.Timestamp(pick_date))]
            st.caption(
                f"{pick_name} @ {pick_date}：分档得分合计 **{total:.2f}**"
                + (f"；`scorecard_scores_v2.csv` 所载 risk_score = **{ref['risk_score'].iloc[0]:.2f}**"
                   if not ref.empty else "（该周不在评分卡分数表内）")
            )
        else:
            st.info("该公司的特征矩阵无记录。")


main()
