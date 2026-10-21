"""
概率校准（阶段 3 · 第 3 步）
运行：python -m src.models.calibrate

在方案 C **留出验证段**上 fit 校准器（Platt / 等渗），把 LightGBM 的原始分数映射为真实出险概率。
输出：校准曲线（reliability diagram，校准前后对比）+ Brier / ECE + 校准后分数。

防泄露（口径纪律）：校准器只 fit 留出验证段（2022-06-02 ~ 2022-11-04）；训练段与测试期仅 transform。
注：可靠性曲线在同一留出段上拟合+展示，属 in-sample，报告须如实声明其乐观性。
"""

import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.config import OUTPUT_DIR, PROJECT_ROOT, active_suffix, log

SUFFIX = active_suffix()   # '' 一期 / '_v2' 二期

# ===== 路径 =====
SCORES_IN = os.path.join(OUTPUT_DIR, f'lgbm_scores{SUFFIX}.csv')
CALIB_OUT = os.path.join(OUTPUT_DIR, f'calibrated_scores{SUFFIX}.csv')
FIG_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')
FIG_OUT = os.path.join(FIG_DIR, f'calibration_before_after{SUFFIX}.png')
FIG_OUT_CV = os.path.join(FIG_DIR, f'calibration_oos_cv{SUFFIX}.png')

N_BINS = 10  # 可靠性图等宽分箱数


def load_scores():
    """读取 LightGBM 原始分数（训练段 + 留出验证段）。"""
    if not os.path.exists(SCORES_IN):
        raise FileNotFoundError(f"未找到 {SCORES_IN}，请先运行 python -m src.models.train_lgbm")
    df = pd.read_csv(SCORES_IN, encoding='utf-8-sig')
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    return df


def fit_platt(raw, y):
    """Platt 校准：对原始分数做一维 Logistic 回归。仅 fit 留出验证段。"""
    lr = LogisticRegression(random_state=42)
    lr.fit(raw.reshape(-1, 1), y)
    return lr


def fit_isotonic(raw, y):
    """等渗校准：拟合单调递增的分段常数映射。仅 fit 留出验证段。"""
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(raw, y)
    return iso


def apply_platt(lr, raw):
    """应用 Platt 校准（仅 transform，不拟合）。"""
    return lr.predict_proba(raw.reshape(-1, 1))[:, 1]


def apply_isotonic(iso, raw):
    """应用等渗校准（仅 transform，不拟合）。"""
    return iso.predict(raw)


def cv_calibrate_oos(raw, y, n_blocks=5):
    """时序分块的**样本外**校准：每块由其余块拟合的校准器预测，返回 (platt_oos, iso_oos)。

    把留出段（须先按交易日期排序）均分为 n_blocks 个连续块；对每块，用其余块 fit 校准器、
    仅对本块 transform。这样得到的是"没见过该点"的校准概率，剔除 in-sample 乐观性
    （等渗在小样本上尤其会高估）。本函数只在留出验证段内拟合，不触碰训练段/测试期。
    """
    n = len(raw)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    p_platt = np.full(n, np.nan)
    p_iso = np.full(n, np.nan)
    for b in range(n_blocks):
        lo, hi = edges[b], edges[b + 1]
        out = np.ones(n, dtype=bool)
        out[lo:hi] = False
        lr = fit_platt(raw[out], y[out])
        iso = fit_isotonic(raw[out], y[out])
        p_platt[lo:hi] = apply_platt(lr, raw[lo:hi])
        p_iso[lo:hi] = apply_isotonic(iso, raw[lo:hi])
    return p_platt, p_iso


def reliability_table(proba, y, n_bins=N_BINS):
    """等宽分箱的可靠性表：每箱的预测均值 vs 实际正例占比。"""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(proba, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        rows.append({
            'bin': f'{edges[b]:.1f}-{edges[b+1]:.1f}',
            'n': n,
            '预测均值': float(proba[m].mean()),
            '实际正例率': float(y[m].mean()),
        })
    return pd.DataFrame(rows)


def brier(proba, y):
    """Brier 分数：概率预测的均方误差，越低越好。"""
    return float(np.mean((proba - y) ** 2))


def ece(proba, y, n_bins=N_BINS):
    """期望校准误差：各箱 |预测均值 − 实际正例率| 按箱样本数加权。"""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(proba, edges[1:-1]), 0, n_bins - 1)
    total, N = 0.0, len(proba)
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        total += n / N * abs(proba[m].mean() - y[m].mean())
    return float(total)


def threshold_metrics(proba, y, thr=0.5):
    """阈值 0.5 下的 Precision / Recall / F1（展示校准如何激活固定阈值）。"""
    pred = (proba >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {'Precision@0.5': round(p, 4), 'Recall@0.5': round(r, 4),
            'F1@0.5': round(f1, 4), 'TP': tp, 'FP': fp, 'FN': fn}


def plot_reliability(curves, report_lines, title, out_path):
    """画可靠性曲线：各方案 + 对角线，保存到 out_path。curves: {名: (proba, y)}。"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 无界面后端
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 中文字体（Windows）
        plt.rcParams['axes.unicode_minus'] = False
    except ImportError:
        log("【警告】未安装 matplotlib，跳过出图（pip install matplotlib 后重跑本模块）", report_lines)
        return

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='完美校准（对角线）')

    palette = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']
    markers = ['o', 's', '^', 'D']
    for i, (name, (proba, yy)) in enumerate(curves.items()):
        tbl = reliability_table(proba, yy)
        ax.plot(tbl['预测均值'], tbl['实际正例率'], marker=markers[i % len(markers)],
                color=palette[i % len(palette)], lw=1.6, ms=6, label=name)

    ax.set_xlabel('模型预测概率（分箱均值）')
    ax.set_ylabel('实际正例占比（出险率）')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"已保存校准曲线：{out_path}", report_lines)


def main():
    report_lines = []
    log("=" * 60, report_lines)
    log("阶段 3 · 概率校准（fit 留出验证段，Platt / 等渗）", report_lines)

    scores = load_scores()
    hold = scores[scores['分段'] == '留出验证段'].reset_index(drop=True)
    raw = hold['proba_raw'].to_numpy()
    y = hold['y'].astype(int).to_numpy()
    log(f"\n留出验证段：{len(hold)} 行 / 正例 {int(y.sum())}（{y.mean():.2%}）；"
        f"原始分数范围 {raw.min():.4f} ~ {raw.max():.4f}", report_lines)

    # fit 校准器（仅留出段）
    lr = fit_platt(raw, y)
    iso = fit_isotonic(raw, y)
    p_platt = apply_platt(lr, raw)
    p_iso = apply_isotonic(iso, raw)

    # 校准质量指标
    qual = pd.DataFrame([
        {'方案': '校准前（原始）', 'Brier': round(brier(raw, y), 4), 'ECE': round(ece(raw, y), 4),
         **threshold_metrics(raw, y)},
        {'方案': 'Platt', 'Brier': round(brier(p_platt, y), 4), 'ECE': round(ece(p_platt, y), 4),
         **threshold_metrics(p_platt, y)},
        {'方案': '等渗', 'Brier': round(brier(p_iso, y), 4), 'ECE': round(ece(p_iso, y), 4),
         **threshold_metrics(p_iso, y)},
    ])
    log("\n校准质量（Brier 越低越好 / ECE 越低越好）：", report_lines)
    log(qual.to_string(index=False), report_lines)

    log("\n校准前 可靠性表（预测均值 vs 实际正例率）：", report_lines)
    log(reliability_table(raw, y).to_string(index=False), report_lines)
    log("\n校准后（Platt）可靠性表：", report_lines)
    log(reliability_table(p_platt, y).to_string(index=False), report_lines)

    plot_reliability({'校准前（原始分数）': (raw, y),
                      '校准后（Platt）': (p_platt, y),
                      '校准后（等渗）': (p_iso, y)}, report_lines,
                     title='可靠性曲线：校准前后对比\n（方案 C 留出验证段，in-sample 校准）',
                     out_path=FIG_OUT)

    log("\n【说明】上图为 in-sample，乐观于真实泛化；测试期只用冻结后的校准器 transform。", report_lines)

    # ===== 样本外（时序分块）对比：剔除 in-sample 乐观性 =====
    hold_sorted = hold.sort_values('交易日期').reset_index(drop=True)
    raw_s = hold_sorted['proba_raw'].to_numpy()
    y_s = hold_sorted['y'].astype(int).to_numpy()
    p_platt_oos, p_iso_oos = cv_calibrate_oos(raw_s, y_s, n_blocks=5)

    oos = pd.DataFrame([
        {'方案': '校准前（原始）', 'Brier(OOS)': round(brier(raw_s, y_s), 4), 'ECE(OOS)': round(ece(raw_s, y_s), 4)},
        {'方案': 'Platt（CV）', 'Brier(OOS)': round(brier(p_platt_oos, y_s), 4), 'ECE(OOS)': round(ece(p_platt_oos, y_s), 4)},
        {'方案': '等渗（CV）', 'Brier(OOS)': round(brier(p_iso_oos, y_s), 4), 'ECE(OOS)': round(ece(p_iso_oos, y_s), 4)},
    ])
    log("\n样本外（时序 5 块交叉）校准质量（越低越好）：", report_lines)
    log(oos.to_string(index=False), report_lines)

    tail = pd.DataFrame([
        {'方案': 'Platt（CV）', 'max': round(float(p_platt_oos.max()), 4),
         'P90': round(float(np.percentile(p_platt_oos, 90)), 4),
         '>0.5 个数': int((p_platt_oos > 0.5).sum()),
         '>0.25 个数': int((p_platt_oos > 0.25).sum())},
        {'方案': '等渗（CV）', 'max': round(float(p_iso_oos.max()), 4),
         'P90': round(float(np.percentile(p_iso_oos, 90)), 4),
         '>0.5 个数': int((p_iso_oos > 0.5).sum()),
         '>0.25 个数': int((p_iso_oos > 0.25).sum())},
    ])
    log("\n校准概率分布（尾部覆盖，样本外）：", report_lines)
    log(tail.to_string(index=False), report_lines)

    log("\n样本外（Platt·CV）可靠性表：", report_lines)
    log(reliability_table(p_platt_oos, y_s).to_string(index=False), report_lines)
    log("\n样本外（等渗·CV）可靠性表：", report_lines)
    log(reliability_table(p_iso_oos, y_s).to_string(index=False), report_lines)

    # 决策相关校准：判定集（p>=p*）的"实际正例率"应≈其"预测均值"（而非≈p*）
    dec_rows = []
    for name, proba in [('Platt(CV)', p_platt_oos), ('等渗(CV)', p_iso_oos)]:
        for k in [2, 3, 5, 7, 10]:
            p_star = 1.0 / (1.0 + k)
            flag = proba >= p_star
            n_flag = int(flag.sum())
            pred_mean = float(proba[flag].mean()) if n_flag else None
            realized = float(y_s[flag].mean()) if n_flag else None
            dec_rows.append({
                '校准器': name, 'k': k, '门槛p*=1/(k+1)': round(p_star, 3),
                '判定数': n_flag,
                '判定集预测均值': round(pred_mean, 3) if pred_mean is not None else None,
                '实际正例率': round(realized, 3) if realized is not None else None,
                '偏差|实际-预测|': round(abs(realized - pred_mean), 3) if n_flag else None,
            })
    log("\n决策相关校准（判定集 实际正例率 vs 预测均值，样本外）：", report_lines)
    log(pd.DataFrame(dec_rows).to_string(index=False), report_lines)

    plot_reliability({'校准前（原始分数）': (raw_s, y_s),
                      '校准后（Platt·CV）': (p_platt_oos, y_s),
                      '校准后（等渗·CV）': (p_iso_oos, y_s)}, report_lines,
                     title='可靠性曲线：校准对比（样本外时序 5 块交叉）',
                     out_path=FIG_OUT_CV)

    log("\n【说明】样本外块用其余块拟合的校准器预测，剔除 in-sample 乐观性；"
        "Platt 更稳、等渗尾部更宽但小样本易过拟合。", report_lines)

    # 保存校准后分数（供第 4 步定阈）
    out = hold[['股票代码', '公司简称', '交易日期', 'y']].copy()
    out['proba_raw'] = raw
    out['proba_platt'] = p_platt
    out['proba_isotonic'] = p_iso
    out.to_csv(CALIB_OUT, index=False, encoding='utf-8-sig')
    log(f"\n已保存校准后分数：{CALIB_OUT}", report_lines)

    with open(os.path.join(OUTPUT_DIR, f'calibration_report{SUFFIX}.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    return qual


if __name__ == '__main__':
    main()
