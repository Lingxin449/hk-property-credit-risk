"""
代价敏感阈值（阶段 3 · 第 4 步）
运行：python -m src.models.cost_threshold

在方案 C **留出验证段**上，按期望代价最小（记 C_FP=1、C_FN=k）扫描阈值并冻结。
阈值冻结在**分数域**（LightGBM 原始分数 cut），以抗留出/测试的基础率漂移（口径纪律）。
k 网格 {2,3,5,7,10}；以 k=3 为标定，确认对 k 不敏感后冻结。

本模块只在留出验证段上选择阈值，不触碰测试期；不拟合任何变换统计量。
"""

import os

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR, PROJECT_ROOT, active_suffix, log

SUFFIX = active_suffix()   # '' 一期 / '_v2' 二期

SCORES_IN = os.path.join(OUTPUT_DIR, f'lgbm_scores{SUFFIX}.csv')
CALIB_IN = os.path.join(OUTPUT_DIR, f'calibrated_scores{SUFFIX}.csv')
FROZEN_OUT = os.path.join(OUTPUT_DIR, f'frozen_threshold{SUFFIX}.csv')
FIG_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')
FIG_OUT = os.path.join(FIG_DIR, f'代价敏感性_阈值扫描{SUFFIX}.png')

KS = [2, 3, 5, 7, 10]
CALIB_PRIMARY = 'proba_isotonic'   # 主校准器（等渗），另存 Platt 作对照


def load_holdout():
    """读取留出验证段的原始分数、等渗/Platt 校准概率与标签。"""
    sc = pd.read_csv(SCORES_IN, encoding='utf-8-sig')
    hold = sc[sc['分段'] == '留出验证段'].reset_index(drop=True)
    raw = hold['proba_raw'].to_numpy()
    y = hold['y'].astype(int).to_numpy()

    cal = pd.read_csv(CALIB_IN, encoding='utf-8-sig')
    # calibrated_scores.csv 的行序与 hold 可能不同（校准模块按日期排过序），按 股票代码+交易日期 对齐
    cal = cal.copy()
    merged = hold[['股票代码', '交易日期']].merge(
        cal, on=['股票代码', '交易日期'], how='left', validate='one_to_one')
    return raw, y, merged


def cost_curve(raw, y, k):
    """按原始分数 cut 扫描阈值：flag = raw >= t。返回各阈值的 FN/FP/代价（每样本）。"""
    n = len(y)
    pos = int((y == 1).sum())
    ts = np.unique(raw)
    rows = []
    # t 取每个唯一分数；另加"高于最大值"以表示一个都不 flag
    for t in list(ts) + [ts.max() + 1e-9]:
        flag = raw >= t
        tp = int((flag & (y == 1)).sum())
        fp = int((flag & (y == 0)).sum())
        fn = int((~flag & (y == 1)).sum())
        rows.append({'cutoff': float(t) if t <= ts.max() else np.inf,
                     'flagged': int(flag.sum()),
                     'tp': tp, 'fp': fp, 'fn': fn,
                     'cost_per_sample': (k * fn + 1 * fp) / n})
    return pd.DataFrame(rows)


def select_best(curve):
    """期望代价最小的阈值；并列时代价相同取更少 flag 的（更保守）。"""
    min_cost = curve['cost_per_sample'].min()
    best = curve[np.isclose(curve['cost_per_sample'], min_cost)]
    return best.iloc[best['flagged'].argmin()]


def eval_flag(flag, y, k):
    """由布尔 flag 掩码输出混淆与指标。"""
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    n = len(y)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {'flagged': int(flag.sum()), 'flag率': round(int(flag.sum()) / n, 4),
            'tp': tp, 'fp': fp, 'fn': fn,
            'Precision': round(p, 4), 'Recall': round(r, 4), 'F1': round(f1, 4),
            'cost_per_sample': round((k * fn + fp) / n, 4)}


def eval_cutoff(raw, y, cutoff, k):
    """给定分数域阈值（flag = raw >= cutoff）输出指标。cutoff=inf 表示一个都不 flag。"""
    flag = (raw >= cutoff) if np.isfinite(cutoff) else np.zeros(len(y), dtype=bool)
    out = {'cutoff': round(float(cutoff), 4) if np.isfinite(cutoff) else np.inf}
    out.update(eval_flag(flag, y, k))
    return out


def plot_cost_curves(curves, report_lines):
    """画期望代价随分数域阈值变化的曲线（每个 k 一条）+ 标出最优点。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except ImportError:
        log("【警告】未安装 matplotlib，跳过出图", report_lines)
        return

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']
    for i, (k, (curve, best)) in enumerate(curves.items()):
        c = curve[np.isfinite(curve['cutoff'])]
        ax.plot(c['cutoff'].to_numpy(), c['cost_per_sample'].to_numpy(),
                color=colors[i % len(colors)], lw=1.8, label=f'k={k}')
        if np.isfinite(best['cutoff']):
            ax.scatter([best['cutoff']], [best['cost_per_sample']], color=colors[i % len(colors)],
                       s=60, zorder=5, edgecolor='k')
    ax.set_xlabel('LightGBM 原始分数阈值（flag = 分数 ≥ 阈值）')
    ax.set_ylabel('期望代价 / 样本（C_FP=1, C_FN=k）')
    ax.set_title('代价敏感性：期望代价随分数域阈值的变化（留出验证段）')
    ax.grid(alpha=0.3)
    ax.legend(title='代价比 k')
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=150)
    plt.close(fig)
    log(f"已保存代价敏感性图：{FIG_OUT}", report_lines)


def main():
    report_lines = []
    log("=" * 60, report_lines)
    log("阶段 3 · 代价敏感阈值（留出验证段，分数域扫描）", report_lines)

    raw, y, merged = load_holdout()
    n = len(y)
    log(f"\n留出验证段：{n} 行 / 正例 {int(y.sum())}（{y.mean():.2%}）；"
        f"原始分数范围 {raw.min():.4f} ~ {raw.max():.4f}", report_lines)

    # 每个 k 选期望代价最小的分数域阈值
    rows, curves = [], {}
    for k in KS:
        curve = cost_curve(raw, y, k)
        best = select_best(curve)
        row = {'k': k, '理论阈值p*=1/(k+1)': round(1 / (1 + k), 4)}
        row.update(eval_cutoff(raw, y, best['cutoff'], k))
        rows.append(row)
        curves[k] = (curve, best)

    table = pd.DataFrame(rows)
    log("\n各 k 的最优分数域阈值（期望代价最小）：", report_lines)
    log(table.to_string(index=False), report_lines)

    # 对照：固定阈值 0.5（作用在等渗校准概率上）
    p_iso = merged[CALIB_PRIMARY].to_numpy()
    fixed05 = eval_flag(p_iso >= 0.5, y, 3)
    log("\n对照·固定阈值 0.5（等渗校准概率 ≥ 0.5 判出险）：", report_lines)
    log(pd.DataFrame([fixed05]).to_string(index=False), report_lines)

    # k 敏感性：最优阈值的 flag 率随 k 变化
    log("\n敏感性：最优阈值的 flag 率随 k 的变化：", report_lines)
    log(table[['k', 'cutoff', 'flag率', 'Recall', 'Precision']].to_string(index=False), report_lines)

    plot_cost_curves(curves, report_lines)

    # 冻结 k=3
    k_frozen = 3
    curve3 = curves[k_frozen][0]
    best3 = curves[k_frozen][1]
    # 分数域阈值对应的等渗校准概率（供解释）
    p_at_cut = float(p_iso[raw >= best3['cutoff']].min()) if (raw >= best3['cutoff']).any() else float('nan')
    frozen = pd.DataFrame([{
        'k': k_frozen, 'C_FN': k_frozen, 'C_FP': 1,
        '原始分数阈值': round(float(best3['cutoff']), 6),
        '对应等渗校准概率': round(p_at_cut, 4),
        'flag率(留出段)': round(best3['flagged'] / n, 4),
        'Recall(留出段)': eval_cutoff(raw, y, best3['cutoff'], k_frozen)['Recall'],
        'Precision(留出段)': eval_cutoff(raw, y, best3['cutoff'], k_frozen)['Precision'],
    }])
    frozen.to_csv(FROZEN_OUT, index=False, encoding='utf-8-sig')
    log(f"\n已冻结 k={k_frozen} 的分数域阈值（{FROZEN_OUT}）：", report_lines)
    log(frozen.to_string(index=False), report_lines)

    log("\n【说明】阈值冻结在**分数域**（原始分数 cut），以抗留出/测试基础率漂移；"
        "测试期只用该冻结阈值，不参与选择（口径纪律）。", report_lines)

    with open(os.path.join(OUTPUT_DIR, f'cost_threshold_report{SUFFIX}.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    return table, frozen


if __name__ == '__main__':
    main()
