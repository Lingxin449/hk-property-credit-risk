"""
SHAP × WOE/IV 双轨互补可解释对照（阶段 4 · 创新点二）
运行：python -m src.eval.shap_vs_iv

把两条解释轨并排对照：
    - SHAP 轨：阶段 3 已冻结 LightGBM 的**多变量、含交互**归因（preprocessed/shap_importance.csv，
      方向取 preprocessed/shap_direction.csv）
    - WOE/IV 轨：阶段 4 评分卡的**单变量、分档**归因（preprocessed/scorecard_iv.csv）
输出「一致 / 方向分歧 / IV 强 SHAP 弱 / SHAP 强 IV 弱」对照表 + Spearman 秩相关 + 分歧项单独解释。

纪律（内部路线图创新点二）：两轨**同一份数据、同一套标签、并不独立**，
故本对照定位为「互补解释 + 时点稳定性」，**不主张"两轨一致即排除伪相关"**。
重点复核阶段 3 已登记的 3 个方向分歧项：成交额 / 经营现金流_净负债 / 流动比率。
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import OUTPUT_DIR, PROJECT_ROOT, active_suffix, log

SUFFIX = active_suffix()   # '' 一期 / '_v2' 二期

SHAP_IMP = os.path.join(OUTPUT_DIR, f'shap_importance{SUFFIX}.csv')
SHAP_DIR = os.path.join(OUTPUT_DIR, f'shap_direction{SUFFIX}.csv')
IV_IN = os.path.join(OUTPUT_DIR, f'scorecard_iv{SUFFIX}.csv')
OUT_CSV = os.path.join(OUTPUT_DIR, f'shap_vs_iv{SUFFIX}.csv')
REPORT_OUT = os.path.join(OUTPUT_DIR, f'shap_vs_iv_report{SUFFIX}.txt')
FIG_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')
FIG_OUT = os.path.join(FIG_DIR, f'shap_vs_iv_crosscheck{SUFFIX}.png')

# SHAP 强 = 排名进前 8（与评分卡"取 IV 前 6-8 维"对齐）
SHAP_STRONG_RANK = 8
REGISTERED_DIVERGENCE = ['成交额', '经营现金流_净负债', '流动比率']


def _read(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {path}")
    return pd.read_csv(path, encoding='utf-8-sig')


def _norm_dir(s):
    """把"值越高/值越大→风险越高/低"归一为统一标记，便于两轨比较。"""
    s = str(s)
    if '风险越高' in s:
        return 'risk_up'
    if '风险越低' in s:
        return 'risk_down'
    return 'na'


def build_table(shap_imp, shap_dir, iv_tbl):
    """合并两轨：SHAP 排名/方向 与 IV 排名/方向，标注一致/分歧类别。"""
    imp = shap_imp.sort_values('mean_abs_SHAP', ascending=False).reset_index(drop=True)
    imp['SHAP排名'] = imp.index + 1
    iv = iv_tbl.sort_values('IV', ascending=False).reset_index(drop=True)
    iv['IV排名'] = iv.index + 1

    tbl = imp[['Feature', 'mean_abs_SHAP', 'SHAP排名']].merge(
        iv[['Feature', 'IV', 'IV排名', 'WOE方向', '入选', 'PSI', '稳定']],
        on='Feature', how='inner')
    tbl = tbl.merge(shap_dir[['Feature', '方向']].rename(columns={'方向': 'SHAP方向'}),
                    on='Feature', how='left')

    tbl['方向一致'] = tbl['SHAP方向'].map(_norm_dir) == tbl['WOE方向'].map(_norm_dir)

    def label(r):
        if not r['方向一致']:
            return '方向分歧'
        ss, si = r['SHAP排名'] <= SHAP_STRONG_RANK, bool(r['入选'])
        if ss and si:
            return '双轨一致(强)'
        if ss and not si:
            return 'SHAP强·IV弱(疑交互)'
        if si and not ss:
            return 'IV强·SHAP弱'
        return '双轨一致(弱)'

    tbl['对照类别'] = tbl.apply(label, axis=1)
    tbl['mean_abs_SHAP'] = tbl['mean_abs_SHAP'].round(4)
    tbl['IV'] = tbl['IV'].round(4)
    return tbl


def plot_rank_scatter(tbl, out_path, report_lines):
    """SHAP 秩 vs IV 秩散点图（对角线=完全一致；偏离=分歧）。"""
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

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    color_map = {'双轨一致(强)': '#2c7fb8', '双轨一致(弱)': '#9ecae1', '方向分歧': '#d62728',
                 'SHAP强·IV弱(疑交互)': '#ff7f0e', 'IV强·SHAP弱': '#9467bd'}
    ax.plot([0.5, len(tbl) + 0.5], [0.5, len(tbl) + 0.5], 'k--', lw=1, label='秩完全一致')
    for cls, grp in tbl.groupby('对照类别'):
        ax.scatter(grp['SHAP排名'], grp['IV排名'], s=70, color=color_map.get(cls, '#7f7f7f'),
                   label=cls, edgecolor='k', linewidth=0.5)
    for _, r in tbl.iterrows():
        ax.annotate(r['Feature'], (r['SHAP排名'], r['IV排名']), fontsize=8,
                    xytext=(4, 4), textcoords='offset points')
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('SHAP 重要度排名（1 = 最重要，向左更强）')
    ax.set_ylabel('IV 排名（1 = 最重要，向上更强）')
    ax.set_title('SHAP × WOE/IV 双轨秩对照（越接近对角=两轨越一致）')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"已保存双轨对照图：{out_path}", report_lines)


def main():
    report_lines = []
    log("=" * 60, report_lines)
    log("阶段 4 · SHAP × WOE/IV 双轨互补对照（创新点二）", report_lines)

    tbl = build_table(_read(SHAP_IMP), _read(SHAP_DIR), _read(IV_IN))
    log(f"\n对照 {len(tbl)} 维特征（SHAP 前 {SHAP_STRONG_RANK} 视为强 / IV≥0.02 视为入选）：", report_lines)
    log(tbl[['Feature', 'SHAP排名', 'mean_abs_SHAP', 'IV排名', 'IV', 'SHAP方向', 'WOE方向', '对照类别']]
        .to_string(index=False), report_lines)

    rho, p = spearmanr(tbl['SHAP排名'], tbl['IV排名'])
    log(f"\n两轨秩相关（Spearman ρ）＝ {rho:.4f}（p={p:.4f}，n={len(tbl)}）", report_lines)
    log("ρ 中等偏低说明两轨**并不等价**：SHAP 反映多变量/交互归因，IV 反映单变量分档判别力，"
        "恰是「互补」而非「互证」（不主张一致即排除伪相关）。", report_lines)

    log("\n各类别计数：", report_lines)
    log(tbl['对照类别'].value_counts().rename_axis('对照类别').reset_index(name='数量')
        .to_string(index=False), report_lines)

    # 双轨共同锁定的核心风险因子
    common = tbl[tbl['对照类别'] == '双轨一致(强)']['Feature'].tolist()
    log(f"\n【双轨共同锁定】两轨均强且方向一致的核心风险因子：{common}", report_lines)

    # 方向分歧项复核：列本次全部分歧项；一期登记的 3 项附标注（口径漂移时可能不在表中）
    log("\n【方向分歧项】（一期登记的 3 项附复核标注）：", report_lines)
    div = tbl[tbl['对照类别'] == '方向分歧']
    if div.empty:
        log("  （本次无方向分歧项：两轨方向全体一致）", report_lines)
    for _, r in div.iterrows():
        mark = '〔一期登记项〕' if r['Feature'] in REGISTERED_DIVERGENCE else ''
        log(f"  - {r['Feature']}{mark}：SHAP 方向={r['SHAP方向']}（排名 {int(r['SHAP排名'])}），"
            f"WOE/IV 方向={r['WOE方向']}（IV {r['IV']:.3f}，排名 {int(r['IV排名'])}）", report_lines)
    missing_reg = [f for f in REGISTERED_DIVERGENCE if f not in set(tbl['Feature'])]
    if missing_reg:
        log(f"  （一期登记项本期不在对照表中：{missing_reg}）", report_lines)

    plot_rank_scatter(tbl, FIG_OUT, report_lines)

    tbl.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    log(f"\n已保存对照表：{OUT_CSV}", report_lines)
    with open(REPORT_OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    return tbl


if __name__ == '__main__':
    main()
