"""
风险层级单调性检验（用事件标签/事件驱动模型分数复核 4 类人工风险分层）
运行：python -m src.eval.tier_monotonicity            # 一期口径（方案 C）
      python -m src.eval.tier_monotonicity --v2       # 二期口径（方案 E）

背景与定位
----------
`风险层级`（稳健龙头 / 中型开发商 / 高杠杆激进型 / 困境典型）是**人工静态标注**的 4 类
风险分类，不是模型输入特征（见 口径纪律）。本模块检验该人工排序是否被**事件驱动口径**
同向复现——即"模型是否学到了真实的风险序"，是描述性复核，不参与任何 fit / 选模型 / 定阈值。

两个口径
--------
A) **标签口径**：各风险层级的入样行中 y=1（预警窗口）占比、及其事件公司占比是否随层级单调递增。
   直接对应旧的「风险层级异常占比单调性」检验，把伪标签换成事件标签 y。
B) **分数口径（主证据）**：公司级聚合分数随风险层级是否单调递增。

口径约定（先定死，防"挑好看"）
------------------------------
- **只用外样本分数**：留出验证段 + 测试期（模型只 fit 训练段）；训练段分数为 in-sample，
  仅作参考列出、不作证据。二期留出段 = 2024、测试期 = 2025；一期留出段 = 2022-06~12、测试期 = 2023-2024。
- **主统计量**：公司级「入样周中位分数」对风险层级序数（0..3）的 **Spearman 秩相关**（ρ 及其 p）。
- **稳健性**：均值替代中位、仅平静期（y=0）行、按分段单独跑、Kendall τ、Kruskal-Wallis。
- **诚实声明**：公司级样本量小（n≈30）、4 层级样本极不均衡，结论只有一次留出 + 一次考卷背书；
  检验的是**横截面排序**，不验证时序预警提前量（后者由事件级命中率覆盖）。

本模块不拟合任何统计量：分数取自已冻结模型产物，分组取自人工标注列。
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    OUTPUT_DIR, PROJECT_ROOT, SUFFIX_V2, active_suffix, log
)

SUFFIX = active_suffix()          # '' 一期（方案 C）/ '_v2' 二期（方案 E）
IS_V2 = SUFFIX == SUFFIX_V2

# ===== 路径 =====
SPLIT_VALID = os.path.join(OUTPUT_DIR, f'split_valid{SUFFIX}.csv')    # 留出验证段（外样本）
SPLIT_TEST = os.path.join(OUTPUT_DIR, f'split_test{SUFFIX}.csv')      # 测试期（外样本）
SPLIT_TRAIN = os.path.join(OUTPUT_DIR, f'split_train{SUFFIX}.csv')    # 训练段（in-sample，仅参考）
SCORES_IN = os.path.join(OUTPUT_DIR, f'lgbm_scores{SUFFIX}.csv')      # 训练段 + 留出段分数
TEST_SCORES_IN = os.path.join(OUTPUT_DIR, f'test_scores{SUFFIX}.csv') # 测试期分数（只跑一次产物）
OUT_COMPANY = os.path.join(OUTPUT_DIR, f'tier_monotonicity{SUFFIX}.csv')
OUT_SUMMARY = os.path.join(OUTPUT_DIR, f'tier_monotonicity_summary{SUFFIX}.csv')
OUT_REPORT = os.path.join(OUTPUT_DIR, f'tier_monotonicity_report{SUFFIX}.txt')
FIG_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')
FIG_OUT = os.path.join(FIG_DIR, f'tier_monotonicity_model_score{SUFFIX}.png')

# 人工风险层级的有序排列（递增 = 风险越高），用于序数编码
TIERS = ['稳健龙头', '中型开发商', '高杠杆激进型', '困境典型']
TIER_RANK = {t: i for i, t in enumerate(TIERS)}


def _read(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    if '交易日期' in df.columns:
        df['交易日期'] = pd.to_datetime(df['交易日期'])
    return df


def build_out_of_sample_panel():
    """拼出外样本打分面板：留出验证段 + 测试期（均未参与模型 fit）。

    返回 DataFrame：股票代码 / 公司简称 / 风险层级 / 交易日期 / y / proba_raw / 分段。
    """
    valid = _read(SPLIT_VALID)[['股票代码', '公司简称', '风险层级', '交易日期', 'y']]
    test = _read(SPLIT_TEST)[['股票代码', '公司简称', '风险层级', '交易日期', 'y']]

    scores = _read(SCORES_IN)
    hold = scores[scores['分段'] == '留出验证段'][['股票代码', '交易日期', 'proba_raw']]
    test_sc = _read(TEST_SCORES_IN)[['股票代码', '交易日期', 'proba_raw']]

    valid = valid.merge(hold, on=['股票代码', '交易日期'], how='left').assign(分段='留出验证段')
    test = test.merge(test_sc, on=['股票代码', '交易日期'], how='left').assign(分段='测试期')
    panel = pd.concat([valid, test], ignore_index=True)
    missing = int(panel['proba_raw'].isna().sum())
    if missing:
        raise ValueError(f"外样本面板有 {missing} 行未匹配到分数，请核对切分与分数产物是否同口径")
    return panel


def company_scores(panel):
    """公司级聚合：每家公司一行，取入样周分数的中位数与均值（外样本）。

    返回 DataFrame：股票代码 / 公司简称 / 风险层级 / tier_rank / n_rows / y=1 行数 / 中位分数 / 均值分数。
    """
    comp = (panel.groupby(['股票代码', '公司简称', '风险层级'])
            .agg(n_rows=('proba_raw', 'size'),
                 n_y1=('y', 'sum'),
                 中位分数=('proba_raw', 'median'),
                 均值分数=('proba_raw', 'mean'))
            .reset_index())
    comp['tier_rank'] = comp['风险层级'].map(TIER_RANK)
    comp['y1占比'] = (comp['n_y1'] / comp['n_rows']).round(4)
    return comp


def ordered_tests(scores, ranks):
    """对（序数层级, 分数）做单调性检验，返回 dict。

    - Spearman ρ：主统计量（序数与分数的秩相关，ρ>0 = 层级越高分数越高）
    - Kendall τ：稳健性
    - Kruskal-Wallis H：仅检验"四组是否来自同一分布"（不要求有序）
    """
    scores = np.asarray(scores, dtype=float)
    ranks = np.asarray(ranks, dtype=float)
    rho, p_rho = stats.spearmanr(ranks, scores)
    tau, p_tau = stats.kendalltau(ranks, scores)
    groups = [scores[ranks == k] for k in sorted(set(ranks))]
    if all(len(g) > 0 for g in groups) and len(groups) > 1:
        h, p_h = stats.kruskal(*groups)
    else:
        h, p_h = float('nan'), float('nan')
    return {'rho': float(rho), 'p_rho': float(p_rho),
            'tau': float(tau), 'p_tau': float(p_tau),
            'H': float(h), 'p_H': float(p_h), 'n': int(len(scores))}


def tier_score_stats(comp, col='中位分数'):
    """按风险层级汇总公司级分数（家数 / 中位 / 均值 / 最小 / 最大）。"""
    g = (comp.groupby('风险层级')[col]
         .agg(公司数='size', 中位=lambda s: s.median(), 均值='mean',
              最小='min', 最大='max')
         .reindex(TIERS).dropna(how='all').reset_index())
    return g.round(4)


def is_non_decreasing(vals):
    """浮点序列是否非降（留 1e-9 容差）。"""
    return all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))


def plot_tier_scores(comp, test_info, report_lines):
    """出箱线 + 散点图：公司级中位分数按风险层级铺开，标注 Spearman ρ。"""
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
    tiers = [t for t in TIERS if t in set(comp['风险层级'])]
    data = [comp.loc[comp['风险层级'] == t, '中位分数'].to_numpy() for t in tiers]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bp = ax.boxplot(data, tick_labels=tiers, widths=0.5, showfliers=False,
                    medianprops=dict(color='#d62728', lw=2))
    for i, t in enumerate(tiers, start=1):
        y = comp.loc[comp['风险层级'] == t, '中位分数'].to_numpy()
        jitter = (np.random.RandomState(42).rand(len(y)) - 0.5) * 0.22
        ax.scatter(i + jitter, y, s=28, color='#1f77b4', alpha=0.75, zorder=3)
    ax.set_ylabel('公司级入样周中位分数（外样本）')
    ax.set_xlabel('人工风险层级（风险递增 →）')
    ax.set_title(f"风险层级 × 事件驱动模型分数单调性（{test_info}）\n"
                 f"Spearman ρ = {comp.attrs.get('rho', float('nan')):.3f}"
                 f"（p = {comp.attrs.get('p_rho', float('nan')):.4f}，n = {len(comp)}）")
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=150)
    plt.close(fig)
    log(f"已保存单调性图：{FIG_OUT}", report_lines)


def main():
    report_lines = []
    log("=" * 60, report_lines)
    log(f"风险层级单调性检验（{'二期 方案 E' if IS_V2 else '一期 方案 C'}）", report_lines)

    panel = build_out_of_sample_panel()
    seg_names = '留出验证段 + 测试期'
    log(f"\n外样本打分面板（{seg_names}）：{len(panel)} 行 / {panel['股票代码'].nunique()} 家"
        f"（留出段 {(panel['分段'] == '留出验证段').sum()} 行 / 测试期 {(panel['分段'] == '测试期').sum()} 行）",
        report_lines)

    # ---- A) 标签口径：各层级 y=1 占比 ----
    lab = (panel.groupby('风险层级')
           .agg(入样行数=('y', 'size'), 正例行数=('y', 'sum'), 公司数=('股票代码', 'nunique'))
           .reindex(TIERS).reset_index())
    lab['y=1占比'] = (lab['正例行数'] / lab['入样行数']).round(4)
    lab_mono = is_non_decreasing(lab['y=1占比'].tolist())
    log("\n[A] 标签口径：各风险层级入样行中 y=1（预警窗口）占比", report_lines)
    log(lab.to_string(index=False), report_lines)
    log(f"    → y=1 占比随层级{'单调递增 ✓' if lab_mono else '未严格单调 ✗'}", report_lines)

    # ---- B) 分数口径（主证据）：公司级中位分数 ----
    comp = company_scores(panel)
    t = ordered_tests(comp['中位分数'], comp['tier_rank'])
    comp.attrs['rho'], comp.attrs['p_rho'] = t['rho'], t['p_rho']
    stats_tbl = tier_score_stats(comp, '中位分数')
    log("\n[B] 分数口径（主证据）：公司级入样周中位分数 按风险层级（只用外样本）", report_lines)
    log(stats_tbl.to_string(index=False), report_lines)
    med = stats_tbl['中位'].tolist()
    log(f"    层级中位数序列：{med} → {'单调递增 ✓' if is_non_decreasing(med) else '未严格单调 ✗'}", report_lines)
    log(f"    Spearman ρ = {t['rho']:.4f}（p = {t['p_rho']:.4f}）；"
        f"Kendall τ = {t['tau']:.4f}（p = {t['p_tau']:.4f}）；"
        f"Kruskal-Wallis H = {t['H']:.3f}（p = {t['p_H']:.4f}），n = {t['n']} 家", report_lines)

    # ---- 稳健性 ----
    log("\n[C] 稳健性（同口径变体）", report_lines)
    rob = []
    t_mean = ordered_tests(comp['均值分数'], comp['tier_rank'])
    rob.append({'变体': '公司级中位分数（主）', 'ρ': round(t['rho'], 4), 'p': round(t['p_rho'], 4), 'n': t['n']})
    rob.append({'变体': '公司级均值分数', 'ρ': round(t_mean['rho'], 4), 'p': round(t_mean['p_rho'], 4), 'n': t_mean['n']})
    calm = panel[panel['y'] == 0]
    if calm['股票代码'].nunique() >= 3 and calm['风险层级'].nunique() > 1:
        ccalm = company_scores(calm)
        t_calm = ordered_tests(ccalm['中位分数'], ccalm['tier_rank'])
        rob.append({'变体': '仅平静期(y=0)中位分数', 'ρ': round(t_calm['rho'], 4),
                    'p': round(t_calm['p_rho'], 4), 'n': t_calm['n']})
    for seg in ['留出验证段', '测试期']:
        sub = panel[panel['分段'] == seg]
        if sub['风险层级'].nunique() > 1 and sub['股票代码'].nunique() >= 3:
            csub = company_scores(sub)
            ts = ordered_tests(csub['中位分数'], csub['tier_rank'])
            rob.append({'变体': f'仅{seg}中位分数', 'ρ': round(ts['rho'], 4),
                        'p': round(ts['p_rho'], 4), 'n': ts['n']})
    rob_tbl = pd.DataFrame(rob)
    log(rob_tbl.to_string(index=False), report_lines)

    # ---- 参考：训练段（in-sample，不作证据） ----
    in_sample = None
    try:
        tr = _read(SPLIT_TRAIN)[['股票代码', '公司简称', '风险层级', '交易日期', 'y']]
        s = _read(SCORES_IN)
        s_tr = s[s['分段'] == '训练段'][['股票代码', '交易日期', 'proba_raw']]
        tr = tr.merge(s_tr, on=['股票代码', '交易日期'], how='left').dropna(subset=['proba_raw'])
        c_tr = company_scores(tr)
        t_tr = ordered_tests(c_tr['中位分数'], c_tr['tier_rank'])
        in_sample = {'ρ': round(t_tr['rho'], 4), 'p': round(t_tr['p_rho'], 4), 'n': t_tr['n']}
        log(f"\n[参考·不作证据] 训练段（in-sample）公司级中位分数："
            f"Spearman ρ = {t_tr['rho']:.4f}（p = {t_tr['p_rho']:.4f}，n = {t_tr['n']} 家）"
            f"——模型 fit 过该段，数值偏高属预期，不构成泛化证据。", report_lines)
    except Exception as e:  # noqa: BLE001  参考项失败不影响主结论
        log(f"\n[参考·不作证据] 训练段 in-sample 计算跳过：{e}", report_lines)

    # ---- 结论 + 局限 ----
    verdict = '支持' if (t['rho'] > 0 and t['p_rho'] < 0.05 and is_non_decreasing(med)) else '不完全支持'
    log("\n【结论】外样本口径下，事件驱动模型分数随人工风险层级"
        f"{'单调递增且统计显著' if verdict == '支持' else '未完全单调/不显著'}——"
        "即模型学到了与人工风险序一致的横截面排序。", report_lines)
    log("【口径声明】本检验为**描述性复核**：分数取自已冻结模型产物，未做任何 fit / 选型 / 改阈值；"
        "分配/分层口径已在上方先定死（外样本+公司级中位），稳健性变体一并列出。", report_lines)
    log("【局限】① 公司级样本量小（n≈30）、4 层级样本极不均衡；"
        "② 只有单折留出 + 一次考卷背书，跨期稳健性证据有限；"
        "③ 只验证**横截面排序**，不验证时序预警提前量（后者由事件级命中率覆盖）。", report_lines)

    # ---- 落盘 ----
    comp = comp.sort_values(['tier_rank', '中位分数'], ascending=[True, False])
    comp.to_csv(OUT_COMPANY, index=False, encoding='utf-8-sig')

    # 汇总：口径 × 风险层级 的长表（样本量 / 指标值）
    sum_rows = []
    for _, r in lab.iterrows():
        sum_rows.append({'口径': 'A 标签：入样行 y=1 占比', '风险层级': r['风险层级'],
                         '样本量': int(r['入样行数']), '指标值': r['y=1占比'], '单调递增': lab_mono})
    for _, r in stats_tbl.iterrows():
        sum_rows.append({'口径': 'B 分数：公司级中位分数（外样本）', '风险层级': r['风险层级'],
                         '样本量': int(r['公司数']), '指标值': r['中位'],
                         '单调递增': is_non_decreasing(med)})
    pd.DataFrame(sum_rows).to_csv(OUT_SUMMARY, index=False, encoding='utf-8-sig')

    # 检验表：主统计量 + 稳健性 + in-sample 参考（另一张表，便于报告引用）
    tests_tbl = pd.DataFrame([{'口径': 'B 分数（外样本，主）',
                               'rho': round(t['rho'], 4), 'p_rho': round(t['p_rho'], 4),
                               'tau': round(t['tau'], 4), 'p_tau': round(t['p_tau'], 4),
                               'H': round(t['H'], 4), 'p_H': round(t['p_H'], 4), 'n': t['n']}])
    rob_t = rob_tbl.rename(columns={'变体': '口径', 'ρ': 'rho', 'p': 'p_rho'})
    rob_t = rob_t[rob_t['口径'] != '公司级中位分数（主）']   # 主项已单独成行，避免重复
    tests_tbl = pd.concat([tests_tbl, rob_t], ignore_index=True)
    if in_sample:
        tests_tbl = pd.concat(
            [tests_tbl, pd.DataFrame([{'口径': '参考：训练段 in-sample（不作证据）',
                                       'rho': in_sample['ρ'], 'p_rho': in_sample['p'], 'n': in_sample['n']}])],
            ignore_index=True)
    tests_tbl.to_csv(os.path.join(OUTPUT_DIR, f'tier_monotonicity_tests{SUFFIX}.csv'),
                     index=False, encoding='utf-8-sig')

    test_info = '二期：留出 2024 + 考卷 2025' if IS_V2 else '一期：留出 2022H2 + 测试期 2023-2024'
    plot_tier_scores(comp, test_info, report_lines)

    with open(OUT_REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    log(f"\n已保存公司级明细：{OUT_COMPANY}", report_lines)
    log(f"已保存汇总/检验：{OUT_SUMMARY}", report_lines)
    return comp, t, lab


if __name__ == '__main__':
    main()
