"""
测试期最终评估（阶段 3 · 第 5 步）
运行：python -m src.eval.evaluate_test

**测试期只跑一次**（AGENTS §4）：加载冻结的模型/校准器/阈值，对测试期做一次评估，不得回头调参。
三方阈值对比：固定 0.5 / AUC 最优（留出段 Youden）/ 代价最优（冻结）。
如实报告测试集 AUC / Precision / Recall / F1、召回率@P≥50%、事件级命中率「N 中 X」。

本模块不做任何拟合选择：模型与阈值均已冻结；校准器 fit 于留出验证段，测试期仅 transform。
"""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import (
    OUTPUT_DIR, PROJECT_ROOT, ID_COLS, EVENTS_2025,
    TEST_SEGMENT_START, TEST_SEGMENT_START_V2, SUFFIX_V2, active_suffix, log
)
from src.labels.build_labels import merge_events_into_clusters, load_events_combined
from src.models.calibrate import apply_isotonic, fit_isotonic

SUFFIX = active_suffix()          # '' 一期 / '_v2' 二期
IS_V2 = SUFFIX == SUFFIX_V2
TEST_START = TEST_SEGMENT_START_V2 if IS_V2 else TEST_SEGMENT_START   # 测试期事件簇判据（锚点 ≥ 此日）

# ===== 路径 =====
MODEL_OUT = os.path.join(OUTPUT_DIR, f'lgbm_model{SUFFIX}.txt')
SCORES_IN = os.path.join(OUTPUT_DIR, f'lgbm_scores{SUFFIX}.csv')
SPLIT_TRAIN = os.path.join(OUTPUT_DIR, f'split_train{SUFFIX}.csv')
SPLIT_TEST = os.path.join(OUTPUT_DIR, f'split_test{SUFFIX}.csv')
FROZEN_IN = os.path.join(OUTPUT_DIR, f'frozen_threshold{SUFFIX}.csv')
EVENTS_FILE = os.path.join(PROJECT_ROOT, 'data/events/events.csv')

NON_FEATURE_COLS = ID_COLS + ['Is_Anomaly', 'y']
PRE_WEEKS = 8


def load_events():
    """读取用于事件级命中评估的事件清单：一期 = events.csv；二期 = events.csv + events_2025.csv 合并。"""
    if IS_V2:
        df, _ = load_events_combined([EVENTS_FILE, EVENTS_2025])
        return df
    df = pd.read_csv(EVENTS_FILE, encoding='utf-8-sig')
    df['公告日'] = pd.to_datetime(df['公告日'])
    return df


def _read(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    if '交易日期' in df.columns:
        df['交易日期'] = pd.to_datetime(df['交易日期'])
    return df


def eval_flag(flag, y):
    """由布尔 flag 掩码输出混淆与指标。"""
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {'flagged': int(flag.sum()), 'flag率': round(int(flag.sum()) / len(y), 4),
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'Precision': round(p, 4), 'Recall': round(r, 4), 'F1': round(f1, 4)}


def youden_threshold(y, proba):
    """在给定集合上按 Youden J = TPR − FPR 最大选阈值（只在留出段用）。"""
    order = np.argsort(proba)
    y_sorted = y[order]
    P, N = int(y.sum()), int((y == 0).sum())
    tpr = np.cumsum(y_sorted) / P
    fpr = np.cumsum(1 - y_sorted) / N
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(proba[order][idx]), float(j[idx])


def recall_at_precision(y, proba, min_prec=0.5):
    """召回率@Precision≥min_prec：按分数降序，可达到的最高召回（precision 达标前提下）。"""
    order = np.argsort(-proba)
    ys = y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    prec = tp / k
    rec = tp / y.sum()
    ok = prec >= min_prec
    return float(rec[ok].max()) if ok.any() else 0.0


def event_level_hit_rate(events_df, test_df, raw, cutoff):
    """事件级命中率：测试期每个事件簇的「锚点前 8 周」窗口内，是否至少一周被 flag。

    返回 (命中簇数, 总簇数, 明细表)。簇按公司全事件序列合并后取锚点 ≥ 测试期起始（TEST_START）者；
    一期 TEST_START=2023-01-01、二期 =2025-01-01。
    """
    start_ts = pd.Timestamp(TEST_START)
    pre = pd.Timedelta(days=PRE_WEEKS * 7)
    rows = []
    for code, grp in events_df.groupby('股票代码'):
        clusters = merge_events_into_clusters(grp['公告日'].tolist())
        firm = test_df[test_df['股票代码'] == code]
        firm_raw = raw[test_df['股票代码'].to_numpy() == code]
        for a, _e in clusters:
            if a < start_ts:
                continue
            win = (firm['交易日期'] >= a - pre) & (firm['交易日期'] <= a)
            if win.sum() == 0:
                rows.append({'股票代码': code, '锚点': pd.Timestamp(a).date(),
                             '窗口周数': 0, '命中': False})
                continue
            hit = bool((firm_raw[win.to_numpy()] >= cutoff).any())
            rows.append({'股票代码': code, '锚点': pd.Timestamp(a).date(),
                         '窗口周数': int(win.sum()), '命中': hit})
    detail = pd.DataFrame(rows)
    return int(detail['命中'].sum()), len(detail), detail


def main():
    report_lines = []
    log("=" * 60, report_lines)
    log("阶段 3 · 测试期最终评估（只跑一次）", report_lines)

    # 1) 加载冻结模型 + 特征顺序
    features = [c for c in _read(SPLIT_TRAIN).columns if c not in NON_FEATURE_COLS]
    booster = lgb.Booster(model_file=MODEL_OUT)

    # 2) 留出段拟合校准器（等渗），供测试期 transform
    scores = _read(SCORES_IN)
    hold = scores[scores['分段'] == '留出验证段']
    raw_h = hold['proba_raw'].to_numpy()
    y_h = hold['y'].astype(int).to_numpy()
    iso = fit_isotonic(raw_h, y_h)

    # 3) 测试期原始分数（模型已冻结，仅 predict）
    test = _read(SPLIT_TEST)
    y_t = test['y'].astype(int).to_numpy()
    raw_t = booster.predict(test[features])
    p_t = apply_isotonic(iso, raw_t)
    log(f"\n测试期：{len(test)} 行 / 正例 {int(y_t.sum())}（{y_t.mean():.2%}）；"
        f"原始分数范围 {raw_t.min():.4f} ~ {raw_t.max():.4f}", report_lines)

    # 4) 三条阈值（均在留出段/固定规则上确定，测试期不参与选择）
    frozen = _read(FROZEN_IN)
    score_cut = float(frozen['原始分数阈值'].iloc[0])
    thr_cost = float(apply_isotonic(iso, np.array([score_cut]))[0])   # 分数域阈值→校准概率域
    thr_youden, j = youden_threshold(y_h, apply_isotonic(iso, raw_h))  # 留出段 Youden
    thr_fixed = 0.5
    log(f"\n阈值（均来自留出段/固定规则）：固定0.5={thr_fixed:.3f}；"
        f"AUC最优(Youden,J={j:.3f})={thr_youden:.3f}；代价最优(冻结分数域 {score_cut:.4f})→校准概率 {thr_cost:.3f}",
        report_lines)

    # 5) 测试集 AUC（如实报告）
    auc = roc_auc_score(y_t, raw_t)
    log(f"\n测试集 AUC = {auc:.4f}（新指标锚点：<0.65 返工 / 0.65-0.85 正常 / >0.90 查泄露）", report_lines)

    # 6) 三方阈值在测试期对比
    rows = []
    for name, thr in [('固定 0.5', thr_fixed), ('AUC 最优', thr_youden), ('代价最优(冻结)', thr_cost)]:
        m = {'阈值方案': name, '阈值(校准概率)': round(thr, 4)}
        m.update(eval_flag(p_t >= thr, y_t))
        rows.append(m)
    cmp_table = pd.DataFrame(rows)
    log("\n三方阈值在测试期的对比：", report_lines)
    log(cmp_table.to_string(index=False), report_lines)

    # 7) 召回率@P≥50%
    r_at_p50 = recall_at_precision(y_t, raw_t, min_prec=0.5)
    log(f"\n测试期 召回率@Precision≥50% = {r_at_p50:.4f}", report_lines)

    # 8) 事件级命中率「N 中 X」（按冻结的代价最优阈值）
    events = load_events()
    hits, total, detail = event_level_hit_rate(events, test, raw_t, score_cut)
    log(f"\n事件级命中率（冻结阈值）：{total} 中 {hits}"
        f"（命中率 {hits / total:.2%}）", report_lines)
    log(detail.to_string(index=False), report_lines)

    # 8.5) 诊断明细：PR 曲线、Top 分数、分公司预警
    scored = test[['股票代码', '公司简称', '交易日期', 'y']].copy()
    scored['proba_raw'] = raw_t
    scored['proba_cal'] = p_t
    scored['flagged'] = p_t >= thr_cost
    scored.to_csv(os.path.join(OUTPUT_DIR, f'test_scores{SUFFIX}.csv'), index=False, encoding='utf-8-sig')

    pr_rows = []
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]:
        m = {'阈值(校准概率)': t}
        m.update(eval_flag(p_t >= t, y_t))
        pr_rows.append(m)
    log("\n测试期 PR 曲线（按校准概率阈值）：", report_lines)
    log(pd.DataFrame(pr_rows).to_string(index=False), report_lines)

    log("\n测试期分数最高的 10 行（raw 分数降序）：", report_lines)
    top10 = scored.sort_values('proba_raw', ascending=False).head(10)
    log(top10.to_string(index=False), report_lines)

    firm = (scored.groupby(['股票代码', '公司简称'])
            .agg(测试行数=('y', 'size'), 正例=('y', 'sum'), 已预警=('flagged', 'sum'))
            .assign(预警率=lambda d: (d['已预警'] / d['测试行数']).round(3))
            .reset_index().sort_values('正例', ascending=False))
    log(f"\n测试期分公司预警明细（全部 {scored['股票代码'].nunique()} 家）：", report_lines)
    log(firm.to_string(index=False), report_lines)

    # 9) 落盘
    with open(os.path.join(OUTPUT_DIR, f'test_eval_report{SUFFIX}.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    cmp_table.to_csv(os.path.join(OUTPUT_DIR, f'test_threshold_compare{SUFFIX}.csv'),
                     index=False, encoding='utf-8-sig')

    log("\n【说明】测试期仅评估、不参与任何 fit 或选择；模型/校准器/阈值均已冻结。", report_lines)
    return cmp_table, auc, (hits, total)


if __name__ == '__main__':
    main()
