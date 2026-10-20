"""
SHAP 可解释性分析（阶段 3 · 第 6 步）
运行：python -m src.eval.shap_analysis

- 全局重要性：`shap.TreeExplainer` 在**事件前 8 周正窗口**上的平均|SHAP|排序
  → 回答业务问题「违约前 8 周，模型认为最重要的预警信号是什么？」
- 蜂群图：正窗口（训练段 + 留出验证段）逐样本 SHAP 分布
- 个案归因（waterfall）：
    一期：合景泰富 1813.HK（事件 2022-09-02，留出验证段）/ 碧桂园 2007.HK（2023-10-10，测试期）
    二期：富力地产 2777.HK（事件 2024-08-11，留出段 2024）/ 合景泰富 1813.HK（2025-08-13，考卷 2025）

本模块只解释已冻结模型，不重训、不调参；SHAP 只用特征、不用标签，故不构成对测试期的信息泄露。
"""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR, PROJECT_ROOT, ID_COLS, SUFFIX_V2, active_suffix, log

SUFFIX = active_suffix()          # '' 一期 / '_v2' 二期
IS_V2 = SUFFIX == SUFFIX_V2

MODEL_OUT = os.path.join(OUTPUT_DIR, f'lgbm_model{SUFFIX}.txt')
SPLIT_TRAIN = os.path.join(OUTPUT_DIR, f'split_train{SUFFIX}.csv')
SPLIT_VALID = os.path.join(OUTPUT_DIR, f'split_valid{SUFFIX}.csv')
SPLIT_TEST = os.path.join(OUTPUT_DIR, f'split_test{SUFFIX}.csv')
IMPORTANCE_OUT = os.path.join(OUTPUT_DIR, f'shap_importance{SUFFIX}.csv')
FIG_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')

NON_FEATURE_COLS = ID_COLS + ['Is_Anomaly', 'y']
PRE_WEEKS = 8

# 个案：公司 / 事件日 / 所在分段（一期：留出段 + 测试期；二期：留出段 2024 + 考卷 2025）
if IS_V2:
    CASES = [
        {'code': '2777.HK', 'name': '富力地产', 'event': '2024-08-11', 'split': SPLIT_VALID},
        {'code': '1813.HK', 'name': '合景泰富', 'event': '2025-08-13', 'split': SPLIT_TEST},
    ]
else:
    CASES = [
        {'code': '1813.HK', 'name': '合景泰富', 'event': '2022-09-02', 'split': SPLIT_VALID},
        {'code': '2007.HK', 'name': '碧桂园', 'event': '2023-10-10', 'split': SPLIT_TEST},
    ]


def _read(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    return df


def _setup_font(report_lines):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        return plt
    except ImportError:
        log("【警告】未安装 matplotlib，跳过出图", report_lines)
        return None


def compute_shap(booster, X):
    """用 TreeExplainer 计算 SHAP 值，返回 (values, base_value)。

    shap 对 LightGBM 二分类可能返回 [class0, class1] 列表，此处统一取正类（class 1）。
    """
    import shap
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    base = float(np.ravel(explainer.expected_value)[-1])
    return np.asarray(sv), base


def importance_table(sv, features):
    """平均|SHAP| 排序表。"""
    mean_abs = np.abs(sv).mean(axis=0)
    return (pd.DataFrame({'Feature': features, 'mean_abs_SHAP': mean_abs})
            .sort_values('mean_abs_SHAP', ascending=False).reset_index(drop=True))


def plot_beeswarm(sv, X, features, base, out_path, report_lines):
    """蜂群图（逐样本 SHAP 分布）。"""
    import shap
    plt = _setup_font(report_lines)
    if plt is None:
        return
    exp = shap.Explanation(values=sv, base_values=base, data=X.values, feature_names=features)
    shap.plots.beeswarm(exp, max_display=18, show=False)
    plt.title('SHAP 蜂群图：事件前 8 周正窗口（训练段 + 留出验证段）')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"已保存蜂群图：{out_path}", report_lines)


def plot_importance_bar(imp, out_path, report_lines, title):
    """平均|SHAP| 横向条形图。"""
    plt = _setup_font(report_lines)
    if plt is None:
        return
    d = imp.iloc[::-1]  # 让最大在顶部
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    ax.barh(d['Feature'], d['mean_abs_SHAP'], color='#2c7fb8')
    ax.set_xlabel('平均 |SHAP|（模型输出 log-odds 单位）')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"已保存重要度条形图：{out_path}", report_lines)


def plot_waterfall_case(booster, df, case, features, report_lines):
    """对个案的"窗口内分数最高那一周"做 waterfall 归因。"""
    import shap
    plt = _setup_font(report_lines)
    if plt is None:
        return None

    ev = pd.Timestamp(case['event'])
    win = df[(df['股票代码'] == case['code']) &
             (df['交易日期'] >= ev - pd.Timedelta(days=PRE_WEEKS * 7)) &
             (df['交易日期'] <= ev)].copy()
    if win.empty:
        log(f"【跳过】{case['name']} 在 {case['event']} 前 8 周无可用行", report_lines)
        return None

    raw = booster.predict(win[features])
    win = win.assign(proba_raw=raw)
    row = win.loc[win['proba_raw'].idxmax()]

    X = pd.DataFrame([row[features].astype(float).values], columns=features)
    sv, base = compute_shap(booster, X)
    exp = shap.Explanation(values=sv[0], base_values=base, data=X.iloc[0].values,
                           feature_names=features)
    shap.plots.waterfall(exp, max_display=12, show=False)
    plt.title(f"个案归因：{case['name']}（{row['交易日期'].date()}，事件 {case['event']} 前 8 周内）")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"shap_waterfall_{case['name']}{SUFFIX}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"已保存个案 waterfall：{out}"
        f"（该周 raw 分数 {row['proba_raw']:.4f}，y={int(row['y'])}）", report_lines)

    top = pd.DataFrame({'Feature': features, 'SHAP': sv[0]}).sort_values(
        'SHAP', key=lambda s: s.abs(), ascending=False).head(6)
    return {'公司': case['name'], '日期': str(row['交易日期'].date()),
            '事件日': case['event'], 'raw分数': round(float(row['proba_raw']), 4),
            'y': int(row['y']),
            'Top信号': '; '.join(f"{r.Feature}({r.SHAP:+.2f})" for r in top.itertuples())}


def main():
    report_lines = []
    log("=" * 60, report_lines)
    log("阶段 3 · SHAP 可解释性分析（只解释已冻结模型）", report_lines)

    os.makedirs(FIG_DIR, exist_ok=True)
    train = _read(SPLIT_TRAIN)
    valid = _read(SPLIT_VALID)
    features = [c for c in train.columns if c not in NON_FEATURE_COLS]
    booster = lgb.Booster(model_file=MODEL_OUT)

    # 1) 正窗口（事件前 8 周）：训练段 + 留出段 的 y=1 行
    pos = pd.concat([train[train['y'] == 1], valid[valid['y'] == 1]], ignore_index=True)
    log(f"\n正窗口样本：{len(pos)} 行（训练段 {int((train['y']==1).sum())} + "
        f"留出段 {int((valid['y']==1).sum())}）", report_lines)

    sv_pos, base_pos = compute_shap(booster, pos[features])
    imp_pos = importance_table(sv_pos, features)

    # 2) 全训练段（对照）
    sv_all, _ = compute_shap(booster, train[features])
    imp_all = importance_table(sv_all, features)

    log("\n【业务问题】违约前 8 周，模型认为最重要的预警信号（按平均|SHAP|）：", report_lines)
    log(imp_pos.head(10).to_string(index=False), report_lines)
    log("\n（对照）全训练段的平均|SHAP| 前 10：", report_lines)
    log(imp_all.head(10).to_string(index=False), report_lines)

    # 方向性：在全训练段（含正负例）上，各特征"低值组 vs 高值组"的平均 SHAP（正=推高风险）
    dir_rows = []
    q1, q4 = train[features].quantile(0.25), train[features].quantile(0.75)
    for j, f in enumerate(features):
        col = train[f].to_numpy()
        lo, hi = col <= q1[f], col >= q4[f]
        if lo.sum() == 0 or hi.sum() == 0:
            continue
        slo, shi = float(sv_all[lo, j].mean()), float(sv_all[hi, j].mean())
        dir_rows.append({'Feature': f, '低值组SHAP': round(slo, 3), '高值组SHAP': round(shi, 3),
                         '方向': '值越高→风险越高' if shi > slo else '值越高→风险越低'})
    order = {f: i for i, f in enumerate(imp_pos['Feature'])}
    direction = pd.DataFrame(dir_rows)
    direction['_rank'] = direction['Feature'].map(order)
    direction = direction.sort_values('_rank').drop(columns='_rank').reset_index(drop=True)
    log("\n方向性（全训练段，低/高四分位组的平均 SHAP；正=推高风险）：", report_lines)
    log(direction.to_string(index=False), report_lines)
    direction.to_csv(os.path.join(OUTPUT_DIR, f'shap_direction{SUFFIX}.csv'), index=False, encoding='utf-8-sig')

    # 单变量方向对照：各特征在 y=1 / y=0 上的标准化均值（交叉验证 SHAP 方向与金融常识）
    uni = train.groupby('y')[features].mean().T
    uni.columns = ['y=0均值', 'y=1均值']
    uni['差(y1−y0)'] = uni['y=1均值'] - uni['y=0均值']
    uni = uni[['y=0均值', 'y=1均值', '差(y1−y0)']].round(3)
    uni['_rank'] = uni.index.map(order)
    uni = uni.sort_values('_rank').drop(columns='_rank').reset_index().rename(columns={'index': 'Feature'})
    log("\n单变量方向对照（标准化后均值；差>0 表示该特征在正窗口更大）：", report_lines)
    log(uni.to_string(index=False), report_lines)
    uni.to_csv(os.path.join(OUTPUT_DIR, f'shap_univariate{SUFFIX}.csv'), index=False, encoding='utf-8-sig')

    imp_pos.to_csv(IMPORTANCE_OUT, index=False, encoding='utf-8-sig')
    log(f"\n已保存重要度表：{IMPORTANCE_OUT}", report_lines)

    # 3) 出图
    plot_beeswarm(sv_pos, pos[features], features, base_pos,
                  os.path.join(FIG_DIR, f'shap_蜂群图_正窗口{SUFFIX}.png'), report_lines)
    plot_importance_bar(imp_pos.head(15), os.path.join(FIG_DIR, f'shap_重要度_正窗口{SUFFIX}.png'),
                        report_lines, '违约前 8 周最重要预警信号（平均 |SHAP|，正窗口）')

    # 4) 个案归因
    cases = []
    cases.append(plot_waterfall_case(booster, valid, CASES[0], features, report_lines))
    cases.append(plot_waterfall_case(booster, _read(SPLIT_TEST), CASES[1], features, report_lines))
    case_rows = [c for c in cases if c]
    if case_rows:
        log("\n个案归因摘要：", report_lines)
        log(pd.DataFrame(case_rows).to_string(index=False), report_lines)

    with open(os.path.join(OUTPUT_DIR, f'shap_report{SUFFIX}.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    return imp_pos


if __name__ == '__main__':
    main()
