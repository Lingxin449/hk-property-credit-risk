"""
WOE/IV 评分卡（阶段 4 · 定位 A：行业通用方法论演示）
运行：python -m src.models.scorecard

流程：
    1) IV 排序   —— 18 维候选特征，分箱边界与 WOE/IV **只在训练段（split_train）拟合**
    2) WOE 分箱  —— 等频初分箱 + 贪心合并强制单调；剔除 IV < 0.02；取 IV 前 8 维
    3) 时点稳定性 —— PSI（训练段 ↔ 留出验证段）与留出段 IV 复核，剔除不稳定变量
    4) 逻辑回归   —— 在训练段 WOE 矩阵上 fit 线性模型（可解释），负系数变量回退剔除
    5) 标准评分卡 —— PDO=20、基准分 600，输出「变量-分档-得分」表
    6) 双方对比   —— 评分卡 vs LightGBM（阶段 3 冻结），测试期**一次性**对照（预声明）

防泄露（口径纪律，最高优先级）：
    - WOE/IV/分箱边界/PSI 只 fit **训练段**（2019-01-01 ~ 2022-05-31，含切点净化带剔除）；
    - 留出验证段仅用于**变量选型**（PSI / 留出 IV 复核），不 fit 分箱；
    - 测试期在变量与分箱**冻结之后**只做一次对照评估，不参与任何 fit 或选择，如实披露。

说明：旧伪标签融合层（IF+GMM+CART+LR 等权）已不再作为对照——其口径为伪标签路线、
与本项目事件标签不一致，旧方法对照职责由阶段 3「伪标签 vs 事件标签」承担。
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import OUTPUT_DIR, PROJECT_ROOT, ID_COLS, active_suffix, log

SUFFIX = active_suffix()   # '' 一期 / '_v2' 二期

# ===== 输入 / 输出路径 =====
SPLIT_TRAIN = os.path.join(OUTPUT_DIR, f'split_train{SUFFIX}.csv')
SPLIT_VALID = os.path.join(OUTPUT_DIR, f'split_valid{SUFFIX}.csv')
SPLIT_TEST = os.path.join(OUTPUT_DIR, f'split_test{SUFFIX}.csv')
TEST_SCORES = os.path.join(OUTPUT_DIR, f'test_scores{SUFFIX}.csv')
TEST_THR_CMP = os.path.join(OUTPUT_DIR, f'test_threshold_compare{SUFFIX}.csv')

IV_OUT = os.path.join(OUTPUT_DIR, f'scorecard_iv{SUFFIX}.csv')
BINS_OUT = os.path.join(OUTPUT_DIR, f'scorecard_woe_bins{SUFFIX}.csv')
POINTS_OUT = os.path.join(OUTPUT_DIR, f'scorecard_points{SUFFIX}.csv')
PSI_OUT = os.path.join(OUTPUT_DIR, f'scorecard_psi{SUFFIX}.csv')
SCORES_OUT = os.path.join(OUTPUT_DIR, f'scorecard_scores{SUFFIX}.csv')
TWOWAY_OUT = os.path.join(OUTPUT_DIR, f'scorecard_twoway{SUFFIX}.csv')
TWOWAY_SCORES_OUT = os.path.join(OUTPUT_DIR, f'scorecard_twoway_scores{SUFFIX}.csv')
TWOWAY_COST_OUT = os.path.join(OUTPUT_DIR, f'scorecard_twoway_cost{SUFFIX}.csv')
TWOWAY_COST_SUM_OUT = os.path.join(OUTPUT_DIR, f'scorecard_twoway_cost_summary{SUFFIX}.csv')
REPORT_OUT = os.path.join(OUTPUT_DIR, f'scorecard_report{SUFFIX}.txt')
FIG_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')

# ===== 参数（机器可读常量；会随进度变的数值事实见 内部数值表）=====
NON_FEATURE_COLS = ID_COLS + ['Is_Anomaly', 'y']
N_INIT_BINS = 10          # 等频初分箱数（贪心合并前）
IV_MIN = 0.02             # IV 入选门槛
MAX_VARS = 8              # 取 IV 前 N 维
PSI_MAX = 0.25            # PSI 时点稳定性门槛（>0.25 判不稳定）
PDO = 20                  # Points to Double the Odds
BASE_SCORE = 600          # 基准分
BASE_ODDS = 1.0           # 基准赔率（600 分对应 odds=1，即 50/50）
K_COST = 3                # 代价视角的业务代价比（与阶段 3 冻结的 k=3 一致）


def _read(path):
    """读取 csv 并把交易日期转 datetime。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {path}")
    df = pd.read_csv(path, encoding='utf-8-sig')
    if '交易日期' in df.columns:
        df['交易日期'] = pd.to_datetime(df['交易日期'])
    return df


def feature_columns(df):
    """推断 18 维建模特征。"""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


# ============================================================
# WOE / IV 分箱（只用训练段拟合）
# ============================================================
def _woe_iv(bad, good):
    """按"坏样本/好样本分布"计算 WOE 与 IV（0.5 平滑防除零）。

    WOE_j = ln( dist_bad_j / dist_good_j )，值越大 → 该档风险越高；
    IV    = Σ (dist_bad_j − dist_good_j) × WOE_j。
    """
    bad_s, good_s = bad + 0.5, good + 0.5
    db, dg = bad_s / bad_s.sum(), good_s / good_s.sum()
    woe = np.log(db / dg)
    return woe, float(np.sum((db - dg) * woe))


def _groups_counts(groups, bad, good):
    """把原始分箱按 group 归并后的坏/好样本计数。"""
    nb = np.array([bad[g].sum() for g in groups], dtype=float)
    ng = np.array([good[g].sum() for g in groups], dtype=float)
    return nb, ng


def _merge_to_monotonic(bad, good, direction):
    """贪心合并相邻分箱直至 WOE 单调（增/减），每次合并代价最小（IV 损失最小）。

    返回归并后的 group 列表（每个 group 是原始分箱索引的列表）。
    """
    groups = [[j] for j in range(len(bad))]
    while len(groups) > 1:
        nb, ng = _groups_counts(groups, bad, good)
        woe, _ = _woe_iv(nb, ng)
        d = np.diff(woe)
        viol = (d < -1e-12) if direction == 'up' else (d > 1e-12)
        if not viol.any():
            break
        best_i, best_iv = None, -np.inf
        for i in np.where(viol)[0]:
            cand = groups[:i] + [groups[i] + groups[i + 1]] + groups[i + 2:]
            nb_c, ng_c = _groups_counts(cand, bad, good)
            _, iv_c = _woe_iv(nb_c, ng_c)
            if iv_c > best_iv:
                best_iv, best_i = iv_c, int(i)
        groups = groups[:best_i] + [groups[best_i] + groups[best_i + 1]] + groups[best_i + 2:]
    return groups


def quantile_edges(x, n_bins=N_INIT_BINS):
    """按训练段分位数取初分箱边界（去重后可能少于 n_bins 个箱）。"""
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    return np.unique(np.quantile(np.asarray(x, dtype=float), qs))


def assign_bin(x, edges):
    """把值映射到分箱索引 0..k-1（右闭区间，越界值夹到端点）。"""
    xc = np.clip(np.asarray(x, dtype=float), edges[0], edges[-1])
    idx = pd.cut(xc, bins=edges, labels=False, include_lowest=True, right=True)
    return np.nan_to_num(np.asarray(idx, dtype=float), nan=0.0).astype(int)


def fit_woe_feature(x, y):
    """在训练段上拟合单变量 WOE 分箱（自动选单调方向：取使 IV 更大的方向）。

    返回 dict：edges(最终分箱边界)、woe(各档 WOE)、iv、direction('up' 值越大风险越高)、
    counts(各档坏/好样本数)。
    """
    edges0 = quantile_edges(x, N_INIT_BINS)
    b = assign_bin(x, edges0)
    k = len(edges0) - 1
    bad = np.array([((b == j) & (y == 1)).sum() for j in range(k)], dtype=float)
    good = np.array([((b == j) & (y == 0)).sum() for j in range(k)], dtype=float)

    best = None
    for direction in ('up', 'down'):
        groups = _merge_to_monotonic(bad, good, direction)
        nb, ng = _groups_counts(groups, bad, good)
        woe, iv = _woe_iv(nb, ng)
        if best is None or iv > best['iv']:
            edges = [edges0[0]] + [edges0[g[-1] + 1] for g in groups]
            best = {'edges': np.array(edges), 'woe': woe, 'iv': iv, 'direction': direction,
                    'bad': nb, 'good': ng}
    return best


def transform_woe(x, fitted):
    """用训练段拟合的分箱把值映射为 WOE（仅 transform）。"""
    return fitted['woe'][assign_bin(x, fitted['edges'])]


def iv_with_frozen_bins(x, y, edges):
    """用训练段冻结的分箱边界、在另一段数据上重算 IV（时点稳定性复核）。"""
    b = assign_bin(x, edges)
    k = len(edges) - 1
    bad = np.array([((b == j) & (y == 1)).sum() for j in range(k)], dtype=float)
    good = np.array([((b == j) & (y == 0)).sum() for j in range(k)], dtype=float)
    return _woe_iv(bad, good)[1]


def psi(expected, actual, edges, eps=1e-6):
    """群体稳定性指标：训练段(expected) 与另一段(actual) 在同一分箱上的分布差异。"""
    be = assign_bin(expected, edges)
    ba = assign_bin(actual, edges)
    k = len(edges) - 1
    pe = np.array([(be == j).mean() for j in range(k)])
    pa = np.array([(ba == j).mean() for j in range(k)])
    pe, pa = np.clip(pe, eps, None), np.clip(pa, eps, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


# ============================================================
# 评分卡刻度（PDO=20、基准分 600）
# ============================================================
def scorecard_scale(pdo=PDO, base_score=BASE_SCORE, base_odds=BASE_ODDS):
    """返回 (factor, offset)：score = offset + factor × ln(odds)。"""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return factor, offset


def score_from_points(base_points, bin_points):
    """由基础分 + 各变量命中档得分求和。"""
    return base_points + float(sum(bin_points))


# ============================================================
# 主流程
# ============================================================
def plot_iv_bar(iv_tbl, out_path, report_lines):
    """IV 排序条形图。"""
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
    d = iv_tbl.iloc[::-1]
    colors = ['#2c7fb8' if s else '#c0c0c0' for s in d['入选']]
    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    ax.barh(d['Feature'], d['IV'], color=colors)
    ax.axvline(IV_MIN, color='#d62728', ls='--', lw=1, label=f'IV 门槛={IV_MIN}')
    ax.set_xlabel('IV（信息量，训练段拟合）')
    ax.set_title('特征 IV 排序（蓝=入选 / 灰=剔除，门槛 IV≥0.02）')
    ax.grid(axis='x', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"已保存 IV 排序图：{out_path}", report_lines)


def plot_twoway_roc(y, score_map, out_path, report_lines):
    """双方（评分卡 / LightGBM）测试期 ROC 叠加图。score_map: {模型名: 分数数组}，分数越高→风险越高。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except ImportError:
        log("【警告】未安装 matplotlib，跳过出图", report_lines)
        return
    from sklearn.metrics import roc_curve

    os.makedirs(FIG_DIR, exist_ok=True)
    colors = {'WOE/IV 评分卡（LR）': '#2c7fb8',
              'LightGBM（阶段 3 冻结）': '#d62728'}
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for name, sc in score_map.items():
        fpr, tpr, _ = roc_curve(y, sc)
        auc = roc_auc_score(y, sc)
        ax.plot(fpr, tpr, color=colors.get(name), lw=1.9, label=f'{name}  AUC={auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='随机基线（AUC=0.5）')
    ax.set_xlabel('假阳性率 FPR')
    ax.set_ylabel('真阳性率 TPR（召回）')
    ax.set_title('测试期 ROC 对比（2023-2024，一次性评估）')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"已保存 ROC 对比图：{out_path}", report_lines)


def cost_curve_by_flagrate(y, score, k, n_points=400):
    """期望代价随**预警频率**变化（按分数降序 flag 前 q 比例）。

    以预警频率为横轴可跨模型比较（分数尺度不同，但按排名 flag 后同口径）。
    代价口径与阶段 3 一致：C_FP=1、C_FN=k，cost = (k×FN + FP) / N。
    """
    order = np.argsort(-np.asarray(score, dtype=float))
    ys = np.asarray(y, dtype=int)[order]
    n, pos = len(ys), int(ys.sum())
    rows = []
    for q in np.linspace(0.0, 1.0, n_points + 1):
        m = int(round(q * n))
        tp = int(ys[:m].sum())
        fp, fn = m - tp, pos - tp
        rows.append({'flag_rate': q, 'cost_per_sample': (k * fn + fp) / n,
                     'recall': tp / pos if pos else 0.0,
                     'precision': tp / m if m else 0.0})
    return pd.DataFrame(rows)


def plot_precision_cost(y, score_map, k, frozen_flag_rate, out_path, report_lines):
    """精度-代价双视角：(a) 三模型测试期 ROC；(b) 期望代价 vs 预警频率（k 固定）。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except ImportError:
        log("【警告】未安装 matplotlib，跳过出图", report_lines)
        return
    from sklearn.metrics import roc_curve

    os.makedirs(FIG_DIR, exist_ok=True)
    colors = {'WOE/IV 评分卡（LR）': '#2c7fb8',
              'LightGBM（阶段 3 冻结）': '#d62728'}
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2))

    # (a) 精度视角：ROC
    ax = axes[0]
    for name, sc in score_map.items():
        fpr, tpr, _ = roc_curve(y, sc)
        ax.plot(fpr, tpr, color=colors.get(name), lw=1.9,
                label=f'{name}  AUC={roc_auc_score(y, sc):.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='随机基线（AUC=0.5）')
    ax.set_xlabel('假阳性率 FPR')
    ax.set_ylabel('真阳性率 TPR（召回）')
    ax.set_title('(a) 精度视角：测试期 ROC 对比')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8.5)

    # (b) 代价视角：期望代价 vs 预警频率
    ax = axes[1]
    for name, sc in score_map.items():
        c = cost_curve_by_flagrate(y, sc, k)
        best = c.loc[c['cost_per_sample'].idxmin()]
        ax.plot(c['flag_rate'], c['cost_per_sample'], color=colors.get(name), lw=1.9,
                label=f'{name}｜最低代价 {best["cost_per_sample"]:.4f} @ 预警率 {best["flag_rate"]:.2f}')
        ax.scatter([best['flag_rate']], [best['cost_per_sample']],
                   color=colors.get(name), s=70, zorder=5, edgecolor='k')
    if frozen_flag_rate is not None:
        ax.axvline(frozen_flag_rate, color='#7f7f7f', ls=':', lw=1.4,
                   label=f'阶段3冻结阈值预警率 {frozen_flag_rate:.1%}')
    ax.set_xlabel('预警频率（按分数降序 flag 的比例）')
    ax.set_ylabel(f'期望代价 / 样本（C_FP=1, C_FN={k}）')
    ax.set_title(f'(b) 代价视角：期望代价随预警频率变化（k={k}）')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper center', fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"已保存精度-代价双视角图：{out_path}", report_lines)


def main():
    report_lines = []
    log("=" * 60, report_lines)
    log("阶段 4 · WOE/IV 评分卡（IV/分箱只 fit 训练段）", report_lines)

    train = _read(SPLIT_TRAIN)
    valid = _read(SPLIT_VALID)
    test = _read(SPLIT_TEST)
    features = feature_columns(train)
    ytr = train['y'].astype(int).to_numpy()
    yva = valid['y'].astype(int).to_numpy()
    yte = test['y'].astype(int).to_numpy()
    log(f"\n训练段 {len(train)} 行 / 正例 {int(ytr.sum())}（{ytr.mean():.2%}）"
        f"｜留出段 {len(valid)} 行 / 正例 {int(yva.sum())}"
        f"｜测试期 {len(test)} 行 / 正例 {int(yte.sum())}", report_lines)
    log(f"候选特征 {len(features)} 维（VIF 后集合）", report_lines)

    # ---------- 1) IV 排序（训练段拟合）----------
    fit = {}
    iv_rows = []
    for f in features:
        fitted = fit_woe_feature(train[f].to_numpy(), ytr)
        fit[f] = fitted
        iv_rows.append({'Feature': f, 'IV': round(fitted['iv'], 4),
                        'WOE方向': '值越大→风险越高' if fitted['direction'] == 'up' else '值越大→风险越低',
                        '档数': len(fitted['woe'])})
    iv_tbl = pd.DataFrame(iv_rows).sort_values('IV', ascending=False).reset_index(drop=True)
    iv_tbl['入选'] = iv_tbl['IV'] >= IV_MIN
    log("\nIV 排序（训练段）：", report_lines)
    log(iv_tbl.to_string(index=False), report_lines)

    # ---------- 2) 选前 8 维 + 时点稳定性（PSI train↔valid）----------
    selected = iv_tbl[iv_tbl['入选']].head(MAX_VARS)['Feature'].tolist()
    log(f"\nIV 前 {MAX_VARS} 维（IV≥{IV_MIN}）：{selected}", report_lines)

    psi_rows = []
    for f in selected:
        edges = fit[f]['edges']
        p = psi(train[f].to_numpy(), valid[f].to_numpy(), edges)
        iv_va = iv_with_frozen_bins(valid[f].to_numpy(), yva, edges)
        # 稳定 = 分布未显著漂移（PSI）且冻结分箱在留出段的信号未坍塌（IV_out ≥ 门槛）
        psi_rows.append({'Feature': f, 'PSI(train↔valid)': round(p, 4),
                         'IV(train)': round(fit[f]['iv'], 4),
                         'IV(valid·冻结分箱)': round(iv_va, 4),
                         '稳定': bool(p <= PSI_MAX and iv_va >= IV_MIN)})
    psi_tbl = pd.DataFrame(psi_rows)
    log(f"\n时点稳定性（不稳定判据：PSI > {PSI_MAX} 或 IV(留出·冻结分箱) < {IV_MIN}）：", report_lines)
    log(psi_tbl.to_string(index=False), report_lines)

    dropped_psi = psi_tbl.loc[~psi_tbl['稳定'], 'Feature'].tolist()
    keep = [f for f in selected if f not in dropped_psi]
    if dropped_psi:
        log(f"\n因时点不稳定剔除：{dropped_psi}", report_lines)
    if not keep:
        raise RuntimeError("时点稳定性检验后无变量留存，请复核 PSI 门槛")

    # 把稳定性结果并回 IV 表
    iv_tbl['PSI'] = iv_tbl['Feature'].map(dict(zip(psi_tbl['Feature'], psi_tbl['PSI(train↔valid)'])))
    iv_tbl['稳定'] = iv_tbl['Feature'].map(dict(zip(psi_tbl['Feature'], psi_tbl['稳定'])))

    # ---------- 3) 逻辑回归（训练段 WOE 矩阵）----------
    def build_woe(df, cols):
        return np.column_stack([transform_woe(df[c].to_numpy(), fit[c]) for c in cols])

    def fit_lr(cols):
        lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs',
                                random_state=42, n_jobs=1)
        lr.fit(build_woe(train, cols), ytr)
        return lr

    lr = fit_lr(keep)
    dropped_neg = []
    it = 0
    while (lr.coef_[0] < 0).any() and it < len(keep) - 1:
        # 负系数 = 该变量在多元模型中的方向与单变量单调 WOE 相悖（共线/交互），回退剔除
        bad = [keep[i] for i in np.where(lr.coef_[0] < 0)[0]]
        dropped_neg += bad
        keep = [f for f in keep if f not in bad]
        lr = fit_lr(keep)
        it += 1
    if dropped_neg:
        log(f"\n因多元系数为负（与单变量单调方向相悖）回退剔除：{dropped_neg}", report_lines)
    coeffs = dict(zip(keep, lr.coef_[0]))
    log(f"\n最终入选变量 {len(keep)} 维：{keep}", report_lines)
    log("逻辑回归系数（均应为正，越大风险越高）：", report_lines)
    log(pd.DataFrame({'Feature': keep, 'coef': np.round(lr.coef_[0], 4)}).to_string(index=False),
        report_lines)

    # ---------- 4) 标准评分卡刻度 ----------
    factor, offset = scorecard_scale()
    base_points = offset + factor * float(lr.intercept_[0])

    points_rows = []
    for i, f in enumerate(keep):
        edges, woe = fit[f]['edges'], fit[f]['woe']
        cnt_bad, cnt_good = fit[f]['bad'], fit[f]['good']
        for j in range(len(woe)):
            lo, hi = edges[j], edges[j + 1]
            pts = factor * coeffs[f] * woe[j]
            points_rows.append({
                '变量': f,
                '分档': f'({lo:.3f}, {hi:.3f}]',
                'WOE': round(float(woe[j]), 4),
                '系数': round(float(coeffs[f]), 4),
                '本档得分': round(float(pts), 2),
                '坏样本数': int(cnt_bad[j]), '好样本数': int(cnt_good[j]),
            })
    points_tbl = pd.DataFrame(points_rows)
    log(f"\n标准评分卡（PDO={PDO}、基准分={BASE_SCORE}、base_odds={BASE_ODDS:g}）：", report_lines)
    log(f"基础分 = {base_points:.2f}（offset={offset:.2f} + factor={factor:.4f} × intercept={lr.intercept_[0]:.4f}）",
        report_lines)
    log(points_tbl.to_string(index=False), report_lines)
    log("\n评分口径：总分 = 基础分 + Σ 命中档得分；分数越高 → 风险越高（odds=p/(1−p)）。", report_lines)

    # ---------- 5) 评分卡打分（三段）----------
    def score_df(df, tag):
        woe_mat = build_woe(df, keep)
        logit = lr.intercept_[0] + woe_mat @ lr.coef_[0]
        sc = offset + factor * logit
        out = df[['股票代码', '公司简称', '交易日期', 'y']].copy()
        out['分段'] = tag
        out['logit'] = logit
        out['risk_score'] = sc
        out['proba_lr'] = 1.0 / (1.0 + np.exp(-logit))
        return out

    scores = pd.concat([score_df(train, '训练段'), score_df(valid, '留出验证段'),
                        score_df(test, '测试期')], ignore_index=True)

    def auc_of(df, tag):
        d = scores[scores['分段'] == tag]
        yy = df['y'].astype(int).to_numpy()
        return round(roc_auc_score(yy, d['risk_score'].to_numpy()), 4)

    log("\n评分卡三段 AUC（分数越高=风险越高）：", report_lines)
    log(pd.DataFrame([
        {'分段': '训练段', 'AUC': auc_of(train, '训练段')},
        {'分段': '留出验证段', 'AUC': auc_of(valid, '留出验证段')},
        {'分段': '测试期', 'AUC': auc_of(test, '测试期')},
    ]).to_string(index=False), report_lines)

    # ---------- 6) 双方对比（测试期一次性）----------
    log("\n" + "-" * 60, report_lines)
    log("【预声明】测试期一次性双方对照：评分卡 / LightGBM。", report_lines)
    log("评分卡的变量、分箱、系数在接触测试期之前已冻结；LightGBM 分数取自阶段 3 冻结模型。", report_lines)
    log("旧伪标签融合层不再作为对照，旧方法对照职责由阶段 3「伪标签 vs 事件标签」承担。", report_lines)

    ts = pd.read_csv(TEST_SCORES, encoding='utf-8-sig')
    ts['交易日期'] = pd.to_datetime(ts['交易日期'])
    yte_s = test[['股票代码', '交易日期', 'y']].copy()
    lgbm = ts[['股票代码', '交易日期', 'proba_raw']].rename(columns={'proba_raw': 'lgbm_raw'})
    merged = yte_s.merge(lgbm, on=['股票代码', '交易日期'], how='left', validate='one_to_one')
    card = scores[scores['分段'] == '测试期'][['股票代码', '交易日期', 'risk_score']]
    merged = merged.merge(card, on=['股票代码', '交易日期'], how='left', validate='one_to_one')
    ym = merged['y'].astype(int).to_numpy()

    twoway = pd.DataFrame([
        {'模型': 'WOE/IV 评分卡（LR）', 'AUC(测试期)': round(roc_auc_score(ym, merged['risk_score'].to_numpy()), 4),
         '可解释性': '单变量分箱+线性，最强'},
        {'模型': 'LightGBM（阶段 3 冻结）', 'AUC(测试期)': round(roc_auc_score(ym, merged['lgbm_raw'].to_numpy()), 4),
         '可解释性': '非线性+交互，SHAP 归因'},
    ])
    log("\n测试期双方对比（可解释性-精度权衡）：", report_lines)
    log(twoway.to_string(index=False), report_lines)
    log(f"对照行数 {len(merged)}（事件标签测试期入样行）。", report_lines)

    # ROC 对比图（同一冻结结果，只出图不改口径）
    score_map = {
        'WOE/IV 评分卡（LR）': merged['risk_score'].to_numpy(),
        'LightGBM（阶段 3 冻结）': merged['lgbm_raw'].to_numpy(),
    }
    plot_twoway_roc(ym, score_map, os.path.join(FIG_DIR, f'scorecard_vs_lgbm_roc{SUFFIX}.png'), report_lines)

    # 精度-代价双视角：代价曲线（k=K_COST，横轴=预警频率）+ 汇总
    cost_frames, summary_rows = [], []
    for name, sc in score_map.items():
        c = cost_curve_by_flagrate(ym, sc, K_COST)
        c.insert(0, '模型', name)
        cost_frames.append(c)
        best = c.loc[c['cost_per_sample'].idxmin()]
        row = {'模型': name, 'AUC(测试期)': round(roc_auc_score(ym, sc), 4),
               '最低代价(k=3)': round(float(best['cost_per_sample']), 4),
               '预警率@最低代价': round(float(best['flag_rate']), 3),
               '召回@最低代价': round(float(best['recall']), 3),
               '精确率@最低代价': round(float(best['precision']), 3)}
        for k2 in (2, 5, 7, 10):
            row[f'最低代价(k={k2})'] = round(float(cost_curve_by_flagrate(ym, sc, k2)['cost_per_sample'].min()), 4)
        summary_rows.append(row)
    cost_all, cost_sum = pd.concat(cost_frames, ignore_index=True), pd.DataFrame(summary_rows)
    log(f"\n代价视角汇总（预警频率口径，C_FP=1；k=3 为业务冻结值）：", report_lines)
    log(cost_sum.to_string(index=False), report_lines)

    frozen_rate = None
    if os.path.exists(TEST_THR_CMP):
        cmp_tbl = pd.read_csv(TEST_THR_CMP, encoding='utf-8-sig')
        sub = cmp_tbl[cmp_tbl['阈值方案'] == '代价最优(冻结)']
        if not sub.empty:
            frozen_rate = float(sub['flag率'].iloc[0])
    plot_precision_cost(ym, score_map, K_COST, frozen_rate,
                        os.path.join(FIG_DIR, f'scorecard_precision_cost_dual_view{SUFFIX}.png'), report_lines)

    # ---------- 落盘 ----------
    iv_tbl.to_csv(IV_OUT, index=False, encoding='utf-8-sig')
    points_tbl.to_csv(POINTS_OUT, index=False, encoding='utf-8-sig')
    psi_tbl.to_csv(PSI_OUT, index=False, encoding='utf-8-sig')
    scores.to_csv(SCORES_OUT, index=False, encoding='utf-8-sig')
    twoway.to_csv(TWOWAY_OUT, index=False, encoding='utf-8-sig')
    merged.to_csv(TWOWAY_SCORES_OUT, index=False, encoding='utf-8-sig')
    cost_all.to_csv(TWOWAY_COST_OUT, index=False, encoding='utf-8-sig')
    cost_sum.to_csv(TWOWAY_COST_SUM_OUT, index=False, encoding='utf-8-sig')

    bins_rows = []
    for f in keep:
        e, w = fit[f]['edges'], fit[f]['woe']
        for j in range(len(w)):
            bins_rows.append({'变量': f, '分档': f'({e[j]:.3f}, {e[j+1]:.3f}]', 'WOE': round(float(w[j]), 4)})
    pd.DataFrame(bins_rows).to_csv(BINS_OUT, index=False, encoding='utf-8-sig')

    plot_iv_bar(iv_tbl, os.path.join(FIG_DIR, f'scorecard_iv_ranking{SUFFIX}.png'), report_lines)
    log(f"\n已保存：{IV_OUT} / {POINTS_OUT} / {PSI_OUT} / {SCORES_OUT} / {TWOWAY_OUT}", report_lines)

    with open(REPORT_OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    return iv_tbl, points_tbl, twoway


if __name__ == '__main__':
    main()
