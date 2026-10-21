"""
黑名单策略回测（阶段 5）
运行：python -m src.backtest.backtest

把阶段 3 冻结的 LightGBM 分数接到**可成交的黑名单决策**上，产出"黑名单组合较基准最大回撤降低 X 个百分点"。

两条决策口径（口径 v2 的参数只从**留出验证段**标定，不碰测试期）：
    - v1 绝对分数阈值：分数 ≥ 冻结阈值 → 剔除（阶段 3 冻结的分数域阈值，抗分布漂移设计）
    - v2 横截面排名阈值：每周只剔**分数最高的前 X%**（X = v1 在留出验证段的实际 flag 率），
      排名只看相互高低、天然抗分数分布跨期漂移；买回用排名滞回带（后 Z%，Z=X/2）
两个分段（一期：留出验证段 2022-06-01 ~ 2022-12-31，参数标定段 / 测试期 2023-01-01 ~ 2024-12-31；
二期：留出段 2024-01-01 ~ 2024-12-31 / 新考卷 2025-01-01 ~ 2025-12-31）。
**测试期 v1 为一次性最终回测；测试期 v2 为探索性结果**（测试期已用于 v1 与阶段 3，须如实标注非无偏确认）。

可成交规则（`内部路线图` 阶段 5）：
    - 某周触发剔除 → **下一周起**生效（决策周 < 执行周，防前视）
    - 被剔除者连续 4 周低于安全线 → 允许买回（滞回带防抖动）
    - 停牌顺延至复牌后首个可交易周（"行在但成交量=0" 与 "整行缺失" 同处理）
    - 双边各 0.3%（`COST_RATE`；Σ|Δw| 已含买卖两侧，故为**单边费率**）
    - 使用复权价格：数据源为**后复权**收盘价，收益 = 周 pct_change

防泄露（`口径纪律`，最高优先级）：
    - **本模块不拟合任何统计量**：模型/阈值取自阶段 3 冻结产物；v2 的唯一参数 X 取自留出验证段。
    - 收益面板取自 `data/raw`（`src.data.load_data`），**不是** `feature_matrix_*.csv` 的 `周收益率` 列
      ——后者已被 Winsorize(±17.9%) + Z-Score 处理，当收益用会把复牌跳空系统性压平。
    - 测试期**只评估、不选参数**；敏感性 / 安慰剂均为**描述性**。

默认口径（用户已确认）：安全线 = 0.5×阈值、连续 4 周；组合 = 事件驱动等权；基准 = 全池等权买入持有（零成交）；
退市/永久停牌按最后可交易价冻结、不回收权重（防幸存者偏差）。
"""

import os
import sys

import numpy as np
import pandas as pd

from src.config import (
    OUTPUT_DIR, PROJECT_ROOT, ID_COLS,
    VALID_SEGMENT_START, VALID_SEGMENT_END,
    TEST_SEGMENT_START, TEST_SEGMENT_END,
    VALID_SEGMENT_START_V2, VALID_SEGMENT_END_V2,
    TEST_SEGMENT_START_V2, TEST_SEGMENT_END_V2,
    SUFFIX_V2, log, ensure_output_dir,
)

# 运行模式：
#   （无参） = 一期方案 C（v1 绝对阈值 / v2 排名阈值）
#   --v2   = 二期方案 E（同 v1/v2 决策口径，输入取 `_v2` 产物）
#   --v3   = 二期数据上的**决策层改造**变体（比例化/分层仓位 + 最小持有期 + 换手惩罚；
#            复用二期 `_v2` 冻结分数与阈值，只改决策映射；参数只在留出段 2024 标定）
IS_V3 = '--v3' in sys.argv
IS_V2 = ('--v2' in sys.argv) or IS_V3
SUFFIX = SUFFIX_V2 if IS_V2 else ''       # 输入产物后缀（v3 复用二期 `_v2` 输入）
OUT = '_v3' if IS_V3 else SUFFIX          # 输出产物后缀

# ===== 输入 =====
SPLIT_TRAIN = os.path.join(OUTPUT_DIR, f'split_train{SUFFIX}.csv')
FM_ALL = os.path.join(OUTPUT_DIR, f'feature_matrix{SUFFIX}.csv')
# 一期训练期/测试期分两个矩阵；二期为单一全量矩阵 feature_matrix_v2.csv（按日期区间过滤）
FM_TRAIN = FM_ALL if IS_V2 else os.path.join(OUTPUT_DIR, 'feature_matrix_train.csv')
FM_TEST = FM_ALL if IS_V2 else os.path.join(OUTPUT_DIR, 'feature_matrix_test.csv')
MODEL_OUT = os.path.join(OUTPUT_DIR, f'lgbm_model{SUFFIX}.txt')
FROZEN_IN = os.path.join(OUTPUT_DIR, f'frozen_threshold{SUFFIX}.csv')
TEST_SCORES = os.path.join(OUTPUT_DIR, f'test_scores{SUFFIX}.csv')

# ===== 输出 =====
NAV_OUT = os.path.join(OUTPUT_DIR, f'backtest_nav{OUT}.csv')
POS_OUT = os.path.join(OUTPUT_DIR, f'backtest_positions{OUT}.csv')
TRADES_OUT = os.path.join(OUTPUT_DIR, f'backtest_trades{OUT}.csv')
V2_TRADES_OUT = os.path.join(OUTPUT_DIR, f'backtest_v2_trades{OUT}.csv')
MEMB_OUT = os.path.join(OUTPUT_DIR, f'backtest_membership{OUT}.csv')
SCORE_PANEL_OUT = os.path.join(OUTPUT_DIR, f'backtest_score_panel{OUT}.csv')
METRICS_OUT = os.path.join(OUTPUT_DIR, f'backtest_metrics{OUT}.csv')
SENS_OUT = os.path.join(OUTPUT_DIR, f'backtest_sensitivity{OUT}.csv')
PLACEBO_OUT = os.path.join(OUTPUT_DIR, f'backtest_placebo{OUT}.csv')
V2_PLACEBO_OUT = os.path.join(OUTPUT_DIR, f'backtest_v2_placebo{OUT}.csv')
REPORT_OUT = os.path.join(OUTPUT_DIR, f'backtest_report{OUT}.txt')
FIG_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')

# ===== 切段（一期：留出 2022-06~12 / 测试 2023-2024；二期：留出 2024 / 考卷 2025）=====
VALID_START, VALID_END = ((VALID_SEGMENT_START_V2, VALID_SEGMENT_END_V2) if IS_V2
                          else (VALID_SEGMENT_START, VALID_SEGMENT_END))
TEST_START, TEST_END = ((TEST_SEGMENT_START_V2, TEST_SEGMENT_END_V2) if IS_V2
                        else (TEST_SEGMENT_START, TEST_SEGMENT_END))

# ===== 参数 =====
NON_FEATURE_COLS = ID_COLS + ['Is_Anomaly', 'y']
COST_RATE = 0.003          # 单边 0.3%（Σ|Δw| 已含买卖两侧 → 往返 0.6%）
SAFE_MULT = 0.5            # v1 安全线 = SAFE_MULT × 冻结阈值
Z_MULT = 0.5               # v2 买回滞回带 Z = Z_MULT × X
NEED_SAFE = 4
WEEKS_PER_YEAR = 52
PLACEBO_RUNS = 200
SENS_COSTS = [0.0015, 0.003, 0.005]
RANDOM_STATE = 42


def _read(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    if '交易日期' in df.columns:
        df['交易日期'] = pd.to_datetime(df['交易日期'])
    return df


def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


# ============================================================
# 加载（全部只读；不拟合）
# ============================================================
def load_market_panel(report_lines, start, end):
    """原始行情 → 指定区间长表。收益取自 data/raw（后复权周 pct_change），非标准化列。

    本函数不拟合任何统计量：只调用 load_data（一期 2019-2024）或 load_data_v2（二期 2019-2025）
    读取与截断（口径见 load_data：最后一个成交量>0 的周）。
    """
    if IS_V2:
        from src.data.load_data import load_data_v2 as _load
    else:
        from src.data.load_data import load_data as _load
    df, _ = _load(report_lines)
    seg = df[(df['交易日期'] >= pd.Timestamp(start)) & (df['交易日期'] <= pd.Timestamp(end))].copy()
    log(f"[{start} ~ {end}] 原始行情：{len(seg)} 行 / {seg['股票代码'].nunique()} 家"
        f"（可交易 {(seg['成交量'] > 0).sum()} / 停牌 {(seg['成交量'] == 0).sum()}）", report_lines)
    return seg[['股票代码', '公司简称', '交易日期', '收盘价', '成交量', '周收益率']]


def load_score_panel(report_lines, fm_path, start, end):
    """用冻结 LightGBM 对指定特征矩阵的**全量行**打分（只 predict，不 fit）。"""
    import lightgbm as lgb
    features = feature_columns(_read(SPLIT_TRAIN))
    fm = _read(fm_path)
    fm = fm[(fm['交易日期'] >= pd.Timestamp(start)) & (fm['交易日期'] <= pd.Timestamp(end))]
    booster = lgb.Booster(model_file=MODEL_OUT)
    out = fm[['股票代码', '交易日期']].copy()
    out['proba_raw'] = booster.predict(fm[features])
    log(f"  冻结模型打分：{len(out)} 行（{len(features)} 维）", report_lines)
    return out


def load_frozen_params(report_lines):
    thr = float(_read(FROZEN_IN)['原始分数阈值'].iloc[0])
    log(f"冻结阈值 THR={thr:.6f}（分数域）；v1 安全线 SAFE={SAFE_MULT * thr:.6f}；"
        f"连续 {NEED_SAFE} 周；单边成本 {COST_RATE:.4%}", report_lines)
    return thr, SAFE_MULT * thr


def build_grid(market, scores, report_lines, name):
    """对齐成 周×公司 的 (weeks, codes, R, T, S) + 完整性自检。本函数不拟合任何统计量。"""
    piv_r = market.pivot(index='交易日期', columns='股票代码', values='周收益率').sort_index()
    piv_v = market.pivot(index='交易日期', columns='股票代码', values='成交量').sort_index()
    codes = list(piv_r.columns)
    T = (piv_v.notna() & (piv_v > 0)).to_numpy()
    R = piv_r.where(T, 0.0).fillna(0.0).to_numpy(dtype=float)

    piv_s = scores.pivot(index='交易日期', columns='股票代码', values='proba_raw')
    S = piv_s.reindex(index=piv_r.index, columns=codes).to_numpy(dtype=float)
    weeks = list(piv_r.index)

    n_mismatch = int((pd.notna(S) != T).sum())
    log(f"  [{name}] 面板：{len(weeks)} 周 × {len(codes)} 家；可交易 {int(T.sum())} / 停牌 {int((~T).sum())}；"
        f"分数非空 {int(pd.notna(S).sum())}", report_lines)
    assert n_mismatch == 0, f"[{name}] 分数非空与可交易不一致 {n_mismatch} 格（管线假设被破坏）"
    assert not np.isnan(R).any(), f"[{name}] 收益矩阵存在 NaN"
    log(f"  [{name}] 收益极值（未经截断）：min={R.min():.4f}, max={R.max():.4f}", report_lines)
    return weeks, codes, R, T, S


def rank_thresholds(S, T, universe, X, Z):
    """每周按**可交易横截面**计算排名阈值：剔除线 = 1−X 分位，买回安全线 = 1−Z 分位。

    本函数不拟合任何统计量（X/Z 由调用方从留出验证段传入）。返回长度 = 周数的两条阈值数组。
    """
    M = S.shape[0]
    thr = np.full(M, np.nan)
    safe = np.full(M, np.nan)
    for tau in range(M):
        mask = universe & T[tau] & ~np.isnan(S[tau])
        if mask.sum() < 2:
            continue
        vals = S[tau][mask]
        thr[tau] = np.quantile(vals, 1.0 - X)
        safe[tau] = np.quantile(vals, 1.0 - Z)
    return thr, safe


# ============================================================
# 组合引擎（防前视状态机）
# ============================================================
def run_portfolio(weeks, codes, R, T, S, thr=None, safe=None, need_safe=NEED_SAFE,
                  cost_rate=COST_RATE, use_signal=True, S_override=None):
    """跑完整分段，返回净值/持仓/交易/诊断。thr/safe 可为标量或长度=周数的数组。

    本函数不拟合任何统计量。use_signal=False → 基准臂（全池等权买入持有、零成交）。
    防前视：信号在 τ 周读取，**执行周严格大于决策周**；停牌（T=False）时待办顺延。
    """
    S = S if S_override is None else S_override
    M, N = R.shape
    thr_arr = np.full(M, np.nan) if thr is None else (
        np.full(M, float(thr)) if np.isscalar(thr) else np.asarray(thr, dtype=float))
    safe_arr = np.full(M, np.nan) if safe is None else (
        np.full(M, float(safe)) if np.isscalar(safe) else np.asarray(safe, dtype=float))

    w = np.zeros(N)
    state = np.array(['OUT'] * N, dtype=object)
    universe = T[0].copy()
    w[universe] = 1.0 / max(universe.sum(), 1)
    state[universe] = 'HELD'

    safe_streak = np.zeros(N, dtype=int)
    decided = np.full(N, -1, dtype=int)
    pending_exit = np.zeros(N, dtype=bool)
    pending_enter = np.zeros(N, dtype=bool)

    nav = np.ones(M)
    n_held = np.zeros(M, dtype=int)
    n_black = np.zeros(M, dtype=int)
    frozen_w = np.zeros(M)
    turnover_hist = np.zeros(M)
    cost_hist = np.zeros(M)
    trades = []

    def read_signals(tau):
        if not use_signal:
            return
        thr_t, safe_t = thr_arr[tau], safe_arr[tau]
        if np.isnan(thr_t):
            return
        s_row = S[tau]
        for j in np.where(universe)[0]:
            s = s_row[j]
            if np.isnan(s):
                continue                     # 停牌周无分数 → 计数暂停（不重置）
            if state[j] == 'HELD':
                if s >= thr_t:
                    if not pending_exit[j]:
                        decided[j] = tau
                    pending_exit[j] = True
            elif state[j] == 'OUT':
                if not np.isnan(safe_t):
                    safe_streak[j] = safe_streak[j] + 1 if s < safe_t else 0
                    if safe_streak[j] >= need_safe:
                        pending_enter[j] = True
                        decided[j] = tau
                        state[j] = 'PENDING'

    def snapshot(tau, turn, cost):
        n_held[tau] = int(((state == 'HELD') & T[tau]).sum())
        n_black[tau] = int(((state == 'OUT') & universe).sum())
        frozen_w[tau] = float(w[(state == 'HELD') & ~T[tau]].sum())
        turnover_hist[tau] = turn
        cost_hist[tau] = cost

    snapshot(0, 0.0, 0.0)
    read_signals(0)

    for tau in range(1, M):
        port_ret = float((w * R[tau]).sum())
        nav[tau] = nav[tau - 1] * (1.0 + port_ret)
        w = w * (1.0 + R[tau]) / (1.0 + port_ret)

        traded, turn, cost = False, 0.0, 0.0
        if use_signal:
            for j in np.where(pending_exit)[0]:
                if T[tau, j]:
                    trades.append({'股票代码': codes[j], '方向': '剔除',
                                   '决策周': str(weeks[decided[j]].date()),
                                   '执行周': str(weeks[tau].date()), '前权重': round(float(w[j]), 6)})
                    w[j] = 0.0
                    state[j] = 'OUT'
                    safe_streak[j] = 0
                    pending_exit[j] = False
                    traded = True
            for j in np.where(pending_enter)[0]:
                if T[tau, j]:
                    trades.append({'股票代码': codes[j], '方向': '买回',
                                   '决策周': str(weeks[decided[j]].date()),
                                   '执行周': str(weeks[tau].date()), '前权重': 0.0})
                    state[j] = 'HELD'
                    pending_enter[j] = False
                    traded = True

            if traded:
                active = universe & (state == 'HELD') & T[tau]
                frozen_mask = universe & (state == 'HELD') & ~T[tau]
                fw = float(w[frozen_mask].sum())
                target = np.zeros(N)
                target[frozen_mask] = w[frozen_mask]
                if active.sum() > 0:
                    target[active] = (1.0 - fw) / active.sum()
                turn = float(np.abs(target - w).sum())
                cost = cost_rate * turn
                nav[tau] *= (1.0 - cost)
                w = target

        snapshot(tau, turn, cost)
        read_signals(tau)

    return {'nav': nav, 'n_held': n_held, 'n_black': n_black, 'frozen_w': frozen_w,
            'turnover': turnover_hist, 'cost': cost_hist, 'trades': trades,
            'state': state, 'weeks': weeks}


def summarize(res, weeks):
    """累计收益 / 最大回撤 / 年化波动率 + 附诊断。"""
    nav = np.asarray(res['nav'], dtype=float)
    r = nav[1:] / nav[:-1] - 1.0
    peak = np.maximum.accumulate(nav)
    mdd = float((nav / peak - 1.0).min())
    vol = float(np.std(r, ddof=1) * np.sqrt(WEEKS_PER_YEAR))
    n_w = len(nav) - 1
    ann = float(nav[-1] ** (WEEKS_PER_YEAR / n_w) - 1.0) if nav[-1] > 0 else -1.0
    return {'累计收益': round(float(nav[-1] - 1.0), 4), '最大回撤': round(mdd, 4),
            '年化波动率': round(vol, 4), '年化收益': round(ann, 4),
            '卡玛': round(ann / abs(mdd), 3) if mdd < 0 else None,
            '平均持仓数': round(float(np.mean(res['n_held'])), 1),
            '成交笔数': len(res['trades']),
            '累计成本占净值': round(float(np.sum(res['cost'])), 4), '周数': n_w}


# ============================================================
# 出图
# ============================================================
def _plt(report_lines):
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


def plot_v1(weeks, bench, strat, m_bench, m_strat, report_lines):
    plt = _plt(report_lines)
    if plt is None:
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    x = [w.date() for w in weeks]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, bench['nav'], label=f"基准·全池等权买入持有（累计 {m_bench['累计收益']:+.2%}）", color='#7f7f7f')
    ax.plot(x, strat['nav'], label=f"策略 v1·绝对阈值（累计 {m_strat['累计收益']:+.2%}）", color='#d62728')
    ax.set_title(f'测试期净值曲线：黑名单组合(v1) vs 全池等权买入持有（{TEST_START[:4]}-{TEST_END[:4]}）')
    ax.set_ylabel('净值（期初=1）')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'backtest_equity_blacklist_vs_benchmark{SUFFIX}.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for tag, res, c in [('基准', bench, '#7f7f7f'), ('策略 v1', strat, '#d62728')]:
        nav = np.asarray(res['nav'])
        ax.plot(x, nav / np.maximum.accumulate(nav) - 1.0,
                label=f"{tag}（最大回撤 {summarize(res, weeks)['最大回撤']:.1%}）", color=c)
    ax.set_title('测试期回撤曲线：黑名单组合(v1) vs 基准')
    ax.set_ylabel('回撤')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'backtest_drawdown_blacklist_vs_benchmark{SUFFIX}.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(x, bench['n_held'], label='基准·持仓数', color='#7f7f7f')
    ax.plot(x, strat['n_held'], label='策略 v1·白名单持仓数', color='#d62728')
    ax.plot(x, strat['n_black'], label='策略 v1·黑名单家数', color='#1f77b4', ls='--')
    ax.set_title('持仓数与黑名单规模时序（v1）')
    ax.set_ylabel('家数')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'backtest_positions_and_blacklist_size{SUFFIX}.png'), dpi=150)
    plt.close(fig)


def plot_v1_v2(weeks, bench, v1, v2, report_lines):
    """v1/v2 口径对比（净值 + 回撤 双栏）。"""
    plt = _plt(report_lines)
    if plt is None:
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    x = [w.date() for w in weeks]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for tag, res, c in [('基准·买入持有', bench, '#7f7f7f'),
                        ('v1·绝对阈值', v1, '#d62728'), ('v2·排名阈值', v2, '#1f77b4')]:
        axes[0].plot(x, np.asarray(res['nav']), label=f"{tag}（累计 {summarize(res, weeks)['累计收益']:+.1%}）", color=c)
        nav = np.asarray(res['nav'])
        axes[1].plot(x, nav / np.maximum.accumulate(nav) - 1.0,
                     label=f"{tag}（回撤 {summarize(res, weeks)['最大回撤']:.1%}）", color=c)
    axes[0].set_title('(a) 测试期净值：基准 / v1 / v2')
    axes[0].set_ylabel('净值（期初=1）')
    axes[1].set_title('(b) 测试期回撤：基准 / v1 / v2')
    axes[1].set_ylabel('回撤')
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'backtest_v1_v2_threshold_comparison{SUFFIX}.png'), dpi=150)
    plt.close(fig)


# ============================================================
# 主流程
# ============================================================
def main():
    report_lines = []
    log("=" * 60, report_lines)
    log("阶段 5 · 黑名单策略回测（v1 绝对阈值 / v2 排名阈值；本模块不拟合任何统计量）", report_lines)

    thr, safe = load_frozen_params(report_lines)

    # ---------- 1) 留出验证段（参数标定段）----------
    log(f"\n[1] 留出验证段（{VALID_START} ~ {VALID_END}，v2 参数标定段）", report_lines)
    m_val = load_market_panel(report_lines, VALID_START, VALID_END)
    s_val = load_score_panel(report_lines, FM_TRAIN, VALID_START, VALID_END)
    wv, cv, Rv, Tv, Sv = build_grid(m_val, s_val, report_lines, '留出段')
    bench_v = run_portfolio(wv, cv, Rv, Tv, Sv, use_signal=False)
    v1_v = run_portfolio(wv, cv, Rv, Tv, Sv, thr=thr, safe=safe)
    m_bv, m_v1v = summarize(bench_v, wv), summarize(v1_v, wv)

    mask_v = Tv & ~np.isnan(Sv)
    X = float((Sv[mask_v] >= thr).mean())            # v2 唯一参数：来自留出段
    Z = X * Z_MULT
    log(f"  v2 参数标定：X = 留出段 flag 率 = {X:.1%}（剔前 X%）；滞回带 Z = {Z:.1%}", report_lines)

    thr_vec, safe_vec = rank_thresholds(Sv, Tv, Tv[0], X, Z)
    v2_v = run_portfolio(wv, cv, Rv, Tv, Sv, thr=thr_vec, safe=safe_vec)
    m_v2v = summarize(v2_v, wv)

    # ---------- 2) 测试期 ----------
    log(f"\n[2] 测试期（{TEST_START} ~ {TEST_END}）", report_lines)
    m_test = load_market_panel(report_lines, TEST_START, TEST_END)
    s_test = load_score_panel(report_lines, FM_TEST, TEST_START, TEST_END)
    wt, ct, Rt, Tt, St = build_grid(m_test, s_test, report_lines, '测试期')
    bench_t = run_portfolio(wt, ct, Rt, Tt, St, use_signal=False)
    v1_t = run_portfolio(wt, ct, Rt, Tt, St, thr=thr, safe=safe)
    thr_vec_t, safe_vec_t = rank_thresholds(St, Tt, Tt[0], X, Z)
    v2_t = run_portfolio(wt, ct, Rt, Tt, St, thr=thr_vec_t, safe=safe_vec_t)
    m_bt, m_v1t, m_v2t = summarize(bench_t, wt), summarize(v1_t, wt), summarize(v2_t, wt)

    # ---------- 3) 汇总 ----------
    rows = []
    for seg, mb, m1, m2 in [('留出验证段', m_bv, m_v1v, m_v2v), ('测试期', m_bt, m_v1t, m_v2t)]:
        for rule, m in [('基准·全池等权买入持有', mb), ('v1·绝对分数阈值', m1), ('v2·排名阈值', m2)]:
            rows.append({'分段': seg, '口径': rule, **m})
    cmp_tbl = pd.DataFrame(rows)
    log("\n【主对比】分段 × 口径：", report_lines)
    log(cmp_tbl.to_string(index=False), report_lines)

    log(f"\n>>> 测试期 v1（一次性最终回测）：回撤降低 {abs(m_v1t['最大回撤'] - m_bt['最大回撤']) * 100:.1f} pp"
        f"（{m_bt['最大回撤']:.1%} → {m_v1t['最大回撤']:.1%}）", report_lines)
    log(f">>> 测试期 v2（**探索性**，非无偏确认）：回撤降低 "
        f"{abs(m_v2t['最大回撤'] - m_bt['最大回撤']) * 100:.1f} pp"
        f"（{m_bt['最大回撤']:.1%} → {m_v2t['最大回撤']:.1%}）", report_lines)
    log(f">>> 留出段 v1/v2：回撤 {m_v1v['最大回撤']:.1%} / {m_v2v['最大回撤']:.1%}"
        f"（基准 {m_bv['最大回撤']:.1%}）", report_lines)

    # ---------- 4) 自检 ----------
    self_checks(bench_t, v1_t, v2_t, St, s_test, report_lines)

    # ---------- 5) 安慰剂（v1 / v2，测试期，描述性）----------
    pl1 = run_placebo(wt, ct, Rt, Tt, St, thr=thr, safe=safe, actual=m_v1t, tag='v1',
                      report_lines=report_lines,
                      thresholds=None)
    pl2 = run_placebo(wt, ct, Rt, Tt, St, thr=thr_vec_t, safe=safe_vec_t, actual=m_v2t, tag='v2',
                      report_lines=report_lines,
                      thresholds=(range(len(wt)), thr_vec_t, safe_vec_t))
    # ---------- 6) 敏感性（v1，描述性）----------
    sens = run_sensitivity(wt, ct, Rt, Tt, St, thr, safe, report_lines)

    # ---------- 7) 出图 + 落盘 ----------
    plot_v1(wt, bench_t, v1_t, m_bt, m_v1t, report_lines)
    plot_v1_v2(wt, bench_t, v1_t, v2_t, report_lines)
    persist(wt, ct, Rt, Tt, St, bench_t, v1_t, v2_t, cmp_tbl, sens, pl1, pl2, report_lines)

    log("\n【口径声明】", report_lines)
    log("① 测试期 v1 为**一次性最终回测**；测试期 v2 为**探索性**结果（测试期已用于 v1 与阶段 3），非无偏确认。", report_lines)
    log("② v2 的参数 X（剔前 X%）与滞回带 Z **只由留出验证段标定**，未使用测试期信息。", report_lines)
    log("③ 成本为**单边** 0.3%（Σ|Δw| 含买卖两侧 → 往返 0.6%），**不含困境股宽价差与冲击成本**，偏乐观。", report_lines)
    log("④ 退市/永久停牌仓位按最后可交易周价冻结、不回收权重（防幸存者偏差）。", report_lines)

    with open(REPORT_OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    log(f"\n已保存回测报告：{REPORT_OUT}", report_lines)
    return cmp_tbl


def self_checks(bench, v1, v2, S, scores, report_lines):
    log("\n" + "-" * 60, report_lines)
    log("自检：", report_lines)
    for tag, res, is_bench in [('基准', bench, True), ('v1', v1, False), ('v2', v2, False)]:
        assert abs(res['nav'][0] - 1.0) < 1e-12, f"{tag} 净值首值≠1"
        if is_bench:
            assert len(res['trades']) == 0 and float(np.sum(res['cost'])) == 0.0, "基准臂出现成交/成本"
        else:
            bad = [t for t in res['trades'] if not (t['决策周'] < t['执行周'])]
            assert not bad, f"{tag} 存在前视交易 {len(bad)} 笔"
    log(f"  ① 三臂净值首值 = 1.0；基准零成交零成本；v1 {len(v1['trades'])} 笔 / v2 {len(v2['trades'])} 笔"
        f" 全部 决策周 < 执行周 ✓", report_lines)
    ts = _read(TEST_SCORES)
    merged = ts[['股票代码', '交易日期', 'proba_raw']].merge(
        scores, on=['股票代码', '交易日期'], how='inner', suffixes=('_file', '_recalc'))
    d = float(np.nanmax(np.abs(merged['proba_raw_file'] - merged['proba_raw_recalc'])))
    assert d < 1e-9, f"分数与 test_scores.csv 不一致 max|Δ|={d}"
    log(f"  ② 与 test_scores.csv 逐行核对 {len(merged)} 行，max|Δ|={d:.2e} ✓", report_lines)


def run_placebo(weeks, codes, R, T, S, thr, safe, actual, tag, report_lines, thresholds=None):
    """安慰剂：每周**置换横截面分数**（同规模 flag、随机指派），跑 N 次，看策略是否优于随机分布。

    对 v2（排名阈值）尤其自然：置换不改变横截面分位，故置换等价于"每周随机剔同比例"。
    """
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for _ in range(PLACEBO_RUNS):
        S_perm = np.empty_like(S)
        for tau in range(S.shape[0]):
            row = S[tau].copy()
            valid = ~np.isnan(row)
            row[valid] = rng.permutation(row[valid])
            S_perm[tau] = row
        r = run_portfolio(weeks, codes, R, T, S, thr=thr, safe=safe, S_override=S_perm)
        m = summarize(r, weeks)
        rows.append({'累计收益': m['累计收益'], '最大回撤': m['最大回撤'],
                     '年化波动率': m['年化波动率'], '成交笔数': m['成交笔数']})
    tbl = pd.DataFrame(rows)
    log(f"\n【描述性】安慰剂（{tag}，随机置换横截面分数 {PLACEBO_RUNS} 次，同 flag 规模）：", report_lines)
    log(f"  成交笔数：随机中位={tbl['成交笔数'].median():.0f}（对照 {tag}={actual['成交笔数']}）", report_lines)
    log(f"  最大回撤：随机 P5={tbl['最大回撤'].quantile(0.05):.1%} / 中位={tbl['最大回撤'].median():.1%} / "
        f"P95={tbl['最大回撤'].quantile(0.95):.1%}", report_lines)
    log(f"  累计收益：随机 P5={tbl['累计收益'].quantile(0.05):.1%} / 中位={tbl['累计收益'].median():.1%} / "
        f"P95={tbl['累计收益'].quantile(0.95):.1%}", report_lines)
    shallower = float((tbl['最大回撤'] > actual['最大回撤']).mean())
    log(f"  策略 {tag} 实际回撤 {actual['最大回撤']:.1%}；随机中比策略**更浅(更好)**占 {shallower:.1%}", report_lines)
    if shallower <= 0.05:
        log(f"  → {tag} 显著优于同规模随机黑名单：改善来自**选得对** ✓", report_lines)
    else:
        log(f"  → {tag} **未显著优于**同规模随机黑名单：改善主要来自**降暴露/降集中**（如实披露）", report_lines)
    return tbl


def run_sensitivity(weeks, codes, R, T, S, thr, safe, report_lines):
    rows = []
    for c in SENS_COSTS:
        r = run_portfolio(weeks, codes, R, T, S, thr=thr, safe=safe, cost_rate=c)
        m = summarize(r, weeks)
        rows.append({'扫描维度': '成本(单边)', '取值': c, '累计收益': m['累计收益'],
                     '最大回撤': m['最大回撤'], '年化波动率': m['年化波动率']})
    for mult in [0.5, 1.0, 2.0]:
        r = run_portfolio(weeks, codes, R, T, S, thr=thr * mult, safe=safe * mult)
        m = summarize(r, weeks)
        rows.append({'扫描维度': '阈值(×冻结)', '取值': mult, '累计收益': m['累计收益'],
                     '最大回撤': m['最大回撤'], '年化波动率': m['年化波动率']})
    for mult in [0.25, 0.5, 0.75]:
        r = run_portfolio(weeks, codes, R, T, S, thr=thr, safe=thr * mult)
        m = summarize(r, weeks)
        rows.append({'扫描维度': '安全线(×阈值)', '取值': mult, '累计收益': m['累计收益'],
                     '最大回撤': m['最大回撤'], '年化波动率': m['年化波动率']})
    for n in [2, 4, 6]:
        r = run_portfolio(weeks, codes, R, T, S, thr=thr, safe=safe, need_safe=n)
        m = summarize(r, weeks)
        rows.append({'扫描维度': '连续周数', '取值': n, '累计收益': m['累计收益'],
                     '最大回撤': m['最大回撤'], '年化波动率': m['年化波动率']})
    tbl = pd.DataFrame(rows)
    log("\n【描述性】敏感性扫描（v1，不用于选型）：", report_lines)
    log(tbl.to_string(index=False), report_lines)
    return tbl


def persist(weeks, codes, R, T, S, bench, v1, v2, cmp_tbl, sens, pl1, pl2, report_lines):
    ensure_output_dir()
    nav = pd.DataFrame({
        '交易日期': [w.date() for w in weeks],
        '基准净值': bench['nav'], 'v1净值': v1['nav'], 'v2净值': v2['nav'],
        '基准持仓数': bench['n_held'], 'v1持仓数': v1['n_held'], 'v2持仓数': v2['n_held'],
        'v1黑名单家数': v1['n_black'], 'v2黑名单家数': v2['n_black'],
        'v1累计成本': np.round(np.cumsum(v1['cost']), 6), 'v2累计成本': np.round(np.cumsum(v2['cost']), 6),
    })
    nav.to_csv(NAV_OUT, index=False, encoding='utf-8-sig')
    pd.DataFrame(v1['trades']).to_csv(TRADES_OUT, index=False, encoding='utf-8-sig')
    pd.DataFrame(v2['trades']).to_csv(V2_TRADES_OUT, index=False, encoding='utf-8-sig')
    cmp_tbl.to_csv(METRICS_OUT, index=False, encoding='utf-8-sig')
    sens.to_csv(SENS_OUT, index=False, encoding='utf-8-sig')
    pl1.to_csv(PLACEBO_OUT, index=False, encoding='utf-8-sig')
    pl2.to_csv(V2_PLACEBO_OUT, index=False, encoding='utf-8-sig')

    long = []
    for tau, wk in enumerate(weeks):
        for j, c in enumerate(codes):
            if T[tau, j] or v1['state'][j] == 'HELD':
                long.append({'交易日期': wk.date(), '股票代码': c, '可交易': bool(T[tau, j]),
                             '分数': None if np.isnan(S[tau, j]) else round(float(S[tau, j]), 6),
                             'v1状态': v1['state'][j], 'v2状态': v2['state'][j]})
    pd.DataFrame(long).to_csv(POS_OUT, index=False, encoding='utf-8-sig')

    mem = pd.DataFrame({'交易日期': [w.date() for w in weeks],
                        'v1黑名单家数': v1['n_black'], 'v1白名单持仓数': v1['n_held'],
                        'v2黑名单家数': v2['n_black'], 'v2白名单持仓数': v2['n_held']})
    mem.to_csv(MEMB_OUT, index=False, encoding='utf-8-sig')

    sp = []
    for tau, wk in enumerate(weeks):
        for j, c in enumerate(codes):
            if not np.isnan(S[tau, j]):
                sp.append({'交易日期': wk.date(), '股票代码': c, 'proba_raw': round(float(S[tau, j]), 6)})
    pd.DataFrame(sp).to_csv(SCORE_PANEL_OUT, index=False, encoding='utf-8-sig')
    log(f"\n已落盘：{NAV_OUT} / {METRICS_OUT} / {TRADES_OUT} / {V2_TRADES_OUT} / {MEMB_OUT} / "
        f"{POS_OUT} / {SCORE_PANEL_OUT} / {SENS_OUT} / {PLACEBO_OUT} / {V2_PLACEBO_OUT}", report_lines)


# ============================================================
# 决策层改造（--v3）：比例化 / 分层仓位 + 最小持有期 + 换手惩罚
# ============================================================
# 只改「决策映射」，复用二期 `_v2` 冻结模型分数与冻结阈值；参数**只在留出段 2024 标定**，
# 2025 只跑一次、标注为「后验探索性」（2025 此前已用于一次性评估，口径纪律/§8）。
V3_HALF_W = 0.5            # 观察层权重系数（安全层 1.0 / 观察层 0.5 / 高风险层 0.0）
V3_GRID = [
    {'x_lo': xl, 'x_hi': xh, 'h': h, 'lam': lam, 'penalty': pen}
    for xl in (0.15, 0.25, 0.35)     # 高风险层（清零）分位宽度：剔分数最高的前 x_lo 比例
    for xh in (0.40, 0.55, 0.70)     # 观察层上界分位（须 > x_lo）
    for h in (0, 4)                  # 最小持有期（周）：未满则不清零、至少半仓
    for lam in (0.5, 1.0)            # 部分调仓比例（<1 抑制 churn）
    for pen in (0.0, 0.003)          # 换手惩罚（叠加在单边成本上）
]
V3_SEL_KEY = ('卡玛', '最大回撤')   # 标定准则：先最大化卡玛、并列时取回撤更浅者


def _pct_rank(vals):
    """按取值升序转 [0,1] 分位（最小=0、最大=1）；单元素返回 0。"""
    n = len(vals)
    if n <= 1:
        return np.zeros(n)
    order = np.argsort(vals, kind='mergesort')
    r = np.empty(n, dtype=float)
    r[order] = np.arange(n) / (n - 1)
    return r


def _v3_target_factors(S_tau, mask, w, entry, tau, x_lo, x_hi, min_hold, f_half=V3_HALF_W):
    """给定某周的分数行，返回 (idx, f_loc)：分位分层的目标权重因子。

    分层（按当周可交易横截面分位 p，p 越大越风险）：
        p < 1 - x_hi        → 1.0（安全层，满仓）
        1-x_hi ≤ p < 1-x_lo → 0.5（观察层，半仓）
        p ≥ 1 - x_lo        → 0.0（高风险层，清零）
    最小持有期保护：入名单未满 min_hold 周者不得一次清零，至少半仓。
    本函数不拟合任何统计量（分位只由当周横截面决定，x_lo/x_hi 由留出段冻结）。
    """
    idx = np.where(mask)[0]
    if idx.size == 0:
        return idx, np.zeros(0)
    p = _pct_rank(S_tau[idx])
    f = np.where(p < (1.0 - x_hi), 1.0,
                 np.where(p < (1.0 - x_lo), f_half, 0.0))
    if min_hold > 0:
        young = (w[idx] > 1e-9) & ((tau - entry[idx]) < min_hold)
        f = np.where(young, np.maximum(f, f_half), f)
    return idx, f


def run_portfolio_v3(weeks, codes, R, T, S, x_lo, x_hi, min_hold, lam, penalty_rate,
                     f_half=V3_HALF_W, cost_rate=COST_RATE, use_signal=True, S_override=None):
    """分层仓位 + 最小持有期 + 换手惩罚的组合引擎（防前视状态机）。

    use_signal=False → 基准臂（全池等权买入持有、零成交）。
    防前视：τ 周读信号 → **τ+1 周执行**（决策周 < 执行周），与 v1/v2 一致。
    成本 = (单边费率 + 换手惩罚) × Σ|Δw|。本函数不拟合任何统计量（参数由调用方冻结传入）。
    """
    S = S if S_override is None else S_override
    M, N = R.shape
    universe = T[0].copy()

    w = np.zeros(N)
    w[universe] = 1.0 / max(int(universe.sum()), 1)
    entry = np.full(N, -1, dtype=int)
    entry[universe] = 0

    nav = np.ones(M)
    n_held = np.zeros(M, dtype=int)
    n_black = np.zeros(M, dtype=int)
    frozen_w = np.zeros(M)
    turnover_hist = np.zeros(M)
    cost_hist = np.zeros(M)
    trades = []
    pending = None      # (idx, f_loc, 决策周索引) 供下一周执行

    def snap(tau, turn, cost):
        n_held[tau] = int((w > 1e-9).sum())
        n_black[tau] = int(((w <= 1e-9) & universe).sum())
        frozen_w[tau] = float(w[(w > 1e-9) & ~T[tau]].sum())
        turnover_hist[tau] = turn
        cost_hist[tau] = cost

    def read_target(tau):
        if not use_signal:
            return None
        mask = universe & T[tau] & ~np.isnan(S[tau])
        if mask.sum() < 2:
            return None
        idx, f = _v3_target_factors(S[tau], mask, w, entry, tau, x_lo, x_hi, min_hold, f_half)
        return idx, f, tau

    snap(0, 0.0, 0.0)
    pending = read_target(0)

    for tau in range(1, M):
        port_ret = float((w * R[tau]).sum())
        nav[tau] = nav[tau - 1] * (1.0 + port_ret)
        w = w * (1.0 + R[tau]) / (1.0 + port_ret)

        turn, cost = 0.0, 0.0
        if use_signal and pending is not None:
            idx, f_loc, dec_tau = pending
            fixed = ~np.zeros(N, dtype=bool)
            fixed[:] = True
            fixed[idx] = False                      # 非当周可交易成员 → 保留原权重（停牌顺延）
            w_fixed = float(w[fixed].sum())
            target = w.copy()
            denom = float(f_loc.sum())
            if denom > 0:
                target[idx] = (1.0 - w_fixed) * f_loc / denom
            else:
                target[idx] = 0.0
            w_new = w + lam * (target - w)
            for pos, j in enumerate(idx):
                if abs(w_new[j] - w[j]) > 0.005:
                    trades.append({
                        '股票代码': codes[j],
                        '方向': '加仓' if w_new[j] > w[j] else '减仓',
                        '决策周': str(weeks[dec_tau].date()), '执行周': str(weeks[tau].date()),
                        '前权重': round(float(w[j]), 6), '后权重': round(float(w_new[j]), 6),
                        '目标因子': round(float(f_loc[pos]), 3),
                    })
                if w_new[j] > 1e-9 and w[j] <= 1e-9:
                    entry[j] = tau
            turn = float(np.abs(w_new - w).sum())
            cost = (cost_rate + penalty_rate) * turn
            if turn > 0:
                nav[tau] *= (1.0 - cost)
            w = w_new

        snap(tau, turn, cost)
        pending = read_target(tau)

    return {'nav': nav, 'n_held': n_held, 'n_black': n_black, 'frozen_w': frozen_w,
            'turnover': turnover_hist, 'cost': cost_hist, 'trades': trades,
            'weeks': weeks}


def run_placebo_v3(weeks, codes, R, T, S, prm, actual, report_lines, runs=PLACEBO_RUNS):
    """安慰剂（v3）：随机置换当周横截面分数（分位结构不变、只换名字），跑 N 次。"""
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for _ in range(runs):
        S_perm = np.empty_like(S)
        for tau in range(S.shape[0]):
            row = S[tau].copy()
            valid = ~np.isnan(row)
            row[valid] = rng.permutation(row[valid])
            S_perm[tau] = row
        r = run_portfolio_v3(weeks, codes, R, T, S, prm['x_lo'], prm['x_hi'], prm['h'],
                             prm['lam'], prm['penalty'], S_override=S_perm)
        m = summarize(r, weeks)
        rows.append({'累计收益': m['累计收益'], '最大回撤': m['最大回撤'],
                     '年化波动率': m['年化波动率'], '成交笔数': m['成交笔数']})
    tbl = pd.DataFrame(rows)
    log(f"\n【描述性】安慰剂（v3，随机置换横截面分数 {runs} 次）：", report_lines)
    log(f"  最大回撤：随机 P5={tbl['最大回撤'].quantile(0.05):.1%} / 中位={tbl['最大回撤'].median():.1%} / "
        f"P95={tbl['最大回撤'].quantile(0.95):.1%}（策略 {actual['最大回撤']:.1%}）", report_lines)
    log(f"  累计收益：随机 P5={tbl['累计收益'].quantile(0.05):.1%} / 中位={tbl['累计收益'].median():.1%} / "
        f"P95={tbl['累计收益'].quantile(0.95):.1%}（策略 {actual['累计收益']:.1%}）", report_lines)
    shallower = float((tbl['最大回撤'] > actual['最大回撤']).mean())
    better_ret = float((tbl['累计收益'] > actual['累计收益']).mean())
    log(f"  随机中「回撤更浅」占 {shallower:.1%}、「收益更高」占 {better_ret:.1%}", report_lines)
    if shallower <= 0.05 and better_ret <= 0.05:
        log("  → v3 同时优于同规模随机的回撤与收益：改善来自**选得对** ✓", report_lines)
    else:
        log("  → v3 未在回撤与收益上同时显著优于同规模随机（如实披露）", report_lines)
    return tbl


def run_sensitivity_v3(weeks, codes, R, T, S, best, report_lines):
    """v3 描述性敏感性：围绕标定值单变量扰动。"""
    rows = []
    specs = [('高风险层宽度 x_lo', 'x_lo', [0.10, 0.25, 0.40]),
             ('观察层上界 x_hi', 'x_hi', [0.40, 0.55, 0.70]),
             ('最小持有期 h', 'h', [0, 4, 8]),
             ('部分调仓 λ', 'lam', [0.3, 0.5, 1.0]),
             ('换手惩罚', 'penalty', [0.0, 0.003, 0.006])]
    for tag, key, vals in specs:
        for v in vals:
            prm = dict(best)
            prm[key] = v
            r = run_portfolio_v3(weeks, codes, R, T, S, prm['x_lo'], prm['x_hi'], prm['h'],
                                 prm['lam'], prm['penalty'])
            m = summarize(r, weeks)
            rows.append({'扫描维度': tag, '取值': v, '累计收益': m['累计收益'],
                         '最大回撤': m['最大回撤'], '年化波动率': m['年化波动率'],
                         '成交笔数': m['成交笔数'], '累计成本占净值': m['累计成本占净值']})
    tbl = pd.DataFrame(rows)
    log("\n【描述性】v3 敏感性扫描（不用于选型）：", report_lines)
    log(tbl.to_string(index=False), report_lines)
    return tbl


def plot_v3(weeks, bench, v3res, m_bench, m_v3, report_lines):
    """v3 净值 + 回撤双图。"""
    plt = _plt(report_lines)
    if plt is None:
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    x = [w.date() for w in weeks]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, bench['nav'], label=f"基准·全池等权买入持有（累计 {m_bench['累计收益']:+.2%}）", color='#7f7f7f')
    ax.plot(x, v3res['nav'], label=f"策略 v3·分层仓位（累计 {m_v3['累计收益']:+.2%}）", color='#1f77b4')
    ax.set_title(f'2025 净值曲线：分层仓位(v3) vs 全池等权买入持有（{TEST_START[:4]}-{TEST_END[:4]}）')
    ax.set_ylabel('净值（期初=1）')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'appendix_v3_equity.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for tag, res, c in [('基准', bench, '#7f7f7f'), ('策略 v3', v3res, '#1f77b4')]:
        nav = np.asarray(res['nav'])
        ax.plot(x, nav / np.maximum.accumulate(nav) - 1.0,
                label=f"{tag}（最大回撤 {summarize(res, weeks)['最大回撤']:.1%}）", color=c)
    ax.set_title('2025 回撤曲线：分层仓位(v3) vs 基准')
    ax.set_ylabel('回撤')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'appendix_v3_drawdown.png'), dpi=150)
    plt.close(fig)
    log("已保存 v3 净值 / 回撤图", report_lines)


def main_v3():
    """决策层改造变体（--v3）：分层仓位 + 最小持有期 + 换手惩罚。

    标定：参数网格只在**留出段 2024** 上按 `V3_SEL_KEY` 选最优 → 冻结；**2025 只跑一次**
    （标注后验探索性，因 2025 此前已用于一次性评估）。本模块不拟合任何变换统计量。
    """
    report_lines = []
    log("=" * 60, report_lines)
    log("决策层改造（--v3）：比例化/分层仓位 + 最小持有期 + 换手惩罚（参数仅在留出段 2024 标定）",
        report_lines)

    thr, _ = load_frozen_params(report_lines)

    # ---------- 1) 留出段 2024：参数标定 ----------
    log(f"\n[1] 留出验证段（{VALID_START} ~ {VALID_END}）标定 v3 参数（网格 {len(V3_GRID)} 组）", report_lines)
    m_val = load_market_panel(report_lines, VALID_START, VALID_END)
    s_val = load_score_panel(report_lines, FM_TRAIN, VALID_START, VALID_END)
    wv, cv, Rv, Tv, Sv = build_grid(m_val, s_val, report_lines, '留出段')
    m_bv = summarize(run_portfolio(wv, cv, Rv, Tv, Sv, use_signal=False), wv)

    grid_rows = []
    best, best_key = None, None
    for prm in V3_GRID:
        r = run_portfolio_v3(wv, cv, Rv, Tv, Sv, prm['x_lo'], prm['x_hi'], prm['h'],
                             prm['lam'], prm['penalty'])
        m = summarize(r, wv)
        grid_rows.append({**prm, **m})
        key = (m['卡玛'] if m['卡玛'] is not None else -1e9, m['最大回撤'])
        if best_key is None or key > best_key:
            best_key, best = key, prm
    grid_tbl = pd.DataFrame(grid_rows).sort_values(['卡玛', '最大回撤'], ascending=False)
    log(f"\n留出段标定：基准回撤 {m_bv['最大回撤']:.1%}；网格前 5 组（按卡玛 / 回撤）：", report_lines)
    log(grid_tbl.head(5)[['x_lo', 'x_hi', 'h', 'lam', 'penalty', '累计收益', '最大回撤', '年化波动率',
                          '平均持仓数', '成交笔数']].to_string(index=False), report_lines)
    log(f"\n>>> 冻结参数：x_lo={best['x_lo']}（高风险层宽度）/ x_hi={best['x_hi']}（观察层上界）"
        f"/ h={best['h']}（最小持有期周）/ λ={best['lam']}（部分调仓）/ 换手惩罚={best['penalty']}", report_lines)
    log(f"    安全层满仓、观察层 {V3_HALF_W:.0%}、高风险层清零；分位按**当周可交易横截面**算（抗分数漂移）。",
        report_lines)

    # ---------- 2) 2025：只跑一次（后验探索性）----------
    log(f"\n[2] 测试期（{TEST_START} ~ {TEST_END}）：v3 一次性应用（**后验探索性**）", report_lines)
    m_test = load_market_panel(report_lines, TEST_START, TEST_END)
    s_test = load_score_panel(report_lines, FM_TEST, TEST_START, TEST_END)
    wt, ct, Rt, Tt, St = build_grid(m_test, s_test, report_lines, '测试期')
    bench_t = run_portfolio(wt, ct, Rt, Tt, St, use_signal=False)
    v1_t = run_portfolio(wt, ct, Rt, Tt, St, thr=thr, safe=SAFE_MULT * thr)
    v3_t = run_portfolio_v3(wt, ct, Rt, Tt, St, best['x_lo'], best['x_hi'], best['h'],
                            best['lam'], best['penalty'])
    m_bt, m_v1t, m_v3t = summarize(bench_t, wt), summarize(v1_t, wt), summarize(v3_t, wt)

    rows = []
    for rule, m in [('基准·全池等权买入持有', m_bt), ('v1·二值剔除（对照）', m_v1t), ('v3·分层仓位', m_v3t)]:
        rows.append({'分段': '测试期(2025)', '口径': rule, **m})
    cmp_tbl = pd.DataFrame(rows)
    log("\n【主对比】2025 测试期：", report_lines)
    log(cmp_tbl.to_string(index=False), report_lines)
    log(f"\n>>> v3（后验探索性）：最大回撤 {m_v3t['最大回撤']:.1%}（基准 {m_bt['最大回撤']:.1%}，"
        f"改善 {abs(m_v3t['最大回撤'] - m_bt['最大回撤']) * 100:.1f}pp）；"
        f"累计收益 {m_v3t['累计收益']:+.1%}（基准 {m_bt['累计收益']:+.1%}）。", report_lines)

    # 自检
    assert abs(v3_t['nav'][0] - 1.0) < 1e-12, "v3 净值首值≠1"
    bad = [t for t in v3_t['trades'] if not (t['决策周'] < t['执行周'])]
    assert not bad, f"v3 存在前视交易 {len(bad)} 笔"
    log(f"  自检：净值首值=1；{len(v3_t['trades'])} 笔调仓全部 决策周 < 执行周 ✓", report_lines)

    # ---------- 3) 安慰剂 + 敏感性 ----------
    pl = run_placebo_v3(wt, ct, Rt, Tt, St, best, m_v3t, report_lines)
    sens = run_sensitivity_v3(wt, ct, Rt, Tt, St, best, report_lines)

    # ---------- 4) 出图 + 落盘 ----------
    plot_v3(wt, bench_t, v3_t, m_bt, m_v3t, report_lines)
    ensure_output_dir()
    cmp_tbl.to_csv(METRICS_OUT, index=False, encoding='utf-8-sig')
    grid_tbl.to_csv(os.path.join(OUTPUT_DIR, f'backtest_v3_grid{OUT}.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(v3_t['trades']).to_csv(TRADES_OUT, index=False, encoding='utf-8-sig')
    sens.to_csv(SENS_OUT, index=False, encoding='utf-8-sig')
    pl.to_csv(PLACEBO_OUT, index=False, encoding='utf-8-sig')
    nav = pd.DataFrame({
        '交易日期': [w.date() for w in wt],
        '基准净值': bench_t['nav'], 'v1净值': v1_t['nav'], 'v3净值': v3_t['nav'],
        '基准持仓数': bench_t['n_held'], 'v1持仓数': v1_t['n_held'], 'v3持仓数': v3_t['n_held'],
        'v3无仓家数': v3_t['n_black'],
        'v3累计成本': np.round(np.cumsum(v3_t['cost']), 6),
    })
    nav.to_csv(NAV_OUT, index=False, encoding='utf-8-sig')
    mem = pd.DataFrame({'交易日期': [w.date() for w in wt],
                        'v1白名单持仓数': v1_t['n_held'], 'v1黑名单家数': v1_t['n_black'],
                        'v3持仓数': v3_t['n_held'], 'v3无仓家数': v3_t['n_black']})
    mem.to_csv(MEMB_OUT, index=False, encoding='utf-8-sig')

    log("\n【口径声明】", report_lines)
    log("① v3 **只改决策映射**（分层仓位 + 最小持有期 + 换手惩罚），复用二期 `_v2` 冻结模型与冻结阈值，"
        "不改模型 / 阈值 / 特征口径（口径纪律）。", report_lines)
    log("② v3 参数网格**只在留出段 2024 标定**、冻结后对 2025 只跑一次；因 2025 此前已用于一次性评估，"
        "本结果为**后验探索性**，非新考卷式无偏确认。", report_lines)
    log("③ 分位按当周可交易横截面计算，天然抗分数分布跨期漂移。", report_lines)
    log("④ 成本 =（单边 0.3% + 换手惩罚）× Σ|Δw|；不含困境股宽价差与冲击成本，偏乐观。", report_lines)

    with open(REPORT_OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    log(f"\n已保存 v3 报告：{REPORT_OUT}", report_lines)
    return cmp_tbl


if __name__ == '__main__':
    if IS_V3:
        main_v3()
    else:
        main()
