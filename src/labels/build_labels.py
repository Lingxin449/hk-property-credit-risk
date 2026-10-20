"""
标签生成入口（事件驱动标签 · 前向 8 周）
运行：python -m src.labels.build_labels

标签来源：`data/events/events.csv`——人工到港交所披露易核实**公告日**后的 5 类硬信号事件。
规则详见 `AGENTS.md §5`、`docs/方案讨论稿.md §3.4`（改良版 a'，2026-10-08 定；2026-10-09 第七轮补三点）。

标签口径（先按公司合并为事件簇，再逐段打标）：
    设簇锚点 a=簇内最早事件日、簇尾 e=簇内最后事件日、下一簇锚点 n
      1. [a-8周, a]        → y=1（预警窗口）
      2. (a, e+4周]        → 剔除（出险后冷却期，右端取簇尾 e 而非锚点 a，防簇尾裸奔）
      3. [e+4周, n-8周]    → 剔除（出险后空窗；最后一簇延伸至数据末尾）
      4. 其余              → y=0
    优先级：两簇区间重叠时（仅当簇间距 e→n ∈ [8,12) 周）**y=1 优先于剔除**。
    簇规则：同一公司事件间隔 ≤56 自然日（8 周）归为一簇、>56 天分簇；
            簇内最大跨度 ≤12 周（84 自然日），超出强制断簇（防传递闭包无限延伸）。
            一律按**日历日**计，不用交易日。

本模块不含任何统计量拟合：标签仅由事件日历 + 固定窗口常数决定，不涉及训练/测试期统计量泄漏。
"""

import os

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, OUTPUT_DIR, EVENTS_2025, SUFFIX_V2, log

# ===== 路径 =====
EVENTS_FILE = os.path.join(PROJECT_ROOT, 'data/events/events.csv')
TRAIN_MATRIX = os.path.join(OUTPUT_DIR, 'feature_matrix_train.csv')
TEST_MATRIX = os.path.join(OUTPUT_DIR, 'feature_matrix_test.csv')
MATRIX_V2 = os.path.join(OUTPUT_DIR, f'feature_matrix{SUFFIX_V2}.csv')

# ===== 标签窗口常数（自然日，非交易日）=====
PRE_WEEKS = 8                    # 预警窗口：事件日前 8 周
POST_WEEKS = 4                   # 出险后冷却期：簇尾后 4 周
MAX_CLUSTER_GAP_DAYS = 56        # 事件间隔 ≤56 自然日（8 周）归为一簇
MAX_CLUSTER_SPAN_DAYS = 84       # 簇内最大跨度 12 周（84 自然日），超出强制断簇

# ===== events.csv 期望列（人工核实后定稿）=====
REQUIRED_EVENT_COLS = ['股票代码', '公司简称', '公告日', '事件类型', '来源链接']


def merge_events_into_clusters(dates,
                               max_gap_days=MAX_CLUSTER_GAP_DAYS,
                               max_span_days=MAX_CLUSTER_SPAN_DAYS):
    """把一家公司的事件日合并为事件簇，返回按时间升序的 [(锚点a, 簇尾e), ...]。

    合并规则：与上一事件间隔 ≤ max_gap_days 且簇内跨度 ≤ max_span_days 才并入当前簇；
    否则另起一簇。跨度硬顶用于切断传递闭包（A→B 55 天、B→C 55 天不应把 A、C 并成一簇）。
    本函数不做任何统计量拟合，仅用固定日历阈值。
    """
    ds = sorted(pd.Timestamp(d).normalize() for d in dates)
    if not ds:
        return []

    clusters = []
    cur = [ds[0]]
    for d in ds[1:]:
        gap_days = (d - cur[-1]).days
        span_days = (d - cur[0]).days
        if gap_days <= max_gap_days and span_days <= max_span_days:
            cur.append(d)
        else:
            clusters.append((cur[0], cur[-1]))
            cur = [d]
    clusters.append((cur[0], cur[-1]))
    return clusters


def assign_company_labels(dates, clusters, pre_weeks=PRE_WEEKS, post_weeks=POST_WEEKS):
    """对单一公司按时间序列打标，返回 (y, excluded)，均为与 dates 等长的 ndarray。

    dates  : 该公司的交易日期（可迭代，任意顺序均可，按位置返回）
    clusters: merge_events_into_clusters 的输出，[(a, e), ...] 升序
    y       : 0/1，1 表示"未来 8 周内出险"
    excluded: True 表示该周不进入样本（出险后冷却期 / 空窗）

    重叠优先级：先算 mask_y1 与 mask_excl，再 `mask_excl &= ~mask_y1`，保证 y=1 不被剔除覆盖。
    本函数不做任何统计量拟合，仅用固定窗口常数。
    """
    t = pd.to_datetime(pd.Series(list(dates))).dt.normalize().values.astype('datetime64[D]')
    pre = np.timedelta64(pre_weeks * 7, 'D')
    post = np.timedelta64(post_weeks * 7, 'D')

    mask_y1 = np.zeros(len(t), dtype=bool)
    mask_excl = np.zeros(len(t), dtype=bool)

    for i, (a, e) in enumerate(clusters):
        a = np.datetime64(pd.Timestamp(a).normalize().date(), 'D')
        e = np.datetime64(pd.Timestamp(e).normalize().date(), 'D')
        n = None
        if i + 1 < len(clusters):
            n = np.datetime64(pd.Timestamp(clusters[i + 1][0]).normalize().date(), 'D')

        # 1. 预警窗口 [a-8周, a]
        mask_y1 |= (t >= a - pre) & (t <= a)
        # 2. 出险后冷却期 (a, e+4周]（右端取簇尾 e，防簇尾裸奔）
        mask_excl |= (t > a) & (t <= e + post)
        # 3. 出险后空窗 [e+4周, n-8周]；最后一簇延伸至数据末尾
        if n is not None:
            mask_excl |= (t > e + post) & (t < n - pre)
        else:
            mask_excl |= (t > e + post)

    # 优先级：y=1 覆盖剔除（布尔合成，避免顺序覆盖把前簇的 y=1 改回 0）
    mask_excl &= ~mask_y1

    y = np.zeros(len(t), dtype=int)
    y[mask_y1] = 1
    return y, mask_excl


def assert_cluster_tails_excluded(dates, clusters, y, excluded):
    """回归防线：簇内 [a, e] 每一周都必须被打标——除锚点 a 可为 y=1 外，其余必须剔除。

    历史 bug：剔除窗口右端只到"锚点 +4 周"，长簇的 [a+4周, e] 段落进 y=0 的"簇尾裸奔"区。
    本函数在打标后校验，防止该 bug 回归。
    """
    t = pd.to_datetime(pd.Series(list(dates))).dt.normalize().values.astype('datetime64[D]')
    y = np.asarray(y)
    excluded = np.asarray(excluded)

    for a, e in clusters:
        a = np.datetime64(pd.Timestamp(a).normalize().date(), 'D')
        e = np.datetime64(pd.Timestamp(e).normalize().date(), 'D')
        seg = (t >= a) & (t <= e)
        covered = excluded[seg] | (y[seg] == 1)
        assert covered.all(), (
            f"簇 [{a} ~ {e}] 内有 {int((~covered).sum())} 周未被打标（簇尾裸奔），"
            f"剔除窗口右端必须覆盖到 e+{POST_WEEKS}周"
        )


def load_events(path=EVENTS_FILE):
    """读取人工核实后的事件清单，返回按公司分组的事件日 dict。

    期望 events.csv 至少包含列：股票代码 / 公司简称 / 公告日 / 事件类型 / 来源链接。
    本函数只做读取与校验，不做统计量拟合。
    """
    df = pd.read_csv(path, encoding='utf-8-sig')
    missing = [c for c in REQUIRED_EVENT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少必要列：{missing}（期望 {REQUIRED_EVENT_COLS}）")
    df['公告日'] = pd.to_datetime(df['公告日'])
    df = df.sort_values(['股票代码', '公告日']).reset_index(drop=True)
    events_by_code = {code: grp['公告日'].tolist() for code, grp in df.groupby('股票代码')}
    return df, events_by_code


def build_labels_for_matrix(matrix, events_by_code, report_lines, tag=''):
    """给一张周频特征矩阵打事件标签，返回 (打标后仅保留非剔除行的 DataFrame, stats)。

    本函数不做任何统计量拟合：标签完全由 events.csv 的事件日历与固定窗口常数决定。
    """
    dfs = []
    n_y1 = n_excl = 0
    for code, grp in matrix.groupby('股票代码'):
        grp = grp.sort_values('交易日期').copy()
        dates = grp['交易日期']
        clusters = merge_events_into_clusters(events_by_code.get(code, []))
        y, excluded = assign_company_labels(dates, clusters)
        assert_cluster_tails_excluded(dates, clusters, y, excluded)

        grp['y'] = y
        grp['is_excluded'] = excluded
        n_y1 += int(y.sum())
        n_excl += int(excluded.sum())
        dfs.append(grp)

    labeled = pd.concat(dfs, ignore_index=True)
    kept = labeled[~labeled['is_excluded']].drop(columns=['is_excluded']).reset_index(drop=True)
    stats = {
        '矩阵': tag,
        '公司数': labeled['股票代码'].nunique(),
        '总周数': len(labeled),
        '正窗口数': n_y1,
        '剔除周数': n_excl,
        '入样周数': len(kept),
        '正样本占比': round(n_y1 / len(kept), 4) if len(kept) else 0.0,
    }
    return kept, stats


def main():
    report_lines = []
    log("=" * 60, report_lines)
    log("事件驱动标签生成（前向 8 周，规则见 AGENTS.md §5）", report_lines)

    if not os.path.exists(EVENTS_FILE):
        log(f"\n【待办】未找到事件清单 {EVENTS_FILE}", report_lines)
        log("请先完成事件核查（见 docs/过程文档/事件核查完整方案_20261014.md），"
            "合并生成 events.csv 后重新运行本脚本。", report_lines)
        return

    events_df, events_by_code = load_events()
    log(f"\n事件清单：{EVENTS_FILE}", report_lines)
    log(f"事件总数：{len(events_df)}，覆盖公司：{len(events_by_code)} 家", report_lines)

    clusters_all = {c: merge_events_into_clusters(ds) for c, ds in events_by_code.items()}
    n_clusters = sum(len(v) for v in clusters_all.values())
    log(f"合并为事件簇：{n_clusters} 个（原 {len(events_df)} 个事件）", report_lines)

    all_stats = []
    for path, tag in [(TRAIN_MATRIX, '训练期'), (TEST_MATRIX, '测试期')]:
        if not os.path.exists(path):
            log(f"\n【跳过】未找到特征矩阵 {path}（请先运行 python -m src.data.build_dataset）", report_lines)
            continue
        matrix = pd.read_csv(path, encoding='utf-8-sig')
        matrix['交易日期'] = pd.to_datetime(matrix['交易日期'])
        kept, stats = build_labels_for_matrix(matrix, events_by_code, report_lines, tag=tag)
        out_path = os.path.join(OUTPUT_DIR, f"labeled_{'train' if tag == '训练期' else 'test'}.csv")
        kept.to_csv(out_path, index=False, encoding='utf-8-sig')
        all_stats.append(stats)
        log(f"\n【{tag}】{out_path}", report_lines)
        log(f"  公司 {stats['公司数']} 家 / 总周 {stats['总周数']} / "
            f"正窗口 {stats['正窗口数']} / 剔除 {stats['剔除周数']} / "
            f"入样 {stats['入样周数']}（正占比 {stats['正样本占比']:.2%}）", report_lines)

    if all_stats:
        log("\n" + "-" * 60, report_lines)
        log(pd.DataFrame(all_stats).to_string(index=False), report_lines)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'label_report.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))


def load_events_combined(paths, report_lines=None):
    """读取并合并多份事件清单（一期 `events.csv` + 二期 `events_2025.csv`），返回 (df, events_by_code)。

    两份文件同 5 列 schema、事件类型字符串均已按 AGENTS §5 归一，可直接合并后按公司分组，
    由 `merge_events_into_clusters` 统一按 §5 簇规则合并。本函数只做读取 / 校验 / 去重，不做统计量拟合。
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p, encoding='utf-8-sig')
        missing = [c for c in REQUIRED_EVENT_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{p} 缺少必要列：{missing}（期望 {REQUIRED_EVENT_COLS}）")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df['公告日'] = pd.to_datetime(df['公告日'])
    n0 = len(df)
    df = (df.drop_duplicates(subset=['股票代码', '公告日', '事件类型'])
            .sort_values(['股票代码', '公告日']).reset_index(drop=True))
    if report_lines is not None and n0 - len(df) > 0:
        report_lines.append(f"事件清单合并：{n0} → {len(df)} 行"
                            f"（按 代码 + 公告日 + 类型 去重 {n0 - len(df)} 条）")
    events_by_code = {code: grp['公告日'].tolist() for code, grp in df.groupby('股票代码')}
    return df, events_by_code


def main_v2():
    """二期标签生成：合并 events.csv + events_2025.csv，给 2019-2025 全量矩阵打标 → labeled_v2.csv。"""
    report_lines = []
    log("=" * 60, report_lines)
    log("事件驱动标签生成（二期：合并 events.csv + events_2025.csv，2019-2025 全量矩阵）", report_lines)

    paths = [EVENTS_FILE, EVENTS_2025]
    events_df, events_by_code = load_events_combined(paths, report_lines)
    log(f"\n事件清单：{paths[0]} + {paths[1]}", report_lines)
    log(f"事件总数：{len(events_df)}，覆盖公司：{len(events_by_code)} 家", report_lines)

    clusters_all = {c: merge_events_into_clusters(ds) for c, ds in events_by_code.items()}
    n_clusters = sum(len(v) for v in clusters_all.values())
    log(f"合并为事件簇：{n_clusters} 个（原 {len(events_df)} 个事件）", report_lines)

    if not os.path.exists(MATRIX_V2):
        log(f"\n【待办】未找到二期特征矩阵 {MATRIX_V2}"
            "（请先运行 python -m src.data.build_dataset --v2）", report_lines)
        return

    matrix = pd.read_csv(MATRIX_V2, encoding='utf-8-sig')
    matrix['交易日期'] = pd.to_datetime(matrix['交易日期'])
    kept, stats = build_labels_for_matrix(matrix, events_by_code, report_lines, tag='二期全量')
    out_path = os.path.join(OUTPUT_DIR, f'labeled{SUFFIX_V2}.csv')
    kept.to_csv(out_path, index=False, encoding='utf-8-sig')
    log(f"\n【二期全量】{out_path}", report_lines)
    log(f"  公司 {stats['公司数']} 家 / 总周 {stats['总周数']} / "
        f"正窗口 {stats['正窗口数']} / 剔除 {stats['剔除周数']} / "
        f"入样 {stats['入样周数']}（正占比 {stats['正样本占比']:.2%}）", report_lines)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f'label{SUFFIX_V2}_report.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))


if __name__ == '__main__':
    import sys
    if '--v2' in sys.argv:
        main_v2()
    else:
        main()
