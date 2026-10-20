"""
方案 C 评估协议切分器（口径纪律）
运行：python -m src.data.split

把事件标签产物切成三段（全部按**日期**切，不拟合任何统计量）：
    训练段      2019-01-01 ~ 2022-05-31   —— 模型训练；亦是所有变换统计量的拟合窗口
    留出验证段  2022-06-01 ~ 2022-12-31   —— 概率校准、代价阈值、特征选型
    测试期      2023-01-01 ~ 2024-12-31   —— 最终回测，全程零接触

切点净化带（≥8 周，**显式按日期剔除**，不得指望"标签天然稀疏"）：
    训练段尾部  2022-04-06 ~ 2022-05-31   （防其标签窗口伸进留出段）
    留出验证段尾部 2022-11-06 ~ 2022-12-31 （防测试期标签信息渗入校准）
净化带宽 = 标签窗口 8 周 = 56 自然日（`config.PURGE_DAYS`），闭区间 (cut−56天, cut]。

本模块不拟合任何统计量：输入 `labeled_*.csv` 的特征已由 build_dataset 按训练段
（2019-01-01 ~ 2022-05-31）统计量变换，此处仅按日期切行。
"""

import os

import pandas as pd

from src.config import (
    OUTPUT_DIR, TRAIN_SEGMENT_END, VALID_SEGMENT_START, VALID_SEGMENT_END, PURGE_DAYS,
    TRAIN_SEGMENT_END_V2, VALID_SEGMENT_START_V2, VALID_SEGMENT_END_V2,
    TEST_SEGMENT_START_V2, TEST_SEGMENT_END_V2, SUFFIX_V2, log
)

# ===== 输入 / 输出路径 =====
TRAIN_LABELED = os.path.join(OUTPUT_DIR, 'labeled_train.csv')   # 2019-2022 入样行
TEST_LABELED = os.path.join(OUTPUT_DIR, 'labeled_test.csv')     # 2023-2024 入样行
LABELED_V2 = os.path.join(OUTPUT_DIR, f'labeled{SUFFIX_V2}.csv')  # 2019-2025 入样行（二期）

# ===== 方案 C 分段边界（引用 config，不另抄数值）=====
SEG_TRAIN_END = TRAIN_SEGMENT_END          # 训练段结束日
SEG_VALID_START = VALID_SEGMENT_START      # 留出验证段起始日
SEG_VALID_END = VALID_SEGMENT_END          # 留出验证段结束日


def _read_labeled(path):
    """读取 labeled 产物并把交易日期转为 datetime。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {path}，请先运行 python -m src.labels.build_labels")
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    return df


def purge_band(df, cut_date, days=PURGE_DAYS):
    """按日期显式剔除净化带 (cut_date − days, cut_date]（闭区间，自然日）。

    cut_date 为切点当日（训练段为 2022-05-31，留出段为 2022-12-31）。
    返回 (剔除后的 DataFrame, 净化带起始日, 净化带结束日)。
    """
    cut = pd.Timestamp(cut_date)
    start = cut - pd.Timedelta(days=days) + pd.Timedelta(days=1)
    mask = (df['交易日期'] >= start) & (df['交易日期'] <= cut)
    kept = df.loc[~mask].reset_index(drop=True)
    return kept, start, cut


def _stats(df, tag):
    """统计某分段的入样行数 / 正例数 / 公司数。"""
    n = len(df)
    pos = int(df['y'].sum())
    return {
        '分段': tag,
        '入样行数': n,
        '正例(y=1)': pos,
        '正例占比': round(pos / n, 4) if n else 0.0,
        '公司数': df['股票代码'].nunique(),
    }


def _assert_no_purge_leak(seg_train, seg_valid):
    """回归防线：两段净化带内不得残留任何行（必须显式剔除）。"""
    tp_start = pd.Timestamp(SEG_TRAIN_END) - pd.Timedelta(days=PURGE_DAYS) + pd.Timedelta(days=1)
    leak_tr = seg_train[seg_train['交易日期'] >= tp_start]
    assert leak_tr.empty, f"训练段残留 {len(leak_tr)} 行净化带数据（{tp_start.date()} 起）"
    vp_start = pd.Timestamp(SEG_VALID_END) - pd.Timedelta(days=PURGE_DAYS) + pd.Timedelta(days=1)
    leak_va = seg_valid[seg_valid['交易日期'] >= vp_start]
    assert leak_va.empty, f"留出验证段残留 {len(leak_va)} 行净化带数据（{vp_start.date()} 起）"


def split_scheme_c(report_lines=None):
    """按方案 C 切分三段，返回 (训练段, 留出验证段, 测试期, stats 列表)。

    训练段只保留 ≤ 2022-05-31 且不在净化带内的行；留出验证段保留 2022-06-01 ~
    2022-12-31 且不在净化带内的行；测试期取其全部（只评估、不参与 fit，无需净化）。
    """
    train_all = _read_labeled(TRAIN_LABELED)
    test_all = _read_labeled(TEST_LABELED)

    # 训练段：≤ 2022-05-31，再显式剔除尾部净化带
    seg_train = train_all[train_all['交易日期'] <= pd.Timestamp(SEG_TRAIN_END)].copy()
    n_train_raw = len(seg_train)
    seg_train, tp_start, tp_end = purge_band(seg_train, SEG_TRAIN_END)

    # 留出验证段：2022-06-01 ~ 2022-12-31，再显式剔除尾部净化带
    seg_valid = train_all[(train_all['交易日期'] >= pd.Timestamp(SEG_VALID_START)) &
                          (train_all['交易日期'] <= pd.Timestamp(SEG_VALID_END))].copy()
    n_valid_raw = len(seg_valid)
    seg_valid, vp_start, vp_end = purge_band(seg_valid, SEG_VALID_END)

    _assert_no_purge_leak(seg_train, seg_valid)

    stats = [_stats(seg_train, '训练段'), _stats(seg_valid, '留出验证段'), _stats(test_all, '测试期')]

    if report_lines is not None:
        log(f"训练段  ≤ {SEG_TRAIN_END}：{n_train_raw} 行 → 剔除净化带 "
            f"{tp_start.date()} ~ {tp_end.date()} 后 {len(seg_train)} 行", report_lines)
        log(f"留出验证段 {SEG_VALID_START} ~ {SEG_VALID_END}：{n_valid_raw} 行 → 剔除净化带 "
            f"{vp_start.date()} ~ {vp_end.date()} 后 {len(seg_valid)} 行", report_lines)
        log(f"测试期  2023-01-01 ~ 2024-12-31：{len(test_all)} 行（只评估、不参与 fit，无需净化）",
            report_lines)

    return seg_train, seg_valid, test_all, stats


def main():
    report_lines = []
    log("=" * 60, report_lines)
    log("方案 C 切分器（训练段 / 留出验证段 / 测试期，含显式净化带）", report_lines)

    seg_train, seg_valid, seg_test, stats = split_scheme_c(report_lines)

    log("\n" + "-" * 60, report_lines)
    log(pd.DataFrame(stats).to_string(index=False), report_lines)
    log(f"\n日期范围：训练段 {seg_train['交易日期'].min().date()} ~ {seg_train['交易日期'].max().date()}"
        f" / 留出段 {seg_valid['交易日期'].min().date()} ~ {seg_valid['交易日期'].max().date()}"
        f" / 测试期 {seg_test['交易日期'].min().date()} ~ {seg_test['交易日期'].max().date()}",
        report_lines)

    for name, df in [('split_train', seg_train), ('split_valid', seg_valid), ('split_test', seg_test)]:
        out = os.path.join(OUTPUT_DIR, f'{name}.csv')
        df.to_csv(out, index=False, encoding='utf-8-sig')
        log(f"已保存 {out}（{len(df)} 行）", report_lines)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'split_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    return seg_train, seg_valid, seg_test


# ===================== 二期（方案 E）：训练段 / 留出段 / 2025 新考卷 =====================
SEG2_TRAIN_END = TRAIN_SEGMENT_END_V2       # 2023-12-31
SEG2_VALID_START = VALID_SEGMENT_START_V2   # 2024-01-01
SEG2_VALID_END = VALID_SEGMENT_END_V2       # 2024-12-31
SEG2_TEST_START = TEST_SEGMENT_START_V2     # 2025-01-01
SEG2_TEST_END = TEST_SEGMENT_END_V2         # 2025-12-31


def _assert_no_purge_leak_v2(seg_train, seg_valid):
    """回归防线：二期两段净化带内不得残留任何行（必须显式剔除）。"""
    tp_start = pd.Timestamp(SEG2_TRAIN_END) - pd.Timedelta(days=PURGE_DAYS) + pd.Timedelta(days=1)
    leak_tr = seg_train[seg_train['交易日期'] >= tp_start]
    assert leak_tr.empty, f"二期训练段残留 {len(leak_tr)} 行净化带数据（{tp_start.date()} 起）"
    vp_start = pd.Timestamp(SEG2_VALID_END) - pd.Timedelta(days=PURGE_DAYS) + pd.Timedelta(days=1)
    leak_va = seg_valid[seg_valid['交易日期'] >= vp_start]
    assert leak_va.empty, f"二期留出段残留 {len(leak_va)} 行净化带数据（{vp_start.date()} 起）"


def split_scheme_e(report_lines=None):
    """按方案 E 切分三段，返回 (训练段, 留出验证段, 测试期(2025), stats 列表)。

    训练段 ≤2023-12-31（显式剔除尾部净化带 2023-11-06 ~ 2023-12-31）；
    留出验证段 2024-01-01 ~ 2024-12-31（显式剔除尾部净化带 2024-11-06 ~ 2024-12-31）；
    测试期 2025-01-01 ~ 2025-12-31（只评估、不参与 fit，无需净化）。
    """
    labeled = _read_labeled(LABELED_V2)

    seg_train = labeled[labeled['交易日期'] <= pd.Timestamp(SEG2_TRAIN_END)].copy()
    n_train_raw = len(seg_train)
    seg_train, tp_start, tp_end = purge_band(seg_train, SEG2_TRAIN_END)

    seg_valid = labeled[(labeled['交易日期'] >= pd.Timestamp(SEG2_VALID_START)) &
                        (labeled['交易日期'] <= pd.Timestamp(SEG2_VALID_END))].copy()
    n_valid_raw = len(seg_valid)
    seg_valid, vp_start, vp_end = purge_band(seg_valid, SEG2_VALID_END)

    seg_test = labeled[(labeled['交易日期'] >= pd.Timestamp(SEG2_TEST_START)) &
                       (labeled['交易日期'] <= pd.Timestamp(SEG2_TEST_END))].copy()

    _assert_no_purge_leak_v2(seg_train, seg_valid)

    stats = [_stats(seg_train, '训练段'), _stats(seg_valid, '留出验证段'), _stats(seg_test, '测试期(2025)')]

    if report_lines is not None:
        log(f"二期训练段  ≤ {SEG2_TRAIN_END}：{n_train_raw} 行 → 剔除净化带 "
            f"{tp_start.date()} ~ {tp_end.date()} 后 {len(seg_train)} 行", report_lines)
        log(f"二期留出验证段 {SEG2_VALID_START} ~ {SEG2_VALID_END}：{n_valid_raw} 行 → 剔除净化带 "
            f"{vp_start.date()} ~ {vp_end.date()} 后 {len(seg_valid)} 行", report_lines)
        log(f"二期测试期  {SEG2_TEST_START} ~ {SEG2_TEST_END}：{len(seg_test)} 行"
            f"（只评估、不参与 fit，无需净化）", report_lines)

    return seg_train, seg_valid, seg_test, stats


def main_v2():
    """执行二期（方案 E）三段切分，输出 split_*_v2.csv。"""
    report_lines = []
    log("=" * 60, report_lines)
    log("方案 E 切分器（二期：训练段 / 留出验证段 / 2025 新考卷，含显式净化带）", report_lines)

    seg_train, seg_valid, seg_test, stats = split_scheme_e(report_lines)

    log("\n" + "-" * 60, report_lines)
    log(pd.DataFrame(stats).to_string(index=False), report_lines)
    log(f"\n日期范围：训练段 {seg_train['交易日期'].min().date()} ~ {seg_train['交易日期'].max().date()}"
        f" / 留出段 {seg_valid['交易日期'].min().date()} ~ {seg_valid['交易日期'].max().date()}"
        f" / 测试期 {seg_test['交易日期'].min().date()} ~ {seg_test['交易日期'].max().date()}",
        report_lines)

    for name, df in [('split_train', seg_train), ('split_valid', seg_valid), ('split_test', seg_test)]:
        out = os.path.join(OUTPUT_DIR, f'{name}{SUFFIX_V2}.csv')
        df.to_csv(out, index=False, encoding='utf-8-sig')
        log(f"已保存 {out}（{len(df)} 行）", report_lines)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f'split{SUFFIX_V2}_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    return seg_train, seg_valid, seg_test


if __name__ == '__main__':
    import sys
    if '--v2' in sys.argv:
        main_v2()
    else:
        main()
