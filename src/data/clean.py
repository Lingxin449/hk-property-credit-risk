"""
缺失值清理与时间切分（Step 2-3）
Step 2 不涉及拟合，Step 3 按日期切分
"""

from src.config import CANDIDATE_FEATURES, TRAIN_CUTOFF, log


def missing_value_clean(df, report_lines, cutoff=TRAIN_CUTOFF):
    """Step 2: 缺失值清理（动态 dropna）

    cutoff：仅用于把记录数按训练/测试期分桶打印；一期 2022-12-31、二期用训练段上界 2023-12-31。
    """
    log("\n" + "=" * 60, report_lines)
    log("Step 2: 缺失值清理（动态 dropna）", report_lines)

    nan_summary = df[CANDIDATE_FEATURES].isnull().sum()
    nan_summary = nan_summary[nan_summary > 0]
    if len(nan_summary) > 0:
        log("缺失值分布：", report_lines)
        log(nan_summary.to_string(), report_lines)
    else:
        log("候选特征中无缺失值", report_lines)

    before = len(df)
    df_clean = df.dropna(subset=CANDIDATE_FEATURES).copy()
    after = len(df_clean)

    log(f"删除前总记录数：{before}", report_lines)
    log(f"删除后总记录数：{after}", report_lines)
    log(f"删除比例：{(before - after) / before:.2%}", report_lines)

    log("\n各公司样本量统计：", report_lines)
    sample_stats = df_clean.groupby(['风险层级', '股票代码', '公司简称']).size().reset_index(name='记录数')
    log(sample_stats.to_string(index=False), report_lines)

    train_size = len(df_clean[df_clean['交易日期'] <= cutoff])
    test_size = len(df_clean[df_clean['交易日期'] > cutoff])
    log(f"\n训练段（<={cutoff}）记录数：{train_size}", report_lines)
    log(f"训练段之后（>{cutoff}）记录数：{test_size}", report_lines)

    if train_size < 4000:
        log("【警告】训练段记录数 < 4000", report_lines)
    if test_size < 2000:
        log("【警告】训练段之后记录数 < 2000", report_lines)

    return df_clean


def time_split(df, report_lines):
    """Step 3: 严格时间切分（防泄露核心）"""
    log("\n" + "=" * 60, report_lines)
    log("Step 3: 严格时间切分（防泄露核心）", report_lines)

    train = df[df['交易日期'] <= TRAIN_CUTOFF].copy()
    test = df[df['交易日期'] > TRAIN_CUTOFF].copy()

    log(f"训练期记录数：{len(train)}", report_lines)
    log(f"测试期记录数：{len(test)}", report_lines)
    log(f"训练期占比：{len(train) / len(df):.2%}", report_lines)

    return train, test
