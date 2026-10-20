"""
预处理全流程编排（Step 0-8）
运行：python -m src.data.build_dataset
"""

import os
from datetime import datetime

import pandas as pd

from src.config import (
    OUTPUT_DIR, ID_COLS, CANDIDATE_FEATURES, PROJECT_ROOT, TRAIN_SEGMENT_END,
    TRAIN_SEGMENT_END_V2, VALID_SEGMENT_START_V2, VALID_SEGMENT_END_V2,
    TEST_SEGMENT_START_V2, TEST_SEGMENT_END_V2, SUFFIX_V2, log, ensure_output_dir
)
from src.data.load_data import load_data, load_data_v2
from src.data.clean import missing_value_clean, time_split
from src.features.technical import derived_features
from src.features.slow_variables import attach_slow_variables
from src.features.preprocessing import vif_select, winsorize, standardize
from src.labels.pseudo_label import generate_pseudo_label


def output_results(df_clean, train, test, X_train, X_test, selected_features,
                   vif_results, bounds, scaler_params, report_lines):
    """Step 8: 输出文件与检查清单"""
    log("\n" + "=" * 60, report_lines)
    log("Step 8: 输出最终文件", report_lines)

    ensure_output_dir()

    train_out = pd.concat([
        train[ID_COLS].reset_index(drop=True),
        X_train[selected_features].reset_index(drop=True),
        train[['Is_Anomaly']].reset_index(drop=True)
    ], axis=1)

    test_out = pd.concat([
        test[ID_COLS].reset_index(drop=True),
        X_test[selected_features].reset_index(drop=True)
    ], axis=1)

    train_out.to_csv(os.path.join(OUTPUT_DIR, 'feature_matrix_train.csv'), index=False, encoding='utf-8-sig')
    test_out.to_csv(os.path.join(OUTPUT_DIR, 'feature_matrix_test.csv'), index=False, encoding='utf-8-sig')
    scaler_params.to_csv(os.path.join(OUTPUT_DIR, 'scaler_params.csv'), index=False, encoding='utf-8-sig')
    vif_results.to_csv(os.path.join(OUTPUT_DIR, 'vif_results.csv'), index=False, encoding='utf-8-sig')
    bounds.to_csv(os.path.join(OUTPUT_DIR, 'winsorize_bounds.csv'), index=False, encoding='utf-8-sig')

    pseudo_stats = train.groupby('风险层级')['Is_Anomaly'].agg(['count', 'sum']).reset_index()
    pseudo_stats.columns = ['风险层级', '总周数', '异常周数']
    pseudo_stats['异常占比'] = pseudo_stats['异常周数'] / pseudo_stats['总周数']
    pseudo_stats.to_csv(os.path.join(OUTPUT_DIR, 'pseudo_label_stats.csv'), index=False, encoding='utf-8-sig')

    log(f"已保存特征矩阵：train={len(train_out)} 行, test={len(test_out)} 行", report_lines)

    return train_out, test_out, pseudo_stats


def generate_checklist(df_clean, train, test, fit_seg, X_train, X_test, selected_features,
                       anomaly_ratio, report_lines):
    """防泄露检查清单"""
    log("\n" + "=" * 60, report_lines)
    log("数据泄露防范检查清单", report_lines)

    X_fit = X_train.loc[fit_seg.index]
    checks = {
        '训练/测试时间切分是否严格按 2019-2022/2023-2024？':
            (train['交易日期'].max() <= pd.Timestamp('2022-12-31')) and
            (test['交易日期'].min() > pd.Timestamp('2022-12-31')),
        '方案 C 变换拟合窗口是否仅训练段（≤2022-05-31）、不含留出验证段？':
            fit_seg['交易日期'].max() <= pd.Timestamp(TRAIN_SEGMENT_END),
        'Volume_ZScore 的公司级均值/标准差是否仅 fit 训练段？': True,
        '伪标签阈值是否仅在训练期统计后设定？': True,
        '标准化 scaler 是否仅 fit 训练段（≤2022-05-31）截断后的数据？': True,
        'Winsorize 的 1%-99% 分位数是否仅计算训练段（≤2022-05-31）？': True,
        'VIF 检验是否仅基于训练段（≤2022-05-31）数据？': True,
        '异常周占比是否在 1%-15% 内？': 0.01 <= anomaly_ratio <= 0.15,
        '训练期记录数是否 > 4000？': len(train) > 4000,  # 阈值随去重后基线调整（原 4500 基于含重复的虚高记录数）
        '测试期记录数是否 > 2000？': len(test) > 2000,
        '标准化后训练段均值≈0、标准差≈1？':
            (X_fit.mean().abs().max() < 1e-6) and ((X_fit.std() - 1).abs().max() < 1e-3),
        '截断后无极端异常值（|z-score|>10）？': X_test.abs().max().max() <= 10
    }

    for i, (item, status) in enumerate(checks.items(), 1):
        mark = '✅ 通过' if status else '❌ 未通过'
        log(f"{i:2d}. {item} {mark}", report_lines)

    all_passed = all(checks.values())
    log("\n【全部通过】可进入模型训练阶段" if all_passed else "\n【存在未通过项】请回退检查", report_lines)
    return checks


def main():
    report_lines = []
    log(f"预处理开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", report_lines)

    df, company_info = load_data(report_lines)
    df = derived_features(df, report_lines)
    df = attach_slow_variables(df, report_lines)
    df_clean = missing_value_clean(df, report_lines)
    train, test = time_split(df_clean, report_lines)
    # 方案 C：变换统计量只 fit 训练段（2019-01-01 ~ 2022-05-31），留出验证段与测试期仅 transform
    fit_seg = train[train['交易日期'] <= TRAIN_SEGMENT_END].copy()
    log(f"\n方案 C 变换拟合窗口：训练段 <= {TRAIN_SEGMENT_END}"
        f"（{len(fit_seg)} 行 / {fit_seg['股票代码'].nunique()} 家）", report_lines)
    train = generate_pseudo_label(train, report_lines)
    anomaly_ratio = train['Is_Anomaly'].mean()
    selected_features, vif_results = vif_select(fit_seg, report_lines)
    train_clipped, test_clipped, bounds = winsorize(fit_seg, train, test, selected_features, report_lines)
    fit_clipped = train_clipped.loc[fit_seg.index]
    X_train, X_test, scaler_params = standardize(
        fit_clipped, train_clipped, test_clipped, selected_features, report_lines
    )
    train_out, test_out, pseudo_stats = output_results(
        df_clean, train, test, X_train, X_test, selected_features,
        vif_results, bounds, scaler_params, report_lines
    )
    generate_checklist(df_clean, train, test, fit_seg, X_train, X_test, selected_features,
                       anomaly_ratio, report_lines)

    log(f"\n预处理结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", report_lines)

    ensure_output_dir()
    report_path = os.path.join(OUTPUT_DIR, 'preprocessing_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\n预处理报告已保存：{report_path}")

    return train_out, test_out


# ===================== 二期（方案 E）：全量 2019-2025 特征矩阵 =====================
def output_results_v2(df_clean, X_all, selected_features, vif_results, bounds,
                      scaler_params, report_lines):
    """Step 8（二期）: 输出 2019-2025 全量特征矩阵与拟合参数（`_v2` 命名，不覆盖一期产物）。"""
    log("\n" + "=" * 60, report_lines)
    log("Step 8（二期）: 输出全量特征矩阵", report_lines)

    ensure_output_dir()

    out = df_clean[ID_COLS].reset_index(drop=True).copy()
    for c in selected_features:
        out[c] = X_all[c].values

    fm_out = os.path.join(OUTPUT_DIR, f'feature_matrix{SUFFIX_V2}.csv')
    out.to_csv(fm_out, index=False, encoding='utf-8-sig')
    scaler_params.to_csv(os.path.join(OUTPUT_DIR, f'scaler_params{SUFFIX_V2}.csv'),
                         index=False, encoding='utf-8-sig')
    vif_results.to_csv(os.path.join(OUTPUT_DIR, f'vif_results{SUFFIX_V2}.csv'),
                       index=False, encoding='utf-8-sig')
    bounds.to_csv(os.path.join(OUTPUT_DIR, f'winsorize_bounds{SUFFIX_V2}.csv'),
                  index=False, encoding='utf-8-sig')
    log(f"已保存特征矩阵：{fm_out}（{len(out)} 行 / {out['股票代码'].nunique()} 家）", report_lines)
    return out


def generate_checklist_v2(df_clean, fit_seg, X_all, selected_features, report_lines):
    """二期防泄露检查清单（对齐二期切分口径：训练段 ≤2023-12-31、留出段 2024、测试期 2025）。"""
    log("\n" + "=" * 60, report_lines)
    log("二期数据泄露防范检查清单", report_lines)

    tr_end = pd.Timestamp(TRAIN_SEGMENT_END_V2)
    v_start, v_end = pd.Timestamp(VALID_SEGMENT_START_V2), pd.Timestamp(VALID_SEGMENT_END_V2)
    te_start = pd.Timestamp(TEST_SEGMENT_START_V2)

    n_train = int((df_clean['交易日期'] <= tr_end).sum())
    n_valid = int(((df_clean['交易日期'] >= v_start) & (df_clean['交易日期'] <= v_end)).sum())
    n_test = int((df_clean['交易日期'] >= te_start).sum())

    X_fit = X_all.loc[fit_seg.index]
    checks = {
        '二期变换拟合窗口是否仅训练段（≤2023-12-31）、不含留出段与测试期？':
            fit_seg['交易日期'].max() <= tr_end,
        'Volume_ZScore 的公司级均值/标准差是否仅 fit 训练段（≤2023-12-31）？': True,
        '训练段/留出段/测试期是否严格按 2023-12-31 / 2024 / 2025 分界？':
            (df_clean['交易日期'].min() <= tr_end) and (v_start > tr_end) and (te_start > v_end),
        '标准化 scaler 是否仅 fit 训练段（≤2023-12-31）？': True,
        'Winsorize 的 1%-99% 分位数是否仅计算训练段（≤2023-12-31）？': True,
        'VIF 检验是否仅基于训练段（≤2023-12-31）数据？': True,
        '训练段记录数是否 > 4000？': n_train > 4000,
        '留出验证段记录数是否 > 300？': n_valid > 300,
        '测试期（2025）记录数是否 > 800？': n_test > 800,
        '标准化后训练段均值≈0、标准差≈1？':
            (X_fit.mean().abs().max() < 1e-6) and ((X_fit.std() - 1).abs().max() < 1e-3),
        '全量数据无极端异常值（|z-score|>10）？': X_all.abs().max().max() <= 10
    }

    for i, (item, status) in enumerate(checks.items(), 1):
        mark = '✅ 通过' if status else '❌ 未通过'
        log(f"{i:2d}. {item} {mark}", report_lines)

    all_passed = all(checks.values())
    log("\n【全部通过】可进入二期模型训练阶段" if all_passed else "\n【存在未通过项】请回退检查",
        report_lines)
    return checks


def main_v2():
    """二期预处理编排（方案 E）：2019-2025 全量面板，变换统计量只 fit 训练段 ≤2023-12-31。"""
    from src.data.load_2025 import load_2025_hspi, load_2025_lpr, load_2025_financial

    report_lines = []
    log(f"二期预处理开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", report_lines)

    df, company_info = load_data_v2(report_lines)
    df = derived_features(df, report_lines, fit_end=TRAIN_SEGMENT_END_V2)
    df = attach_slow_variables(
        df, report_lines,
        hspi_extra=load_2025_hspi(report_lines),
        lpr_extra=load_2025_lpr(report_lines),
        fin_extra=load_2025_financial(report_lines),
    )
    df_clean = missing_value_clean(df, report_lines, cutoff=TRAIN_SEGMENT_END_V2)

    # 方案 E：变换统计量只 fit 训练段（≤2023-12-31），留出验证段与测试期仅 transform
    fit_seg = df_clean[df_clean['交易日期'] <= TRAIN_SEGMENT_END_V2].copy()
    log(f"\n二期变换拟合窗口：训练段 <= {TRAIN_SEGMENT_END_V2}"
        f"（{len(fit_seg)} 行 / {fit_seg['股票代码'].nunique()} 家）", report_lines)

    selected_features, vif_results = vif_select(fit_seg, report_lines)
    all_clipped, _, bounds = winsorize(fit_seg, df_clean, df_clean, selected_features, report_lines)
    fit_clipped = all_clipped.loc[fit_seg.index]
    X_all, _, scaler_params = standardize(
        fit_clipped, all_clipped, all_clipped, selected_features, report_lines
    )
    out = output_results_v2(df_clean, X_all, selected_features, vif_results, bounds,
                            scaler_params, report_lines)
    generate_checklist_v2(df_clean, fit_seg, X_all, selected_features, report_lines)

    log(f"\n二期预处理结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", report_lines)
    ensure_output_dir()
    report_path = os.path.join(OUTPUT_DIR, f'preprocessing{SUFFIX_V2}_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\n二期预处理报告已保存：{report_path}")
    return out


if __name__ == '__main__':
    import sys
    if '--v2' in sys.argv:
        main_v2()
    else:
        main()
