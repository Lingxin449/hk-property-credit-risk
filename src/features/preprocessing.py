"""
VIF 共线性检验 / Winsorize 截断 / Z-Score 标准化（Step 5-7）
所有统计量（VIF、分位数、均值/标准差）仅基于方案 C 训练段（2019-01-01 ~ 2022-05-31）拟合，
留出验证段（2022-06-01 ~ 2022-12-31）与测试期（2023-2024）仅 transform。
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.config import VIF_THRESHOLD, WINSOR_LOWER, WINSOR_UPPER, CANDIDATE_FEATURES, log


def vif_select(fit_df, report_lines):
    """Step 5: VIF 共线性检验与动态特征筛选。统计量仅基于训练段 fit_df 拟合。"""
    log("\n" + "=" * 60, report_lines)
    log("Step 5: VIF 共线性检验与动态特征筛选", report_lines)

    selected = CANDIDATE_FEATURES.copy()

    X_vif = fit_df[selected].values
    vif_data = pd.DataFrame({
        'Feature': selected,
        'VIF': [variance_inflation_factor(X_vif, i) for i in range(X_vif.shape[1])]
    }).sort_values('VIF', ascending=False)

    log("初始 VIF 结果：", report_lines)
    log(vif_data.to_string(index=False), report_lines)

    max_vif = vif_data['VIF'].max()
    while max_vif > VIF_THRESHOLD and len(selected) > 2:
        drop_feature = vif_data.iloc[0]['Feature']
        log(f"删除高 VIF 特征：{drop_feature}（VIF={vif_data.iloc[0]['VIF']:.2f}）", report_lines)
        selected.remove(drop_feature)

        X_vif = fit_df[selected].values
        vif_data = pd.DataFrame({
            'Feature': selected,
            'VIF': [variance_inflation_factor(X_vif, i) for i in range(X_vif.shape[1])]
        }).sort_values('VIF', ascending=False)
        max_vif = vif_data['VIF'].max()

    log(f"\n最终入选特征：{selected}（共 {len(selected)} 维）", report_lines)
    log("最终 VIF 结果：", report_lines)
    log(vif_data.to_string(index=False), report_lines)

    vif_results = vif_data.copy()
    vif_results['Status'] = 'Kept'
    for feat in set(CANDIDATE_FEATURES) - set(selected):
        vif_results = pd.concat([
            vif_results,
            pd.DataFrame({'Feature': [feat], 'VIF': [np.nan], 'Status': ['Dropped']})
        ], ignore_index=True)

    return selected, vif_results


def winsorize(fit_df, train, test, selected_features, report_lines):
    """Step 6: 极端值截断（Winsorize）。分位数仅基于训练段 fit_df 拟合；train/test 仅 transform。"""
    log("\n" + "=" * 60, report_lines)
    log("Step 6: 极端值截断（Winsorize）", report_lines)

    lower_bounds = fit_df[selected_features].quantile(WINSOR_LOWER)
    upper_bounds = fit_df[selected_features].quantile(WINSOR_UPPER)

    bounds = pd.DataFrame({
        'Feature': selected_features,
        'Lower_1%': [lower_bounds[f] for f in selected_features],
        'Upper_99%': [upper_bounds[f] for f in selected_features]
    })
    log("Winsorize 阈值（训练段 1%-99% 分位）：", report_lines)
    log(bounds.to_string(index=False), report_lines)

    train_clipped = train[selected_features].copy()
    test_clipped = test[selected_features].copy()
    for col in selected_features:
        train_clipped[col] = train_clipped[col].clip(lower=lower_bounds[col], upper=upper_bounds[col])
        test_clipped[col] = test_clipped[col].clip(lower=lower_bounds[col], upper=upper_bounds[col])

    return train_clipped, test_clipped, bounds


def standardize(fit_clipped, train_clipped, test_clipped, selected_features, report_lines):
    """Step 7: Z-Score 标准化。均值/标准差仅基于训练段 fit_clipped 拟合；train/test 仅 transform。"""
    log("\n" + "=" * 60, report_lines)
    log("Step 7: Z-Score 标准化（训练段 fit，其余 transform）", report_lines)

    scaler = StandardScaler()
    scaler.fit(fit_clipped)

    X_train = pd.DataFrame(scaler.transform(train_clipped), columns=selected_features, index=train_clipped.index)
    X_test = pd.DataFrame(scaler.transform(test_clipped), columns=selected_features, index=test_clipped.index)

    scaler_params = pd.DataFrame({
        'Feature': selected_features,
        'Mean': scaler.mean_,
        'Scale': scaler.scale_
    })
    log("标准化参数（训练段）：", report_lines)
    log(scaler_params.to_string(index=False), report_lines)

    # 标准化后均值≈0、标准差≈1 的校验只在训练段成立（fit 窗口），故以 fit_clipped 为准
    X_fit = pd.DataFrame(scaler.transform(fit_clipped), columns=selected_features, index=fit_clipped.index)
    train_means = X_fit.mean().abs().max()
    train_stds = (X_fit.std() - 1).abs().max()
    log(f"\n训练段标准化后最大绝对均值：{train_means:.6f}（应≈0）", report_lines)
    log(f"训练段标准化后最大标准差偏差：{train_stds:.6f}（应≈0）", report_lines)

    test_max_abs_z = X_test.abs().max().max()
    log(f"测试集最大绝对 Z-Score：{test_max_abs_z:.2f}", report_lines)
    if test_max_abs_z > 10:
        log("【警告】测试集存在 |z-score| > 10 的极端值", report_lines)
    else:
        log("【通过】截断后无极端异常值", report_lines)

    return X_train, X_test, scaler_params
