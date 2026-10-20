"""
伪标签生成（Step 4）
标签规则：收益率<-15% OR Volume_ZScore>1.645 OR 振幅>20%
含自动阈值调整机制。仅基于训练期生成。
"""

from src.config import ANOMALY_THRESH_RET, ANOMALY_THRESH_VOLZ, ANOMALY_THRESH_AMP, log


def generate_pseudo_label(train, report_lines):
    """Step 4: 伪标签生成与占比硬检查"""
    log("\n" + "=" * 60, report_lines)
    log("Step 4: 伪标签生成与占比硬检查", report_lines)

    train['Is_Anomaly'] = (
        (train['周收益率'] < ANOMALY_THRESH_RET) |
        (train['Volume_ZScore'] > ANOMALY_THRESH_VOLZ) |
        (train['振幅'] > ANOMALY_THRESH_AMP)
    ).astype(int)

    anomaly_ratio = train['Is_Anomaly'].mean()
    log(f"初始异常周占比：{anomaly_ratio:.2%}", report_lines)

    if anomaly_ratio < 0.01:
        log("【警告】异常占比过低，放宽阈值至跌幅>10%、振幅>15%", report_lines)
        train['Is_Anomaly'] = (
            (train['周收益率'] < -0.10) |
            (train['Volume_ZScore'] > 1.645) |
            (train['振幅'] > 0.15)
        ).astype(int)
    elif anomaly_ratio > 0.15:
        log("【警告】异常占比过高，收紧阈值至跌幅>20%、Volume_ZScore>2.0、振幅>25%", report_lines)
        train['Is_Anomaly'] = (
            (train['周收益率'] < -0.20) |
            (train['Volume_ZScore'] > 2.0) |
            (train['振幅'] > 0.25)
        ).astype(int)

    final_anomaly_ratio = train['Is_Anomaly'].mean()
    log(f"调整后异常周占比：{final_anomaly_ratio:.2%}", report_lines)
    log(f"异常周数：{train['Is_Anomaly'].sum()} / {len(train)}", report_lines)

    log("\n各风险层级异常周占比：", report_lines)
    anomaly_by_tier = train.groupby('风险层级')['Is_Anomaly'].agg(['sum', 'count', 'mean']).reset_index()
    anomaly_by_tier.columns = ['风险层级', '异常周数', '总周数', '异常占比']
    log(anomaly_by_tier.to_string(index=False), report_lines)

    expected_order = ['稳健龙头', '中型开发商', '高杠杆激进型', '困境典型']
    actual_order = [t for t in expected_order if t in anomaly_by_tier['风险层级'].values]
    ratios = [anomaly_by_tier[anomaly_by_tier['风险层级'] == t]['异常占比'].values[0] for t in actual_order]
    is_monotonic = all(ratios[i] <= ratios[i + 1] for i in range(len(ratios) - 1))
    log("\n【通过】风险层级异常占比单调递增" if is_monotonic else "\n【注意】风险层级异常占比未严格单调", report_lines)

    return train
