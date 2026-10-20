"""
LightGBM 训练（阶段 3 · 第 2 步）
运行：python -m src.models.train_lgbm

- 监督目标：事件标签 y（新标签，定义见 AGENTS §5），不再用伪标签 Is_Anomaly
- 只 fit 方案 C **训练段**（preprocessed/split_train.csv）；留出验证段仅 predict（供第 3~4 步校准/定阈）
- **测试期本轮不评估**，留待阈值冻结后的最终回测（防泄露铁律 AGENTS §4；阶段 3 计划书）
- 不复用 src/models/train_all.py —— 那是旧伪标签四层模型入口

本模块的模型只 fit 训练段（一期训练段 ≤2022-05-31；二期训练段 ≤2023-12-31）；特征已由
build_dataset 按对应训练段统计量变换并切分，此处不拟合任何统计量。
"""

import os

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support, roc_auc_score
)

from src.config import OUTPUT_DIR, ID_COLS, active_suffix, log

SUFFIX = active_suffix()   # '' 一期（方案 C）/ '_v2' 二期（方案 E，由命令行 --v2 激活）

# ===== 输入 / 输出路径 =====
SPLIT_TRAIN = os.path.join(OUTPUT_DIR, f'split_train{SUFFIX}.csv')
SPLIT_VALID = os.path.join(OUTPUT_DIR, f'split_valid{SUFFIX}.csv')
SCORES_OUT = os.path.join(OUTPUT_DIR, f'lgbm_scores{SUFFIX}.csv')
MODEL_OUT = os.path.join(OUTPUT_DIR, f'lgbm_model{SUFFIX}.txt')

# 非特征列（标识列 + 伪标签列 + 监督目标）
NON_FEATURE_COLS = ID_COLS + ['Is_Anomaly', 'y']

# ===== 超参数（按样本量定保守值，不在留出/测试上调参）=====
# 瘦身依据（2026-10-16）：原 300 树 × 15 叶在 143 正例上训练段 AUC=1.0（过拟合），
# 原始分数挤堆 → 校准器被迫退化。按样本量下调容量至 80 树 / 8 叶 / min_child_samples=40。
LGBM_PARAMS = dict(
    n_estimators=80,         # 固定树数，避免用留出段做 early stopping（防泄露）
    learning_rate=0.05,
    num_leaves=8,            # 小叶子数，适配 5k 行 / 143 正例的小样本
    min_child_samples=40,    # 每叶最少样本提高，抑制把个例记进叶子
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight='balanced', # 处理 2.7% 正例的不平衡
    random_state=42,
    n_jobs=1,                # Windows 兼容（AGENTS §6）
    verbose=-1,
)


def load_split(path):
    """读取方案 C 分段产物并把交易日期转为 datetime。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {path}，请先运行 python -m src.data.split")
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    return df


def feature_columns(df):
    """从列名推断 18 维建模特征（排除标识列 / 伪标签 / 标签）。"""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def train_lgbm(train, features):
    """在训练段上 fit LightGBM。本函数的拟合仅用训练段（2019-04-04 ~ 2022-04-01）样本。"""
    clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf.fit(train[features], train['y'].astype(int))
    return clf


def predict_proba(clf, df, features):
    """输出正类（y=1）原始概率。本函数不拟合，仅 transform。"""
    return clf.predict_proba(df[features])[:, 1]


def segment_metrics(df, proba, tag):
    """计算某分段的 AUC 与阈值 0.5 下的 Precision/Recall/F1/混淆矩阵。"""
    y = df['y'].astype(int).values
    auc = roc_auc_score(y, proba) if y.min() != y.max() else float('nan')
    pred = (proba >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        '分段': tag,
        '样本数': len(df),
        '正例数': int(y.sum()),
        '正例率': round(y.mean(), 4),
        'AUC': round(auc, 4),
        'Precision@0.5': round(p, 4),
        'Recall@0.5': round(r, 4),
        'F1@0.5': round(f1, 4),
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
    }


def feature_importance(clf, features):
    """输出 LightGBM gain 重要性排序（前 18 维）。"""
    booster = clf.booster_
    gain = booster.feature_importance(importance_type='gain')
    return (pd.DataFrame({'Feature': features, 'Gain': gain})
            .sort_values('Gain', ascending=False)
            .reset_index(drop=True))


def main():
    report_lines = []
    log("=" * 60, report_lines)
    log("阶段 3 · LightGBM 训练（事件标签 y，只 fit 训练段）", report_lines)

    train = load_split(SPLIT_TRAIN)
    valid = load_split(SPLIT_VALID)
    features = feature_columns(train)
    log(f"\n训练段：{len(train)} 行 / 正例 {int(train['y'].sum())}"
        f"（{train['y'].mean():.2%}）；留出验证段：{len(valid)} 行 / 正例 {int(valid['y'].sum())}"
        f"（{valid['y'].mean():.2%}）", report_lines)
    log(f"特征 {len(features)} 维：{features}", report_lines)

    clf = train_lgbm(train, features)
    proba_tr = predict_proba(clf, train, features)
    proba_va = predict_proba(clf, valid, features)

    metrics = pd.DataFrame([
        segment_metrics(train, proba_tr, '训练段'),
        segment_metrics(valid, proba_va, '留出验证段'),
    ])
    log("\n" + "-" * 60, report_lines)
    log(metrics.to_string(index=False), report_lines)

    log("\n特征重要性（gain，前 8）：", report_lines)
    log(feature_importance(clf, features).head(8).to_string(index=False), report_lines)

    log("\n【说明】测试期本轮不评估——模型/校准/阈值冻结后只跑一次最终回测（AGENTS §4）。", report_lines)

    # 保存模型与分数（留出段原始概率供第 3 步校准）
    ensure = os.path.isdir(OUTPUT_DIR) or os.makedirs(OUTPUT_DIR, exist_ok=True)
    clf.booster_.save_model(MODEL_OUT)

    scores = pd.concat([
        _score_frame(train, proba_tr, '训练段'),
        _score_frame(valid, proba_va, '留出验证段'),
    ], ignore_index=True)
    scores.to_csv(SCORES_OUT, index=False, encoding='utf-8-sig')

    with open(os.path.join(OUTPUT_DIR, f'lgbm_report{SUFFIX}.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    log(f"\n已保存模型：{MODEL_OUT}", report_lines)
    log(f"已保存分数：{SCORES_OUT}", report_lines)
    return clf, metrics


def _score_frame(df, proba, tag):
    """组装某分段的分数表（标识 + 标签 + 原始概率）。"""
    out = df[[c for c in ['股票代码', '公司简称', '交易日期', 'y'] if c in df.columns]].copy()
    out['分段'] = tag
    out['proba_raw'] = proba
    return out


if __name__ == '__main__':
    main()
