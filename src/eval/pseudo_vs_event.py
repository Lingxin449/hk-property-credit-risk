"""
伪标签 vs 事件标签 对照（阶段 3 · 第 7 步）
运行：python -m src.eval.pseudo_vs_event

控制变量对照：**同一特征集（18 维）、同一时间切分、同一模型与超参**，只改监督目标——
    Model P：目标 = 伪标签 Is_Anomaly（周收益率<-15% OR Volume_ZScore>1.645 OR 振幅>20%）
    Model E：目标 = 事件标签 y（阶段 3 冻结模型）
伪标签的 3 个判定条件本身即 3 个输入特征 → 模型只是复述规则（**标签循环**）。

补充口径：伪标签仅在训练期生成，测试期特征矩阵未存 Is_Anomaly 列。因 3 个阈值均落在
Winsorize 1%/99% 边界之内，可用标准化阈值从**已标准化特征**等价复算（分类不受截断影响），
先在训练段与存档列比对校验，再应用于留出段/测试期。
"""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import (
    OUTPUT_DIR, PROJECT_ROOT, ID_COLS, ANOMALY_THRESH_RET, ANOMALY_THRESH_VOLZ,
    ANOMALY_THRESH_AMP, log
)
from src.models.train_lgbm import LGBM_PARAMS

MODEL_OUT = os.path.join(OUTPUT_DIR, 'lgbm_model.txt')
SCALER_IN = os.path.join(OUTPUT_DIR, 'scaler_params.csv')
SPLITS = {
    '训练段': os.path.join(OUTPUT_DIR, 'split_train.csv'),
    '留出验证段': os.path.join(OUTPUT_DIR, 'split_valid.csv'),
    '测试期': os.path.join(OUTPUT_DIR, 'split_test.csv'),
}
OUT_CSV = os.path.join(OUTPUT_DIR, 'pseudo_vs_event.csv')
FIG_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')

NON_FEATURE_COLS = ID_COLS + ['Is_Anomaly', 'y']
# 伪标签的 3 个判定条件特征（与输入特征重合 → 循环）
DEFINING = ['周收益率', 'Volume_ZScore', '振幅']
THRESH = {'周收益率': ('<', ANOMALY_THRESH_RET), 'Volume_ZScore': ('>', ANOMALY_THRESH_VOLZ),
          '振幅': ('>', ANOMALY_THRESH_AMP)}


def _read(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    return df


def recompute_pseudo_label(df, scaler):
    """用标准化阈值从已标准化特征等价复算伪标签。"""
    thr = {}
    for f, (op, raw_thr) in THRESH.items():
        m = scaler.loc[scaler['Feature'] == f, 'Mean'].iloc[0]
        s = scaler.loc[scaler['Feature'] == f, 'Scale'].iloc[0]
        thr[f] = (raw_thr - m) / s
    return ((df['周收益率'] < thr['周收益率']) |
            (df['Volume_ZScore'] > thr['Volume_ZScore']) |
            (df['振幅'] > thr['振幅'])).astype(int)


def _auc(y, p):
    y = np.asarray(y)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan')


def plot_auc_bar(table, report_lines):
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
    segs = list(table['分段'])
    x = np.arange(len(segs))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(x - 0.2, table['Model P（伪标签）AUC'], width=0.4, label='Model P：伪标签 Is_Anomaly', color='#d62728')
    ax.bar(x + 0.2, table['Model E（事件标签）AUC'], width=0.4, label='Model E：事件标签 y', color='#1f77b4')
    ax.axhline(0.90, ls='--', lw=1, color='gray')
    ax.text(len(segs) - 0.5, 0.905, '查泄露线 0.90', ha='right', color='gray', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(segs)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('AUC')
    ax.set_title('伪标签 vs 事件标签：同一特征/切分/模型，只换监督目标')
    ax.legend(loc='lower left')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'pseudo_vs_event_label_auc.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"已保存 AUC 对比图：{out}", report_lines)


def main():
    report_lines = []
    log("=" * 60, report_lines)
    log("阶段 3 · 伪标签 vs 事件标签 对照（控制变量）", report_lines)

    scaler = pd.read_csv(SCALER_IN, encoding='utf-8-sig')
    train = _read(SPLITS['训练段'])
    features = [c for c in train.columns if c not in NON_FEATURE_COLS]

    # 校验：标准化阈值复算的伪标签 vs 存档列
    recomputed = recompute_pseudo_label(train, scaler)
    agree = float((recomputed == train['Is_Anomaly']).mean())
    log(f"\n伪标签复算校验（训练段）：与存档 Is_Anomaly 一致率 {agree:.4%}"
        f"；判定条件特征 {DEFINING} 均在输入特征中（循环根源）", report_lines)

    # Model P：只换目标为 Is_Anomaly
    model_p = lgb.LGBMClassifier(**LGBM_PARAMS)
    model_p.fit(train[features], recomputed)

    # Model E：冻结模型
    model_e = lgb.Booster(model_file=MODEL_OUT)

    rows = []
    for seg, path in SPLITS.items():
        df = _read(path)
        p_p = model_p.predict_proba(df[features])[:, 1]
        p_e = model_e.predict(df[features])
        y_p = recompute_pseudo_label(df, scaler).to_numpy()   # 伪标签（特征派生，非测试期事件标签）
        y_e = df['y'].astype(int).to_numpy()                 # 事件标签
        rows.append({
            '分段': seg, '样本数': len(df),
            'Model P（伪标签）AUC': round(_auc(y_p, p_p), 4),
            'Model E（事件标签）AUC': round(_auc(y_e, p_e), 4),
            '伪标签正例率': round(float(y_p.mean()), 4),
            '事件正例率': round(float(y_e.mean()), 4),
        })

    table = pd.DataFrame(rows)
    log("\n对照结果：", report_lines)
    log(table.to_string(index=False), report_lines)

    log("\n【解读】Model P 三段 AUC 均≈1.0，说明它只是**复述标签规则**（标签循环）；"
        "Model E 的 AUC 才是模型的真实泛化能力。二者差异即'假分'与'真分'之别。", report_lines)
    log("【说明】本对照为诊断用途，不参与任何模型/阈值选择；"
        "Model P 评估所用的伪标签由特征派生，不涉及测试期事件标签。", report_lines)

    table.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    plot_auc_bar(table, report_lines)
    with open(os.path.join(OUTPUT_DIR, 'pseudo_vs_event_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    log(f"\n已保存对照表：{OUT_CSV}", report_lines)

    return table


if __name__ == '__main__':
    main()
