"""
统一配置：路径、阈值、候选特征、公共工具函数
本文件的统计量拟合自整个项目周期，不涉及时间切分
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ===== 路径 =====
INPUT_FILE = str(PROJECT_ROOT / 'data/raw/港股地产24家公司_周频行情数据_2019_2024.xlsx')
COMPANY_INFO_FILE = str(PROJECT_ROOT / 'data/raw/公司信息表_24家.xlsx')
# 新增候选池（阶段 2 扩池）：行情取自 iFinD；公司信息表已剔除不满足 §2.8 第三层事件可得性的主体
# 注：文件名中的 “24家 / 新增22家” 是原始批次命名，data/raw 只读，不改名
# 公司池规模与剔除口径（银城国际 1902.HK / 明发集团 0846.HK / 天誉置业 0059.HK）见 内部数值表NEW_INPUT_FILE = str(PROJECT_ROOT / 'data/raw/港股地产新增24家公司_周频行情数据_2019_2024.xlsx')
NEW_COMPANY_INFO_FILE = str(PROJECT_ROOT / 'data/raw/公司信息表_新增22家.xlsx')
# 新增池半年度财务的公告日期（人工到港交所披露易核实；与行情工作簿「财务指标原始」按 代码+报告期 连接）
NEW_FIN_DATE_FILE = str(PROJECT_ROOT / 'data/raw/港股地产新增24家_业绩公告日期_2019_2024.xlsx')
# 慢变量（阶段 1）：指数来源 Yahoo ^HSNP、LPR 来源中国银行、财务来源 iFinD（美元口径）
HSPI_FILE = str(PROJECT_ROOT / 'data/raw/恒生地产指数_周频_2019_2024.xlsx')
LPR_FILE = str(PROJECT_ROOT / 'data/raw/LPR_月度_2019_2024.xlsx')
FIN_FILE = str(PROJECT_ROOT / 'data/raw/港股地产24家_财务指标_2018H2_2024H2.xlsx')
OUTPUT_DIR = str(PROJECT_ROOT / 'preprocessed/')

# ===== 二期：2025 数据（归档于 data/raw/；复权口径与衔接方案见 内部设计说明）=====
QUOTE_FILE_2025 = str(PROJECT_ROOT / 'data/raw/港股地产全池_周频行情_2025.xlsx')
FIN_FILE_2025 = str(PROJECT_ROOT / 'data/raw/港股地产全池_财务指标_2024H2_2025H2.xlsx')
HSPI_FILE_2025 = str(PROJECT_ROOT / 'data/raw/恒生地产指数_周频_2025.xlsx')
LPR_FILE_2025 = str(PROJECT_ROOT / 'data/raw/LPR_月度_2025.xlsx')
EVENTS_CAND_2025 = str(PROJECT_ROOT / 'data/raw/2025信用事件候选_初筛.csv')

# ===== 时间切分 =====
TRAIN_CUTOFF = '2022-12-31'          # 特征矩阵 / 标签的期界（训练期 2019-2022 / 测试期 2023-2024）
TEST_SEGMENT_START = '2023-01-01'    # 一期测试期起始（供按日期二分的流程引用；二期见 TEST_SEGMENT_START_V2）
TEST_SEGMENT_END = '2024-12-31'      # 一期测试期结束

# 方案 C（口径纪律）：训练段 / 留出验证段 / 切点净化带
TRAIN_SEGMENT_END = '2022-05-31'      # 训练段结束日；所有变换统计量（Volume_ZScore / VIF / Winsorize / Z-Score）的拟合窗口上界
VALID_SEGMENT_START = '2022-06-01'    # 留出验证段起始日（仅用于校准 / 代价阈值 / 特征选型）
VALID_SEGMENT_END = '2022-12-31'      # 留出验证段结束日
PURGE_DAYS = 56                       # 切点净化带宽度 = 标签窗口 8 周（自然日）

# ===== 二期（方案 E：重训 + 新考卷）切分常量：一期常量保留不动（见 内部设计说明）=====
# 切点 2023-12-31 于 2026-10-19 按正例账核定（推导见 内部设计说明、数值见 内部数值表）
SUFFIX_V2 = '_v2'                          # 二期产物统一后缀，避免覆盖一期冻结产物（§8.3）
EVENTS_2025 = str(PROJECT_ROOT / 'data/events/events_2025.csv')   # 二期事件清单（与 events.csv 同 5 列 schema）
TRAIN_SEGMENT_END_V2 = '2023-12-31'       # 二期训练段结束（切点）；亦是二期变换统计量（Volume_ZScore / VIF / Winsorize / Z-Score）的拟合窗口上界
VALID_SEGMENT_START_V2 = '2024-01-01'     # 二期留出验证段起始（仅用于校准 / 代价阈值 / 特征选型）
VALID_SEGMENT_END_V2 = '2024-12-31'       # 二期留出验证段结束
TEST_SEGMENT_START_V2 = '2025-01-01'      # 新考卷（一次性评估，全程零接触）
TEST_SEGMENT_END_V2 = '2025-12-31'        # 新考卷结束

# ===== 伪标签阈值（小数形式）=====
# 仅用于事后验证 / 对比实验（供 src/labels/pseudo_label.py 使用），不作为监督模型训练目标（口径纪律）
ANOMALY_THRESH_RET = -0.15    # 周收益率 < -15%
ANOMALY_THRESH_VOLZ = 1.645   # Volume_ZScore > 1.645
ANOMALY_THRESH_AMP = 0.20     # 振幅 > 20%

# ===== VIF =====
VIF_THRESHOLD = 10

# ===== Winsorize =====
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

# ===== 候选特征 =====
CANDIDATE_FEATURES = [
    '周收益率', '成交量', '振幅', 'RSI_14W', 'MACD_HIST',
    'Volatility_4W', 'Volume_ZScore', 'Price_Position',
    '成交额', 'Amihud', '资产负债率', '流动比率', '经营现金流_净负债',
    # 阶段 1 慢变量（HSPI 指数 / LPR / 半年度财务趋势）
    'HSPI_周收益率', 'HSPI_波动率_4W', 'HSPI_动量_4W',
    'LPR_1Y', 'LPR_5Y_1Y_利差', 'LPR_1Y_12W变动',
    '资产负债率_环比变动', '流动比率_环比变动'
]

# ===== 标识列 =====
ID_COLS = ['股票代码', '公司简称', '风险层级', '交易日期']


# ===== 公共工具 =====
def log(msg, report_lines):
    print(msg)
    report_lines.append(msg)


def active_suffix(argv=None):
    """按命令行是否含 `--v2` 返回产物后缀（`''` = 一期 / `SUFFIX_V2` = 二期）。

    各下游入口模块（train_lgbm / calibrate / cost_threshold / evaluate_test / shap_analysis /
    shap_vs_iv / scorecard / backtest）在 import 期调用本函数，把产物路径与切段参数切到二期，
    从而在不覆盖一期冻结产物的前提下复用同一份代码（口径纪律 二期协议、内部设计说明）。
    """
    import sys
    argv = sys.argv if argv is None else argv
    return SUFFIX_V2 if '--v2' in argv else ''


def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def get_id_and_feature_cols(df):
    """从 DataFrame 列名推断标识列和特征列"""
    id_cols = [c for c in ID_COLS if c in df.columns]
    if 'Is_Anomaly' in df.columns:
        id_cols = id_cols + ['Is_Anomaly']
    feature_cols = [c for c in df.columns if c not in id_cols]
    return id_cols, feature_cols
