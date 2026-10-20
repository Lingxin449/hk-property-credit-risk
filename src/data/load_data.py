"""
数据加载与基础探查（Step 0）
本函数不涉及统计量拟合，不产生泄露风险

阶段 2 扩池：原 24 家 + 新增候选池（剔除不满足 §2.8 第三层事件可得性的银城国际、明发集团后为 22 家），合并为 46 家。
新增池的三处口径对齐在本模块统一处理：
  1. 补算原表自带、新增表缺失的「周收益率」「振幅」，公式与原 24 家完全一致；
  2. 成交额统一按「收盘价 × 成交量」近似重算（新增表无成交额字段，全池必须同口径）；
  3. 按「最后一个 成交量>0 的周」截断，其后记录全部丢弃（停牌/摘牌后 iFinD 将收盘价置 0 或冻结）。
"""

import os

import numpy as np
import pandas as pd
from src.config import (
    INPUT_FILE, NEW_INPUT_FILE, COMPANY_INFO_FILE, NEW_COMPANY_INFO_FILE,
    TRAIN_CUTOFF, PROJECT_ROOT, log
)

# §2.8 第三层（事件可得性）判定的数据源：新候选入池须在 events.csv 中有 ≥1 个留用事件
EVENTS_FILE = os.path.join(PROJECT_ROOT, 'data/events/events.csv')

# 合并后的行情字段（资产负债率/流动比率/经营现金流_净负债 由 slow_variables 按公告日期重建）
MARKET_COLS = ['股票代码', '公司简称', '交易日期',
               '开盘价', '收盘价', '最高价', '最低价', '成交量', '成交额', '振幅', '周收益率']


def _normalize_code(series):
    """统一股票代码格式为「4 位数字 + .HK」：原始导出个别公司为 5 位（如 00884.HK），
    与其余 4 位写法不一致会导致与事件清单（events.csv）按代码关联时漏配。"""
    return (series.astype(str)
            .str.replace(r'\.HK$', '', regex=True)
            .str.strip()
            .str.lstrip('0').str.zfill(4) + '.HK')


def _to_datetime(series):
    """兼容 Excel 日期序列号与文本日期两种写法。"""
    if series.dtype in ['int64', 'float64']:
        return pd.to_datetime(series, unit='D', origin='1899-12-30')
    return pd.to_datetime(series)


def recalc_volume_zscore(df, train_cutoff):
    """
    重新计算 Volume_ZScore。
    对每家公司，使用拟合窗口（<= train_cutoff，方案 C 下为训练段 2019-01-01 ~ 2022-05-31）
    的成交量均值和标准差，对全量数据做 z-score 变换，防止未来信息泄露。
    本函数的统计量拟合自 <= train_cutoff 的拟合窗口；留出验证段与测试期仅 transform。
    """
    df = df.copy()
    train_mask = df['交易日期'] <= train_cutoff

    train_stats = df[train_mask].groupby('股票代码')['成交量'].agg(['mean', 'std']).reset_index()
    train_stats.columns = ['股票代码', 'vol_mean_train', 'vol_std_train']

    df = df.merge(train_stats, on='股票代码', how='left')
    df['vol_std_train'] = df['vol_std_train'].replace(0, np.nan)
    df['Volume_ZScore'] = (df['成交量'] - df['vol_mean_train']) / df['vol_std_train']
    df = df.drop(columns=['vol_mean_train', 'vol_std_train'])

    return df


def _load_market_sources(report_lines):
    """读入原 24 家与新增池行情，统一字段口径后纵向合并。不做任何统计量拟合。"""
    old = pd.read_excel(INPUT_FILE)
    old['股票代码'] = _normalize_code(old['股票代码'])
    old['交易日期'] = _to_datetime(old['交易日期'])

    # 入池名单：新增池公司信息表 ∩ 在 events.csv 中有留用事件的公司（§2.8 第三层事件可得性）
    # 数据驱动、不硬编码公司；天誉置业 0059.HK 无留用事件据此出池（46 → 45）
    new_info_codes = set(_normalize_code(pd.read_excel(NEW_COMPANY_INFO_FILE)['股票代码']))
    events_codes = set(pd.read_csv(EVENTS_FILE, encoding='utf-8-sig')['股票代码'].astype(str))
    keep_codes = new_info_codes & events_codes
    no_event = sorted(new_info_codes - keep_codes)
    log(f"新增池事件可得性过滤：信息表 {len(new_info_codes)} 家 ∩ events.csv → 入池 {len(keep_codes)} 家，"
        f"无硬信号事件出池 {len(no_event)} 家 {no_event or ''}", report_lines)
    new = pd.read_excel(NEW_INPUT_FILE, sheet_name='周频行情')
    new['股票代码'] = _normalize_code(new['股票代码'])
    dropped = sorted(set(new['股票代码']) - keep_codes)
    new = new[new['股票代码'].isin(keep_codes)].copy()
    new['交易日期'] = _to_datetime(new['交易日期'])
    log(f"新增池：行情导出 {new['股票代码'].nunique() + len(dropped)} 家，"
        f"入池 {new['股票代码'].nunique()} 家，剔除 {len(dropped)} 家 {dropped or ''}", report_lines)

    # 新增表缺「周收益率」「振幅」，按原 24 家同口径补算（百分数，与原始列一致，后续统一 /100）
    new = new.sort_values(['股票代码', '交易日期'])
    new['周收益率'] = new.groupby('股票代码')['收盘价'].pct_change() * 100
    new['振幅'] = (new['最高价'] - new['最低价']) / new['最低价'] * 100

    return pd.concat([old[MARKET_COLS], new[MARKET_COLS]], ignore_index=True)


def load_data(report_lines):
    """Step 0: 读取原始数据并做基础探查"""
    log("=" * 60, report_lines)
    log("Step 0: 读取原始数据", report_lines)

    df = _load_market_sources(report_lines)

    # 去重：同一公司同一交易日的重复记录（iFinD 导出时个别公司会整表重复）
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['股票代码', '交易日期'], keep='first').reset_index(drop=True)
    log(f"去重：删除重复（股票代码+交易日期）记录 {before_dedup - len(df)} 条，剩余 {len(df)} 条", report_lines)

    # 成交额统一口径：新增表无成交额字段，全池一律按「收盘价 × 成交量」近似重算
    df['成交额'] = df['收盘价'] * df['成交量']

    # 截断：只保留「最后一个 成交量>0 的周」及其之前的记录
    last_tradable = df[df['成交量'] > 0].groupby('股票代码')['交易日期'].max()
    no_trade = sorted(set(df['股票代码']) - set(last_tradable.index))
    if no_trade:
        log(f"【警告】{len(no_trade)} 家公司无成交量>0 的周，全部剔除：{no_trade}", report_lines)
    before_cut = len(df)
    df = df[df['交易日期'] <= df['股票代码'].map(last_tradable)].reset_index(drop=True)
    log(f"截断：删除停牌/摘牌后的记录 {before_cut - len(df)} 条，剩余 {len(df)} 条", report_lines)

    df = df.sort_values(['股票代码', '交易日期']).reset_index(drop=True)

    # 风险层级：统一取公司信息表（原 24 家表 + 新增池表），行情表自带的旧标注弃用
    info = pd.concat([pd.read_excel(COMPANY_INFO_FILE), pd.read_excel(NEW_COMPANY_INFO_FILE)],
                     ignore_index=True)
    info['股票代码'] = _normalize_code(info['股票代码'])
    df = df.drop(columns=[c for c in ['风险层级'] if c in df.columns])
    df = df.merge(info[['股票代码', '风险层级']], on='股票代码', how='left')
    missing = df.loc[df['风险层级'].isnull(), '股票代码'].unique().tolist()
    if missing:
        raise ValueError(f"以下公司缺少风险层级标注：{missing}")

    # 百分数转小数
    df['周收益率'] = df['周收益率'] / 100.0
    df['振幅'] = df['振幅'] / 100.0
    # Volume_ZScore 由 technical.derived_features 按训练期统一重算，此处不留原始列

    log(f"原始行情数据量：{len(df)} 条", report_lines)
    log(f"字段数：{df.shape[1]} 列", report_lines)
    log(f"公司数：{df['股票代码'].nunique()} 家", report_lines)
    log(f"日期范围：{df['交易日期'].min().date()} ~ {df['交易日期'].max().date()}", report_lines)
    log(f"风险层级分布：\n{df['风险层级'].value_counts().to_string()}", report_lines)

    return df, info


def load_data_v2(report_lines):
    """Step 0（二期）：读取 2019-2025 全量市场面板（一期 2019-2024 截断结果 + 2025 行情衔接）。

    一期侧一列不改：`load_data` 的产出原样保留，仅纵向追加 2025 行——`load_2025_market` 已做
    复权对齐、跨年边界去重、停牌截断与派生列（周收益率 / 振幅 / 成交额）连续计算。风险层级沿用
    公司信息表。本函数不做任何统计量拟合。
    """
    from src.data.load_2025 import load_2025_market   # 局部导入，避免与本模块的循环依赖

    df, info = load_data(report_lines)

    log("\n" + "=" * 60, report_lines)
    log("Step 0（二期）: 追加 2025 行情（复权对齐 + 边界去重 + 截断）", report_lines)
    q25 = load_2025_market(report_lines, base=df)
    q25 = q25.merge(info[['股票代码', '风险层级']], on='股票代码', how='left')
    missing = q25.loc[q25['风险层级'].isnull(), '股票代码'].unique().tolist()
    if missing:
        raise ValueError(f"以下 2025 公司缺少风险层级标注：{missing}")

    df = pd.concat([df, q25], ignore_index=True)
    df = df.sort_values(['股票代码', '交易日期']).reset_index(drop=True)
    log(f"2019-2025 合并面板：{len(df)} 条 / {df['股票代码'].nunique()} 家 / "
        f"{df['交易日期'].min().date()} ~ {df['交易日期'].max().date()}", report_lines)
    return df, info