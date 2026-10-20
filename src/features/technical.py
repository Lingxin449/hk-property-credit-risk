"""
技术指标衍生特征计算（Step 1）
本函数使用滚动窗口计算，不涉及未来信息
"""

import numpy as np
import pandas as pd
from src.config import log, TRAIN_SEGMENT_END
from src.data.load_data import recalc_volume_zscore


def calc_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calc_macd_hist(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif - dea


def calc_price_position(g):
    """含除零保护的 Price_Position"""
    low_4w = g['最低价'].rolling(4).min()
    high_4w = g['最高价'].rolling(4).max()
    pos = (g['收盘价'] - low_4w) / (high_4w - low_4w)
    pos = pos.replace([np.inf, -np.inf], 0.5)
    pos = pos.fillna(0.5)
    return pos


def derived_features(df, report_lines, fit_end=TRAIN_SEGMENT_END):
    """Step 1: 计算衍生特征

    fit_end：Volume_ZScore 的公司级均值/标准差拟合窗口上界（一期 2022-05-31、二期 2023-12-31）。
    """
    log("\n" + "=" * 60, report_lines)
    log("Step 1: 计算衍生特征", report_lines)

    df = df.sort_values(['股票代码', '交易日期']).reset_index(drop=True)

    df['RSI_14W'] = df.groupby('股票代码')['收盘价'].transform(lambda x: calc_rsi(x, 14))
    df['MACD_HIST'] = df.groupby('股票代码')['收盘价'].transform(lambda x: calc_macd_hist(x))
    df['Volatility_4W'] = df.groupby('股票代码')['周收益率'].transform(lambda x: x.rolling(4).std())
    df['Price_Position'] = df.groupby('股票代码', group_keys=False).apply(calc_price_position)
    # Amihud 非流动性：停牌周 成交量=0 → 成交额=0 → log(1)=0，除零产生 inf；
    # 无成交的周本无有效非流动性可言，置 NaN 交后续 dropna。
    df['Amihud'] = (np.abs(df['周收益率']) / np.log(df['成交额'] + 1)).replace([np.inf, -np.inf], np.nan)

    n_bad = int(df['Amihud'].isnull().sum() - df['周收益率'].isnull().sum())
    log("已计算特征：RSI_14W、MACD_HIST、Volatility_4W、Price_Position、Amihud", report_lines)
    if n_bad > 0:
        log(f"Amihud 除零保护：{n_bad} 个无成交周已置 NaN", report_lines)

    # 重新计算 Volume_ZScore（拟合窗口上界 fit_end，全量 transform）
    df = recalc_volume_zscore(df, fit_end)
    log(f"已重新计算 Volume_ZScore（拟合窗口上界 {fit_end}）", report_lines)

    return df
