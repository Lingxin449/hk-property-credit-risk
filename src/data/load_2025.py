# -*- coding: utf-8 -*-
"""二期：2025 行情接入（衔接对齐 + 池过滤 + 边界去重）。

复权口径背景（详见 `docs/方案讨论稿.md §7`）：
  一期两份行情导出的复权口径并不一致——`data/raw/港股地产24家公司_周频行情数据_2019_2024.xlsx`
  是**未复权**（价格即真实股价），`data/raw/港股地产新增24家公司_周频行情数据_2019_2024.xlsx` 为**后复权**；
  而 2025 导出 `data/raw/港股地产全池_周频行情_2025.xlsx` 为**后复权**。
  两侧在年界处是**同一根周线**（成交量逐位相同；iFinD 按"区间末交易日"给周线打标签，故 ≤2024-12-31 的导出
  把它标成 2024-12-31、2025 导出把它标成 2025-01-03）。

本模块的处置（方案 A：本地换算，**不改动 `data/raw/` 下任何文件**）：
  1. 只保留一期已在池的公司（自动剔除 2025 导出多带的 0846.HK 明发集团、1902.HK 银城国际）；
  2. 删除 2025 的首行（与一期 `2024-12-31` 为同一根周线，避免同一周被记两次）；
  3. 按 **一期末周收盘 ÷ 2025 首周收盘** 把 2025 的价格列归一到"与一期逐公司一致"的基准
     （新增池公司该比值恒为 1，属空操作；原 24 家公司修正约一个数量级的尺度差）。
     收益率不受该换算影响（尺度无关），仅成交额 / Amihud 的绝对水平归位。

本模块不拟合任何统计量；2025 行的派生列（周收益率 / 振幅 / 成交额）以"一期最后一根周线"为起点连续计算，
以保证边界周的收益率正确。2019-2024 侧一列都不改。
"""

import numpy as np
import pandas as pd

from src.config import QUOTE_FILE_2025, FIN_FILE_2025, HSPI_FILE_2025, LPR_FILE_2025
from src.data.load_data import _load_market_sources, _normalize_code, _to_datetime
from src.features.slow_variables import _read_sheet, FIN_LEVEL_FEATURES

PRICE_COLS = ['开盘价', '最高价', '最低价', '收盘价']
BOUNDARY_DATE = pd.Timestamp('2024-12-31')   # 一期最后一根周线的日期标签（实为跨年周）


def _read_quote_2025():
    """读取 2025 行情导出（不做任何统计量拟合）。"""
    q = pd.read_excel(QUOTE_FILE_2025)
    q['股票代码'] = _normalize_code(q['股票代码'])
    q['交易日期'] = _to_datetime(q['交易日期'])
    return q.sort_values(['股票代码', '交易日期']).reset_index(drop=True)


def load_2025_market(report_lines=None, base=None):
    """
    返回 2025 年行情（已衔接对齐），列 = 股票代码 / 公司简称 / 交易日期 / 开盘价 / 最高价 / 最低价 /
    收盘价 / 成交量 / 成交额 / 周收益率 / 振幅。

    - 周收益率、振幅为小数（与一期 `load_data` 输出一致，1.0 = 100%）；
    - 只含 2025-01-01 起的行；边界周（跨年那一周）保留在一期侧（日期标签 2024-12-31），
      因此 2025 首行的周收益率 = 相对该边界周收盘的涨跌幅，不是 NaN。
    - base：一期市场面板（可传 `load_data` 的截断结果以复用、免二次读盘）；缺省时本函数自行读取。
    本函数的换算因子 `scale` 只用到一期与 2025 的**行情价格**，不涉及标签、不涉及任何 fit。
    """
    rl = report_lines if report_lines is not None else []
    base = _load_market_sources(rl) if base is None else base          # 一期 2019-2024 市场面板
    q = _read_quote_2025()

    # 1) 池过滤：只保留一期已在池的公司
    codes = set(base['股票代码'])
    extra = sorted(set(q['股票代码']) - codes)
    q = q[q['股票代码'].isin(codes)].copy()
    rl.append(f"2025 行情：导出 {q['股票代码'].nunique() + len(extra)} 家，入池 "
              f"{q['股票代码'].nunique()} 家，剔除池外 {len(extra)} 家 {extra or ''}")

    # 2) 换算因子 = 一期末周收盘 ÷ 2025 首周收盘（两侧同一根周线）
    base_last = (base.sort_values('交易日期').groupby('股票代码').tail(1)
                 .set_index('股票代码')['收盘价'])
    head_2025 = q.groupby('股票代码').head(1).set_index('股票代码')['收盘价']
    scale = base_last / head_2025

    # 3) 删除 2025 首行（与一期重复的边界周），再施加换算
    first_date = q.groupby('股票代码')['交易日期'].transform('min')
    q = q[q['交易日期'] > first_date].copy()
    fac = q['股票代码'].map(scale)
    for c in PRICE_COLS:
        q[c] = q[c] * fac
    q['成交额'] = q['收盘价'] * q['成交量']

    # 4) 逐公司以"一期最后一根周线"为种子，连续计算周收益率（不改动一期侧任何列）
    frames = []
    for code, g in q.groupby('股票代码', sort=True):
        seed = pd.DataFrame({
            '股票代码': [code], '公司简称': [g['公司简称'].iloc[0]], '交易日期': [BOUNDARY_DATE],
            '开盘价': [np.nan], '最高价': [np.nan], '最低价': [np.nan],
            '收盘价': [base_last[code]], '成交量': [np.nan], '成交额': [np.nan],
        })
        gg = pd.concat([seed, g[['股票代码', '公司简称', '交易日期'] + PRICE_COLS + ['成交量', '成交额']]],
                       ignore_index=True)
        gg['周收益率'] = gg['收盘价'].pct_change()
        gg['振幅'] = (gg['最高价'] - gg['最低价']) / gg['最低价']
        frames.append(gg[gg['交易日期'] > BOUNDARY_DATE])
    out = pd.concat(frames, ignore_index=True)

    # 5) 截断到"最后一个成交量>0 的周"（与一期同口径）
    last_tradable = out[out['成交量'] > 0].groupby('股票代码')['交易日期'].max()
    no_trade = sorted(set(out['股票代码']) - set(last_tradable.index))
    if no_trade:
        rl.append(f'【说明】{len(no_trade)} 家公司在 2025 年无任何成交量>0 的周（长期停牌/已摘牌），'
                  f'按「最后一个成交量>0 的周」规则全部剔除：{no_trade}')
    before = len(out)
    out = out[out['交易日期'] <= out['股票代码'].map(last_tradable)].reset_index(drop=True)
    rl.append(f"2025 行情：截断停牌/摘牌后记录 {before - len(out)} 条，剩余 {len(out)} 条；"
              f"公司 {out['股票代码'].nunique()} 家，日期 "
              f"{out['交易日期'].min().date()} ~ {out['交易日期'].max().date()}")
    return out


# ===== 二期慢变量入接：2025 HSPI / LPR / 半年度财务 =====
def load_2025_hspi(report_lines=None):
    """读 2025 恒生地产指数周线原始行（日期 / 收盘价），用于与一期序列拼接后统一计算滚动特征。

    只返回原始价格行、不计算滚动统计量：4 周波动率 / 动量必须在一期（2019-2024）+ 2025 拼接后的
    连续序列上统一计算，否则 2025 年初的滚动窗口会因缺前 4 周而失真。本函数不做任何 fit。
    """
    idx = _read_sheet(HSPI_FILE_2025, '行情数据', ['日期', '开盘价'])
    idx['日期'] = pd.to_datetime(idx['日期'], errors='coerce')
    idx['收盘价'] = pd.to_numeric(idx['收盘价'], errors='coerce')
    idx = idx.dropna(subset=['日期', '收盘价']).sort_values('日期').reset_index(drop=True)
    if report_lines is not None:
        report_lines.append(f"2025 HSPI：{len(idx)} 周原始行 "
                            f"（{idx['日期'].min().date()} ~ {idx['日期'].max().date()}），待与一期拼接")
    return idx[['日期', '收盘价']]


def load_2025_lpr(report_lines=None):
    """读 2025 LPR 月度报价（生效日期 / LPR_1Y / LPR_5Y）。

    2025 文件已为规整 schema，无需基准利率衔接（衔接行仅一期 2019-08 前需要）。
    """
    lpr = _read_sheet(LPR_FILE_2025, 'LPR月度', ['生效日期'])
    lpr = lpr[['生效日期', 'LPR_1Y', 'LPR_5Y']].copy()
    lpr['生效日期'] = pd.to_datetime(lpr['生效日期'], errors='coerce')
    lpr['LPR_1Y'] = pd.to_numeric(lpr['LPR_1Y'], errors='coerce')
    lpr['LPR_5Y'] = pd.to_numeric(lpr['LPR_5Y'], errors='coerce')
    lpr = lpr.dropna(subset=['生效日期']).sort_values('生效日期').reset_index(drop=True)
    if report_lines is not None:
        report_lines.append(f"2025 LPR：{len(lpr)} 期月度报价 "
                            f"（{lpr['生效日期'].min().date()} ~ {lpr['生效日期'].max().date()}）")
    return lpr


def load_2025_financial(report_lines=None):
    """读 2025 全池半年度财务（2024H2-2025H2，含公告日期列），按与原 24 家同口径计算三个比率。

    金额列为原始报表口径（_oas），仅比值入模；公告日期为港交所披露易核实的**最早正式业绩公告日**，
    按公告日对齐以避免前视偏差。本函数不做任何统计量拟合，仅做键连接与逐期比率计算。
    """
    fin = _read_sheet(FIN_FILE_2025, '财务数据', ['股票代码', '报告期'])
    fin['股票代码'] = _normalize_code(fin['股票代码'])
    fin['报告期'] = pd.to_datetime(fin['报告期'], errors='coerce')
    fin['公告日期'] = pd.to_datetime(fin['公告日期'], errors='coerce')

    amount_cols = ['总资产', '总负债', '流动资产', '流动负债', '货币资金', '短期借款',
                   '长期借款', '经营活动产生的现金流量净额']
    for c in amount_cols:
        fin[c] = pd.to_numeric(fin[c], errors='coerce')

    fin['资产负债率'] = fin['总负债'] / fin['总资产'] * 100
    fin['流动比率'] = fin['流动资产'] / fin['流动负债']
    net_debt = fin['短期借款'] + fin['长期借款'] - fin['货币资金']
    fin['经营现金流_净负债'] = fin['经营活动产生的现金流量净额'] / net_debt

    n_all = len(fin)
    fin = fin.dropna(subset=['公告日期'])
    if report_lines is not None:
        report_lines.append(f"2025 财务面板：{len(fin)}/{n_all} 条有公告日期，"
                            f"覆盖 {fin['股票代码'].nunique()} 家 × {fin['报告期'].nunique()} 报告期")
    return fin[['股票代码', '报告期', '公告日期'] + FIN_LEVEL_FEATURES]


if __name__ == '__main__':
    lines = []
    d = load_2025_market(lines)
    print('=' * 70)
    for ln in lines:
        print(ln)
    print('=' * 70)
    print('行数', len(d), '| 公司数', d['股票代码'].nunique(),
          '| 日期', d['交易日期'].min().date(), '~', d['交易日期'].max().date())
    print('\n【衔接自检】2025 首周周收益率（应无数量级异常）')
    first = d.sort_values('交易日期').groupby('股票代码').head(1)
    bad = first[first['周收益率'].abs() > 0.5]
    print('  首周收益率绝对值 >50% 的公司数：', len(bad), '/', len(first),
          '（期望为 0：换算已消除复权尺度差）')
    if len(bad):
        print(bad[['股票代码', '公司简称', '交易日期', '收盘价', '周收益率']].to_string(index=False))
    print('\n【样例】恒基地产 2025 前 3 行（与一期末周 23.25 同基准）')
    print(d[d['股票代码'] == '0012.HK'].head(3)[
        ['交易日期', '开盘价', '最高价', '最低价', '收盘价', '成交量', '成交额', '周收益率', '振幅']].to_string(index=False))
