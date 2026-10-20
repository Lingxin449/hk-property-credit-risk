"""
阶段 1 慢变量接入：恒生地产分类指数（HSPI）、LPR 利率、半年度财务面板。

对齐纪律（防泄露铁律）：
- HSPI 行情口径是「周起始日（周一）」，与公司行情的「周结束日」相反，故按 ISO 年 + 周号对齐，
  不按日期直接 merge。滚动统计量（4 周波动率 / 4 周动量）只依赖当期及之前窗口。
- LPR 为月度报价，按「生效日期 <= 交易日期」取最近一次报价（前向填充）；
  2019-08-20 之前无 LPR 报价，用 2015-10-24 生效的贷款基准利率 4.35 / 4.90 衔接。
- 财务面板按**公告日期**对齐到周（公告日 <= 交易日期），并按字段做 LOCF（缺报期续用上一期值），
  不使用任何未来信息。财务金额为美元口径，模型只使用比值特征。
本模块的对齐与滚动统计量均只用当期及之前信息，不拟合任何跨期统计量。
"""

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, log

RAW_DIR = PROJECT_ROOT / 'data/raw'
HSPI_FILE = RAW_DIR / '恒生地产指数_周频_2019_2024.xlsx'
LPR_FILE = RAW_DIR / 'LPR_月度_2019_2024.xlsx'
FIN_FILE = RAW_DIR / '港股地产24家_财务指标_2018H2_2024H2.xlsx'
# 新增池：比率取自行情工作簿「财务指标原始」sheet，公告日期取独立核实文件
NEW_MARKET_FILE = RAW_DIR / '港股地产新增24家公司_周频行情数据_2019_2024.xlsx'
NEW_FIN_DATE_FILE = RAW_DIR / '港股地产新增24家_业绩公告日期_2019_2024.xlsx'

# 新增候选特征（与 src/config.py 的 CANDIDATE_FEATURES 保持一致）
HSPI_FEATURES = ['HSPI_周收益率', 'HSPI_波动率_4W', 'HSPI_动量_4W']
LPR_FEATURES = ['LPR_1Y', 'LPR_5Y_1Y_利差', 'LPR_1Y_12W变动']
FIN_LEVEL_FEATURES = ['资产负债率', '流动比率', '经营现金流_净负债']
FIN_DIFF_FEATURES = ['资产负债率_环比变动', '流动比率_环比变动']
FIN_FEATURES = FIN_LEVEL_FEATURES + FIN_DIFF_FEATURES
SLOW_FEATURES = HSPI_FEATURES + LPR_FEATURES + FIN_FEATURES


def _read_sheet(path, sheet, keys, min_nonempty=3):
    """智能定位表头：找含关键字且非空单元格 >= min_nonempty 的行作为表头。
    三份新文件的表头都不在第 0 行（前面是合并标题块）。"""
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    header_row = None
    for i in range(min(20, len(raw))):
        vals = [str(v) for v in raw.iloc[i].tolist()]
        joined = ''.join(vals)
        if any(k in joined for k in keys) and sum(1 for v in vals if v and v != 'nan') >= min_nonempty:
            header_row = i
            break
    if header_row is None:
        raise ValueError(f'{path.name} / {sheet} 未定位到表头行')

    data = raw.iloc[header_row + 1:].reset_index(drop=True)
    data.columns = [str(c) for c in raw.iloc[header_row].tolist()]
    keep = [c for c in data.columns if not c.startswith('Unnamed') and c != 'nan']
    return data[keep].copy()


def _iso_week_key(dt_series):
    """把日期转成 ISO「年*100+周号」整数键，用于跨数据源按周对齐（年份边界也稳定）。"""
    iso = pd.to_datetime(dt_series).dt.isocalendar()
    return iso['year'].astype(int) * 100 + iso['week'].astype(int)


def _normalize_code(s):
    """统一股票代码为「4 位数字 + .HK」。"""
    return (s.astype(str)
            .str.replace(r'\.HK$', '', regex=True)
            .str.strip()
            .str.lstrip('0').str.zfill(4) + '.HK')


# ===== HSPI =====
def _hspi_raw(path):
    """读一份 HSPI 周线原始行（日期 / 收盘价），只做类型规整、不做统计量。"""
    idx = _read_sheet(path, '行情数据', ['日期', '开盘价'])
    idx['日期'] = pd.to_datetime(idx['日期'], errors='coerce')
    idx['收盘价'] = pd.to_numeric(idx['收盘价'], errors='coerce')
    return idx[['日期', '收盘价']]


def build_hspi_features(report_lines, extra_raw=None):
    """构建指数周频特征并按 ISO 周对齐（来源：Yahoo ^HSNP，非 iFinD）。成交量列弃用。

    extra_raw：可选的额外周线原始行（DataFrame[日期, 收盘价]）。二期把 2025 序列并进一期后再**统一**
    计算 4 周滚动量，保证 2025 年初窗口不缺前 4 周。缺省（一期）只读 2019-2024，行为不变。
    """
    idx = _hspi_raw(HSPI_FILE)
    if extra_raw is not None:
        idx = pd.concat([idx, extra_raw[['日期', '收盘价']]], ignore_index=True)
    idx = (idx.dropna(subset=['日期', '收盘价'])
              .drop_duplicates('日期')
              .sort_values('日期')
              .reset_index(drop=True))

    idx['HSPI_周收益率'] = idx['收盘价'].pct_change()
    idx['HSPI_波动率_4W'] = idx['HSPI_周收益率'].rolling(4).std()
    idx['HSPI_动量_4W'] = idx['收盘价'].pct_change(4)
    idx['iso_key'] = _iso_week_key(idx['日期'])

    out = idx[['iso_key'] + HSPI_FEATURES].drop_duplicates('iso_key')
    log(f"HSPI：{len(idx)} 周（{idx['日期'].min().date()} ~ {idx['日期'].max().date()}），"
        f"按 ISO 周对齐，成交量列弃用", report_lines)
    return out


# ===== LPR =====
def _load_lpr_monthly(report_lines, extra=None):
    """LPR 月度报价 + 2015-10-24 贷款基准利率衔接，得到 生效日期 -> LPR_1Y / LPR_5Y 序列。

    extra：可选的 2025 月度报价行（DataFrame[生效日期, LPR_1Y, LPR_5Y]），无基准衔接段。
    """
    lpr = _read_sheet(LPR_FILE, 'LPR月度', ['日期', '1年期'])
    lpr = lpr.rename(columns={'日期': '生效日期', '1年期LPR（%）': 'LPR_1Y',
                              '5年期以上LPR（%）': 'LPR_5Y'})
    lpr = lpr[['生效日期', 'LPR_1Y', 'LPR_5Y']]
    lpr['生效日期'] = pd.to_datetime(lpr['生效日期'], errors='coerce')
    lpr['LPR_1Y'] = pd.to_numeric(lpr['LPR_1Y'], errors='coerce')
    lpr['LPR_5Y'] = pd.to_numeric(lpr['LPR_5Y'], errors='coerce')

    base = _read_sheet(LPR_FILE, '基准利率衔接', ['生效日期'])
    base = base.rename(columns={'生效日期': '生效日期',
                                '一年以内（含一年）贷款基准利率（%）': 'LPR_1Y',
                                '五年以上贷款基准利率（%）': 'LPR_5Y'})
    base = base[['生效日期', 'LPR_1Y', 'LPR_5Y']].copy()
    base['生效日期'] = pd.to_datetime(base['生效日期'], errors='coerce')
    for c in ['LPR_1Y', 'LPR_5Y']:
        base[c] = pd.to_numeric(base[c], errors='coerce')

    parts = [base, lpr]
    if extra is not None:
        parts.append(extra[['生效日期', 'LPR_1Y', 'LPR_5Y']].copy())
    comb = pd.concat(parts, ignore_index=True)
    comb = comb.dropna(subset=['生效日期']).sort_values('生效日期').reset_index(drop=True)
    log(f"LPR：{len(lpr)} 期月度报价 + {len(base)} 行基准利率衔接"
        + (f" + {len(extra)} 期 2025 报价" if extra is not None else "")
        + f"，覆盖 {comb['生效日期'].min().date()} 起", report_lines)
    return comb


def build_lpr_features(df, report_lines, extra=None):
    """LPR 前向填充到周频：按 交易日期 取最近一次生效报价，再构造利差与 12 周变动。"""
    comb = _load_lpr_monthly(report_lines, extra=extra)

    uniq = pd.DataFrame({'交易日期': np.sort(df['交易日期'].unique())})
    uniq = pd.merge_asof(uniq, comb, left_on='交易日期', right_on='生效日期',
                         direction='backward')
    uniq['LPR_1Y'] = uniq['LPR_1Y'].ffill()
    uniq['LPR_5Y'] = uniq['LPR_5Y'].ffill()
    uniq['LPR_5Y_1Y_利差'] = uniq['LPR_5Y'] - uniq['LPR_1Y']
    uniq['LPR_1Y_12W变动'] = uniq['LPR_1Y'] - uniq['LPR_1Y'].shift(12)
    return uniq[['交易日期'] + LPR_FEATURES]


# ===== 半年度财务 =====
def _build_financial_original(report_lines):
    """原 24 家：由 iFinD 原始金额列计算三个比率，按「公告日期」对齐（美元口径，仅比值入模）。"""
    fin = _read_sheet(FIN_FILE, '财务数据', ['股票代码', '报告期'])
    fin['股票代码'] = _normalize_code(fin['股票代码'])
    fin['报告期'] = pd.to_datetime(fin['报告期'], errors='coerce')
    fin['公告日期'] = pd.to_datetime(fin['公告日期'], errors='coerce')

    # iFinD 导出金额列可能混入文本，统一转数值
    amount_cols = ['总资产', '总负债', '流动资产', '流动负债', '货币资金', '短期借款',
                   '长期借款', '经营活动产生的现金流量净额']
    for c in amount_cols:
        fin[c] = pd.to_numeric(fin[c], errors='coerce')

    fin['资产负债率'] = fin['总负债'] / fin['总资产'] * 100
    fin['流动比率'] = fin['流动资产'] / fin['流动负债']
    net_debt = fin['短期借款'] + fin['长期借款'] - fin['货币资金']
    fin['经营现金流_净负债'] = fin['经营活动产生的现金流量净额'] / net_debt

    fin = fin.dropna(subset=['公告日期'])
    log(f"原 24 家财务面板：{fin['股票代码'].nunique()} 家 × {fin['报告期'].nunique()} 报告期，"
        f"按公告日期对齐（美元口径，仅比值入模）", report_lines)
    return fin[['股票代码', '报告期', '公告日期'] + FIN_LEVEL_FEATURES]


def _build_financial_new(report_lines):
    """新增池：三比率已在行情工作簿「财务指标原始」sheet，公告日期来自人工核实文件，
    按 (股票代码, 报告期) 左连接；缺失公告日期者（上市前期间 / 延迟刊发未出正式业绩）剔除。
    本函数不做统计量拟合，仅做键连接与逐期差分。"""
    ratio = _read_sheet(NEW_MARKET_FILE, '财务指标原始', ['股票代码', '报告期'])
    ratio['股票代码'] = _normalize_code(ratio['股票代码'])
    ratio['报告期'] = pd.to_datetime(ratio['报告期'], errors='coerce')
    for c in ['资产负债率', '流动比率', '经营现金流_净负债']:
        ratio[c] = pd.to_numeric(ratio[c], errors='coerce')

    dates = _read_sheet(NEW_FIN_DATE_FILE, '公告日期', ['股票代码', '报告期'])
    dates['股票代码'] = _normalize_code(dates['股票代码'])
    dates['报告期'] = pd.to_datetime(dates['报告期'], errors='coerce')
    dates['公告日期'] = pd.to_datetime(dates['公告日期'], errors='coerce')

    fin = ratio.merge(dates[['股票代码', '报告期', '公告日期']],
                      on=['股票代码', '报告期'], how='left')
    n_all = len(fin)
    fin = fin.dropna(subset=['公告日期'])
    log(f"新增池财务面板：{len(fin)}/{n_all} 条有公告日期（其余为上市前期间 / 未披露，已剔除），"
        f"覆盖 {fin['股票代码'].nunique()} 家 × {fin['报告期'].nunique()} 报告期", report_lines)
    return fin[['股票代码', '报告期', '公告日期'] + FIN_LEVEL_FEATURES]


def build_financial_features(report_lines, extra_panel=None):
    """合并原 24 家与新增池财务面板，做与期对齐的环比变动后返回统一面板。

    同一公司多期共用同一公告日（困境公司一次性补发多期）时，保留报告期最新的一行，
    使 merge_asof 取值确定且符合"当时可得信息最新"的语义。

    extra_panel：可选的 2025 全池面板（DataFrame[股票代码, 报告期, 公告日期] + FIN_LEVEL_FEATURES）。
    与一期面板在 2024H2 有重叠；用稳定排序 + keep='last' 让 2025 批次（后拼接）覆盖重叠行，取值确定。
    """
    fin = pd.concat([_build_financial_original(report_lines),
                     _build_financial_new(report_lines)], ignore_index=True)
    if extra_panel is not None:
        fin = pd.concat([fin, extra_panel], ignore_index=True)
    fin = fin.sort_values(['股票代码', '报告期'], kind='mergesort')
    before = len(fin)
    fin = fin.drop_duplicates(subset=['股票代码', '公告日期'], keep='last')
    if before - len(fin) > 0:
        log(f"同一公告日多期合并：保留最新报告期，去重 {before - len(fin)} 条", report_lines)

    fin = fin.sort_values(['股票代码', '报告期'], kind='mergesort')
    fin['资产负债率_环比变动'] = fin.groupby('股票代码')['资产负债率'].diff()
    fin['流动比率_环比变动'] = fin.groupby('股票代码')['流动比率'].diff()

    fin = fin.sort_values('公告日期').reset_index(drop=True)
    log(f"财务面板合计：{fin['股票代码'].nunique()} 家，按公告日期对齐", report_lines)
    return fin[['股票代码', '公告日期'] + FIN_FEATURES]


def attach_slow_variables(df, report_lines, hspi_extra=None, lpr_extra=None, fin_extra=None):
    """Step 1.5: 接入指数 / LPR / 半年度财务慢变量。所有对齐只用当期及之前信息。

    hspi_extra / lpr_extra / fin_extra：二期 2025 来源（缺省为空 = 一期行为，只覆盖 2019-2024）。
    """
    log("\n" + "=" * 60, report_lines)
    log("Step 1.5: 接入慢变量（HSPI / LPR / 半年度财务）", report_lines)

    df = df.copy()
    df['iso_key'] = _iso_week_key(df['交易日期'])

    hspi = build_hspi_features(report_lines, extra_raw=hspi_extra)
    df = df.merge(hspi, on='iso_key', how='left').drop(columns=['iso_key'])

    lpr = build_lpr_features(df, report_lines, extra=lpr_extra)
    df = df.merge(lpr, on='交易日期', how='left')

    fin = build_financial_features(report_lines, extra_panel=fin_extra)
    before = len(df)
    # 原始行情表内自带的财务三列为「末期值冻结」（等于 2024-12-31 截面值，等于把未来值灌进训练期），
    # 先整体丢弃，再由本模块按公告日期重建为半年度时变特征。
    df = df.drop(columns=[c for c in FIN_FEATURES if c in df.columns])
    # merge_asof（pandas 2.x）要求 on 键全局有序，故按交易日期全局排序后合并，再还原行序
    df['_row_id'] = np.arange(before)
    df = df.sort_values('交易日期', kind='mergesort')
    df = pd.merge_asof(df, fin, left_on='交易日期', right_on='公告日期',
                       by='股票代码', direction='backward')
    df = df.drop(columns=['公告日期'])
    # 按公司对字段做 LOCF（use 上一期可用值，不用未来值）
    df = df.sort_values(['股票代码', '交易日期'], kind='mergesort')
    df[FIN_FEATURES] = df.groupby('股票代码')[FIN_FEATURES].ffill()
    df = df.sort_values('_row_id', kind='mergesort').drop(columns=['_row_id']).reset_index(drop=True)

    log(f"慢变量接入完成：新增 {len(SLOW_FEATURES)} 列，记录数 {before} 行（未增删行）", report_lines)
    nan_cnt = df[SLOW_FEATURES].isnull().sum()
    nan_cnt = nan_cnt[nan_cnt > 0]
    if len(nan_cnt) > 0:
        log("慢变量缺失分布（由后续 dropna 处理）：", report_lines)
        log(nan_cnt.to_string(), report_lines)
    else:
        log("慢变量无缺失", report_lines)

    return df
