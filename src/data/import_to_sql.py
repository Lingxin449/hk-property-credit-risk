"""
将行情数据导入 SQL Server HK_Stock_RiskDB
运行：python -m src.data.import_to_sql
"""

import pandas as pd
from sqlalchemy import create_engine

from src.config import PROJECT_ROOT


def main():
    xlsx_path = str(PROJECT_ROOT / 'data/raw/港股地产24家公司_周频行情数据_2019_2024.xlsx')
    df = pd.read_excel(xlsx_path)

    df_sql = df.rename(columns={
        '股票代码': 'Ticker', '公司简称': 'Name', '风险层级': 'SubCategory',
        '交易日期': 'Trade_Date', '开盘价': 'Open_Price', '收盘价': 'Close_Price',
        '最高价': 'High_Price', '最低价': 'Low_Price', '成交量': 'Volume',
        '成交额': 'Turnover', '振幅': 'Amplitude', '周收益率': 'Weekly_Return',
        'Volume_ZScore': 'Volume_ZScore', '资产负债率': 'Debt_Ratio',
        '流动比率': 'Current_Ratio', '经营现金流_净负债': 'CF_NetDebt'
    })

    engine = create_engine('mssql+pyodbc://localhost/HK_Stock_RiskDB?driver=SQL+Server+Native+Client+11.0')
    df_sql.to_sql('Weekly_Raw', engine, if_exists='append', index=False)
    print(f"导入完成，共 {len(df_sql)} 行")


if __name__ == '__main__':
    main()
