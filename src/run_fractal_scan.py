import pandas as pd
from calculator import ChanlunCalculator

# 加载数据
df = pd.read_csv('../outputs/exports/sh512660_daily_20251124_195330.csv')

# 创建必要的配置
config = {
    'data_validation_enabled': True,
    'min_data_points': 20
}

# 创建计算器实例
calculator = ChanlunCalculator(config=config)

# 计算分型
df_with_fractals = calculator.calculate_fractals(df)

# 输出结果
print('\n完整分型扫描结果：')
print(f'顶分型总数: {df_with_fractals["top_fractal"].sum()}')
print(f'底分型总数: {df_with_fractals["bottom_fractal"].sum()}')

# 显示最近的顶分型
print('\n最近5个顶分型:')
top_fractals = df_with_fractals[df_with_fractals['top_fractal']].sort_values('date', ascending=False).head(5)
if not top_fractals.empty:
    for _, row in top_fractals.iterrows():
        print(f'  {row["date"]} - 价格: {row["fractal_price"]:.4f}')
else:
    print('  无顶分型')

# 显示最近的底分型
print('\n最近5个底分型:')
bottom_fractals = df_with_fractals[df_with_fractals['bottom_fractal']].sort_values('date', ascending=False).head(5)
if not bottom_fractals.empty:
    for _, row in bottom_fractals.iterrows():
        print(f'  {row["date"]} - 价格: {row["fractal_price"]:.4f}')
else:
    print('  无底分型')

# 特别验证11月17日和11月24日的分型
print('\n特别验证：')
nov17 = df_with_fractals[df_with_fractals['date'] == '2025-11-17']
if not nov17.empty:
    print(f'11月17日 - 顶分型: {nov17["top_fractal"].values[0]}, 底分型: {nov17["bottom_fractal"].values[0]}')

nov24 = df_with_fractals[df_with_fractals['date'] == '2025-11-24']
if not nov24.empty:
    print(f'11月24日 - 顶分型: {nov24["top_fractal"].values[0]}, 底分型: {nov24["bottom_fractal"].values[0]}')