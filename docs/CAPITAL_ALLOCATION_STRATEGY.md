# 资金分配与建仓加仓策略配置指南

本文档详细说明缠论交易系统中资金分配和建仓加仓策略的实现机制，以及如何根据您的需求进行配置。

## 1. 资金分配机制概述

系统使用动态资金分配策略，主要包括以下核心机制：

- **基于信号强度的仓位计算**：根据缠论信号的强度自动调整建仓比例
- **最大单仓限制**：防止单个股票占用过多资金，控制风险
- **分散投资原则**：自动在多个股票之间分配资金
- **保留现金缓冲**：确保有足够资金应对新机会
- **动态再平衡**：根据市场变化调整各持仓比例

## 2. 核心配置参数

所有资金分配相关的配置都可以在 `system.yaml` 文件中设置：

```yaml
# 监控器资金配置部分
monitor:
  initial_capital: 600000      # 初始资金，单位：元
  max_position_percent: 0.2    # 单个股票最大仓位比例
  min_position_percent: 0.05   # 最小建仓比例
  default_position_percent: 0.1 # 默认建仓比例
  signal_strength_threshold: 60 # 信号强度阈值（0-100）
  capital_allocation_strategy: "dynamic" # 资金分配策略："dynamic", "fixed", "aggressive"
  rebalance_interval: 5        # 资金再平衡间隔（交易日）

# 风险控制配置
risk_control:
  stop_loss_percent: 0.05      # 止损比例
  take_profit_percent: 0.15    # 止盈比例
  max_drawdown: 0.10           # 最大回撤限制
  max_open_positions: 10       # 最大持仓数量
  position_sizing_method: "fixed_fraction" # 仓位计算方法
  risk_per_trade: 0.01         # 每笔交易风险比例
  volatility_adjustment: true  # 是否根据波动率调整仓位
```

## 3. 建仓加仓策略详解

### 3.1 信号强度与仓位计算

系统使用 `calculate_position_size` 方法根据信号强度计算仓位大小：

```python
# 简化的仓位计算逻辑
def calculate_position_size(self, signal_type, signal_strength, current_price):
    # 基础仓位比例
    base_percent = self.config.get('default_position_percent', 0.1)
    
    # 根据信号类型调整
    if signal_type == 'BUY_STRONG':
        base_percent *= 1.5  # 强买入信号加大仓位
    elif signal_type == 'BUY_WEAK':
        base_percent *= 0.7  # 弱买入信号减小仓位
    
    # 根据信号强度进一步调整
    adjusted_percent = base_percent * (signal_strength / 100)
    
    # 确保在最小和最大仓位范围内
    min_percent = self.config.get('min_position_percent', 0.05)
    max_percent = self.config.get('max_position_percent', 0.2)
    final_percent = max(min(adjusted_percent, max_percent), min_percent)
    
    # 计算实际资金量
    position_capital = self.available_capital * final_percent
    
    # 计算可以购买的股数
    shares = int(position_capital / current_price)
    
    return shares, position_capital
```

### 3.2 资金分配策略类型

系统支持三种主要的资金分配策略：

#### 3.2.1 动态分配（dynamic）

这是默认策略，根据信号强度和市场状况动态调整资金分配：

- 信号越强，分配资金越多
- 根据市场整体状况（牛市/熊市）调整总体仓位
- 自动为强势股票配置更多资金

配置示例：
```yaml
monitor:
  capital_allocation_strategy: "dynamic"
  max_position_percent: 0.25  # 牛市中可提高最大仓位
  market_adaptation: true     # 启用市场适应
```

#### 3.2.2 固定分配（fixed）

为每只股票分配固定比例的资金，适合稳定的投资组合：

- 每只股票使用固定比例的资金
- 更容易管理和预测风险
- 不考虑信号强度的差异

配置示例：
```yaml
monitor:
  capital_allocation_strategy: "fixed"
  default_position_percent: 0.1  # 每只股票固定10%资金
  max_open_positions: 8          # 最多8只股票，总计80%资金
```

#### 3.2.3 激进策略（aggressive）

为强烈信号分配更多资金，适合追求高收益、能承受高风险的投资者：

- 强信号可能占用更多资金（最高可达30%）
- 弱信号依然配置基础资金
- 允许更集中的投资组合

配置示例：
```yaml
monitor:
  capital_allocation_strategy: "aggressive"
  max_position_percent: 0.3    # 单个股票最高30%仓位
  signal_strength_threshold: 70 # 更高的信号强度阈值
```

## 4. 加仓策略配置

系统支持自动加仓功能，可以通过以下参数配置：

```yaml
monitor:
  # 加仓配置
  enable_additional_position: true   # 启用加仓功能
  additional_position_threshold: 0.03 # 上涨超过此比例时考虑加仓
  max_additional_times: 2            # 最多加仓次数
  additional_position_factor: 0.8    # 加仓仓位是初始仓位的比例
  additional_signal_strength: 65     # 加仓所需的最低信号强度
```

加仓触发条件：
1. 股票价格上涨超过阈值（默认为3%）
2. 依然存在买入信号且强度足够
3. 未达到最大加仓次数限制
4. 账户有足够可用资金

## 5. 风险控制与资金保护

### 5.1 止损设置

系统为每个持仓自动设置止损点：

```yaml
risk_control:
  stop_loss_percent: 0.05      # 相对买入价止损5%
  trailing_stop_enabled: true  # 启用追踪止损
  trailing_stop_percent: 0.03  # 追踪止损幅度3%
```

### 5.2 资金回撤控制

当整体资金回撤达到设定阈值时，系统会自动降低仓位：

```yaml
risk_control:
  max_drawdown: 0.10           # 最大回撤10%
  drawdown_response: "reduce_positions"  # 回撤响应策略
  drawdown_position_reduction: 0.5  # 回撤时仓位缩减比例
```

### 5.3 波动率调整

系统可以根据市场波动率自动调整仓位大小：

```yaml
risk_control:
  volatility_adjustment: true
  max_volatility: 0.03         # 最大可接受的日波动率
  volatility_lookback_days: 20 # 计算波动率的回看天数
```

## 6. 实际配置示例

### 6.1 平衡型投资策略

适合大多数投资者的平衡型配置：

```yaml
monitor:
  initial_capital: 600000
  max_position_percent: 0.2
  min_position_percent: 0.05
  default_position_percent: 0.1
  capital_allocation_strategy: "dynamic"
  enable_additional_position: true
  max_additional_times: 2
  signal_strength_threshold: 60

risk_control:
  stop_loss_percent: 0.05
  take_profit_percent: 0.15
  max_open_positions: 8
  volatility_adjustment: true
```

### 6.2 保守型投资策略

适合风险承受能力较低的投资者：

```yaml
monitor:
  initial_capital: 600000
  max_position_percent: 0.15
  min_position_percent: 0.03
  default_position_percent: 0.08
  capital_allocation_strategy: "fixed"
  enable_additional_position: false
  signal_strength_threshold: 70

risk_control:
  stop_loss_percent: 0.04
  take_profit_percent: 0.10
  max_open_positions: 10
  max_drawdown: 0.08
```

### 6.3 激进型投资策略

适合风险承受能力高、追求高收益的投资者：

```yaml
monitor:
  initial_capital: 600000
  max_position_percent: 0.3
  min_position_percent: 0.05
  default_position_percent: 0.15
  capital_allocation_strategy: "aggressive"
  enable_additional_position: true
  max_additional_times: 3
  signal_strength_threshold: 65

risk_control:
  stop_loss_percent: 0.06
  take_profit_percent: 0.20
  max_open_positions: 6
  risk_per_trade: 0.02
```

## 7. 资金状态监控

### 7.1 资金状态文件

系统会保存当前资金状态到文件，以便程序重启后恢复：

```yaml
monitor:
  save_capital_state: true
  capital_state_file: "outputs/capital_state.json"
  state_save_interval: 300  # 保存间隔（秒）
```

### 7.2 资金统计报告

配置每日资金统计报告：

```yaml
monitor:
  daily_capital_report: true
  report_time: "15:00"  # 收盘后生成报告
  report_file: "outputs/daily_capital/{date}.json"
```

## 8. 自定义资金分配逻辑

如果内置策略不满足需求，您可以通过修改代码实现自定义逻辑。主要涉及以下文件：

1. **src/monitor.py** - 修改 `calculate_position_size` 方法
2. **src/strategy.py** - 添加自定义策略类

示例：实现基于波动率的动态仓位调整

```python
def calculate_position_size_with_volatility(self, signal_type, current_price, volatility):
    # 基础风险资金（总资金的1%）
    risk_amount = self.total_capital * self.risk_per_trade
    
    # 根据波动率计算仓位（波动率越高，仓位越小）
    position_size = risk_amount / (current_price * volatility * 2)  # 2倍波动率作为止损空间
    
    # 确保仓位在合理范围内
    max_size = self.total_capital * self.max_position_percent / current_price
    min_size = self.total_capital * self.min_position_percent / current_price
    
    return max(min(position_size, max_size), min_size)
```

## 9. 常见问题

**Q: 如何设置初始资金？**
A: 可以通过命令行参数 `--capital` 或在配置文件中设置 `monitor.initial_capital`。

**Q: 为什么实际建仓资金与配置的百分比不符？**
A: 可能有以下原因：1) 信号强度不足；2) 可用资金不足；3) 达到单仓上限；4) 波动率调整。

**Q: 如何禁用自动加仓功能？**
A: 在配置文件中设置 `monitor.enable_additional_position: false`。

**Q: 资金回撤时系统会自动减仓吗？**
A: 是的，当回撤达到 `risk_control.max_drawdown` 时，系统会根据 `drawdown_response` 设置采取相应措施。

**Q: 如何查看当前资金分配状态？**
A: 资金状态保存在 `outputs/capital_state.json` 文件中，也可以通过日志查看定期统计信息。

## 10. 性能优化建议

1. **定期调整策略参数**：根据市场环境变化调整配置
2. **监控信号质量**：跟踪信号强度与实际收益的关系
3. **资金曲线分析**：定期分析资金曲线，识别策略弱点
4. **设置合理的最大持仓数**：避免过度分散或过度集中
5. **平衡风险与收益**：根据个人风险偏好调整止损和仓位参数

通过合理配置上述参数，您可以根据自己的投资风格和风险偏好，实现个性化的资金分配和建仓加仓策略。