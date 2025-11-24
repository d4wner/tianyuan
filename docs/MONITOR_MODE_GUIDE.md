# 监控模式使用指南

本文档详细说明如何使用系统的监控模式来实时观测市场信号、动态分配资金并执行建仓加仓操作。

## 1. 监控模式功能概述

监控模式可以：
- 实时观测多个股票的市场信号
- 根据缠论算法计算买卖信号强度
- 基于预设策略动态分配资金
- 执行自动化的买入和卖出操作
- 发送信号通知（通过钉钉）
- 保存交易记录和资金状态

## 2. 启动监控模式的方法

### 2.1 通过命令行启动

最直接的方式是使用命令行参数启动监控模式：

```bash
# 基本监控模式（默认300秒扫描间隔）
python -m src.main --mode monitor --symbols 600000.SH 600036.SH 601318.SH --capital 600000

# 日内高频监控模式（1分钟扫描间隔）
python -m src.main --mode intraday --symbols 600000.SH 600036.SH 601318.SH --capital 600000

# 指定扫描间隔（秒）
python -m src.main --mode monitor --symbols 600000.SH 600036.SH --interval 120 --capital 600000

# 盘后继续扫描
export MONITOR_AFTER_HOURS=true
python -m src.main --mode monitor --symbols 600000.SH
```

### 2.2 命令行参数说明

| 参数 | 说明 | 是否必需 | 默认值 |
|------|------|----------|--------|
| `--mode` | 运行模式，设置为 `monitor` 或 `intraday` | 是 | - |
| `--symbols` | 要监控的股票代码，多个用空格分隔 | 否（如不指定则使用配置文件中的默认股票） | - |
| `--capital` | 初始资金，单位：元 | 否 | 600000 |
| `--interval` | 扫描间隔，单位：秒 | 否 | 300（monitor模式）/60（intraday模式） |
| `--config` | 配置文件路径 | 否 | `system.yaml` |

### 2.3 从配置文件加载股票列表

如果不想在命令行中指定股票列表，可以在 `system.yaml` 中配置：

```yaml
monitor:
  symbols:
    - 600000.SH
    - 600036.SH
    - 601318.SH
```

然后直接启动：

```bash
python -m src.main --mode monitor --capital 600000
```

## 3. 配置说明

### 3.1 系统配置文件（system.yaml）

监控模式使用的主要配置在 `system.yaml` 文件中：

```yaml
# 系统配置
system:
  name: "缠论交易系统"
  version: "1.0.0"
  log_level: "INFO"

# 数据获取配置
data_fetcher:
  api_key: "your_api_key"
  timeout: 10
  retry_times: 3

# 缠论计算配置
chanlun:
  kline_period: "5m"  # 5分钟K线
  days: 3             # 历史数据天数
  use_volatility: true

# 监控器配置
monitor:
  initial_capital: 600000    # 初始资金，单位：元
  interval: 300              # 监控间隔，单位：秒
  symbols:                   # 监控股票列表
    - 600000.SH
    - 600036.SH
  scan_after_hours: false    # 是否在非交易时间也扫描
  max_position_percent: 0.2  # 单个股票最大仓位比例
  min_position_percent: 0.05 # 最小建仓比例

# 风险控制配置
risk_control:
  stop_loss_percent: 0.05    # 止损比例
  take_profit_percent: 0.15  # 止盈比例
  max_drawdown: 0.10         # 最大回撤比例
  max_open_positions: 10     # 最大持仓数量

# 通知配置
dingding:
  webhook_url: "https://oapi.dingtalk.com/robot/send?access_token=xxx"
  secret: "xxx"
  send_signals: true
  send_errors: true
```

### 3.2 环境变量配置

除了配置文件外，还可以通过环境变量覆盖某些配置：

| 环境变量 | 说明 | 对应配置 |
|---------|------|----------|
| `MONITOR_INITIAL_CAPITAL` | 初始资金 | `monitor.initial_capital` |
| `MONITOR_INTERVAL` | 监控间隔（秒） | `monitor.interval` |
| `MONITOR_AFTER_HOURS` | 是否盘后扫描 | `monitor.scan_after_hours` |
| `RISK_STOP_LOSS` | 止损比例 | `risk_control.stop_loss_percent` |

## 4. 资金分配和建仓逻辑

监控模式使用以下策略进行资金分配和建仓：

1. **动态仓位计算**：根据信号强度自动调整仓位大小
2. **分散投资**：单个股票最大仓位不超过总资金的20%
3. **分批建仓**：对于强力信号，可以分多次加仓
4. **止损保护**：每个持仓都设置自动止损

## 5. 输出和日志

### 5.1 信号文件

检测到的交易信号保存在 `outputs/signals/` 目录下，文件名格式为：`{股票代码}_{时间戳}.json`。

### 5.2 交易记录

交易操作记录保存在 `outputs/trades/` 目录下，文件名格式为：`trades_{日期}.json`。

### 5.3 日志

系统日志输出到控制台和 `logs/` 目录，日志级别可在配置文件中设置。

## 6. 使用注意事项

1. **确保API密钥有效**：在 `system.yaml` 中配置有效的数据API密钥
2. **资金配置一致性**：命令行参数 `--capital` 优先级高于配置文件中的 `monitor.initial_capital`
3. **交易时间注意**：默认只在交易时间扫描，非交易时间扫描需设置相关配置
4. **止损策略**：系统提供自动止损功能，建议保持开启状态
5. **监控资源消耗**：高频监控模式（intraday）会增加API调用次数和系统资源消耗

## 7. 示例：启动实时监控

### 7.1 基本监控

```bash
# 使用默认配置，监控配置文件中定义的股票
python -m src.main --mode monitor --capital 500000
```

### 7.2 针对特定股票的监控

```bash
# 监控特定的几只股票，设置较短的扫描间隔
python -m src.main --mode monitor --symbols 600000.SH 600036.SH 601318.SH --interval 120 --capital 800000
```

### 7.3 日内高频监控

```bash
# 日内模式，每分钟扫描一次
python -m src.main --mode intraday --symbols 600000.SH 600036.SH
```

## 8. 监控模式与回测模式的关系

监控模式使用与回测模式相同的缠论计算引擎，但有以下主要区别：

- 监控模式实时运行，回测模式基于历史数据
- 监控模式执行实际交易决策，回测模式模拟历史交易
- 监控模式使用当前市场数据，回测模式使用历史数据集

## 9. 常见问题

**Q: 如何修改默认的初始资金？**
A: 可以通过命令行参数 `--capital` 或修改配置文件中的 `monitor.initial_capital` 来设置。

**Q: 监控模式会自动执行交易吗？**
A: 目前监控模式会生成交易信号并记录模拟交易，但实际交易执行需要额外配置（如券商API）。

**Q: 如何调整信号强度阈值？**
A: 可以在 `system.yaml` 中的 `monitor` 部分添加 `signal_strength_threshold` 配置项。

**Q: 监控模式消耗多少系统资源？**
A: 资源消耗与监控的股票数量和扫描间隔有关，建议在服务器上运行时监控CPU和内存使用情况。