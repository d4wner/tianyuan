# 实时小时/分钟级别信号检测系统

## 系统概述

该系统已修复并完全验证，可以在盘中实时识别小时和分钟级别的买入信号，并将信号发送到钉钉通知。

## 已完成的修复

1. **钉钉通知配置加载修复**
   - 修复了`DingdingNotifier`类的配置加载逻辑
   - 修复了`HourlySignalDetector`类的配置加载逻辑
   - 确保系统能正确读取`system.yaml`中的`access_token`配置

2. **功能验证**
   - ✅ 小时级别信号检测功能正常
   - ✅ 分钟级别信号检测功能正常（15min/30min）
   - ✅ 钉钉通知功能正常工作
   - ✅ 实时检测机制已验证

## 系统功能

### 1. 小时级别信号检测
- 检测小时级别的底分型信号
- 结合MACD指标进行背离分析
- 提供买入置信度评估
- 自动发送钉钉通知

### 2. 分钟级别信号检测
- 支持15分钟和30分钟级别的信号检测
- 识别向上笔完成信号
- 检测回撤买点
- 支持子仓位分配建议

### 3. 钉钉通知功能
- 发送小时级别买入预警
- 包含标的名称、价格、低点、置信度等信息
- 提供交易建议
- 支持实时消息推送

## 运行方式

### 方式1：使用monitor.py实时监控
```bash
python -m src.monitor
```
- 24小时实时监控
- 自动检测交易时间
- 支持多标的监控

### 方式2：配置定时任务(crontab)
```bash
# 每15分钟检测一次
0,15,30,45 9-15 * * 1-5 python -m src.hourly_signal_detector
```
- 灵活的检测间隔设置
- 只在交易时间运行
- 节省系统资源

### 方式3：直接运行检测器
```bash
# 运行小时级别信号检测器
python -m src.hourly_signal_detector

# 运行分钟级别信号检测器
python -m src.minute_position_allocator
```
- 适合手动触发检测
- 支持调试和测试

## 配置说明

主要配置文件：`config/system.yaml`

```yaml
dingding:
  access_token: "your-access-token"  # 钉钉机器人access_token
  position_units: 50  # 持仓单位
  alert_levels:  # 预警级别
    emergency: true
    warning: true
    info: true
  weekly_report: true  # 是否发送周报

hourly_signal:
  enabled: true  # 是否启用小时级别信号检测
  symbols: ["512660", "512480", "510300"]  # 监控标的
  scan_interval: 15  # 扫描间隔（分钟）
  sensitivity: 3  # 分型敏感度
  prediction_threshold: 0.6  # 预测阈值
  trading_hours_only: true  # 只在交易时间运行
```

## 测试结果

已通过以下测试验证系统功能：

1. **钉钉通知测试** - ✅ 成功
2. **小时级别信号检测** - ✅ 成功
3. **分钟级别信号检测** - ✅ 成功
4. **完整流程测试** - ✅ 成功
5. **模拟盘中运行** - ✅ 成功（部分测试）

## 维护建议

1. 定期检查钉钉机器人的access_token是否有效
2. 根据市场情况调整信号检测的敏感度和阈值
3. 定期更新系统依赖包
4. 监控系统日志，及时处理异常情况

## 故障排除

### 常见问题

1. **钉钉通知发送失败**
   - 检查`system.yaml`中的`access_token`是否正确
   - 确保钉钉机器人的安全设置允许发送通知
   - 检查网络连接是否正常

2. **信号检测不准确**
   - 调整信号检测的敏感度和阈值
   - 检查数据获取是否正常
   - 考虑增加历史数据量

3. **系统运行缓慢**
   - 减少监控标的数量
   - 增加检测间隔时间
   - 优化系统资源配置

## 项目结构

```
tianyuan/
├── src/                 # 主源码目录
│   ├── main.py         # 主程序入口
│   ├── hourly_signal_detector.py  # 小时级别信号检测器
│   ├── minute_position_allocator.py  # 分钟级别信号检测器
│   ├── dingding_notifier.py  # 钉钉通知模块
│   ├── monitor.py      # 实时监控模块
│   └── ...
├── tests/              # 测试脚本目录
│   ├── test_hourly_minute_signal_flow.py  # 小时/分钟信号流程测试
│   ├── test_real_time_signal_detection.py  # 实时信号检测测试
│   ├── test_dingding_fix.py  # 钉钉通知修复测试
│   └── ...
├── config/             # 配置文件目录
│   └── system.yaml     # 系统配置文件
├── cache/              # 缓存目录
├── data/               # 数据目录
│   ├── daily/          # 日线数据
│   ├── minute/         # 分钟线数据
│   └── signals/        # 信号数据
├── outputs/            # 输出目录
│   ├── analysis/       # 分析结果
│   ├── backtest/       # 回测结果
│   └── reports/        # 报告
├── logs/               # 日志目录
└── REAL_TIME_SIGNAL_SYSTEM_README.md  # 系统文档
```

## 联系信息

如有问题或建议，请联系系统管理员。

---

**更新日期：** 2025-12-05
**版本：** 1.0.0（修复版）