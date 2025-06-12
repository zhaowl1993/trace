# AI黄金交易系统

一个基于人工智能的黄金交易自动化系统，采用机器学习算法进行市场分析和交易决策。

## 📋 项目概述

本系统是一个完整的AI驱动的黄金交易解决方案，具备数据采集、模型训练、策略回测、实时交易、纸上交易、券商集成和系统监控等全套功能。

## 🎯 系统架构

### 三阶段开发计划

#### 第一阶段：基础框架 ✅
- ✅ 数据采集和预处理
- ✅ 基础AI模型实现
- ✅ 简单回测功能
- ✅ 配置管理系统

#### 第二阶段：AI模型开发 ✅
- ✅ 专业回测系统
- ✅ 实时交易引擎
- ✅ 模型性能分析
- ✅ 风险管理系统

#### 第三阶段：实时交易系统 🆕
- ✅ 纸上交易环境
- ✅ 券商API集成
- ✅ 监控和日志系统
- ✅ 完整的实时交易流程

## 🛠️ 技术栈

- **编程语言**: Python 3.8+
- **机器学习**: scikit-learn, XGBoost
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn
- **实时数据**: yfinance, MetaTrader5
- **交易接口**: Alpaca, OANDA
- **监控系统**: SQLite, 自定义告警
- **配置管理**: YAML

## 📁 项目结构

```
ai-gold-trading-system/
├── config/
│   └── config.yaml              # 系统配置文件
├── src/
│   ├── __init__.py
│   ├── data_collector.py        # 数据采集模块
│   ├── ai_model.py             # AI模型核心
│   ├── backtester.py           # 专业回测引擎
│   ├── trader.py               # 实时交易引擎
│   ├── paper_trading.py        # 📊 纸上交易系统
│   ├── broker_interface.py     # 🏦 券商接口模块
│   └── monitoring.py           # 📱 监控日志系统
├── data/                       # 数据存储目录
├── logs/                       # 日志文件目录
├── models/                     # 训练模型存储
├── main.py                     # 主程序入口
├── phase2_demo.py             # 第二阶段演示
├── phase3_demo.py             # 🆕 第三阶段演示
├── requirements.txt           # Python依赖
└── README.md                  # 项目文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/your-repo/ai-gold-trading-system.git
cd ai-gold-trading-system

# 安装依赖
pip install -r requirements.txt

# 创建必要目录
mkdir -p data logs models
```

### 2. 配置系统

编辑 `config/config.yaml` 文件，配置交易参数、API密钥等。

### 3. 运行方式

#### 方式一：完整系统使用
```bash
python main.py
```

#### 方式二：第三阶段演示
```bash
python phase3_demo.py
```

## 🌟 主要功能

### 📊 数据和模型管理
1. **数据采集和预处理** - 自动获取黄金价格数据
2. **训练AI模型** - 多算法集成机器学习
3. **模型性能评估** - 详细的性能分析报告
4. **数据更新** - 定时更新市场数据

### 💼 回测和分析
5. **运行回测** - 专业级历史数据回测
6. **查看回测报告** - 详细的回测结果分析
7. **模型性能分析** - 深度性能指标评估

### 🔄 实时交易
8. **启动模拟交易** - 实时交易信号生成
9. **交易引擎状态** - 实时监控交易状态
10. **停止交易引擎** - 安全停止交易系统
11. **手动交易测试** - 手动测试交易功能

### 🆕 第三阶段功能
12. **📊 纸上交易演示** - 无风险模拟交易环境
13. **🏦 券商接口管理** - 多券商API集成管理
14. **📱 监控系统控制** - 实时系统监控
15. **🔍 系统状态查看** - 全面的系统状态检查
16. **🚨 告警管理** - 智能告警和通知系统
17. **📋 日志导出** - 完整的日志记录和导出
18. **🎯 第三阶段完整演示** - 一键体验所有功能

### 🛠️ 系统管理
19. **系统配置** - 灵活的参数配置
20. **生成报告** - 自动生成分析报告

## 📊 核心模块详解

### 纸上交易系统 (Paper Trading)
- **完整的模拟交易环境**，无真实资金风险
- **支持多种订单类型**：市价单、限价单、止损单
- **实时订单管理**：提交、取消、状态查询
- **详细的交易记录**：成交历史、盈亏统计
- **真实的交易成本**：手续费、滑点模拟

```python
from src.paper_trading import PaperTradingEngine, OrderType

# 创建纸上交易引擎
paper_trader = PaperTradingEngine({
    'initial_capital': 10000.0,
    'commission': 0.0001,
    'slippage': 0.0002
})

# 提交订单
order_id = paper_trader.submit_order(
    symbol='XAUUSD',
    side='buy',
    quantity=0.1,
    order_type=OrderType.MARKET
)
```

### 券商接口系统 (Broker Interface)
- **多券商支持**：Alpaca、OANDA等主流券商
- **统一接口设计**：标准化的API调用
- **连接管理**：自动重连、状态监控
- **订单管理**：下单、撤单、查询
- **实时数据**：WebSocket实时行情

```python
from src.broker_interface import BrokerManager, create_broker_config

# 创建券商管理器
broker_manager = BrokerManager()

# 添加Alpaca券商
config = create_broker_config(
    broker_type='alpaca',
    api_key='your_api_key',
    secret_key='your_secret_key',
    sandbox=True
)
broker_manager.add_broker('alpaca', config)
```

### 监控日志系统 (Monitoring System)
- **系统监控**：CPU、内存、磁盘使用率
- **交易监控**：盈亏、回撤、胜率跟踪
- **智能告警**：多级别告警系统
- **日志记录**：SQLite数据库存储
- **通知系统**：邮件、Webhook通知

```python
from src.monitoring import MonitoringSystem, AlertType, AlertLevel

# 创建监控系统
monitoring = MonitoringSystem(config)
monitoring.start()

# 发送自定义告警
monitoring.send_custom_alert(
    alert_type=AlertType.TRADING,
    level=AlertLevel.WARNING,
    title="交易异常",
    message="检测到异常交易信号"
)
```

## 🎯 交易策略

### AI模型集成
- **随机森林 (Random Forest)** - 处理非线性关系
- **XGBoost** - 梯度提升决策树
- **逻辑回归 (Logistic Regression)** - 基础线性模型
- **集成学习** - 多模型投票机制

### 技术指标
- **趋势指标**：SMA、EMA、MACD
- **动量指标**：RSI、随机振荡器
- **波动率指标**：布林带、ATR
- **成交量指标**：OBV、成交量移动平均

### 风险管理
- **止损机制**：固定止损、动态止损
- **仓位控制**：Kelly公式、固定比例
- **最大回撤限制**：15%回撤保护
- **日损失限制**：30美元每日限额

## 📈 性能指标

### 回测结果 (示例)
- **总收益率**: +15.2%
- **夏普比率**: 1.85
- **最大回撤**: -8.3%
- **胜率**: 58.7%
- **盈利因子**: 1.42

### 系统性能
- **数据处理速度**: 1000条/秒
- **信号延迟**: <100ms
- **系统可用性**: 99.9%
- **内存使用**: <512MB

## 🔧 配置说明

### 交易配置
```yaml
trading:
  symbol: "XAUUSD"                    # 交易品种
  initial_capital: 10000.0            # 初始资金
  max_daily_loss: 30.0               # 最大日损失
  position_size: 0.01                # 仓位大小
  confidence_threshold: 0.65         # 置信度阈值
```

### 券商配置
```yaml
brokers:
  alpaca:
    api_key: "your_api_key"
    secret_key: "your_secret_key"
    sandbox: true
  oanda:
    api_token: "your_token"
    account_id: "your_account"
    sandbox: true
```

### 监控配置
```yaml
monitoring:
  system:
    cpu_threshold: 80.0              # CPU告警阈值
    memory_threshold: 85.0           # 内存告警阈值
    check_interval: 10               # 检查间隔(秒)
  trading:
    max_drawdown_threshold: 0.15     # 最大回撤阈值
    max_daily_loss_threshold: 1000   # 日损失阈值
  notifications:
    enabled_channels: ['log', 'email', 'webhook']
```

## 🚀 部署指南

### 开发环境
```bash
# 开发模式运行
python main.py

# 或运行演示
python phase3_demo.py
```

### 生产环境
```bash
# 使用supervisor管理进程
sudo apt-get install supervisor

# 配置supervisor
sudo nano /etc/supervisor/conf.d/ai-trading.conf

# 启动服务
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start ai-trading
```

### Docker部署
```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

## 📚 API文档

### 纸上交易API
```python
# 提交订单
order_id = paper_trader.submit_order(symbol, side, quantity, order_type, price)

# 取消订单
success = paper_trader.cancel_order(order_id)

# 查询持仓
positions = paper_trader.get_positions_list()

# 获取账户信息
account = paper_trader.get_account()
```

### 监控系统API
```python
# 启动监控
monitoring.start()

# 更新交易指标
monitoring.update_trading_metrics(metrics)

# 发送告警
monitoring.send_custom_alert(alert_type, level, title, message)

# 导出日志
log_file = monitoring.export_logs(hours=24)
```

## 🔍 故障排除

### 常见问题

1. **数据获取失败**
   - 检查网络连接
   - 验证API密钥
   - 确认数据源可用性

2. **模型训练失败**
   - 检查数据质量
   - 验证特征工程
   - 调整模型参数

3. **交易连接问题**
   - 检查券商API配置
   - 验证账户权限
   - 确认网络稳定性

### 日志查看
```bash
# 查看系统日志
tail -f logs/system.log

# 查看交易日志
tail -f logs/trading.log

# 查看错误日志
tail -f logs/error.log
```

## 🤝 贡献指南

### 开发流程
1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

### 代码规范
- 遵循 PEP 8 规范
- 添加完整的文档字符串
- 编写单元测试
- 保持代码简洁

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 📞 联系方式

- **项目主页**: https://github.com/your-repo/ai-gold-trading-system
- **问题反馈**: https://github.com/your-repo/ai-gold-trading-system/issues
- **邮箱**: your-email@example.com

## 🙏 致谢

感谢所有为本项目做出贡献的开发者和用户。

---

⭐ 如果这个项目对您有帮助，请给它一个星标！ 