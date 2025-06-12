"""
AI黄金交易系统
==================

一个基于人工智能的实时黄金交易系统，专注于短期交易策略。

主要特性:
- 实时数据采集和处理
- 多种机器学习模型集成
- 严格的风险管理
- 完整的回测框架
- 模块化设计

作者: AI Trading System
版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Trading System"

# 导入核心模块
from .data_collector import DataCollector
from .feature_engineer import FeatureEngineer
from .ai_models import AIModelManager
from .risk_manager import RiskManager
from .backtester import Backtester
from .trader import TradingEngine 