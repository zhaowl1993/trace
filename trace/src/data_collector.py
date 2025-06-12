"""
数据采集模块
负责从多个数据源获取黄金价格和相关市场数据
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
from loguru import logger
import sqlite3
from sqlalchemy import create_engine
import pytz


class DataCollector:
    """数据采集器 - 获取实时和历史市场数据"""
    
    def __init__(self, config: Dict):
        """
        初始化数据采集器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.symbol = config['trading']['symbol']
        self.timeframe = config['trading']['timeframe']
        self.update_interval = config['data_sources']['update_interval']
        
        # 初始化数据库连接
        self.db_engine = create_engine(f"sqlite:///{config['database']['path']}")
        
        # 时区设置
        self.timezone = pytz.timezone('UTC')
        
        logger.info(f"数据采集器初始化完成 - 交易品种: {self.symbol}")
    
    def get_historical_data(self, 
                          period: str = "1y", 
                          interval: str = "1m") -> pd.DataFrame:
        """
        获取历史数据
        
        Args:
            period: 数据周期 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: 数据间隔 (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        try:
            # 将XAUUSD转换为Yahoo Finance格式
            yahoo_symbol = "GC=F"  # 黄金期货
            if self.symbol == "XAUUSD":
                yahoo_symbol = "GC=F"
            
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"未获取到 {yahoo_symbol} 的历史数据")
                return pd.DataFrame()
            
            # 重命名列以保持一致性
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            data.index.name = 'DateTime'
            
            # 添加时间戳
            data['Timestamp'] = data.index
            
            logger.info(f"成功获取历史数据: {len(data)} 条记录")
            return data
            
        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            return pd.DataFrame()
    
    def get_realtime_data(self) -> Optional[Dict]:
        """
        获取实时数据
        
        Returns:
            包含当前价格信息的字典
        """
        try:
            yahoo_symbol = "GC=F"
            ticker = yf.Ticker(yahoo_symbol)
            
            # 获取实时报价
            info = ticker.fast_info
            history = ticker.history(period="1d", interval="1m")
            
            if history.empty:
                return None
            
            latest = history.iloc[-1]
            
            realtime_data = {
                'symbol': self.symbol,
                'timestamp': datetime.now(self.timezone),
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'close': float(latest['Close']),
                'volume': float(latest['Volume']),
                'bid': float(info.get('bid', latest['Close'])),
                'ask': float(info.get('ask', latest['Close'])),
                'spread': abs(float(info.get('ask', latest['Close'])) - 
                             float(info.get('bid', latest['Close'])))
            }
            
            return realtime_data
            
        except Exception as e:
            logger.error(f"获取实时数据失败: {e}")
            return None
    
    def get_market_sentiment_data(self) -> Dict:
        """
        获取市场情绪相关数据
        
        Returns:
            包含VIX、DXY等指标的字典
        """
        sentiment_data = {}
        
        try:
            # VIX恐慌指数
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="1d", interval="1m")
            if not vix_data.empty:
                sentiment_data['vix'] = float(vix_data['Close'].iloc[-1])
            
            # 美元指数 (DXY)
            dxy_ticker = yf.Ticker("DX-Y.NYB")
            dxy_data = dxy_ticker.history(period="1d", interval="1m")
            if not dxy_data.empty:
                sentiment_data['dxy'] = float(dxy_data['Close'].iloc[-1])
            
            # 添加时间戳
            sentiment_data['timestamp'] = datetime.now(self.timezone)
            
            logger.debug(f"获取市场情绪数据: {sentiment_data}")
            
        except Exception as e:
            logger.error(f"获取市场情绪数据失败: {e}")
        
        return sentiment_data
    
    def save_data_to_db(self, data: pd.DataFrame, table_name: str):
        """
        保存数据到数据库
        
        Args:
            data: 要保存的数据
            table_name: 表名
        """
        try:
            data.to_sql(table_name, self.db_engine, 
                       if_exists='replace', index=True)
            logger.info(f"数据已保存到数据库表: {table_name}")
        except Exception as e:
            logger.error(f"保存数据到数据库失败: {e}")
    
    def load_data_from_db(self, table_name: str, 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """
        从数据库加载数据
        
        Args:
            table_name: 表名
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame
        """
        try:
            query = f"SELECT * FROM {table_name}"
            
            if start_date and end_date:
                query += f" WHERE DateTime BETWEEN '{start_date}' AND '{end_date}'"
            
            data = pd.read_sql(query, self.db_engine, index_col='DateTime')
            logger.info(f"从数据库加载数据: {len(data)} 条记录")
            return data
            
        except Exception as e:
            logger.error(f"从数据库加载数据失败: {e}")
            return pd.DataFrame()
    
    def start_realtime_collection(self, callback_func=None):
        """
        启动实时数据采集
        
        Args:
            callback_func: 数据更新时的回调函数
        """
        logger.info("开始实时数据采集...")
        
        while True:
            try:
                # 获取实时数据
                realtime_data = self.get_realtime_data()
                
                if realtime_data:
                    # 获取市场情绪数据
                    sentiment_data = self.get_market_sentiment_data()
                    realtime_data.update(sentiment_data)
                    
                    # 调用回调函数
                    if callback_func:
                        callback_func(realtime_data)
                    
                    logger.debug(f"实时数据更新: {realtime_data['close']}")
                
                # 等待下次更新
                time.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                logger.info("实时数据采集已停止")
                break
            except Exception as e:
                logger.error(f"实时数据采集出错: {e}")
                time.sleep(self.update_interval)
    
    def validate_data_quality(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        验证数据质量
        
        Args:
            data: 要验证的数据
            
        Returns:
            (是否通过验证, 问题列表)
        """
        issues = []
        
        # 检查必要的列
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"缺失必要的列: {missing_columns}")
        
        # 检查空值
        null_counts = data.isnull().sum()
        if null_counts.any():
            issues.append(f"存在空值: {null_counts[null_counts > 0].to_dict()}")
        
        # 检查价格逻辑
        if 'High' in data.columns and 'Low' in data.columns:
            invalid_price_logic = (data['High'] < data['Low']).sum()
            if invalid_price_logic > 0:
                issues.append(f"价格逻辑错误: {invalid_price_logic} 条记录")
        
        # 检查重复时间戳
        if data.index.duplicated().sum() > 0:
            issues.append(f"重复时间戳: {data.index.duplicated().sum()} 条")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"数据质量问题: {issues}")
        else:
            logger.info("数据质量验证通过")
        
        return is_valid, issues 