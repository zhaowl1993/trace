"""
交易引擎模块
整合AI预测、风险管理和交易执行的核心引擎
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from loguru import logger
import time
import threading
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """信号类型"""
    BUY = "buy"
    SELL = "sell"
    CLOSE_BUY = "close_buy"
    CLOSE_SELL = "close_sell"
    HOLD = "hold"


@dataclass
class TradingSignal:
    """交易信号"""
    timestamp: datetime
    signal_type: SignalType
    confidence: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    model_predictions: Dict = None


class TradingEngine:
    """交易引擎 - 实时交易的核心模块"""
    
    def __init__(self, config: Dict, data_collector, feature_engineer, 
                 ai_model_manager, risk_manager):
        """
        初始化交易引擎
        
        Args:
            config: 配置字典
            data_collector: 数据采集器
            feature_engineer: 特征工程器
            ai_model_manager: AI模型管理器
            risk_manager: 风险管理器
        """
        self.config = config
        self.data_collector = data_collector
        self.feature_engineer = feature_engineer
        self.ai_model_manager = ai_model_manager
        self.risk_manager = risk_manager
        
        # 交易参数
        self.symbol = config['trading']['symbol']
        self.position_size = config['trading']['position_size']
        self.stop_loss_pips = config['trading']['stop_loss_pips']
        self.take_profit_pips = config['trading']['take_profit_pips']
        
        # 运行状态
        self.is_running = False
        self.is_trading_enabled = True
        self.last_signal_time = None
        self.min_signal_interval = 60  # 最小信号间隔(秒)
        
        # 历史数据缓存
        self.price_history = []
        self.signal_history = []
        self.max_history_length = 1000
        
        # 当前市场数据
        self.current_price = 0.0
        self.current_features = None
        
        logger.info("交易引擎初始化完成")
    
    def start_trading(self):
        """启动实时交易"""
        if self.is_running:
            logger.warning("交易引擎已在运行")
            return
        
        if not self.ai_model_manager.models:
            logger.error("AI模型未训练，无法启动交易")
            return
        
        logger.info("启动实时交易引擎...")
        self.is_running = True
        
        # 启动数据采集线程
        data_thread = threading.Thread(target=self._data_collection_loop)
        data_thread.daemon = True
        data_thread.start()
        
        # 启动交易决策线程
        trading_thread = threading.Thread(target=self._trading_decision_loop)
        trading_thread.daemon = True
        trading_thread.start()
        
        logger.info("交易引擎已启动")
    
    def stop_trading(self):
        """停止交易"""
        logger.info("停止交易引擎...")
        self.is_running = False
        
        # 平仓所有持仓
        self._close_all_positions("system_stop")
        
        logger.info("交易引擎已停止")
    
    def enable_trading(self):
        """启用交易"""
        self.is_trading_enabled = True
        logger.info("交易已启用")
    
    def disable_trading(self):
        """禁用交易"""
        self.is_trading_enabled = False
        logger.info("交易已禁用")
    
    def _data_collection_loop(self):
        """数据采集循环"""
        logger.info("数据采集循环启动")
        
        while self.is_running:
            try:
                # 获取实时数据
                realtime_data = self.data_collector.get_realtime_data()
                
                if realtime_data:
                    self.current_price = realtime_data['close']
                    
                    # 更新价格历史
                    self._update_price_history(realtime_data)
                    
                    # 计算当前特征
                    self._calculate_current_features()
                    
                    # 更新持仓盈亏
                    self._update_positions_pnl()
                
                # 等待下次数据更新
                time.sleep(self.data_collector.update_interval)
                
            except Exception as e:
                logger.error(f"数据采集循环出错: {e}")
                time.sleep(10)  # 出错后等待10秒
    
    def _trading_decision_loop(self):
        """交易决策循环"""
        logger.info("交易决策循环启动")
        
        while self.is_running:
            try:
                if not self.is_trading_enabled:
                    time.sleep(5)
                    continue
                
                # 检查风险限制
                if not self.risk_manager.check_daily_risk_limit():
                    logger.warning("达到日风险限制，停止交易")
                    self.disable_trading()
                    continue
                
                # 生成交易信号
                signal = self._generate_trading_signal()
                
                if signal:
                    # 记录信号
                    self.signal_history.append(signal)
                    
                    # 执行交易决策
                    self._execute_trading_decision(signal)
                
                # 检查持仓状态
                self._check_position_status()
                
                # 等待下次决策
                time.sleep(30)  # 30秒决策间隔
                
            except Exception as e:
                logger.error(f"交易决策循环出错: {e}")
                time.sleep(30)
    
    def _update_price_history(self, market_data: Dict):
        """更新价格历史"""
        try:
            # 添加新数据
            self.price_history.append({
                'timestamp': market_data['timestamp'],
                'open': market_data.get('open', market_data['close']),
                'high': market_data.get('high', market_data['close']),
                'low': market_data.get('low', market_data['close']),
                'close': market_data['close'],
                'volume': market_data.get('volume', 0)
            })
            
            # 保持历史长度限制
            if len(self.price_history) > self.max_history_length:
                self.price_history = self.price_history[-self.max_history_length:]
                
        except Exception as e:
            logger.error(f"更新价格历史失败: {e}")
    
    def _calculate_current_features(self):
        """计算当前特征"""
        try:
            if len(self.price_history) < 50:  # 需要足够的历史数据
                return
            
            # 转换为DataFrame
            df = pd.DataFrame(self.price_history)
            df.set_index('timestamp', inplace=True)
            
            # 计算特征
            feature_data = self.feature_engineer.create_feature_matrix(
                df, include_targets=False
            )
            
            if not feature_data.empty:
                # 获取最新特征
                latest_features = feature_data.iloc[-1]
                selected_features = self.ai_model_manager.feature_columns
                
                if all(col in latest_features.index for col in selected_features):
                    self.current_features = latest_features[selected_features].values
                else:
                    logger.debug("特征列不匹配")
                    
        except Exception as e:
            logger.debug(f"计算当前特征失败: {e}")
    
    def _generate_trading_signal(self) -> Optional[TradingSignal]:
        """生成交易信号"""
        try:
            if self.current_features is None or len(self.price_history) < 10:
                return None
            
            # 检查信号频率限制
            current_time = datetime.now()
            if (self.last_signal_time and 
                (current_time - self.last_signal_time).seconds < self.min_signal_interval):
                return None
            
            # 跳过包含NaN的特征
            if np.isnan(self.current_features).any():
                return None
            
            # AI预测
            prediction, details = self.ai_model_manager.predict_ensemble(
                self.current_features, method='weighted'
            )
            
            if prediction is None:
                return None
            
            # 计算置信度
            confidence = self._calculate_signal_confidence(details)
            
            # 置信度过低则不交易
            if confidence < 0.65:  # 65%置信度阈值
                return None
            
            # 确定信号类型
            signal_type = self._determine_signal_type(prediction, confidence)
            
            if signal_type == SignalType.HOLD:
                return None
            
            # 计算止损止盈
            stop_loss, take_profit = self._calculate_stop_levels(signal_type)
            
            # 创建交易信号
            signal = TradingSignal(
                timestamp=current_time,
                signal_type=signal_type,
                confidence=confidence,
                price=self.current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"AI预测: {prediction}, 置信度: {confidence:.3f}",
                model_predictions=details
            )
            
            self.last_signal_time = current_time
            
            logger.info(f"生成交易信号: {signal_type.value} @ {self.current_price:.2f}, 置信度: {confidence:.3f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"生成交易信号失败: {e}")
            return None
    
    def _calculate_signal_confidence(self, details: Dict) -> float:
        """计算信号置信度"""
        try:
            probabilities = details.get('individual_probabilities', {})
            
            if not probabilities:
                return 0.5
            
            # 计算各模型的平均置信度
            all_confidences = []
            
            for model_name, probs in probabilities.items():
                if isinstance(probs, (list, np.ndarray)) and len(probs) > 1:
                    # 获取模型性能权重
                    performance = self.ai_model_manager.model_performance.get(model_name, {})
                    weight = performance.get('test_accuracy', 0.5)
                    
                    # 计算该模型的置信度
                    model_confidence = max(probs) * weight
                    all_confidences.append(model_confidence)
            
            if all_confidences:
                return np.mean(all_confidences)
            else:
                return 0.5
                
        except Exception as e:
            logger.debug(f"计算信号置信度失败: {e}")
            return 0.5
    
    def _determine_signal_type(self, prediction: int, confidence: float) -> SignalType:
        """确定信号类型"""
        try:
            # 检查当前持仓状态
            current_positions = list(self.risk_manager.positions.keys())
            
            # 如果有持仓，考虑平仓信号
            if current_positions:
                # 简单策略：反向信号时平仓
                for position_id, position in self.risk_manager.positions.items():
                    if position['side'] == 'buy' and prediction == 0:
                        return SignalType.CLOSE_BUY
                    elif position['side'] == 'sell' and prediction == 1:
                        return SignalType.CLOSE_SELL
            
            # 无持仓时的开仓信号
            else:
                if prediction == 1:  # 看涨
                    return SignalType.BUY
                elif prediction == 0:  # 看跌
                    return SignalType.SELL
            
            return SignalType.HOLD
            
        except Exception as e:
            logger.error(f"确定信号类型失败: {e}")
            return SignalType.HOLD
    
    def _calculate_stop_levels(self, signal_type: SignalType) -> Tuple[float, float]:
        """计算止损止盈水平"""
        try:
            current_price = self.current_price
            
            if signal_type == SignalType.BUY:
                stop_loss = current_price - self.stop_loss_pips
                take_profit = current_price + self.take_profit_pips
            elif signal_type == SignalType.SELL:
                stop_loss = current_price + self.stop_loss_pips
                take_profit = current_price - self.take_profit_pips
            else:
                stop_loss = None
                take_profit = None
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"计算止损止盈失败: {e}")
            return None, None
    
    def _execute_trading_decision(self, signal: TradingSignal):
        """执行交易决策"""
        try:
            logger.info(f"执行交易决策: {signal.signal_type.value}")
            
            if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                # 开仓
                self._open_position(signal)
                
            elif signal.signal_type in [SignalType.CLOSE_BUY, SignalType.CLOSE_SELL]:
                # 平仓
                self._close_positions_by_side(
                    'buy' if signal.signal_type == SignalType.CLOSE_BUY else 'sell'
                )
                
        except Exception as e:
            logger.error(f"执行交易决策失败: {e}")
    
    def _open_position(self, signal: TradingSignal):
        """开仓"""
        try:
            side = 'buy' if signal.signal_type == SignalType.BUY else 'sell'
            
            # 风险检查
            can_trade, risk_msg, adjusted_size = self.risk_manager.check_position_risk(
                self.symbol, side, signal.price, signal.stop_loss, self.position_size
            )
            
            if not can_trade:
                logger.warning(f"风险检查失败，无法开仓: {risk_msg}")
                return
            
            # 添加持仓记录
            position_id = f"pos_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            position_data = {
                'symbol': self.symbol,
                'side': side,
                'size': adjusted_size,
                'entry_price': signal.price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit
            }
            
            self.risk_manager.add_position(position_id, position_data)
            
            logger.info(f"开仓成功: {position_id} {side} {adjusted_size} @ {signal.price:.2f}")
            
        except Exception as e:
            logger.error(f"开仓失败: {e}")
    
    def _close_positions_by_side(self, side: str):
        """按方向平仓"""
        try:
            positions_to_close = []
            
            for position_id, position in self.risk_manager.positions.items():
                if position['side'] == side:
                    positions_to_close.append(position_id)
            
            for position_id in positions_to_close:
                pnl = self.risk_manager.close_position(position_id, self.current_price)
                logger.info(f"平仓: {position_id}, 盈亏: ${pnl:.2f}")
                
        except Exception as e:
            logger.error(f"平仓失败: {e}")
    
    def _close_all_positions(self, reason: str):
        """平仓所有持仓"""
        try:
            position_ids = list(self.risk_manager.positions.keys())
            
            for position_id in position_ids:
                pnl = self.risk_manager.close_position(position_id, self.current_price)
                logger.info(f"平仓 ({reason}): {position_id}, 盈亏: ${pnl:.2f}")
                
        except Exception as e:
            logger.error(f"平仓所有持仓失败: {e}")
    
    def _update_positions_pnl(self):
        """更新持仓盈亏"""
        try:
            positions_to_close = []
            
            for position_id in list(self.risk_manager.positions.keys()):
                should_close = self.risk_manager.update_position_pnl(
                    position_id, self.current_price
                )
                
                if should_close:
                    positions_to_close.append(position_id)
            
            # 处理需要平仓的持仓
            for position_id in positions_to_close:
                pnl = self.risk_manager.close_position(position_id, self.current_price)
                logger.info(f"止损平仓: {position_id}, 盈亏: ${pnl:.2f}")
                
        except Exception as e:
            logger.error(f"更新持仓盈亏失败: {e}")
    
    def _check_position_status(self):
        """检查持仓状态"""
        try:
            # 获取风险汇总
            risk_summary = self.risk_manager.get_risk_summary()
            
            # 检查是否达到风险限制
            if not risk_summary['can_trade']:
                logger.warning("达到风险限制")
                self._close_all_positions("risk_limit")
                self.disable_trading()
            
            # 定期记录状态
            current_time = datetime.now()
            if current_time.minute % 15 == 0:  # 每15分钟记录一次
                logger.info(f"交易状态 - 持仓: {risk_summary['current_positions']}, "
                          f"日盈亏: ${risk_summary['daily_pnl']:.2f}")
                
        except Exception as e:
            logger.error(f"检查持仓状态失败: {e}")
    
    def get_trading_status(self) -> Dict:
        """获取交易状态"""
        try:
            risk_summary = self.risk_manager.get_risk_summary()
            
            status = {
                'engine_running': self.is_running,
                'trading_enabled': self.is_trading_enabled,
                'current_price': self.current_price,
                'current_positions': risk_summary['current_positions'],
                'daily_pnl': risk_summary['daily_pnl'],
                'total_pnl': risk_summary['total_pnl'],
                'can_trade': risk_summary['can_trade'],
                'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
                'signal_count': len(self.signal_history),
                'price_history_length': len(self.price_history)
            }
            
            return status
            
        except Exception as e:
            logger.error(f"获取交易状态失败: {e}")
            return {}
    
    def get_recent_signals(self, count: int = 10) -> List[Dict]:
        """获取最近的交易信号"""
        try:
            recent_signals = self.signal_history[-count:] if self.signal_history else []
            
            signal_list = []
            for signal in recent_signals:
                signal_dict = {
                    'timestamp': signal.timestamp.isoformat(),
                    'signal_type': signal.signal_type.value,
                    'confidence': signal.confidence,
                    'price': signal.price,
                    'reason': signal.reason
                }
                signal_list.append(signal_dict)
            
            return signal_list
            
        except Exception as e:
            logger.error(f"获取最近信号失败: {e}")
            return []
    
    def manual_trade(self, side: str, size: float = None) -> bool:
        """手动交易"""
        try:
            if not self.is_trading_enabled:
                logger.warning("交易已禁用，无法手动交易")
                return False
            
            if size is None:
                size = self.position_size
            
            # 创建手动信号
            signal_type = SignalType.BUY if side.lower() == 'buy' else SignalType.SELL
            stop_loss, take_profit = self._calculate_stop_levels(signal_type)
            
            manual_signal = TradingSignal(
                timestamp=datetime.now(),
                signal_type=signal_type,
                confidence=1.0,  # 手动交易置信度为100%
                price=self.current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason="手动交易"
            )
            
            # 执行交易
            self._execute_trading_decision(manual_signal)
            
            logger.info(f"手动交易执行: {side} {size} @ {self.current_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"手动交易失败: {e}")
            return False
    
    def emergency_stop(self):
        """紧急停止"""
        logger.critical("执行紧急停止")
        
        # 停止交易
        self.disable_trading()
        
        # 平仓所有持仓
        self._close_all_positions("emergency_stop")
        
        # 通知风险管理器
        self.risk_manager.emergency_stop("手动紧急停止")
        
        logger.critical("紧急停止执行完毕") 