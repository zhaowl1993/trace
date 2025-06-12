"""
风险管理模块
负责控制交易风险，确保不超过设定的最大损失限额
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import json


class RiskManager:
    """风险管理器 - 控制交易风险"""
    
    def __init__(self, config: Dict):
        """
        初始化风险管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.max_daily_loss = config['risk_management']['max_daily_loss']  # 30美元
        self.max_positions = config['risk_management']['max_positions']
        self.risk_per_trade = config['risk_management']['risk_per_trade']
        self.drawdown_limit = config['risk_management']['drawdown_limit']
        
        # 交易记录
        self.positions = {}  # 当前持仓
        self.daily_pnl = 0.0  # 当日盈亏
        self.total_pnl = 0.0  # 总盈亏
        self.peak_equity = 0.0  # 权益峰值
        self.current_drawdown = 0.0  # 当前回撤
        
        # 风险状态
        self.risk_status = {
            'daily_loss_exceeded': False,
            'max_positions_reached': False,
            'drawdown_exceeded': False,
            'emergency_stop': False
        }
        
        logger.info(f"风险管理器初始化完成 - 最大日损失: ${self.max_daily_loss}")
    
    def check_position_risk(self, symbol: str, side: str, 
                          entry_price: float, stop_loss: float,
                          position_size: float) -> Tuple[bool, str, float]:
        """
        检查单笔交易风险
        
        Args:
            symbol: 交易品种
            side: 交易方向 ('buy' or 'sell')
            entry_price: 入场价格
            stop_loss: 止损价格
            position_size: 仓位大小
            
        Returns:
            (是否通过风险检查, 风险信息, 调整后的仓位大小)
        """
        try:
            # 计算单笔潜在损失
            if side.lower() == 'buy':
                potential_loss = (entry_price - stop_loss) * position_size
            else:
                potential_loss = (stop_loss - entry_price) * position_size
            
            potential_loss = abs(potential_loss)
            
            # 检查单笔损失是否超过限制
            max_risk_per_trade = self.max_daily_loss * self.risk_per_trade
            if potential_loss > max_risk_per_trade:
                # 调整仓位大小
                if side.lower() == 'buy':
                    adjusted_size = max_risk_per_trade / abs(entry_price - stop_loss)
                else:
                    adjusted_size = max_risk_per_trade / abs(stop_loss - entry_price)
                
                logger.warning(f"仓位风险过大，调整仓位: {position_size:.4f} -> {adjusted_size:.4f}")
                return True, f"仓位已调整至风险限制内", adjusted_size
            
            # 检查是否会导致日损失超限
            if self.daily_pnl - potential_loss < -self.max_daily_loss:
                return False, f"交易可能导致日损失超过${self.max_daily_loss}限制", 0.0
            
            # 检查最大持仓数
            if len(self.positions) >= self.max_positions:
                return False, f"已达到最大持仓数限制: {self.max_positions}", 0.0
            
            logger.info(f"交易风险检查通过 - 潜在损失: ${potential_loss:.2f}")
            return True, "风险检查通过", position_size
            
        except Exception as e:
            logger.error(f"检查交易风险失败: {e}")
            return False, f"风险检查出错: {e}", 0.0
    
    def check_daily_risk_limit(self) -> bool:
        """
        检查是否达到日风险限制
        
        Returns:
            是否可以继续交易
        """
        if self.daily_pnl <= -self.max_daily_loss:
            self.risk_status['daily_loss_exceeded'] = True
            logger.error(f"已达到日最大损失限制: ${abs(self.daily_pnl):.2f}/${self.max_daily_loss}")
            return False
        
        return True
    
    def calculate_position_size(self, account_balance: float,
                              entry_price: float, stop_loss: float,
                              risk_percentage: float = None) -> float:
        """
        计算合理的仓位大小
        
        Args:
            account_balance: 账户余额
            entry_price: 入场价格
            stop_loss: 止损价格
            risk_percentage: 风险百分比
            
        Returns:
            建议的仓位大小
        """
        try:
            if risk_percentage is None:
                risk_percentage = self.risk_per_trade
            
            # 可承受的损失金额
            risk_amount = account_balance * risk_percentage
            
            # 确保不超过日风险限制
            remaining_daily_risk = self.max_daily_loss + self.daily_pnl
            risk_amount = min(risk_amount, remaining_daily_risk)
            
            # 计算每点价值对应的仓位
            price_difference = abs(entry_price - stop_loss)
            if price_difference == 0:
                return 0.0
            
            position_size = risk_amount / price_difference
            
            logger.info(f"计算仓位大小: ${risk_amount:.2f} 风险 -> {position_size:.4f} 仓位")
            return position_size
            
        except Exception as e:
            logger.error(f"计算仓位大小失败: {e}")
            return 0.0
    
    def add_position(self, position_id: str, position_data: Dict):
        """
        添加持仓记录
        
        Args:
            position_id: 持仓ID
            position_data: 持仓数据
        """
        try:
            self.positions[position_id] = {
                'symbol': position_data['symbol'],
                'side': position_data['side'],
                'size': position_data['size'],
                'entry_price': position_data['entry_price'],
                'stop_loss': position_data.get('stop_loss', 0),
                'take_profit': position_data.get('take_profit', 0),
                'timestamp': datetime.now(),
                'unrealized_pnl': 0.0
            }
            
            logger.info(f"添加持仓: {position_id} - {position_data['symbol']}")
            
        except Exception as e:
            logger.error(f"添加持仓失败: {e}")
    
    def update_position_pnl(self, position_id: str, current_price: float):
        """
        更新持仓盈亏
        
        Args:
            position_id: 持仓ID
            current_price: 当前价格
        """
        try:
            if position_id not in self.positions:
                return
            
            position = self.positions[position_id]
            entry_price = position['entry_price']
            size = position['size']
            side = position['side']
            
            if side.lower() == 'buy':
                unrealized_pnl = (current_price - entry_price) * size
            else:
                unrealized_pnl = (entry_price - current_price) * size
            
            position['unrealized_pnl'] = unrealized_pnl
            
            # 检查止损
            if self._should_trigger_stop_loss(position, current_price):
                logger.warning(f"持仓 {position_id} 触发止损")
                return True  # 需要平仓
            
            return False
            
        except Exception as e:
            logger.error(f"更新持仓盈亏失败: {e}")
            return False
    
    def close_position(self, position_id: str, exit_price: float) -> float:
        """
        平仓并计算实际盈亏
        
        Args:
            position_id: 持仓ID
            exit_price: 平仓价格
            
        Returns:
            实际盈亏
        """
        try:
            if position_id not in self.positions:
                return 0.0
            
            position = self.positions[position_id]
            entry_price = position['entry_price']
            size = position['size']
            side = position['side']
            
            if side.lower() == 'buy':
                realized_pnl = (exit_price - entry_price) * size
            else:
                realized_pnl = (entry_price - exit_price) * size
            
            # 更新盈亏统计
            self.daily_pnl += realized_pnl
            self.total_pnl += realized_pnl
            
            # 更新权益峰值和回撤
            current_equity = self.total_pnl
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            
            # 移除持仓
            del self.positions[position_id]
            
            logger.info(f"平仓 {position_id}: ${realized_pnl:.2f} | 日盈亏: ${self.daily_pnl:.2f}")
            
            return realized_pnl
            
        except Exception as e:
            logger.error(f"平仓失败: {e}")
            return 0.0
    
    def _should_trigger_stop_loss(self, position: Dict, current_price: float) -> bool:
        """
        检查是否应该触发止损
        
        Args:
            position: 持仓信息
            current_price: 当前价格
            
        Returns:
            是否触发止损
        """
        if position['stop_loss'] == 0:
            return False
        
        if position['side'].lower() == 'buy':
            return current_price <= position['stop_loss']
        else:
            return current_price >= position['stop_loss']
    
    def check_drawdown_limit(self) -> bool:
        """
        检查回撤限制
        
        Returns:
            是否超过回撤限制
        """
        if self.current_drawdown > self.drawdown_limit:
            self.risk_status['drawdown_exceeded'] = True
            logger.error(f"回撤超过限制: {self.current_drawdown:.2%} > {self.drawdown_limit:.2%}")
            return False
        
        return True
    
    def get_risk_summary(self) -> Dict:
        """
        获取风险状况汇总
        
        Returns:
            风险状况字典
        """
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        
        return {
            'timestamp': datetime.now().isoformat(),
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'current_positions': len(self.positions),
            'max_positions': self.max_positions,
            'current_drawdown': self.current_drawdown,
            'drawdown_limit': self.drawdown_limit,
            'daily_loss_limit': self.max_daily_loss,
            'risk_status': self.risk_status.copy(),
            'can_trade': self._can_continue_trading()
        }
    
    def _can_continue_trading(self) -> bool:
        """
        检查是否可以继续交易
        
        Returns:
            是否可以继续交易
        """
        # 检查各项风险限制
        if self.risk_status['daily_loss_exceeded']:
            return False
        
        if self.risk_status['drawdown_exceeded']:
            return False
        
        if self.risk_status['emergency_stop']:
            return False
        
        if len(self.positions) >= self.max_positions:
            self.risk_status['max_positions_reached'] = True
            return False
        
        return True
    
    def reset_daily_stats(self):
        """重置日统计数据"""
        self.daily_pnl = 0.0
        self.risk_status['daily_loss_exceeded'] = False
        logger.info("日统计数据已重置")
    
    def emergency_stop(self, reason: str):
        """
        紧急停止交易
        
        Args:
            reason: 停止原因
        """
        self.risk_status['emergency_stop'] = True
        logger.critical(f"紧急停止交易: {reason}")
    
    def save_risk_log(self, file_path: str):
        """
        保存风险日志
        
        Args:
            file_path: 日志文件路径
        """
        try:
            risk_data = {
                'timestamp': datetime.now().isoformat(),
                'risk_summary': self.get_risk_summary(),
                'positions': self.positions.copy()
            }
            
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(risk_data, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"保存风险日志失败: {e}") 