"""
纸上交易系统
实现完整的模拟交易环境，无真实资金风险
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import uuid
import json
import threading
import time


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"      # 市价单
    LIMIT = "limit"        # 限价单
    STOP = "stop"          # 止损单
    STOP_LIMIT = "stop_limit"  # 止损限价单


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"        # 等待中
    FILLED = "filled"          # 已成交
    PARTIALLY_FILLED = "partially_filled"  # 部分成交
    CANCELLED = "cancelled"    # 已取消
    REJECTED = "rejected"      # 已拒绝
    EXPIRED = "expired"        # 已过期


@dataclass
class Order:
    """订单对象"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: str = ""  # "buy" or "sell"
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    client_order_id: str = ""
    tag: str = ""  # 用于标识订单来源


@dataclass
class Position:
    """持仓对象"""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    side: str = "long"  # "long" or "short"
    cost_basis: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class Account:
    """账户对象"""
    account_id: str
    buying_power: float = 10000.0  # 购买力
    cash: float = 10000.0          # 现金
    portfolio_value: float = 10000.0  # 总资产
    day_trading_buying_power: float = 10000.0
    maintenance_margin: float = 0.0
    initial_margin: float = 0.0
    equity: float = 10000.0
    last_equity: float = 10000.0
    multiplier: int = 1
    currency: str = "USD"


class PaperTradingEngine:
    """纸上交易引擎"""
    
    def __init__(self, config: Dict):
        """
        初始化纸上交易引擎
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.account = Account(
            account_id="PAPER_ACCOUNT_001",
            buying_power=config.get('initial_capital', 10000.0),
            cash=config.get('initial_capital', 10000.0),
            portfolio_value=config.get('initial_capital', 10000.0),
            equity=config.get('initial_capital', 10000.0),
            last_equity=config.get('initial_capital', 10000.0)
        )
        
        # 订单和持仓管理
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.order_history: List[Order] = []
        
        # 市场数据
        self.current_prices: Dict[str, float] = {}
        self.price_history: Dict[str, List[Dict]] = {}
        
        # 交易参数
        self.commission_rate = config.get('commission', 0.0001)
        self.slippage_rate = config.get('slippage', 0.0002)
        self.margin_requirement = config.get('margin_requirement', 0.05)
        
        # 运行状态
        self.is_running = False
        self.last_update = datetime.now()
        
        # 性能统计
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_commission = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = self.account.equity
        
        logger.info(f"纸上交易引擎初始化完成 - 初始资金: ${self.account.cash:.2f}")
    
    def submit_order(self, symbol: str, side: str, quantity: float,
                    order_type: OrderType = OrderType.MARKET,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    tag: str = "") -> str:
        """
        提交订单
        
        Args:
            symbol: 交易品种
            side: 交易方向 ("buy" or "sell")
            quantity: 数量
            order_type: 订单类型
            price: 限价单价格
            stop_price: 止损价格
            tag: 订单标签
            
        Returns:
            订单ID
        """
        try:
            # 创建订单
            order = Order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                tag=tag,
                client_order_id=f"client_{len(self.orders) + 1}"
            )
            
            # 验证订单
            if not self._validate_order(order):
                order.status = OrderStatus.REJECTED
                logger.warning(f"订单被拒绝: {order.id}")
                return order.id
            
            # 添加到订单簿
            self.orders[order.id] = order
            
            # 如果是市价单，立即执行
            if order_type == OrderType.MARKET:
                self._execute_market_order(order)
            
            logger.info(f"订单已提交: {order.id} {side} {quantity} {symbol}")
            return order.id
            
        except Exception as e:
            logger.error(f"提交订单失败: {e}")
            return ""
    
    def cancel_order(self, order_id: str) -> bool:
        """
        取消订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            是否成功取消
        """
        try:
            if order_id not in self.orders:
                logger.warning(f"订单不存在: {order_id}")
                return False
            
            order = self.orders[order_id]
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                logger.warning(f"订单无法取消: {order_id}, 状态: {order.status}")
                return False
            
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            
            logger.info(f"订单已取消: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单信息"""
        return self.orders.get(order_id)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取持仓信息"""
        return self.positions.get(symbol)
    
    def get_account(self) -> Account:
        """获取账户信息"""
        return self.account
    
    def update_market_data(self, symbol: str, price: float, timestamp: datetime = None):
        """
        更新市场数据
        
        Args:
            symbol: 交易品种
            price: 当前价格
            timestamp: 时间戳
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            self.current_prices[symbol] = price
            
            # 记录价格历史
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append({
                'timestamp': timestamp,
                'price': price
            })
            
            # 保持历史长度限制
            if len(self.price_history[symbol]) > 1000:
                self.price_history[symbol] = self.price_history[symbol][-1000:]
            
            # 更新持仓市值
            self._update_positions_market_value()
            
            # 处理待执行订单
            self._process_pending_orders()
            
            # 更新账户信息
            self._update_account()
            
        except Exception as e:
            logger.error(f"更新市场数据失败: {e}")
    
    def _validate_order(self, order: Order) -> bool:
        """验证订单有效性"""
        try:
            # 检查基本参数
            if order.quantity <= 0:
                logger.warning("订单数量必须大于0")
                return False
            
            if order.symbol not in self.current_prices and order.order_type == OrderType.MARKET:
                logger.warning(f"缺少市场数据: {order.symbol}")
                return False
            
            # 检查购买力
            if order.side == "buy":
                estimated_cost = self._estimate_order_cost(order)
                if estimated_cost > self.account.buying_power:
                    logger.warning(f"购买力不足: 需要${estimated_cost:.2f}, 可用${self.account.buying_power:.2f}")
                    return False
            
            # 检查持仓（卖出时）
            if order.side == "sell":
                position = self.positions.get(order.symbol)
                if not position or position.quantity < order.quantity:
                    available_qty = position.quantity if position else 0
                    logger.warning(f"持仓不足: 需要{order.quantity}, 可用{available_qty}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"订单验证失败: {e}")
            return False
    
    def _estimate_order_cost(self, order: Order) -> float:
        """估算订单成本"""
        try:
            if order.order_type == OrderType.MARKET:
                price = self.current_prices.get(order.symbol, 0)
            else:
                price = order.price or 0
            
            cost = price * order.quantity
            commission = cost * self.commission_rate
            
            return cost + commission
            
        except Exception as e:
            logger.error(f"估算订单成本失败: {e}")
            return 0.0
    
    def _execute_market_order(self, order: Order):
        """执行市价单"""
        try:
            if order.symbol not in self.current_prices:
                order.status = OrderStatus.REJECTED
                logger.warning(f"无法执行市价单，缺少市场数据: {order.symbol}")
                return
            
            # 获取执行价格（考虑滑点）
            market_price = self.current_prices[order.symbol]
            if order.side == "buy":
                execution_price = market_price * (1 + self.slippage_rate)
            else:
                execution_price = market_price * (1 - self.slippage_rate)
            
            # 执行交易
            self._fill_order(order, execution_price, order.quantity)
            
        except Exception as e:
            logger.error(f"执行市价单失败: {e}")
            order.status = OrderStatus.REJECTED
    
    def _fill_order(self, order: Order, fill_price: float, fill_quantity: float):
        """成交订单"""
        try:
            # 计算手续费
            trade_value = fill_price * fill_quantity
            commission = trade_value * self.commission_rate
            
            # 更新订单状态
            order.filled_quantity += fill_quantity
            order.filled_price = fill_price
            order.commission += commission
            order.filled_at = datetime.now()
            order.updated_at = datetime.now()
            
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
            
            # 更新持仓
            self._update_position(order.symbol, order.side, fill_quantity, fill_price, commission)
            
            # 更新账户
            if order.side == "buy":
                self.account.cash -= (trade_value + commission)
            else:
                self.account.cash += (trade_value - commission)
            
            self.total_commission += commission
            self.trade_count += 1
            
            # 添加到历史记录
            if order.status == OrderStatus.FILLED:
                self.order_history.append(order)
            
            logger.info(f"订单成交: {order.id} {fill_quantity}@{fill_price:.2f}, 手续费: ${commission:.2f}")
            
        except Exception as e:
            logger.error(f"订单成交处理失败: {e}")
    
    def _update_position(self, symbol: str, side: str, quantity: float, price: float, commission: float):
        """更新持仓"""
        try:
            if symbol not in self.positions:
                self.positions[symbol] = Position(symbol=symbol)
            
            position = self.positions[symbol]
            
            if side == "buy":
                # 买入
                total_cost = position.avg_price * position.quantity + price * quantity + commission
                position.quantity += quantity
                position.avg_price = total_cost / position.quantity if position.quantity > 0 else 0
                position.cost_basis += price * quantity + commission
                
            else:
                # 卖出
                if position.quantity >= quantity:
                    # 计算已实现盈亏
                    realized_pnl = (price - position.avg_price) * quantity - commission
                    position.realized_pnl += realized_pnl
                    position.quantity -= quantity
                    position.cost_basis -= position.avg_price * quantity
                    
                    # 更新胜负统计
                    if realized_pnl > 0:
                        self.win_count += 1
                    elif realized_pnl < 0:
                        self.loss_count += 1
                else:
                    logger.warning(f"卖出数量超过持仓: {symbol}")
            
            # 如果持仓为0，清除记录
            if position.quantity <= 0.001:  # 考虑浮点数精度
                if symbol in self.positions:
                    del self.positions[symbol]
            else:
                position.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"更新持仓失败: {e}")
    
    def _update_positions_market_value(self):
        """更新持仓市值"""
        try:
            for symbol, position in self.positions.items():
                if symbol in self.current_prices:
                    current_price = self.current_prices[symbol]
                    position.market_value = position.quantity * current_price
                    position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                    
        except Exception as e:
            logger.error(f"更新持仓市值失败: {e}")
    
    def _process_pending_orders(self):
        """处理待执行订单"""
        try:
            for order_id, order in list(self.orders.items()):
                if order.status != OrderStatus.PENDING:
                    continue
                
                if order.symbol not in self.current_prices:
                    continue
                
                current_price = self.current_prices[order.symbol]
                
                # 处理限价单
                if order.order_type == OrderType.LIMIT:
                    should_fill = False
                    if order.side == "buy" and current_price <= order.price:
                        should_fill = True
                    elif order.side == "sell" and current_price >= order.price:
                        should_fill = True
                    
                    if should_fill:
                        self._fill_order(order, order.price, order.quantity)
                
                # 处理止损单
                elif order.order_type == OrderType.STOP:
                    should_trigger = False
                    if order.side == "buy" and current_price >= order.stop_price:
                        should_trigger = True
                    elif order.side == "sell" and current_price <= order.stop_price:
                        should_trigger = True
                    
                    if should_trigger:
                        # 转为市价单执行
                        execution_price = current_price
                        if order.side == "buy":
                            execution_price *= (1 + self.slippage_rate)
                        else:
                            execution_price *= (1 - self.slippage_rate)
                        
                        self._fill_order(order, execution_price, order.quantity)
                
        except Exception as e:
            logger.error(f"处理待执行订单失败: {e}")
    
    def _update_account(self):
        """更新账户信息"""
        try:
            # 计算总持仓市值
            total_position_value = sum(pos.market_value for pos in self.positions.values())
            
            # 更新账户权益
            self.account.equity = self.account.cash + total_position_value
            self.account.portfolio_value = self.account.equity
            
            # 计算购买力（简化版本）
            used_margin = sum(pos.market_value * self.margin_requirement for pos in self.positions.values())
            self.account.buying_power = max(0, self.account.cash - used_margin)
            
            # 更新最大回撤
            if self.account.equity > self.peak_equity:
                self.peak_equity = self.account.equity
            else:
                current_drawdown = (self.peak_equity - self.account.equity) / self.peak_equity
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"更新账户信息失败: {e}")
    
    def get_portfolio_summary(self) -> Dict:
        """获取投资组合摘要"""
        try:
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            
            summary = {
                'account_id': self.account.account_id,
                'cash': self.account.cash,
                'equity': self.account.equity,
                'portfolio_value': self.account.portfolio_value,
                'buying_power': self.account.buying_power,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_realized_pnl': total_realized_pnl,
                'total_pnl': total_unrealized_pnl + total_realized_pnl,
                'total_commission': self.total_commission,
                'max_drawdown': self.max_drawdown,
                'trade_count': self.trade_count,
                'win_count': self.win_count,
                'loss_count': self.loss_count,
                'win_rate': self.win_count / max(1, self.trade_count),
                'positions_count': len(self.positions),
                'active_orders_count': len([o for o in self.orders.values() if o.status == OrderStatus.PENDING]),
                'last_update': self.last_update.isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"获取投资组合摘要失败: {e}")
            return {}
    
    def get_positions_list(self) -> List[Dict]:
        """获取持仓列表"""
        try:
            positions_list = []
            
            for symbol, position in self.positions.items():
                current_price = self.current_prices.get(symbol, 0)
                
                position_dict = {
                    'symbol': position.symbol,
                    'quantity': position.quantity,
                    'avg_price': position.avg_price,
                    'current_price': current_price,
                    'market_value': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl,
                    'cost_basis': position.cost_basis,
                    'pnl_percentage': (position.unrealized_pnl / position.cost_basis * 100) if position.cost_basis > 0 else 0,
                    'last_update': position.last_update.isoformat()
                }
                
                positions_list.append(position_dict)
            
            return positions_list
            
        except Exception as e:
            logger.error(f"获取持仓列表失败: {e}")
            return []
    
    def get_orders_list(self, status: Optional[OrderStatus] = None) -> List[Dict]:
        """获取订单列表"""
        try:
            orders_list = []
            
            for order in self.orders.values():
                if status and order.status != status:
                    continue
                
                order_dict = {
                    'id': order.id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'order_type': order.order_type.value,
                    'quantity': order.quantity,
                    'price': order.price,
                    'stop_price': order.stop_price,
                    'status': order.status.value,
                    'filled_quantity': order.filled_quantity,
                    'filled_price': order.filled_price,
                    'commission': order.commission,
                    'created_at': order.created_at.isoformat(),
                    'updated_at': order.updated_at.isoformat(),
                    'filled_at': order.filled_at.isoformat() if order.filled_at else None,
                    'tag': order.tag
                }
                
                orders_list.append(order_dict)
            
            return orders_list
            
        except Exception as e:
            logger.error(f"获取订单列表失败: {e}")
            return []
    
    def reset_account(self):
        """重置账户（用于重新开始模拟）"""
        try:
            initial_capital = self.config.get('initial_capital', 10000.0)
            
            self.account = Account(
                account_id=self.account.account_id,
                buying_power=initial_capital,
                cash=initial_capital,
                portfolio_value=initial_capital,
                equity=initial_capital,
                last_equity=initial_capital
            )
            
            self.orders.clear()
            self.positions.clear()
            self.order_history.clear()
            self.current_prices.clear()
            self.price_history.clear()
            
            self.trade_count = 0
            self.win_count = 0
            self.loss_count = 0
            self.total_commission = 0.0
            self.max_drawdown = 0.0
            self.peak_equity = initial_capital
            
            logger.info(f"账户已重置 - 初始资金: ${initial_capital:.2f}")
            
        except Exception as e:
            logger.error(f"重置账户失败: {e}")
    
    def export_trading_history(self) -> Dict:
        """导出交易历史"""
        try:
            history = {
                'account_summary': self.get_portfolio_summary(),
                'orders_history': [
                    {
                        'id': order.id,
                        'symbol': order.symbol,
                        'side': order.side,
                        'order_type': order.order_type.value,
                        'quantity': order.quantity,
                        'filled_price': order.filled_price,
                        'commission': order.commission,
                        'filled_at': order.filled_at.isoformat() if order.filled_at else None,
                        'tag': order.tag
                    }
                    for order in self.order_history
                ],
                'positions_history': self.get_positions_list(),
                'performance_metrics': {
                    'total_trades': self.trade_count,
                    'winning_trades': self.win_count,
                    'losing_trades': self.loss_count,
                    'win_rate': self.win_count / max(1, self.trade_count),
                    'total_commission': self.total_commission,
                    'max_drawdown': self.max_drawdown,
                    'return_percentage': ((self.account.equity - self.config.get('initial_capital', 10000)) / self.config.get('initial_capital', 10000)) * 100
                }
            }
            
            return history
            
        except Exception as e:
            logger.error(f"导出交易历史失败: {e}")
            return {} 