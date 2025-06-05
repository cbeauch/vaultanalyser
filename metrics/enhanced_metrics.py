"""
Enhanced metrics for vault analysis - additional trading statistics.
"""

import numpy as np
from typing import List, Dict, Tuple


def calculate_daily_returns(pnl_values: List[float]) -> List[float]:
    """Calculate daily returns from PnL values."""
    if len(pnl_values) < 2:
        return []
    
    returns = []
    for i in range(1, len(pnl_values)):
        if pnl_values[i-1] != 0:
            daily_return = (pnl_values[i] - pnl_values[i-1]) / pnl_values[i-1]
            returns.append(daily_return)
    
    return returns


def calculate_win_rate(pnl_values: List[float]) -> float:
    """
    Calculate win rate (percentage of profitable periods).
    
    :param pnl_values: List of daily PnL values
    :return: Win rate as percentage (0-100)
    """
    daily_returns = calculate_daily_returns(pnl_values)
    
    if not daily_returns:
        return 0.0
    
    winning_days = sum(1 for ret in daily_returns if ret > 0)
    win_rate = (winning_days / len(daily_returns)) * 100
    
    return round(win_rate, 2)


def calculate_profit_factor(pnl_values: List[float]) -> float:
    """
    Calculate profit factor (total gains / total losses).
    
    :param pnl_values: List of daily PnL values
    :return: Profit factor
    """
    daily_returns = calculate_daily_returns(pnl_values)
    
    if not daily_returns:
        return 0.0
    
    total_gains = sum(ret for ret in daily_returns if ret > 0)
    total_losses = abs(sum(ret for ret in daily_returns if ret < 0))
    
    if total_losses == 0:
        return float('inf') if total_gains > 0 else 0.0
    
    return round(total_gains / total_losses, 2)


def calculate_average_win_loss(pnl_values: List[float]) -> Tuple[float, float]:
    """
    Calculate average win and average loss.
    
    :param pnl_values: List of daily PnL values
    :return: (average_win_pct, average_loss_pct)
    """
    daily_returns = calculate_daily_returns(pnl_values)
    
    if not daily_returns:
        return 0.0, 0.0
    
    wins = [ret for ret in daily_returns if ret > 0]
    losses = [ret for ret in daily_returns if ret < 0]
    
    avg_win = (sum(wins) / len(wins) * 100) if wins else 0.0
    avg_loss = (sum(losses) / len(losses) * 100) if losses else 0.0
    
    return round(avg_win, 2), round(abs(avg_loss), 2)


def calculate_consecutive_stats(pnl_values: List[float]) -> Dict[str, int]:
    """
    Calculate maximum consecutive wins and losses.
    
    :param pnl_values: List of daily PnL values
    :return: Dictionary with consecutive win/loss stats
    """
    daily_returns = calculate_daily_returns(pnl_values)
    
    if not daily_returns:
        return {"max_consecutive_wins": 0, "max_consecutive_losses": 0}
    
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_consecutive_wins = 0
    current_consecutive_losses = 0
    
    for ret in daily_returns:
        if ret > 0:
            current_consecutive_wins += 1
            current_consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_consecutive_wins)
        elif ret < 0:
            current_consecutive_losses += 1
            current_consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
        else:
            current_consecutive_wins = 0
            current_consecutive_losses = 0
    
    return {
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses
    }


def calculate_volatility(pnl_values: List[float], annualize: bool = True) -> float:
    """
    Calculate volatility (standard deviation of returns).
    
    :param pnl_values: List of daily PnL values
    :param annualize: Whether to annualize the volatility (multiply by sqrt(365))
    :return: Volatility as percentage
    """
    daily_returns = calculate_daily_returns(pnl_values)
    
    if len(daily_returns) < 2:
        return 0.0
    
    volatility = np.std(daily_returns, ddof=1)
    
    if annualize:
        volatility *= np.sqrt(365)  # Annualize assuming daily data
    
    return round(float(volatility * 100), 2)


def calculate_calmar_ratio(pnl_values: List[float], days_since: int) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    :param pnl_values: List of daily PnL values
    :param days_since: Number of days since inception
    :return: Calmar ratio
    """
    if len(pnl_values) < 2 or days_since <= 0:
        return 0.0
    
    # Calculate annualized return
    total_return = (pnl_values[-1] - pnl_values[0]) / pnl_values[0]
    annualized_return = (1 + total_return) ** (365 / days_since) - 1
    
    # Calculate max drawdown
    from .drawdown import calculate_max_drawdown_on_accountValue
    max_dd = calculate_max_drawdown_on_accountValue(pnl_values) / 100
    
    if max_dd == 0:
        return float('inf') if annualized_return > 0 else 0.0
    
    return round(annualized_return / max_dd, 2)


def calculate_recovery_factor(pnl_values: List[float]) -> float:
    """
    Calculate recovery factor (total return / max drawdown).
    
    :param pnl_values: List of daily PnL values
    :return: Recovery factor
    """
    if len(pnl_values) < 2:
        return 0.0
    
    total_return = (pnl_values[-1] - pnl_values[0]) / pnl_values[0] * 100
    
    from .drawdown import calculate_max_drawdown_on_accountValue
    max_dd = calculate_max_drawdown_on_accountValue(pnl_values)
    
    if max_dd == 0:
        return float('inf') if total_return > 0 else 0.0
    
    return round(total_return / max_dd, 2)


def calculate_ulcer_index(pnl_values: List[float]) -> float:
    """
    Calculate Ulcer Index (measure of downside risk).
    
    :param pnl_values: List of daily PnL values
    :return: Ulcer Index as percentage
    """
    if len(pnl_values) < 2:
        return 0.0
    
    running_max = pnl_values[0]
    drawdown_squares = []
    
    for value in pnl_values:
        running_max = max(running_max, value)
        drawdown_pct = ((running_max - value) / running_max) * 100 if running_max > 0 else 0
        drawdown_squares.append(drawdown_pct ** 2)
    
    ulcer_index = (sum(drawdown_squares) / len(drawdown_squares)) ** 0.5
    return round(ulcer_index, 2)


def calculate_sterling_ratio(pnl_values: List[float], days_since: int) -> float:
    """
    Calculate Sterling ratio (annualized return / average drawdown).
    
    :param pnl_values: List of daily PnL values
    :param days_since: Number of days since inception
    :return: Sterling ratio
    """
    if len(pnl_values) < 2 or days_since <= 0:
        return 0.0
    
    # Calculate annualized return
    total_return = (pnl_values[-1] - pnl_values[0]) / pnl_values[0]
    annualized_return = (1 + total_return) ** (365 / days_since) - 1
    
    # Calculate average drawdown
    running_max = pnl_values[0]
    drawdowns = []
    
    for value in pnl_values:
        running_max = max(running_max, value)
        drawdown_pct = ((running_max - value) / running_max) * 100 if running_max > 0 else 0
        drawdowns.append(drawdown_pct)
    
    avg_drawdown = sum(drawdowns) / len(drawdowns) if drawdowns else 0
    
    if avg_drawdown == 0:
        return float('inf') if annualized_return > 0 else 0.0
    
    return round((annualized_return * 100) / avg_drawdown, 2)


def calculate_daily_var(pnl_values: List[float], confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) at specified confidence level.
    
    :param pnl_values: List of daily PnL values
    :param confidence_level: Confidence level (default 95%)
    :return: VaR as percentage
    """
    daily_returns = calculate_daily_returns(pnl_values)
    
    if len(daily_returns) < 2:
        return 0.0
    
    var = np.percentile(daily_returns, (1 - confidence_level) * 100)
    return round(float(abs(var) * 100), 2)


def calculate_expectancy(pnl_values: List[float]) -> float:
    """
    Calculate expectancy (average return per trade).
    
    :param pnl_values: List of daily PnL values
    :return: Expectancy as percentage
    """
    daily_returns = calculate_daily_returns(pnl_values)
    
    if not daily_returns:
        return 0.0
    
    expectancy = sum(daily_returns) / len(daily_returns) * 100
    return round(expectancy, 4)


def calculate_all_enhanced_metrics(pnl_values: List[float], days_since: int) -> Dict[str, float]:
    """
    Calculate all enhanced metrics at once.
    
    :param pnl_values: List of daily PnL values
    :param days_since: Number of days since inception
    :return: Dictionary with all calculated metrics
    """
    if not pnl_values or len(pnl_values) < 2:
        return {}
    
    avg_win, avg_loss = calculate_average_win_loss(pnl_values)
    consecutive_stats = calculate_consecutive_stats(pnl_values)
    
    metrics = {
        "Win Rate %": calculate_win_rate(pnl_values),
        "Profit Factor": calculate_profit_factor(pnl_values),
        "Avg Win %": avg_win,
        "Avg Loss %": avg_loss,
        "Max Consecutive Wins": consecutive_stats["max_consecutive_wins"],
        "Max Consecutive Losses": consecutive_stats["max_consecutive_losses"],
        "Volatility %": calculate_volatility(pnl_values),
        "Calmar Ratio": calculate_calmar_ratio(pnl_values, days_since),
        "Recovery Factor": calculate_recovery_factor(pnl_values),
        "Ulcer Index": calculate_ulcer_index(pnl_values),
        "Sterling Ratio": calculate_sterling_ratio(pnl_values, days_since),
        "VaR 95% %": calculate_daily_var(pnl_values, 0.95),
        "Expectancy %": calculate_expectancy(pnl_values),
    }
    
    return metrics 