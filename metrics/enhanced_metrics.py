"""
Enhanced metrics for vault analysis - additional trading statistics.
Uses actual timestamp data to calculate proper time-based metrics.
"""

import numpy as np
from typing import List, Dict, Tuple


def calculate_time_based_returns(pnl_history: List[List], account_value_history: List[List]) -> Tuple[List[float], List[float]]:
    """
    Calculate returns and time intervals from timestamp data.
    
    :param pnl_history: List of [timestamp, pnl_value] pairs
    :param account_value_history: List of [timestamp, account_value] pairs
    :return: (returns_list, time_intervals_in_days)
    """
    if len(pnl_history) < 2 or len(account_value_history) < 2:
        return [], []
    
    returns = []
    time_intervals = []
    
    for i in range(1, len(pnl_history)):
        prev_timestamp = pnl_history[i-1][0]
        curr_timestamp = pnl_history[i][0]
        
        prev_value = float(account_value_history[i-1][1])
        curr_value = float(account_value_history[i][1])
        
        if prev_value != 0:
            period_return = (curr_value - prev_value) / prev_value
            returns.append(period_return)
            
            # Calculate time interval in days
            time_interval_ms = curr_timestamp - prev_timestamp
            time_interval_days = time_interval_ms / (1000 * 60 * 60 * 24)  # Convert ms to days
            time_intervals.append(time_interval_days)
    
    return returns, time_intervals


def calculate_win_rate_with_timestamps(pnl_history: List[List], account_value_history: List[List]) -> float:
    """
    Calculate win rate (percentage of profitable periods) using timestamp data.
    
    :param pnl_history: List of [timestamp, pnl_value] pairs
    :param account_value_history: List of [timestamp, account_value] pairs
    :return: Win rate as percentage (0-100)
    """
    returns, _ = calculate_time_based_returns(pnl_history, account_value_history)
    
    if not returns:
        return 0.0
    
    winning_periods = sum(1 for ret in returns if ret > 0)
    win_rate = (winning_periods / len(returns)) * 100
    
    return round(win_rate, 2)


def calculate_profit_factor_with_timestamps(pnl_history: List[List], account_value_history: List[List]) -> float:
    """
    Calculate profit factor (total gains / total losses) using timestamp data.
    
    :param pnl_history: List of [timestamp, pnl_value] pairs
    :param account_value_history: List of [timestamp, account_value] pairs
    :return: Profit factor
    """
    returns, _ = calculate_time_based_returns(pnl_history, account_value_history)
    
    if not returns:
        return 0.0
    
    total_gains = sum(ret for ret in returns if ret > 0)
    total_losses = abs(sum(ret for ret in returns if ret < 0))
    
    if total_losses == 0:
        return float('inf') if total_gains > 0 else 0.0
    
    return round(total_gains / total_losses, 2)


def calculate_average_win_loss_with_timestamps(pnl_history: List[List], account_value_history: List[List]) -> Tuple[float, float]:
    """
    Calculate average win and average loss using timestamp data.
    
    :param pnl_history: List of [timestamp, pnl_value] pairs
    :param account_value_history: List of [timestamp, account_value] pairs
    :return: (average_win_pct, average_loss_pct)
    """
    returns, _ = calculate_time_based_returns(pnl_history, account_value_history)
    
    if not returns:
        return 0.0, 0.0
    
    wins = [ret for ret in returns if ret > 0]
    losses = [ret for ret in returns if ret < 0]
    
    avg_win = (sum(wins) / len(wins) * 100) if wins else 0.0
    avg_loss = (sum(losses) / len(losses) * 100) if losses else 0.0
    
    return round(avg_win, 2), round(abs(avg_loss), 2)


def calculate_consecutive_stats_with_timestamps(pnl_history: List[List], account_value_history: List[List]) -> Dict[str, int]:
    """
    Calculate maximum consecutive wins and losses using timestamp data.
    
    :param pnl_history: List of [timestamp, pnl_value] pairs
    :param account_value_history: List of [timestamp, account_value] pairs
    :return: Dictionary with consecutive win/loss stats
    """
    returns, _ = calculate_time_based_returns(pnl_history, account_value_history)
    
    if not returns:
        return {"max_consecutive_wins": 0, "max_consecutive_losses": 0}
    
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_consecutive_wins = 0
    current_consecutive_losses = 0
    
    for ret in returns:
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


def calculate_volatility_with_timestamps(pnl_history: List[List], account_value_history: List[List], annualize: bool = True) -> float:
    """
    Calculate volatility using actual time intervals from timestamp data.
    
    :param pnl_history: List of [timestamp, pnl_value] pairs
    :param account_value_history: List of [timestamp, account_value] pairs
    :param annualize: Whether to annualize the volatility
    :return: Volatility as percentage
    """
    returns, time_intervals = calculate_time_based_returns(pnl_history, account_value_history)
    
    if len(returns) < 2:
        return 0.0
    
    volatility = np.std(returns, ddof=1)
    
    if annualize and time_intervals:
        # Calculate average time interval and use it for proper annualization
        avg_interval_days = np.mean(time_intervals)
        if avg_interval_days > 0:
            # Cap the annualization to prevent extreme values from high-frequency data
            # If data is more frequent than daily, use daily volatility scaling
            if avg_interval_days < 1.0:
                # For intraday data, scale to daily first, then annualize
                daily_volatility = volatility * np.sqrt(1.0 / avg_interval_days)
                # Then annualize daily volatility (√252 for trading days)
                volatility = daily_volatility * np.sqrt(252)
            else:
                # For data with intervals >= 1 day, use normal annualization
                periods_per_year = 365 / avg_interval_days
                volatility *= np.sqrt(periods_per_year)
            
            # Cap maximum volatility at 500% to prevent unrealistic values
            volatility = min(volatility, 5.0)
    
    return round(float(volatility * 100), 2)


def calculate_calmar_ratio_with_timestamps(pnl_values: List[float], total_days: int) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown) using proper time period.
    
    :param pnl_values: List of reconstructed PnL values
    :param total_days: Total number of days since inception
    :return: Calmar ratio
    """
    if len(pnl_values) < 2 or total_days <= 0:
        return 0.0
    
    # Calculate annualized return using actual time period
    total_return = (pnl_values[-1] - pnl_values[0]) / pnl_values[0]
    annualized_return = (1 + total_return) ** (365 / total_days) - 1
    
    # Calculate max drawdown
    from .drawdown import calculate_max_drawdown_on_accountValue
    max_dd_pct = calculate_max_drawdown_on_accountValue(pnl_values)
    max_dd = max_dd_pct / 100
    
    # Handle edge cases and prevent extreme values
    if max_dd <= 0.001:  # If drawdown is less than 0.1%
        # Cap the Calmar ratio at 100 for very low drawdowns
        return 100.0 if annualized_return > 0 else 0.0
    
    calmar_ratio = annualized_return / max_dd
    
    # Cap maximum Calmar ratio at 100 to prevent extreme values
    calmar_ratio = min(abs(calmar_ratio), 100.0)
    
    # Return negative if annualized return is negative
    if annualized_return < 0:
        calmar_ratio = -calmar_ratio
    
    return round(float(calmar_ratio), 2)


def calculate_recovery_factor(pnl_values: List[float]) -> float:
    """
    Calculate recovery factor (total return / max drawdown).
    
    :param pnl_values: List of reconstructed PnL values
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
    
    :param pnl_values: List of reconstructed PnL values
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


def calculate_sterling_ratio_with_timestamps(pnl_values: List[float], total_days: int) -> float:
    """
    Calculate Sterling ratio (annualized return / average drawdown) using proper time period.
    
    :param pnl_values: List of reconstructed PnL values
    :param total_days: Total number of days since inception
    :return: Sterling ratio
    """
    if len(pnl_values) < 2 or total_days <= 0:
        return 0.0
    
    # Calculate annualized return using actual time period
    total_return = (pnl_values[-1] - pnl_values[0]) / pnl_values[0]
    annualized_return = (1 + total_return) ** (365 / total_days) - 1
    
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


def calculate_var_with_timestamps(pnl_history: List[List], account_value_history: List[List], confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) at specified confidence level using timestamp data.
    
    :param pnl_history: List of [timestamp, pnl_value] pairs
    :param account_value_history: List of [timestamp, account_value] pairs
    :param confidence_level: Confidence level (default 95%)
    :return: VaR as percentage
    """
    returns, _ = calculate_time_based_returns(pnl_history, account_value_history)
    
    if len(returns) < 2:
        return 0.0
    
    var = np.percentile(returns, (1 - confidence_level) * 100)
    return round(float(abs(var) * 100), 2)


def calculate_expectancy_with_timestamps(pnl_history: List[List], account_value_history: List[List], total_days: int) -> float:
    """
    Calculate expectancy (average return per period, annualized).
    
    :param pnl_history: List of [timestamp, pnl_value] pairs
    :param account_value_history: List of [timestamp, account_value] pairs
    :param total_days: Total number of days since inception
    :return: Expectancy as percentage (annualized)
    """
    returns, time_intervals = calculate_time_based_returns(pnl_history, account_value_history)
    
    if not returns or not time_intervals:
        return 0.0
    
    avg_return_per_period = sum(returns) / len(returns)
    avg_interval_days = np.mean(time_intervals)
    
    if avg_interval_days > 0:
        # Annualize the expectancy
        periods_per_year = 365 / avg_interval_days
        annualized_expectancy = avg_return_per_period * periods_per_year * 100
        return round(annualized_expectancy, 4)
    
    return 0.0


def calculate_all_enhanced_metrics_with_timestamps(
    pnl_history: List[List], 
    account_value_history: List[List], 
    rebuilded_pnl: List[float], 
    total_days: int
) -> Dict[str, float]:
    """
    Calculate all enhanced metrics using timestamp data for accurate time-based calculations.
    
    :param pnl_history: List of [timestamp, pnl_value] pairs from raw vault data
    :param account_value_history: List of [timestamp, account_value] pairs from raw vault data
    :param rebuilded_pnl: List of reconstructed PnL values for drawdown calculations
    :param total_days: Total number of days since inception
    :return: Dictionary with all calculated metrics
    """
    if not pnl_history or len(pnl_history) < 2:
        return {}
    
    avg_win, avg_loss = calculate_average_win_loss_with_timestamps(pnl_history, account_value_history)
    consecutive_stats = calculate_consecutive_stats_with_timestamps(pnl_history, account_value_history)
    
    metrics = {
        "Win Rate %": calculate_win_rate_with_timestamps(pnl_history, account_value_history),
        "Profit Factor": calculate_profit_factor_with_timestamps(pnl_history, account_value_history),
        "Avg Win %": avg_win,
        "Avg Loss %": avg_loss,
        "Max Consecutive Wins": consecutive_stats["max_consecutive_wins"],
        "Max Consecutive Losses": consecutive_stats["max_consecutive_losses"],
        "Volatility %": calculate_volatility_with_timestamps(pnl_history, account_value_history),
        "Calmar Ratio": calculate_calmar_ratio_with_timestamps(rebuilded_pnl, total_days),
        "Recovery Factor": calculate_recovery_factor(rebuilded_pnl),
        "Ulcer Index": calculate_ulcer_index(rebuilded_pnl),
        "Sterling Ratio": calculate_sterling_ratio_with_timestamps(rebuilded_pnl, total_days),
        "VaR 95% %": calculate_var_with_timestamps(pnl_history, account_value_history, 0.95),
        "Expectancy %": calculate_expectancy_with_timestamps(pnl_history, account_value_history, total_days),
    }
    
    return metrics


# Legacy function for backward compatibility (fallback to old calculation if timestamps not available)
def calculate_all_enhanced_metrics(pnl_values: List[float], days_since: int) -> Dict[str, float]:
    """
    Legacy function for backward compatibility. 
    This should not be used anymore as it assumes daily intervals.
    """
    # Return empty dict to signal that timestamp data is needed
    return {} 