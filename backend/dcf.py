import numpy as np

def calculate_dcf(
    fcf: float,
    growth_rate: float,
    discount_rate: float,
    terminal_growth_rate: float,
    years: int = 5,
    shares_outstanding: float = 1.0,
    net_debt: float = 0.0
) -> dict:
    """
    Calculate the intrinsic value of a stock using the Discounted Cash Flow (DCF) model.
    
    Args:
        fcf (float): Free Cash Flow for the most recent year.
        growth_rate (float): Expected annual growth rate of FCF (decimal, e.g., 0.05 for 5%).
        discount_rate (float): Discount rate (WACC) (decimal).
        terminal_growth_rate (float): Terminal growth rate (decimal).
        years (int): Number of years to project.
        shares_outstanding (float): Number of shares outstanding.
        net_debt (float): Total Debt - Cash & Equivalents.
        
    Returns:
        dict: Breakdown of the DCF calculation.
    """
    
    future_cash_flows = []
    discounted_cash_flows = []
    
    current_fcf = fcf
    
    # Calculate projected FCF and discount them
    for i in range(1, years + 1):
        projected_fcf = current_fcf * (1 + growth_rate)
        discounted_fcf = projected_fcf / ((1 + discount_rate) ** i)
        
        future_cash_flows.append(projected_fcf)
        discounted_cash_flows.append(discounted_fcf)
        
        current_fcf = projected_fcf
        
    # Terminal Value
    terminal_value = (future_cash_flows[-1] * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
    discounted_terminal_value = terminal_value / ((1 + discount_rate) ** years)
    
    # Sum of discounted cash flows
    sum_discounted_cf = sum(discounted_cash_flows)
    
    # Enterprise Value
    enterprise_value = sum_discounted_cf + discounted_terminal_value
    
    # Equity Value
    equity_value = enterprise_value - net_debt
    
    # Intrinsic Value per Share
    intrinsic_value_per_share = equity_value / shares_outstanding
    
    return {
        "projected_cash_flows": future_cash_flows,
        "discounted_cash_flows": discounted_cash_flows,
        "terminal_value": terminal_value,
        "discounted_terminal_value": discounted_terminal_value,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "intrinsic_value_per_share": intrinsic_value_per_share
    }
