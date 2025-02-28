import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import seaborn as sns
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent popups

# Set plotting style
plt.style.use('fivethirtyeight')
sns.set_palette("deep")

def get_stock_data(ticker, period="1y", start_date=None, end_date=None):
    """
    Fetch stock data for the specified ticker and period or date range.
    
    Parameters:
    ticker (str): Stock ticker symbol
    period (str): Time period to fetch (default: "1y" for 1 year)
    start_date (str): Start date in YYYY-MM-DD format (overrides period if provided)
    end_date (str): End date in YYYY-MM-DD format (overrides period if provided)
    
    Returns:
    pandas.DataFrame: DataFrame with stock data
    """
    stock = yf.Ticker(ticker)
    
    if start_date and end_date:
        data = stock.history(start=start_date, end=end_date)
        print(f"Retrieved {len(data)} days of data for {ticker} from {start_date} to {end_date}")
    else:
        data = stock.history(period=period)
        print(f"Retrieved {len(data)} days of data for {ticker} with period={period}")
    
    # Check if data is empty
    if data.empty:
        print(f"No data found for {ticker}. Please check if the ticker is valid.")
        return None
    
    return data

def calculate_statistics(data):
    """
    Calculate key statistics for the stock price data.
    
    Parameters:
    data (pandas.DataFrame): DataFrame with stock data
    
    Returns:
    dict: Dictionary containing key statistics
    """
    stats_dict = {
        'Mean': data['Close'].mean(),
        'Median': data['Close'].median(),
        'Standard Deviation': data['Close'].std(),
        'Min': data['Close'].min(),
        'Max': data['Close'].max(),
        'Current Price': data['Close'].iloc[-1],
        'Period Return (%)': ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100,
        'Annualized Return (%)': (((data['Close'].iloc[-1] / data['Close'].iloc[0]) ** (365 / len(data))) - 1) * 100,
        'Daily Return Mean (%)': data['Close'].pct_change().mean() * 100,
        'Daily Return Std (%)': data['Close'].pct_change().std() * 100,
        'Daily Volume Mean': data['Volume'].mean(),
        'Volatility (Annualized %)': data['Close'].pct_change().std() * np.sqrt(252) * 100,
        'Sharpe Ratio': (data['Close'].pct_change().mean() / data['Close'].pct_change().std()) * np.sqrt(252),
        'Skewness': stats.skew(data['Close'].pct_change().dropna()),
        'Kurtosis': stats.kurtosis(data['Close'].pct_change().dropna()),
    }
    
    return stats_dict

def calculate_moving_averages(data):
    """
    Calculate moving averages for different windows.
    
    Parameters:
    data (pandas.DataFrame): DataFrame with stock data
    
    Returns:
    pandas.DataFrame: DataFrame with added moving average columns
    """
    ma_data = data.copy()
    ma_windows = [5, 20, 50, 100, 200]
    
    for window in ma_windows:
        if len(data) >= window:
            ma_data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
    
    return ma_data

def calculate_trendline(data, days_to_project=30):
    """
    Calculate linear trendline and project it forward.
    
    Parameters:
    data (pandas.DataFrame): DataFrame with stock data
    days_to_project (int): Number of days to project the trendline
    
    Returns:
    tuple: (slope, intercept, projected_dates, projected_values, r_value, p_value)
    """
    # Prepare data for trendline
    data = data.reset_index()
    data['Day'] = np.arange(len(data))
    
    # Calculate trendline
    slope, intercept, r_value, p_value, std_err = stats.linregress(data['Day'], data['Close'])
    
    # Project trendline forward
    last_day = data['Day'].iloc[-1]
    projected_days = np.arange(last_day + 1, last_day + days_to_project + 1)
    projected_values = slope * projected_days + intercept
    
    # Create projected dates
    last_date = data['Date'].iloc[-1]
    projected_dates = [last_date + timedelta(days=i+1) for i in range(days_to_project)]
    
    return slope, intercept, projected_dates, projected_values, r_value, p_value

def project_price_range(data, days=30, confidence=0.95):
    """
    Project price range for the specified number of days based on historical volatility.
    
    Parameters:
    data (pandas.DataFrame): DataFrame with stock data
    days (int): Number of days to project
    confidence (float): Confidence level for the projection
    
    Returns:
    tuple: (current_price, lower_bound, upper_bound)
    """
    # Calculate daily returns
    returns = data['Close'].pct_change().dropna()
    
    # Current price
    current_price = data['Close'].iloc[-1]
    
    # Mean and standard deviation of daily returns
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Z-score for the specified confidence level
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    # Project price range
    daily_movement = mean_return * days
    daily_volatility = std_return * np.sqrt(days) * z_score
    
    lower_bound = current_price * (1 + daily_movement - daily_volatility)
    upper_bound = current_price * (1 + daily_movement + daily_volatility)
    
    return current_price, lower_bound, upper_bound, confidence

def analyze_event_impact(data, event_date, event_description, event_type, window=10):
    """
    Analyze the impact of an event on stock price.
    
    Parameters:
    data (pandas.DataFrame): DataFrame with stock data
    event_date (datetime.date): Date of the event
    event_description (str): Description of the event
    event_type (str): Type of the event
    window (int): Number of days to analyze before and after the event
    
    Returns:
    dict: Dictionary with event impact statistics
    """
    # Reset index to make date operations easier
    data_reset = data.reset_index()
    data_dates = [date.date() for date in data_reset['Date']]
    
    # Find the closest trading day to the event
    closest_idx = None
    min_diff = float('inf')
    
    for i, date in enumerate(data_dates):
        diff = abs((date - event_date).days)
        if diff < min_diff:
            min_diff = diff
            closest_idx = i
    
    # If no suitable date found or too far from the event
    if closest_idx is None or min_diff > 5:
        return {
            'event_date': event_date,
            'description': event_description,
            'type': event_type,
            'found': False,
            'reason': f"No suitable trading day found within 5 days of event (closest: {min_diff} days)"
        }
    
    # Get the actual date used for analysis
    actual_date = data_dates[closest_idx]
    price_at_event = data_reset.loc[closest_idx, 'Close']
    
    # Define pre and post event windows
    pre_start = max(0, closest_idx - window)
    post_end = min(len(data_reset) - 1, closest_idx + window)
    
    # Get pre and post event data
    pre_event_data = data_reset.iloc[pre_start:closest_idx]
    post_event_data = data_reset.iloc[closest_idx:post_end+1]
    
    # Calculate statistics
    if len(pre_event_data) > 1 and len(post_event_data) > 1:
        # Pre-event metrics
        pre_mean = pre_event_data['Close'].mean()
        pre_std = pre_event_data['Close'].std()
        pre_returns = pre_event_data['Close'].pct_change().dropna()
        pre_volatility = pre_returns.std() * 100
        
        # Post-event metrics
        post_mean = post_event_data['Close'].mean()
        post_std = post_event_data['Close'].std()
        post_returns = post_event_data['Close'].pct_change().dropna()
        post_volatility = post_returns.std() * 100
        
        # Calculate immediate impact (1-day)
        if closest_idx > 0:
            price_day_before = data_reset.loc[closest_idx-1, 'Close']
            immediate_change = ((price_at_event / price_day_before) - 1) * 100
        else:
            immediate_change = None
        
        # Calculate short-term impact (window-day)
        first_post_price = post_event_data['Close'].iloc[0]
        last_post_price = post_event_data['Close'].iloc[-1]
        short_term_change = ((last_post_price / first_post_price) - 1) * 100
        
        # Statistical test: t-test for pre vs post returns
        if len(pre_returns) > 1 and len(post_returns) > 1:
            t_stat, p_value = stats.ttest_ind(pre_returns, post_returns, equal_var=False)
            significant = p_value < 0.05
        else:
            t_stat, p_value, significant = None, None, None
        
        # F-test for change in volatility
        if len(pre_returns) > 1 and len(post_returns) > 1:
            f_stat = np.var(post_returns) / np.var(pre_returns)
            volatility_increase = post_volatility > pre_volatility
        else:
            f_stat, volatility_increase = None, None
        
        return {
            'event_date': event_date,
            'actual_date': actual_date,
            'description': event_description,
            'type': event_type,
            'found': True,
            'price_at_event': price_at_event,
            'immediate_change_pct': immediate_change,
            'short_term_change_pct': short_term_change,
            'pre_mean': pre_mean,
            'post_mean': post_mean,
            'mean_change_pct': ((post_mean / pre_mean) - 1) * 100,
            'pre_volatility': pre_volatility,
            'post_volatility': post_volatility,
            'volatility_change_pct': ((post_volatility / pre_volatility) - 1) * 100 if pre_volatility > 0 else None,
            'volatility_increased': volatility_increase,
            't_stat': t_stat,
            'p_value': p_value,
            'statistically_significant': significant,
            'days_from_event': min_diff
        }
    else:
        return {
            'event_date': event_date,
            'description': event_description,
            'type': event_type,
            'found': False,
            'reason': f"Insufficient data points for analysis (pre: {len(pre_event_data)}, post: {len(post_event_data)})"
        }

def plot_stock_data(data, ma_data, ticker, stats_dict, events=None):
    """
    Create visualizations for stock data analysis.
    
    Parameters:
    data (pandas.DataFrame): DataFrame with stock data
    ma_data (pandas.DataFrame): DataFrame with moving averages
    ticker (str): Stock ticker symbol
    stats_dict (dict): Dictionary with key statistics
    events (list): List of tuples (date, description, type) for key events
    """
    print("Creating price chart...")
    
    # Create ticker-specific directory for saving charts
    save_dir = f"stock_analysis_{ticker}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 1. Plot stock price with moving averages
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], linewidth=2, label=f'{ticker} Close Price')
    
    # Plot moving averages
    for col in ma_data.columns:
        if col.startswith('MA_'):
            window = col.split('_')[1]
            plt.plot(ma_data.index, ma_data[col], label=f'{window}-Day MA', alpha=0.7)
    
    # Add events if provided
    if events:
        print(f"Processing {len(events)} events...")
        # Create list of date objects from the DataFrame index
        data_dates = [date.date() for date in data.index.to_pydatetime()]
        
        # Define colors by event type
        event_colors = {
            'Earnings': 'red',
            'M&A': 'green',
            'Stock Split': 'purple',
            'Geopolitical': 'orange',
            'Economic': 'blue',
            'Product': 'brown',
            'Management': 'magenta',
            'Legal': 'cyan',
            'Industry': 'pink',
            'Other': 'gray'
        }
        
        for event_date, description, event_type in events:
            print(f"Event: {description} ({event_type}) on {event_date}")
            # Try to find the exact date
            try:
                if event_date in data_dates:
                    idx = data_dates.index(event_date)
                    chart_date = data.index[idx]
                    price_at_event = data['Close'].iloc[idx]
                    
                    color = event_colors.get(event_type, 'gray')
                    plt.axvline(x=chart_date, color=color, linestyle='--', alpha=0.5)
                    plt.annotate(f"{description} ({event_type})", xy=(chart_date, price_at_event), 
                                xytext=(15, 15), textcoords='offset points',
                                arrowprops=dict(arrowstyle='->', color=color),
                                fontsize=8, rotation=45)
                    print(f"  Found exact match for date {event_date}")
                else:
                    # Find the closest trading day
                    closest_idx = None
                    min_diff = float('inf')
                    
                    for i, date in enumerate(data_dates):
                        diff = abs((date - event_date).days)
                        if diff < min_diff:
                            min_diff = diff
                            closest_idx = i
                    
                    if closest_idx is not None and min_diff <= 5:  # Within 5 days
                        chart_date = data.index[closest_idx]
                        price_at_closest = data['Close'].iloc[closest_idx]
                        
                        color = event_colors.get(event_type, 'gray')
                        plt.axvline(x=chart_date, color=color, linestyle='--', alpha=0.5)
                        plt.annotate(f"{description} ({event_type})", xy=(chart_date, price_at_closest), 
                                    xytext=(15, 15), textcoords='offset points',
                                    arrowprops=dict(arrowstyle='->', color=color),
                                    fontsize=8, rotation=45)
                        print(f"  Using closest date {data_dates[closest_idx]} (diff: {min_diff} days)")
                    else:
                        print(f"  No suitable date found within 5 days (closest diff: {min_diff} days)")
            except Exception as e:
                print(f"  Error processing event: {e}")
    
    plt.title(f'{ticker} Stock Price with Moving Averages')
    plt.ylabel('Price ($)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the chart
    save_path = os.path.join(save_dir, f"{ticker}_price_chart.png")
    plt.savefig(save_path, dpi=300)
    print(f"Price chart saved to {save_path}")
    
    # No plt.show() to prevent popups
    plt.close()
    
    # 2. Plot volume
    print("Creating volume chart...")
    plt.figure(figsize=(12, 5))
    plt.bar(data.index, data['Volume'], color='blue', alpha=0.5)
    plt.title(f'{ticker} Trading Volume')
    plt.ylabel('Volume')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the chart
    save_path = os.path.join(save_dir, f"{ticker}_volume_chart.png")
    plt.savefig(save_path, dpi=300)
    print(f"Volume chart saved to {save_path}")
    
    # No plt.show() to prevent popups
    plt.close()
    
    # 3. Plot daily returns histogram
    print("Creating returns histogram...")
    plt.figure(figsize=(12, 5))
    daily_returns = data['Close'].pct_change().dropna() * 100
    plt.hist(daily_returns, bins=50, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'{ticker} Daily Returns Distribution')
    plt.xlabel('Daily Returns (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the chart
    save_path = os.path.join(save_dir, f"{ticker}_returns_histogram.png")
    plt.savefig(save_path, dpi=300)
    print(f"Returns histogram saved to {save_path}")
    
    # No plt.show() to prevent popups
    plt.close()
    
    # 4. Plot price distribution
    print("Creating price distribution chart...")
    plt.figure(figsize=(12, 5))
    sns.kdeplot(data['Close'], fill=True)
    plt.axvline(x=stats_dict['Mean'], color='r', linestyle='--', label=f"Mean: ${stats_dict['Mean']:.2f}")
    plt.axvline(x=stats_dict['Median'], color='g', linestyle='--', label=f"Median: ${stats_dict['Median']:.2f}")
    plt.title(f'{ticker} Price Distribution')
    plt.xlabel('Price ($)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the chart
    save_path = os.path.join(save_dir, f"{ticker}_price_distribution.png")
    plt.savefig(save_path, dpi=300)
    print(f"Price distribution chart saved to {save_path}")
    
    # No plt.show() to prevent popups
    plt.close()

def plot_price_projection(data, ticker, slope, intercept, projected_dates, projected_values, lower_bound, upper_bound, r_value, p_value):
    """
    Plot price projection with confidence interval.
    
    Parameters:
    data (pandas.DataFrame): DataFrame with stock data
    ticker (str): Stock ticker symbol
    slope, intercept: Trendline parameters
    projected_dates, projected_values: Projected trendline data
    lower_bound, upper_bound: Projected price range bounds
    r_value, p_value: Statistical measures of the trendline
    """
    print("Creating price projection chart...")
    
    # Create ticker-specific directory for saving charts
    save_dir = f"stock_analysis_{ticker}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data
    data_reset = data.reset_index()
    ax.plot(data_reset['Date'], data_reset['Close'], label=f'{ticker} Historical', color='blue')
    
    # Create Day column for trendline calculation
    data_reset['Day'] = np.arange(len(data_reset))
    
    # Plot trendline on historical data
    trendline_historical = data_reset['Day'] * slope + intercept
    ax.plot(data_reset['Date'], trendline_historical, 'r--', 
            label=f'Historical Trendline (R²={r_value**2:.3f}, p={p_value:.4f})', alpha=0.7)
    
    # Plot projected trendline
    ax.plot(projected_dates, projected_values, 'g--', label='Projected Trendline', alpha=0.7)
    
    # Shade the projected price range
    ax.axhline(y=lower_bound, color='orange', linestyle=':', label=f'Lower Bound: ${lower_bound:.2f}')
    ax.axhline(y=upper_bound, color='purple', linestyle=':', label=f'Upper Bound: ${upper_bound:.2f}')
    
    # Fill the range between lower and upper bounds
    projection_x = [projected_dates[0], projected_dates[-1], projected_dates[-1], projected_dates[0]]
    projection_y = [lower_bound, lower_bound, upper_bound, upper_bound]
    ax.fill(projection_x, projection_y, alpha=0.2, color='green', label='Projected Price Range')
    
    # Format and display
    ax.set_title(f'{ticker} 30-Day Price Projection')
    ax.set_xlabel('Date') 
    ax.set_ylabel('Price ($)')
    ax.legend(loc='best')
    ax.grid(True)
    
    plt.tight_layout()
    
    # Save the chart
    save_path = os.path.join(save_dir, f"{ticker}_projection.png")
    plt.savefig(save_path, dpi=300)
    print(f"Price projection chart saved to {save_path}")
    
    # No plt.show() to prevent popups
    plt.close()

def generate_written_analysis(ticker, data, stats_dict, projected_range, trend_data, events=None, event_analyses=None):
    """
    Generate a written analysis of the stock with academic financial language.
    
    Parameters:
    ticker (str): Stock ticker symbol
    data (pandas.DataFrame): DataFrame with stock data
    stats_dict (dict): Dictionary with key statistics
    projected_range (tuple): (current_price, lower_bound, upper_bound, confidence)
    trend_data (tuple): (slope, intercept, r_value, p_value)
    events (list): List of tuples (date, description, type) for key events
    event_analyses (list): List of dictionaries with event impact analyses
    
    Returns:
    str: Written analysis text
    """
    print("Generating written analysis...")
    current_price, lower_bound, upper_bound, confidence = projected_range
    slope, intercept, r_value, p_value = trend_data
    
    # Determine the trend
    if stats_dict['Period Return (%)'] > 0:
        trend = "positive"
        trend_detail = f"appreciated by {stats_dict['Period Return (%)']:.2f}%"
    else:
        trend = "negative"
        trend_detail = f"declined by {abs(stats_dict['Period Return (%)']):.2f}%"
    
    # Calculate volatility characterization
    sp500_avg_volatility = 1.0  # Approximate daily volatility for S&P 500 (%)
    market_vol_comparison = ""
    if stats_dict['Daily Return Std (%)'] < sp500_avg_volatility * 0.8:
        volatility = "significantly lower than market average"
        market_vol_comparison = "exhibiting defensive characteristics"
    elif stats_dict['Daily Return Std (%)'] < sp500_avg_volatility:
        volatility = "lower than market average"
        market_vol_comparison = "demonstrating relatively stable price action"
    elif stats_dict['Daily Return Std (%)'] < sp500_avg_volatility * 1.5:
        volatility = "in line with market average"
        market_vol_comparison = "following typical market behavior"
    elif stats_dict['Daily Return Std (%)'] < sp500_avg_volatility * 2:
        volatility = "higher than market average"
        market_vol_comparison = "showing heightened price sensitivity"
    else:
        volatility = "significantly higher than market average"
        market_vol_comparison = "demonstrating substantial price variability"
    
    # Interpret Sharpe Ratio
    sharpe_interpretation = ""
    if stats_dict['Sharpe Ratio'] < 0:
        sharpe_interpretation = "negative risk-adjusted return, indicating underperformance relative to risk-free alternatives"
    elif stats_dict['Sharpe Ratio'] < 0.5:
        sharpe_interpretation = "poor risk-adjusted return, suggesting inadequate compensation for the volatility risk"
    elif stats_dict['Sharpe Ratio'] < 1.0:
        sharpe_interpretation = "suboptimal risk-adjusted return, although positive"
    elif stats_dict['Sharpe Ratio'] < 2.0:
        sharpe_interpretation = "good risk-adjusted return, indicating efficient risk compensation"
    else:
        sharpe_interpretation = "excellent risk-adjusted return, demonstrating superior risk-reward characteristics"
    
    # Interpret trend statistical significance
    trend_significance = ""
    if p_value < 0.01:
        trend_significance = "highly statistically significant (p<0.01)"
    elif p_value < 0.05:
        trend_significance = "statistically significant (p<0.05)"
    elif p_value < 0.1:
        trend_significance = "marginally significant (p<0.1)"
    else:
        trend_significance = "not statistically significant (p>0.1)"
    
    # Calculate projection characterization
    projection_range_percent = ((upper_bound - lower_bound) / current_price) * 100
    
    # Build the analysis
    analysis = f"""# {ticker} Stock Analysis

## Key Findings Summary
Over the analyzed period, {ticker} has exhibited a {trend} trend, having {trend_detail} from the beginning of the analyzed period. The stock demonstrated volatility {volatility} ({stats_dict['Daily Return Std (%)']:.2f}%), {market_vol_comparison}. Daily trading volume averaged {stats_dict['Daily Volume Mean']:,.0f} shares, with the share price fluctuating between a minimum of ${stats_dict['Min']:.2f} and a maximum of ${stats_dict['Max']:.2f}.

The stock's Sharpe Ratio of {stats_dict['Sharpe Ratio']:.2f} indicates a {sharpe_interpretation}. The returns distribution exhibits a skewness of {stats_dict['Skewness']:.2f} and kurtosis of {stats_dict['Kurtosis']:.2f}, suggesting {'a relatively normal distribution' if abs(stats_dict['Skewness']) < 0.5 and abs(stats_dict['Kurtosis']) < 3 else 'deviations from normality that could imply heightened tail risk'}.

## Price Range Projection Rationale
Linear regression analysis of historical price movements reveals a {trend_significance} trend with a slope coefficient of {slope:.4f} (R² = {r_value**2:.3f}). Based on this trendline analysis combined with historical volatility patterns, {ticker} is projected to trade between ${lower_bound:.2f} and ${upper_bound:.2f} over the next 30 days, representing a range of {((lower_bound/current_price)-1)*100:.2f}% to {((upper_bound/current_price)-1)*100:.2f}% from the current price of ${current_price:.2f}. This projection incorporates historical volatility and trend patterns, with a {confidence*100:.0f}% confidence interval.

The width of this projected range ({projection_range_percent:.1f}%) {'suggests significant price uncertainty' if projection_range_percent > 20 else 'indicates moderate price stability' if projection_range_percent > 10 else 'reflects relatively constrained expected price movement'}, consistent with the stock's historical volatility profile.
"""
    
    # Add event impact analysis if provided
    if event_analyses and len(event_analyses) > 0:
        analysis += "## Event Impact Analysis\n"
        
        # Group events by type for structured analysis
        event_types = {}
        for event in event_analyses:
            if event['found']:
                event_type = event['type']
                if event_type not in event_types:
                    event_types[event_type] = []
                event_types[event_type].append(event)
        
        if event_types:
            # Analyze each event type
            for event_type, events_of_type in event_types.items():
                analysis += f"### {event_type} Events\n"
                
                # For each event type, provide a summary
                significant_events = [e for e in events_of_type if e.get('statistically_significant')]
                
                if len(events_of_type) > 1:
                    avg_immediate_change = np.mean([e['immediate_change_pct'] for e in events_of_type if e.get('immediate_change_pct') is not None])
                    avg_volatility_change = np.mean([e['volatility_change_pct'] for e in events_of_type if e.get('volatility_change_pct') is not None])
                    
                    analysis += f"Analysis of {len(events_of_type)} {event_type} events reveals an average immediate price change of {avg_immediate_change:.2f}% and an average volatility change of {avg_volatility_change:.2f}%. "
                    if significant_events:
                        analysis += f"{len(significant_events)} of these events showed statistically significant impacts on price patterns.\n\n"
                    else:
                        analysis += "None of these events demonstrated statistically significant impacts on price patterns.\n\n"
                
                # Provide details for each individual event
                for event in events_of_type:
                    date_str = event['event_date'].strftime('%Y-%m-%d')
                    desc = event['description']
                    
                    if event['days_from_event'] > 0:
                        date_approx = f" (analyzed using closest trading day {event['actual_date'].strftime('%Y-%m-%d')})"
                    else:
                        date_approx = ""
                    
                    if event.get('immediate_change_pct') is not None:
                        immediate_change = event['immediate_change_pct']
                        if abs(immediate_change) > 5:
                            impact_desc = f"substantial immediate impact of {immediate_change:.2f}%"
                        elif abs(immediate_change) > 2:
                            impact_desc = f"moderate immediate impact of {immediate_change:.2f}%"
                        else:
                            impact_desc = f"limited immediate impact of {immediate_change:.2f}%"
                            
                        volatility_impact = ""
                        if event.get('volatility_change_pct') is not None and abs(event['volatility_change_pct']) > 15:
                            vol_change = event['volatility_change_pct']
                            volatility_impact = f" and {'increased' if vol_change > 0 else 'decreased'} volatility by {abs(vol_change):.2f}%"
                        
                        significance = ""
                        if event.get('statistically_significant'):
                            significance = " with statistical significance (p<0.05)"
                        
                        analysis += f"The {desc} on {date_str}{date_approx} had a {impact_desc}{volatility_impact}{significance}. "
                        
                        # Add short-term impact if available
                        if event.get('short_term_change_pct') is not None:
                            st_change = event['short_term_change_pct']
                            if abs(st_change) > 10:
                                st_impact = f"substantial"
                            elif abs(st_change) > 5:
                                st_impact = f"notable"
                            else:
                                st_impact = f"modest"
                            
                            analysis += f"Over the subsequent trading window, the stock demonstrated a {st_impact} {st_change:.2f}% {'gain' if st_change > 0 else 'decline'}.\n\n"
                        else:
                            analysis += "\n\n"
                    else:
                        analysis += f"The {desc} on {date_str}{date_approx} did not have sufficient data to calculate immediate price impact.\n\n"
        else:
            analysis += "No events with quantifiable impact were found in the analyzed time period.\n\n"
    else:
        analysis += "## Event Impact Analysis\nNo event data was provided for impact analysis.\n\n"
    
    # Add categorical summaries if multiple events were analyzed
    if event_analyses and len(event_analyses) > 2:
        analysis += "## Categorical Event Impact Summary\n"
        analysis += "The following table summarizes the impact of different event categories on price action:\n\n"
        
        # Create a summary table
        analysis += "| Event Category | Count | Avg. Immediate Impact | Avg. Short-term Impact | Volatility Effect | Statistically Significant |\n"
        analysis += "|---------------|-------|----------------------|----------------------|------------------|---------------------------|\n"
        
        # Group events by type
        event_types_summary = {}
        for event in event_analyses:
            if event['found']:
                event_type = event['type']
                if event_type not in event_types_summary:
                    event_types_summary[event_type] = {
                        'count': 0,
                        'immediate_impacts': [],
                        'short_term_impacts': [],
                        'volatility_changes': [],
                        'significant_count': 0
                    }
                
                event_types_summary[event_type]['count'] += 1
                
                if event.get('immediate_change_pct') is not None:
                    event_types_summary[event_type]['immediate_impacts'].append(event['immediate_change_pct'])
                
                if event.get('short_term_change_pct') is not None:
                    event_types_summary[event_type]['short_term_impacts'].append(event['short_term_change_pct'])
                
                if event.get('volatility_change_pct') is not None:
                    event_types_summary[event_type]['volatility_changes'].append(event['volatility_change_pct'])
                
                if event.get('statistically_significant'):
                    event_types_summary[event_type]['significant_count'] += 1
        
        # Add each event type to the table
        for event_type, summary in event_types_summary.items():
            count = summary['count']
            
            # Calculate averages
            if summary['immediate_impacts']:
                avg_immediate = np.mean(summary['immediate_impacts'])
                immediate_str = f"{avg_immediate:.2f}%"
            else:
                immediate_str = "N/A"
            
            if summary['short_term_impacts']:
                avg_short_term = np.mean(summary['short_term_impacts'])
                short_term_str = f"{avg_short_term:.2f}%"
            else:
                short_term_str = "N/A"
            
            if summary['volatility_changes']:
                avg_vol_change = np.mean(summary['volatility_changes'])
                if avg_vol_change > 15:
                    vol_str = "Significant increase"
                elif avg_vol_change > 5:
                    vol_str = "Moderate increase"
                elif avg_vol_change > -5:
                    vol_str = "Minimal change"
                elif avg_vol_change > -15:
                    vol_str = "Moderate decrease"
                else:
                    vol_str = "Significant decrease"
            else:
                vol_str = "N/A"
            
            significant_pct = (summary['significant_count'] / count) * 100 if count > 0 else 0
            sig_str = f"{summary['significant_count']}/{count} ({significant_pct:.0f}%)"
            
            analysis += f"| {event_type} | {count} | {immediate_str} | {short_term_str} | {vol_str} | {sig_str} |\n"
    
    analysis += """## Analysis Limitations
This analysis is based on historical price and volume data, with statistical inferences that should be interpreted within proper context. Several limitations should be considered:

1. **Historical Context**: Past performance is not necessarily indicative of future results. Market conditions, economic environments, and company-specific factors can change rapidly.

2. **Statistical Assumptions**: The projection models assume normal distribution of returns and homoscedasticity, which may not hold during extreme market conditions or during structural breaks in market behavior.

3. **Event Analysis Limitations**: The causality between identified events and price movements cannot be definitively established due to potential confounding variables and the presence of other concurrent market influences.

4. **Methodological Constraints**: Technical analysis inherently does not incorporate fundamental factors such as company financials, industry dynamics, macroeconomic indicators, or qualitative market sentiment.

5. **Confidence Intervals**: The projected price ranges represent probabilistic outcomes rather than deterministic predictions and should be interpreted as zones of likelihood rather than precise forecasts.

Investors should combine these technical findings with fundamental analysis, broader market context, and appropriate risk management strategies before making investment decisions."""
    
    print("Written analysis generated successfully.")
    return analysis

def analyze_stock(ticker, events=None, period="1y", start_date=None, end_date=None):
    """
    Main function to perform comprehensive stock analysis.
    
    Parameters:
    ticker (str): Stock ticker symbol
    events (list): List of tuples (date, description, type) for key events
    period (str): Time period for data fetching (e.g., "1y", "6mo")
    start_date (str): Start date in YYYY-MM-DD format (overrides period if provided)
    end_date (str): End date in YYYY-MM-DD format (overrides period if provided)
    
    Returns:
    tuple: (data, stats_dict, projected_range, trend_data, event_analyses)
    """
    # Get stock data
    data = get_stock_data(ticker, period, start_date, end_date)
    if data is None:
        return None, None, None, None, None
    
    # Calculate statistics
    print("Calculating statistics...")
    stats_dict = calculate_statistics(data)
    
    # Calculate moving averages
    print("Calculating moving averages...")
    ma_data = calculate_moving_averages(data)
    
    # Calculate trendline and projection
    print("Calculating trendline and projection...")
    slope, intercept, projected_dates, projected_values, r_value, p_value = calculate_trendline(data)
    trend_data = (slope, intercept, r_value, p_value)
    
    # Project price range
    print("Projecting price range...")
    current_price, lower_bound, upper_bound, confidence = project_price_range(data)
    projected_range = (current_price, lower_bound, upper_bound, confidence)
    
    # Analyze event impacts
    event_analyses = []
    if events:
        print("Analyzing event impacts...")
        for event_date, description, event_type in events:
            event_analysis = analyze_event_impact(data, event_date, description, event_type)
            event_analyses.append(event_analysis)
    
    # Create visualizations
    plot_stock_data(data, ma_data, ticker, stats_dict, events)
    plot_price_projection(data, ticker, slope, intercept, projected_dates, projected_values, 
                          lower_bound, upper_bound, r_value, p_value)
    
    # Print analysis results
    print("\n" + "="*50)
    print(f"ANALYSIS RESULTS FOR {ticker}")
    print("="*50)
    
    print("\nKey Statistics:")
    for stat, value in stats_dict.items():
        print(f"{stat}: {value:.2f}")
    
    print("\nPrice Projection (30 Days):")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Projected Range: ${lower_bound:.2f} to ${upper_bound:.2f}")
    print(f"Percent Change Range: {((lower_bound/current_price)-1)*100:.2f}% to {((upper_bound/current_price)-1)*100:.2f}%")
    
    if slope > 0:
        trend_direction = "upward"
    else:
        trend_direction = "downward"
    
    print(f"\nTrendline Analysis: The stock shows a {trend_direction} trend with a slope of {slope:.6f}")
    print(f"R-squared: {r_value**2:.3f}, p-value: {p_value:.4f}")
    
    return data, stats_dict, projected_range, trend_data, event_analyses

def analyze_another_stock():
    """Ask the user if they want to analyze another stock."""
    print("\nWould you like to analyze another stock? (y/n)")
    response = input().strip().lower()
    return response == 'y' or response == 'yes'

def get_event_type():
    """Get the event type from user input."""
    event_types = {
        "1": "Earnings",
        "2": "M&A",
        "3": "Stock Split",
        "4": "Geopolitical",
        "5": "Economic",
        "6": "Product",
        "7": "Management",
        "8": "Legal",
        "9": "Industry",
        "0": "Other"
    }
    
    print("\nSelect event type:")
    for key, value in event_types.items():
        print(f"  {key} = {value}")
    
    while True:
        choice = input("Enter event type number: ").strip()
        if choice in event_types:
            return event_types[choice]
        else:
            try:
                # If they typed the name directly
                choice_upper = choice.upper()
                for value in event_types.values():
                    if value.upper() == choice_upper:
                        return value
            except:
                pass
            print("Invalid selection. Please try again.")

def get_events_from_user():
    """
    Get event information from the user.
    
    Returns:
    list: List of tuples (date, description, type) for key events
    """
    events = []
    
    print("\nWould you like to add events to your analysis? (y/n)")
    add_events = input().strip().lower()

    if add_events == 'y' or add_events == 'yes':
        print("\nHow many events would you like to add?")
        try:
            num_events_str = input().strip()
            num_events = int(num_events_str)
            
            if num_events <= 0:
                print("Number must be positive. No events added.")
                return events
            
            print(f"Adding {num_events} event(s)...")
            
            for i in range(num_events):
                print(f"\nEvent #{i+1}:")
                
                # Get date with validation
                while True:
                    print("Enter event date (YYYY-MM-DD):")
                    date_str = input().strip()
                    
                    try:
                        event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                        break  # Valid date, exit the loop
                    except ValueError:
                        print("Invalid date format. Please use YYYY-MM-DD format.")
                
                # Get description
                print("Enter event description (e.g., 'Q1 Earnings Release'):")
                description = input().strip()
                
                if not description:
                    description = "Unspecified Event"  # Default if user enters nothing
                
                # Get event type
                event_type = get_event_type()
                
                events.append((event_date, description, event_type))
                print(f"Event added: {event_date} - {description} ({event_type})")
            
            print(f"\nTotal events added: {len(events)}")
            
        except ValueError:
            print(f"Invalid number '{num_events_str}'. No events added.")
    else:
        print("No events will be added to the analysis.")
    
    return events

def get_date_range():
    """
    Get date range from user input.
    
    Returns:
    tuple: (use_period, period, start_date, end_date)
    """
    print("\nWould you like to specify a custom date range? (y/n)")
    print("(Default is 1-year of data)")
    response = input().strip().lower()
    
    if response == 'y' or response == 'yes':
        # Allow user to choose between period or exact dates
        print("\nDo you want to use a standard period or specific dates?")
        print("1: Standard period (e.g., '1y', '6mo', '1d')")
        print("2: Specific date range (YYYY-MM-DD)")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '1':
            print("\nEnter period (e.g., 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max):")
            period = input().strip()
            if not period:
                print("Using default period (1y)")
                period = "1y"
            return True, period, None, None
        else:
            start_date, end_date = None, None
            
            # Get start date
            while True:
                print("\nEnter start date (YYYY-MM-DD):")
                start_str = input().strip()
                try:
                    datetime.strptime(start_str, "%Y-%m-%d")
                    start_date = start_str
                    break
                except ValueError:
                    print("Invalid date format. Please use YYYY-MM-DD format.")
            
            # Get end date
            while True:
                print("Enter end date (YYYY-MM-DD) or press Enter for today:")
                end_str = input().strip()
                if not end_str:
                    end_date = datetime.now().strftime("%Y-%m-%d")
                    break
                try:
                    datetime.strptime(end_str, "%Y-%m-%d")
                    end_date = end_str
                    break
                except ValueError:
                    print("Invalid date format. Please use YYYY-MM-DD format.")
            
            return False, None, start_date, end_date
    else:
        print("Using default period (1y)")
        return True, "1y", None, None

# Main program
def main():
    print("\n" + "="*50)
    print("===== STOCK PRICE ANALYSIS TOOL =====")
    print("="*50)
    
    analyze_more = True
    
    while analyze_more:
        # Get ticker symbol from user
        print("\nEnter the stock ticker symbol you want to analyze (e.g., AAPL, MSFT, NVDA):")
        ticker_symbol = input().strip().upper()  # Convert to uppercase and remove whitespace
        
        if not ticker_symbol:
            print("Error: Ticker symbol cannot be empty.")
            continue
            
        print(f"Analyzing stock: {ticker_symbol}")
        
        # Get date range
        use_period, period, start_date, end_date = get_date_range()
        
        # Get events from user with improved error handling
        events = get_events_from_user()
        
        # Current date for reference
        current_date = datetime.now()
        
        if use_period:
            print(f"\nUsing period: {period}")
        else:
            print(f"\nAnalyzing data from {start_date} to {end_date}")
        
        # Run the analysis
        data, stats_dict, projected_range, trend_data, event_analyses = analyze_stock(
            ticker_symbol, events, 
            period if use_period else None,
            start_date if not use_period else None,
            end_date if not use_period else None
        )
        
        if data is not None:
            print("Analysis complete. Generating written report...")
            # Generate the written analysis
            written_analysis = generate_written_analysis(
                ticker_symbol, data, stats_dict, projected_range, trend_data, events, event_analyses
            )
            
            # Create ticker-specific directory for saving reports
            report_dir = f"stock_analysis_{ticker_symbol}"
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            
            # Save the written analysis to a file
            report_path = os.path.join(report_dir, f"{ticker_symbol}_analysis.md")
            with open(report_path, "w") as f:
                f.write(written_analysis)
            
            print(f"\nWritten analysis saved to {report_path}")
            
            # Display the written analysis
            print("\n" + "="*50)
            print("WRITTEN ANALYSIS")
            print("="*50)
            print(written_analysis)
            
            print(f"\nAnalysis complete! All files saved in folder: stock_analysis_{ticker_symbol}")
        else:
            print(f"Analysis failed for {ticker_symbol}. Please check the ticker symbol and try again.")
        
        # Ask if user wants to analyze another stock
        analyze_more = analyze_another_stock()
    
    print("\nThank you for using the Stock Price Analysis Tool!")

# Run the main program
if __name__ == "__main__":
    main()
