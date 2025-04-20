# stock_simulator.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.spatial.distance import euclidean
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Stock Pattern Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .stApp {
        max-width: 100%;
    }
    .download-button {
        text-align: center;
        margin: 10px 0;
    }
    .disclaimer {
        font-size: 0.8em;
        color: #666;
        border-top: 1px solid #ddd;
        padding-top: 10px;
        margin-top: 20px;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .summary-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# Caching decorator for expensive computations
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_stock_data(ticker, lookback_years=10):
    """Load stock data with caching"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*lookback_years)
        hist_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if hist_data.empty:
            return None, f"No data found for ticker {ticker}"
        
        hist_data['Return'] = hist_data['Close'].pct_change()
        hist_data = hist_data.dropna()
        
        ticker_info = yf.Ticker(ticker).info
        company_name = ticker_info.get('shortName', ticker)
        
        return hist_data, company_name
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

def normalize_series(series):
    """Normalize a price series to range [0,1] for better shape comparison"""
    min_val = np.min(series)
    max_val = np.max(series)
    if max_val == min_val:
        return np.zeros_like(series)
    return (series - min_val) / (max_val - min_val)

def calculate_euclidean_similarity(recent_prices, hist_prices):
    """Calculate Euclidean distance-based similarity between two price sequences"""
    recent_norm = normalize_series(recent_prices)
    hist_norm = normalize_series(hist_prices)
    recent_norm = np.ravel(recent_norm)
    hist_norm = np.ravel(hist_norm)
    distance = euclidean(recent_norm, hist_norm)
    return 1.0 / (1.0 + distance)

def run_simulation(hist_data, horizon_days, period_length, num_simulations):
    """Run the pattern matching simulation"""
    # Get recent period data
    recent_period_data = hist_data.tail(period_length)
    recent_period_prices = recent_period_data['Close'].values.flatten()
    
    # Pattern matching
    max_similarity = -1
    best_start_idx = 0
    similarities = []
    
    # Define lookback window
    lookback_years = 10
    min_start_idx = max(0, len(hist_data) - 252 * lookback_years)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Search for similar patterns
    total_iterations = len(hist_data) - period_length - horizon_days - min_start_idx
    
    for i in range(min_start_idx, len(hist_data) - period_length - horizon_days):
        # Update progress
        progress = (i - min_start_idx) / total_iterations
        progress_bar.progress(progress)
        status_text.text(f"Analyzing patterns: {progress*100:.1f}% complete")
        
        # Skip recent period
        if i > len(hist_data) - period_length * 2:
            continue
            
        historical_period_prices = hist_data['Close'].iloc[i:i+period_length].values.flatten()
        
        if len(historical_period_prices) == period_length:
            similarity = calculate_euclidean_similarity(recent_period_prices, historical_period_prices)
            similarities.append((i, similarity))
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_start_idx = i
    
    progress_bar.progress(1.0)
    status_text.text("Analysis complete!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    # Get the matching period
    matching_period_start = hist_data.index[best_start_idx]
    matching_period_end = hist_data.index[best_start_idx + period_length - 1]
    
    # Get simulation basis returns
    simulation_basis_returns = hist_data['Return'].iloc[best_start_idx+period_length:best_start_idx+period_length+horizon_days].values
    
    # If insufficient data, pad with randomly selected returns
    if len(simulation_basis_returns) < horizon_days:
        additional_days = horizon_days - len(simulation_basis_returns)
        padding = np.random.choice(simulation_basis_returns, size=additional_days)
        simulation_basis_returns = np.concatenate([simulation_basis_returns, padding])
    
    # Run simulations
    current_price = float(hist_data['Close'].iloc[-1])
    all_simulations = np.zeros((horizon_days, num_simulations))
    all_simulations[0, :] = current_price
    
    # Run the bootstrap simulations
    for sim in range(num_simulations):
        for day in range(1, horizon_days):
            random_return = np.random.choice(simulation_basis_returns)
            all_simulations[day, sim] = all_simulations[day-1, sim] * (1 + random_return)
    
    # Calculate percentiles
    percentiles = {
        '1st': np.percentile(all_simulations, 1, axis=1),
        '10th': np.percentile(all_simulations, 10, axis=1),
        '25th': np.percentile(all_simulations, 25, axis=1),
        '50th': np.percentile(all_simulations, 50, axis=1),  # Median
        '75th': np.percentile(all_simulations, 75, axis=1),
        '90th': np.percentile(all_simulations, 90, axis=1),
        '99th': np.percentile(all_simulations, 99, axis=1)
    }
    
    # Get matched historical period prices
    matched_period_prices = hist_data['Close'].iloc[best_start_idx:best_start_idx+period_length].values.flatten()
    
    # Scale matched prices to align with current price
    scale_factor = current_price / matched_period_prices[-1]
    matched_period_prices = matched_period_prices * scale_factor
    
    # Prepare follow-up historical period
    follow_up_prices = []
    for i in range(horizon_days):
        idx = best_start_idx + period_length + i
        if idx < len(hist_data):
            follow_up_prices.append(hist_data['Close'].iloc[idx] * scale_factor)
        else:
            follow_up_prices.append(follow_up_prices[-1] if follow_up_prices else current_price)
    
    # Create date ranges
    last_date = datetime.now()
    future_dates = pd.date_range(start=last_date, periods=horizon_days, freq='B')[:horizon_days]
    
    # Sort similarities to find top matches
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_5_matches = similarities[:5]
    
    return {
        'current_price': current_price,
        'matching_period_start': matching_period_start,
        'matching_period_end': matching_period_end,
        'similarity_score': max_similarity,
        'percentiles': percentiles,
        'future_dates': future_dates,
        'recent_period_data': recent_period_data,
        'matched_period_prices': matched_period_prices,
        'follow_up_prices': follow_up_prices,
        'all_simulations': all_simulations,
        'top_5_matches': top_5_matches,
        'best_start_idx': best_start_idx
    }

def create_plot(results, ticker, horizon_display):
    """Create interactive Plotly visualization"""
    fig = make_subplots()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=results['recent_period_data'].index,
        y=results['recent_period_data']['Close'],
        mode='lines',
        name=f'Last {len(results["recent_period_data"])} Days',
        line=dict(color='darkred', width=2)
    ))
    
    # Add matched period
    matched_dates = pd.date_range(end=datetime.now(), periods=len(results['matched_period_prices']), freq='B')
    fig.add_trace(go.Scatter(
        x=matched_dates,
        y=results['matched_period_prices'],
        mode='lines',
        name='Best Historical Match',
        line=dict(color='darkblue', width=2)
    ))
    
    # Add vertical line for today
    fig.add_vline(x=datetime.now(), line=dict(color='black', width=2, dash='solid'))
    
    # Add historical continuation
    fig.add_trace(go.Scatter(
        x=results['future_dates'],
        y=results['follow_up_prices'],
        mode='lines',
        name='Historical Continuation',
        line=dict(color='darkblue', width=2, dash='dash')
    ))
    
    # Add percentile lines
    colors = {
        '1st': 'red',
        '10th': 'orangered',
        '25th': 'orange',
        '50th': 'green',
        '75th': 'lightblue',
        '90th': 'blue',
        '99th': 'darkblue'
    }
    
    for percentile, values in results['percentiles'].items():
        fig.add_trace(go.Scatter(
            x=results['future_dates'],
            y=values,
            mode='lines',
            name=f'{percentile} Percentile',
            line=dict(color=colors[percentile], width=2 if percentile == '50th' else 1)
        ))
    
    # Add shaded areas
    fig.add_trace(go.Scatter(
        x=list(results['future_dates']) + list(results['future_dates'])[::-1],
        y=list(results['percentiles']['10th']) + list(results['percentiles']['90th'])[::-1],
        fill='toself',
        fillcolor='rgba(128,128,128,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name='10-90 Percentile Range'
    ))
    
    # Add confidence band for 25th-75th percentile
    fig.add_trace(go.Scatter(
        x=list(results['future_dates']) + list(results['future_dates'])[::-1],
        y=list(results['percentiles']['25th']) + list(results['percentiles']['75th'])[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name='25-75 Percentile Range'
    ))
    
    # Configure layout
    fig.update_layout(
        title=f"{ticker} Price Simulation - {horizon_display} Horizon",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=600,
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Format hover data
    fig.update_traces(
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:.2f}<extra></extra>"
    )
    
    return fig

def create_static_plot(results, ticker, horizon_display):
    """Create a static matplotlib plot for download"""
    plt.figure(figsize=(14, 8))
    
    # Plot historical data
    plt.plot(results['recent_period_data'].index, results['recent_period_data']['Close'], 
             'darkred', linewidth=2, label=f'Last {len(results["recent_period_data"])} Days')
    
    # Plot matched period
    matched_dates = pd.date_range(end=datetime.now(), periods=len(results['matched_period_prices']), freq='B')
    plt.plot(matched_dates, results['matched_period_prices'], 'darkblue', linewidth=2,
             label='Best Historical Match')
    
    # Add vertical line for today
    plt.axvline(x=datetime.now(), color='black', linestyle='-', linewidth=2, label='Today')
    
    # Plot historical continuation
    plt.plot(results['future_dates'], results['follow_up_prices'], 'darkblue', linewidth=2, 
             linestyle='--', label='Historical Continuation')
    
    # Plot percentile lines
    plt.plot(results['future_dates'], results['percentiles']['10th'], 'r--', linewidth=1, 
             label='10th Percentile')
    plt.plot(results['future_dates'], results['percentiles']['50th'], 'g-', linewidth=3, 
             label='Median (50th)')
    plt.plot(results['future_dates'], results['percentiles']['90th'], 'b--', linewidth=1, 
             label='90th Percentile')
    
    # Fill between percentiles
    plt.fill_between(results['future_dates'], 
                    results['percentiles']['10th'], 
                    results['percentiles']['90th'], 
                    color='gray', alpha=0.3, label='10-90 Percentile Range')
    
    plt.title(f'{ticker} Price Simulation - {horizon_display} Horizon\n' +
              f'Current Price: ${results["current_price"]:.2f} | ' +
              f'Similarity Score: {results["similarity_score"]:.4f}',
              fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return buf

# Main app layout
st.title("📈 Stock Pattern Simulator")
st.markdown("""
This interactive simulator analyzes historical stock patterns to forecast potential future price movements.
It uses a pattern matching algorithm to identify similar price movements in the past and project potential outcomes.
""")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    
    # Ticker input
    ticker = st.text_input("Stock Ticker Symbol", "SPY", help="Enter a valid stock ticker (e.g., AAPL, MSFT, GOOGL)").upper()
    
    # Horizon selection
    horizon_options = {
        "5d": "5 Days",
        "1mo": "1 Month",
        "2mo": "2 Months",
        "3mo": "3 Months",
        "6mo": "6 Months",
        "9mo": "9 Months",
        "12mo": "12 Months"
    }
    
    horizon_days_map = {
        "5d": 5,
        "1mo": 21,
        "2mo": 42,
        "3mo": 63,
        "6mo": 126,
        "9mo": 189,
        "12mo": 252
    }
    
    horizon = st.selectbox(
        "Forecast Horizon", 
        options=list(horizon_options.keys()),
        format_func=lambda x: horizon_options[x],
        index=3  # Default to 3 months
    )
    
    # Advanced settings (collapsible)
    with st.expander("Advanced Settings"):
        period_length = st.slider(
            "Pattern Matching Period (Trading Days)", 
            min_value=21, 
            max_value=126, 
            value=63, 
            help="Number of trading days to use for pattern matching (default: 63 ≈ 3 months)"
        )
        
        num_simulations = st.slider(
            "Number of Simulations", 
            min_value=500, 
            max_value=5000, 
            value=2000, 
            step=500,
            help="More simulations provide smoother results but take longer"
        )
        
        lookback_years = st.slider(
            "Historical Lookback (Years)", 
            min_value=3, 
            max_value=20, 
            value=10, 
            help="How many years of historical data to analyze"
        )
    
    # Run button
    run_simulation_button = st.button("Run Simulation", type="primary", use_container_width=True)

# Main content area
if run_simulation_button:
    # Check if ticker is provided
    if not ticker:
        st.error("Please enter a stock ticker symbol")
    else:
        # Load data
        with st.spinner(f"Loading data for {ticker}..."):
            hist_data, company_name = load_stock_data(ticker, lookback_years)
        
        if hist_data is None:
            st.error(company_name)  # company_name contains error message
        else:
            # Display company name and current price
            current_price = float(hist_data['Close'].iloc[-1])
            st.markdown(f"### {company_name} ({ticker})")
            st.metric("Current Price", f"${current_price:.2f}")
            
            # Run simulation
            with st.spinner("Running simulation..."):
                results = run_simulation(
                    hist_data, 
                    horizon_days_map[horizon], 
                    period_length, 
                    num_simulations
                )
            
            # Display the interactive plot
            st.plotly_chart(create_plot(results, ticker, horizon_options[horizon]), use_container_width=True)
            
            # Results summary
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Simulation Results")
                
                # Price scenarios table
                final_percentiles = {key: values[-1] for key, values in results['percentiles'].items()}
                scenarios = {
                    'Percentile': list(final_percentiles.keys()),
                    'Projected Price': [f"${price:.2f}" for price in final_percentiles.values()],
                    'Return': [f"{((price/current_price)-1)*100:.1f}%" for price in final_percentiles.values()]
                }
                
                scenarios_df = pd.DataFrame(scenarios)
                st.table(scenarios_df)
                
                # Top matching periods
                st.markdown("### Top 5 Historical Matches")
                for i, (idx, score) in enumerate(results['top_5_matches'], 1):
                    match_start = hist_data.index[idx]
                    match_end = hist_data.index[idx + period_length - 1]
                    st.markdown(f"**{i}.** {match_start.strftime('%Y-%m-%d')} to {match_end.strftime('%Y-%m-%d')} - Score: {score:.4f}")
            
            with col2:
                st.markdown("### Key Metrics")
                
                # Base case metrics
                median_price = results['percentiles']['50th'][-1]
                median_return = ((median_price/current_price)-1)*100
                
                st.metric("Base Case Price", f"${median_price:.2f}", f"{median_return:.1f}%")
                st.metric("Best Historical Match", 
                         f"{results['matching_period_start'].strftime('%Y-%m-%d')} to {results['matching_period_end'].strftime('%Y-%m-%d')}")
                st.metric("Similarity Score", f"{results['similarity_score']:.4f}", help="Higher score indicates better pattern match")
                
                # Download options
                st.markdown("### Download Options")
                
                # Static chart download
                static_chart = create_static_plot(results, ticker, horizon_options[horizon])
                st.download_button(
                    label="Download Chart (PNG)",
                    data=static_chart,
                    file_name=f"{ticker}_simulation_{horizon}_{datetime.now().strftime('%Y%m%d')}.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                # Data download (CSV)
                simulation_data = pd.DataFrame({
                    'Date': results['future_dates'],
                    **{f'{key} Percentile': values for key, values in results['percentiles'].items()}
                })
                
                csv = simulation_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Data (CSV)",
                    data=csv,
                    file_name=f"{ticker}_simulation_data_{horizon}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Monte Carlo simulation data download (optional)
                with st.expander("Download Raw Simulation Data"):
                    mc_data = pd.DataFrame(results['all_simulations'], 
                                         index=results['future_dates'],
                                         columns=[f'Simulation_{i+1}' for i in range(num_simulations)])
                    
                    mc_csv = mc_data.to_csv().encode('utf-8')
                    st.download_button(
                        label="Download Monte Carlo Data (CSV)",
                        data=mc_csv,
                        file_name=f"{ticker}_monte_carlo_{horizon}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            # Technical details (collapsible)
            with st.expander("Technical Details"):
                st.markdown(f"""
                ### Simulation Parameters
                - **Pattern Matching Period**: {period_length} trading days
                - **Number of Simulations**: {num_simulations}
                - **Historical Lookback**: {lookback_years} years
                - **Forecast Horizon**: {horizon_options[horizon]}
                - **Total Historical Data Points**: {len(hist_data)}
                
                ### Best Historical Match
                - **Period**: {results['matching_period_start'].strftime('%Y-%m-%d')} to {results['matching_period_end'].strftime('%Y-%m-%d')}
                - **Similarity Score**: {results['similarity_score']:.4f}
                
                ### Methodology
                1. Euclidean distance pattern matching on normalized price series
                2. Bootstrap sampling from historical returns following similar patterns
                3. Monte Carlo simulation with {num_simulations} paths
                """)
            
            # Disclaimer
            st.markdown('<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> This simulator is for educational purposes only. Past performance does not guarantee future results. This should not be used as the sole basis for investment decisions. Always consult with a qualified financial advisor before making investment decisions.</div>', unsafe_allow_html=True)

else:
    # Default view when no simulation is running
    st.info("👈 Configure the simulation in the sidebar and click 'Run Simulation' to begin")
    
    # Show example or placeholder
    st.markdown("### How It Works")
    st.markdown("""
    1. **Pattern Recognition**: The algorithm identifies similar price patterns in historical data
    2. **Simulation**: Uses returns following those patterns to generate potential future scenarios
    3. **Visualization**: Displays percentile bands showing the range of possible outcomes
    """)
    
    # Example image (optional)
    st.image("https://via.placeholder.com/800x400.png?text=Example+Simulation+Output", 
             caption="Sample output showing historical pattern matching and future projections")
    
    # Footer
    st.markdown("---")
    st.markdown("Created with Streamlit • Data provided by Yahoo Finance")

# For embedding
if st.query_params.get('embed') == 'true':
    # Hide sidebar in embed mode
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
