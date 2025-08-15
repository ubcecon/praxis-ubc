# Create a plot with the forecasted data, historical sentiment, and stock price changes
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_plot(combined_df, forecast_mean, forecast_ci, forecast_index, actuals_df=None):
    combined_df = combined_df.copy()
    
    # Aggregate data by date to show one clear point per day
    daily_agg = combined_df.groupby(combined_df.index).agg({
        'lagged_sentiment': 'mean',
        'Pct_Change': 'mean'
    }).reset_index()
    daily_agg.set_index('DateOnly', inplace=True)
    
    sentiment_std = (daily_agg['lagged_sentiment'] - daily_agg['lagged_sentiment'].mean()) \
                    / daily_agg['lagged_sentiment'].std()

    fig = go.Figure()

    # Historical sentiment
    fig.add_trace(go.Scatter(
        x=daily_agg.index,
        y=sentiment_std,
        name='Standardized Sentiment',
        mode='lines+markers',
        line=dict(color='royalblue', width=2),
        marker=dict(color='royalblue', size=4),
        hovertemplate='Date: %{x}<br>Sentiment: %{y:.2f}<extra></extra>'
    ))

    # Historical % change
    fig.add_trace(go.Scatter(
        x=daily_agg.index,
        y=daily_agg['Pct_Change'],
        name='Stock % Change',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='limegreen', width=2),
        marker=dict(color='limegreen', size=4),
        hovertemplate='Date: %{x}<br>% Change: %{y:.2f}%<extra></extra>'
    ))

    # Forecast mean (NOW on y2)
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=forecast_mean,
        name='Forecasted % Change',
        yaxis='y2',
        line=dict(color='tomato', width=2, dash='dash'),
        hovertemplate='Date: %{x}<br>Forecast: %{y:.2f}%<extra></extra>'
    ))

    # Forecast confidence interval (NOW on y2)
    fig.add_trace(go.Scatter(
        x=np.concatenate([forecast_index, forecast_index[::-1]]),
        y=np.concatenate([forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1][::-1]]),
        name='95% CI',
        yaxis='y2',
        fill='toself',
        fillcolor='rgba(255,99,71,0.2)',  # softened tomato
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True
    ))

    # Add actual data if provided
    if actuals_df is not None and not actuals_df.empty:
        # If actuals_df has multiple tickers, aggregate by date (take mean)
        if 'Ticker' in actuals_df.columns and len(actuals_df['Ticker'].unique()) > 1:
            actual_agg = actuals_df.groupby('DateOnly')['Actual_Pct_Change'].mean().reset_index()
            actual_dates = actual_agg['DateOnly']
            actual_values = actual_agg['Actual_Pct_Change']
        else:
            actual_dates = actuals_df['DateOnly']
            actual_values = actuals_df['Actual_Pct_Change']
        
        # Actual data points
        fig.add_trace(go.Scatter(
            x=actual_dates,
            y=actual_values,
            name='Actual % Change',
            yaxis='y2',
            mode='markers',
            marker=dict(color='gold', size=8, symbol='circle'),
            hovertemplate='Date: %{x}<br>Actual: %{y:.2f}%<extra></extra>'
        ))
        
        # Calculate and display gaps (forecast vs actual) where dates overlap
        forecast_df = pd.DataFrame({
            'DateOnly': pd.to_datetime(forecast_index).normalize(),
            'Forecast': forecast_mean
        })
        
        # Prepare actual data for merging
        if 'Ticker' in actuals_df.columns and len(actuals_df['Ticker'].unique()) > 1:
            actual_for_gap = actual_agg.copy()
            actual_for_gap.rename(columns={'Actual_Pct_Change': 'Actual'}, inplace=True)
        else:
            actual_for_gap = actuals_df[['DateOnly', 'Actual_Pct_Change']].copy()
            actual_for_gap.rename(columns={'Actual_Pct_Change': 'Actual'}, inplace=True)
        
        # Merge to find overlapping dates
        gap_df = pd.merge(forecast_df, actual_for_gap, on='DateOnly', how='inner')
        
        if not gap_df.empty:
            gap_df['Gap'] = gap_df['Forecast'] - gap_df['Actual']
            
            # Add gap visualization (error bars or connecting lines)
            for _, row in gap_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['DateOnly'], row['DateOnly']],
                    y=[row['Forecast'], row['Actual']],
                    name='Forecast Gap',
                    yaxis='y2',
                    mode='lines',
                    line=dict(color='orange', width=2, dash='dot'),
                    showlegend=False,
                    hoverinfo="none"
                ))
            
            # Add invisible points at gap midpoints for targeted hover
            for _, row in gap_df.iterrows():
                midpoint = (row['Forecast'] + row['Actual']) / 2
                fig.add_trace(go.Scatter(
                    x=[row['DateOnly']],
                    y=[midpoint],
                    name='Gap Info',
                    yaxis='y2',
                    mode='markers',
                    marker=dict(color='orange', size=8, opacity=0),
                    showlegend=False,
                    hovertemplate=f'Date: %{{x}}<br>Gap: {row["Gap"]:.2f}%<extra></extra>'
                ))
            
            # Add a single legend entry for gaps
            if len(gap_df) > 0:
                fig.add_trace(go.Scatter(
                    x=[gap_df.iloc[0]['DateOnly']],
                    y=[gap_df.iloc[0]['Forecast']],
                    name='Forecast Gap',
                    yaxis='y2',
                    mode='lines',
                    line=dict(color='orange', width=2, dash='dot'),
                    showlegend=True,
                    hoverinfo="skip"
                ))

    fig.update_layout(
        title=dict(
            text='Sentiment vs Stock Change Forecast',
            font=dict(size=20),
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=60, r=60, t=140, b=60),
        template='plotly_dark',
        xaxis=dict(
            title='Date',
            showgrid=False,
            tickangle=-45,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title='Standardized Sentiment',
            title_font=dict(color='royalblue'),
            tickfont=dict(color='royalblue'),
            range=[-2, 2]   # sentiment fixed range
        ),
        yaxis2=dict(
            title='Stock % Changes',
            title_font=dict(color='limegreen'),
            tickfont=dict(color='limegreen'),
            overlaying='y',
            side='right'
        )
    )

    fig.show()