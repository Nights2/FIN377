import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind
import pickle
import os

st.set_page_config(layout="wide", page_title="NFL Spending vs. Performance")

# --- Page Title and Introduction ---
st.title("🏈 NFL Team Spending vs. Performance Dashboard")
st.markdown("""
This interactive dashboard explores the relationship between NFL team spending patterns
on key positions (Cap and Average Contract) and their on-field performance (Win-Loss Percentage).
Discover high-ROI positions, analyze spending trends, and compare top, bottom, and 'other' spenders.
""")

# --- Data Loading ---
st.header("📈 Data Loading")

@st.cache_data
def load_data():
    try:
        df_combined = pd.read_csv('df_combined.csv')
        avg_wl_top_spenders = pd.read_csv('avg_wl_top_spenders.csv')
        avg_wl_bottom_spenders = pd.read_csv('avg_wl_bottom_spenders.csv')
        avg_wl_other_spenders = pd.read_csv('avg_wl_other_spenders.csv')
        correlation_matrix = pd.read_csv('correlation_matrix.csv', index_col=0)
        avg_top_performers_spending = pd.read_csv('avg_top_performers_spending.csv')
        avg_cap_pct_trends = pd.read_csv('avg_cap_pct_trends.csv')

        with open('ttest_results.pkl', 'rb') as f:
            ttest_results = pickle.load(f)
        st.success("All data loaded successfully!")
        return df_combined, avg_wl_top_spenders, avg_wl_bottom_spenders, avg_wl_other_spenders, correlation_matrix, ttest_results, avg_top_performers_spending, avg_cap_pct_trends
    except FileNotFoundError as e:
        st.error(f"Error loading data file: {e}. Please ensure all CSVs and pickle file are in the same directory as this script.")
        st.stop()

df_combined, avg_wl_top_spenders, avg_wl_bottom_spenders, avg_wl_other_spenders, correlation_matrix, ttest_results, avg_top_performers_spending, avg_cap_pct_trends = load_data()

# --- Sidebar Filters ---
st.sidebar.header("⚙️ Filter Options")

all_years = sorted(df_combined['Year'].unique())
all_teams = sorted(df_combined['Team'].unique())
all_spending_metrics = sorted(avg_wl_top_spenders['Spending_Metric'].unique())
all_cap_pct_metrics = [col for col in avg_cap_pct_trends.columns if col.endswith('_cap_pct')]

selected_years = st.sidebar.multiselect(
    "Select Year(s)",
    options=all_years,
    default=all_years
)

selected_teams = st.sidebar.multiselect(
    "Select Team(s)",
    options=all_teams,
    default=all_teams
)

selected_metrics = st.sidebar.multiselect(
    "Select Spending Metric(s) for W-L% Comparison",
    options=all_spending_metrics,
    default=all_spending_metrics
)

selected_cap_pct_metrics = st.sidebar.multiselect(
    "Select Cap % Metrics for Trends",
    options=all_cap_pct_metrics,
    default=all_cap_pct_metrics
)

# Filter relevant dataframes based on sidebar selections
filtered_df_combined = df_combined[df_combined['Year'].isin(selected_years) & df_combined['Team'].isin(selected_teams)]
filtered_avg_wl_top = avg_wl_top_spenders[avg_wl_top_spenders['Year'].isin(selected_years) & avg_wl_top_spenders['Spending_Metric'].isin(selected_metrics)]
filtered_avg_wl_bottom = avg_wl_bottom_spenders[avg_wl_bottom_spenders['Year'].isin(selected_years) & avg_wl_bottom_spenders['Spending_Metric'].isin(selected_metrics)]
filtered_avg_wl_other = avg_wl_other_spenders[avg_wl_other_spenders['Year'].isin(selected_years) & avg_wl_other_spenders['Spending_Metric'].isin(selected_metrics)]
filtered_avg_top_performers_spending = avg_top_performers_spending[avg_top_performers_spending['Year'].isin(selected_years)]
filtered_avg_cap_pct_trends = avg_cap_pct_trends[avg_cap_pct_trends['Year'].isin(selected_years)]

# --- Metric Cards ---
st.header("📊 Key Performance Indicators")

if not filtered_df_combined.empty:
    avg_wl = filtered_df_combined['W-L%'].mean()
    avg_pf = filtered_df_combined['PF'].mean()
    avg_pa = filtered_df_combined['PA'].mean()
    total_teams_years = len(filtered_df_combined)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Avg. Win-Loss %", value=f"{avg_wl:.3f}")
    with col2:
        st.metric(label="Avg. Points For", value=f"{avg_pf:.1f}")
    with col3:
        st.metric(label="Avg. Points Against", value=f"{avg_pa:.1f}")
    with col4:
        st.metric(label="Total Team-Years Analyzed", value=total_teams_years)
else:
    st.info("No data available for selected filters to display KPIs.")


# --- Summary Statistics ---
st.header("🔍 Data Overview & Statistics")

with st.expander("View Overall Summary Statistics"):
    st.subheader("Overall Summary Statistics for Combined Data")
    st.write(df_combined.describe())

# --- Correlations ---
with st.expander("View Spending vs. Performance Correlations"):
    st.subheader("Correlations of Position Spending Metrics with W-L%")
    spending_columns_all = [col for col in df_combined.columns if col.endswith(('_cap', '_avg')) and not col.startswith('Total')] # Total_cap, Total_avg
    correlation_with_wl_percentage = correlation_matrix.loc[spending_columns_all, 'W-L%']
    st.dataframe(correlation_with_wl_percentage.sort_values(ascending=False))

# --- Statistical Significance Test Results ---
with st.expander("View Statistical Significance Test Results (Top vs. Bottom Spenders)"):
    st.header("Statistical Significance Test Results")
    p_value_threshold = 0.05

    for metric, results in ttest_results.items():
        t_statistic = results.get('t_statistic')
        p_value = results.get('p_value')
        mean_top = results.get('mean_w_l_top_spenders')
        mean_bottom = results.get('mean_w_l_bottom_spenders')
        note = results.get('note', '')

        st.subheader(f"Metric: {metric}")
        if p_value is not None:
            significance = 'Statistically Significant' if p_value < p_value_threshold else 'Not Statistically Significant'
            st.write(f"  Mean W-L% (Top 3 Spenders): {mean_top:.3f}")
            st.write(f"  Mean W-L% (Bottom 3 Spenders): {mean_bottom:.3f}")
            st.write(f"  T-Statistic: {t_statistic:.3f}")
            st.write(f"  P-Value: {p_value:.3e}") # Scientific notation for p-value
            st.markdown(f"**Significance (alpha={p_value_threshold}): {significance}**")
        else:
            st.write(f"  Note: {note}")
        st.markdown("--- ")

# --- Visualizations ---
st.header("📈 Interactive Visualizations")

# Calculate and filter year_average
year_average = df_combined.groupby('Year').mean(numeric_only=True).reset_index()
year_average = year_average[year_average['Year'].isin(selected_years)]


# 1. Plot League Average Total Cap and Total Contract Value Over Time
st.subheader("League Average Total Spending Over Time")
if not year_average.empty:
    fig_total_avg = go.Figure()
    fig_total_avg.add_trace(go.Scatter(x=year_average['Year'], y=year_average['Total_cap'], mode='lines+markers', name='Average Total Cap'))
    fig_total_avg.add_trace(go.Scatter(x=year_average['Year'], y=year_average['Total_avg'], mode='lines+markers', name='Average Total Contract Value'))
    fig_total_avg.update_layout(
        title='League Average Total Cap and Total Contract Value Over Time',
        xaxis_title='Year',
        yaxis_title='Value (Millions $)',
        hovermode='x unified'
    )
    st.plotly_chart(fig_total_avg, use_container_width=True)
else:
    st.info("No data available for League Average Total Spending based on current filters.")


# 2. Plot League Average Position Spending Over Time
st.subheader("League Average Position Spending Over Time")
positions_to_plot_avg_spending = ['QB', 'RB', 'WR', 'OL', 'DL', 'S', 'ED', 'TE'] # Using all high-ROI and main positions

for pos in positions_to_plot_avg_spending:
    cap_col = f'{pos}_cap'
    avg_col = f'{pos}_avg'

    if cap_col in year_average.columns and avg_col in year_average.columns:
        fig_pos_avg = go.Figure()
        fig_pos_avg.add_trace(go.Scatter(x=year_average['Year'], y=year_average[cap_col], mode='lines+markers', name=f'{pos} Cap'))
        fig_pos_avg.add_trace(go.Scatter(x=year_average['Year'], y=year_average[avg_col], mode='lines+markers', name=f'{pos} Contract Average'))
        fig_pos_avg.update_layout(
            title=f'League Average {pos} Spending Over Time',
            xaxis_title='Year',
            yaxis_title='Value (Millions $)',
            hovermode='x unified'
        )
        st.plotly_chart(fig_pos_avg, use_container_width=True)
    else:
        pass # Skip if columns are not present after filtering.


# 3. Plot Comparative W-L% for Top 3, Bottom 3, and Other Spenders
st.subheader("Comparative W-L% for Top 3, Bottom 3, and Other Spenders")

if not selected_metrics:
    st.info("Please select at least one Spending Metric to view comparative W-L% plots.")
else:
    for metric in selected_metrics:
        top_data = filtered_avg_wl_top[filtered_avg_wl_top['Spending_Metric'] == metric]
        bottom_data = filtered_avg_wl_bottom[filtered_avg_wl_bottom['Spending_Metric'] == metric]
        other_data = filtered_avg_wl_other[filtered_avg_wl_other['Spending_Metric'] == metric]

        if not top_data.empty or not bottom_data.empty or not other_data.empty:
            fig_comp_wl = go.Figure()
            if not top_data.empty:
                fig_comp_wl.add_trace(go.Scatter(x=top_data['Year'], y=top_data['W-L%'], mode='lines+markers', name='Top 3 Spenders'))
            if not bottom_data.empty:
                fig_comp_wl.add_trace(go.Scatter(x=bottom_data['Year'], y=bottom_data['W-L%'], mode='lines+markers', name='Bottom 3 Spenders'))
            if not other_data.empty:
                fig_comp_wl.add_trace(go.Scatter(x=other_data['Year'], y=other_data['W-L%'], mode='lines+markers', name='Other Teams'))

            fig_comp_wl.update_layout(
                title=f'Average W-L% for Top 3, Bottom 3, and Other Spenders in {metric}',
                xaxis_title='Year',
                yaxis_title='Average Win-Loss Percentage',
                hovermode='x unified'
            )
            st.plotly_chart(fig_comp_wl, use_container_width=True)
        else:
            st.info(f"No comparative W-L% data available for {metric} based on current filters.")

# 4. Position Trends by % of Cap
st.subheader("Position Trends by Percentage of Cap Spending")
st.info("This chart shows how the average percentage of total cap spent on specific positions has changed over the selected years.")

if not filtered_avg_cap_pct_trends.empty and selected_cap_pct_metrics:
    fig_pct_trends = go.Figure()
    for metric_pct in selected_cap_pct_metrics:
        if metric_pct in filtered_avg_cap_pct_trends.columns:
            fig_pct_trends.add_trace(go.Scatter(x=filtered_avg_cap_pct_trends['Year'], y=filtered_avg_cap_pct_trends[metric_pct], mode='lines+markers', name=metric_pct.replace('_cap_pct', ' Cap %')))
    fig_pct_trends.update_layout(
        title='Average Percentage of Cap Spent per Position Over Time',
        xaxis_title='Year',
        yaxis_title='Average % of Cap',
        hovermode='x unified'
    )
    st.plotly_chart(fig_pct_trends, use_container_width=True)
else:
    st.info("Select at least one Cap % Metric to view spending trends.")


# 5. Radar Chart for Top Performers' Cap Composition
st.subheader("Cap Composition of Top 3 Performing Teams")
st.info("This radar chart visualizes the average cap spending allocation across positions for the top 3 performing teams in a selected year.")

# Dropdown for selecting a single year for the radar chart
radar_year_selection = st.selectbox(
    "Select a Year for Radar Chart",
    options=sorted(filtered_avg_top_performers_spending['Year'].unique(), reverse=True) if not filtered_avg_top_performers_spending.empty else all_years,
    index=0 if not filtered_avg_top_performers_spending.empty else None
)

if radar_year_selection and not filtered_avg_top_performers_spending.empty:
    radar_data = filtered_avg_top_performers_spending[filtered_avg_top_performers_spending['Year'] == radar_year_selection]

    if not radar_data.empty:
        # Ensure all cap spending columns are numeric for plotting
        cap_cols_for_radar = [col for col in radar_data.columns if col.endswith('_cap') and col != 'Total_cap']
        radar_df = radar_data[cap_cols_for_radar].mean(numeric_only=True)
        radar_df = radar_df.reset_index()
        radar_df.columns = ['Position', 'Average_Cap']
        radar_df['Position'] = radar_df['Position'].str.replace('_cap', '')

        fig_radar = px.line_polar(radar_df, r='Average_Cap', theta='Position', line_close=True, 
                                range_r=[0, radar_df['Average_Cap'].max() * 1.1],
                                title=f'Average Cap Spending Composition of Top 3 Performing Teams in {radar_year_selection}')
        fig_radar.update_traces(fill='toself')
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info(f"No top performer cap composition data for {radar_year_selection} based on current filters.")
else:
    st.info("Select at least one year and ensure data is available for top performing teams to view the radar chart.")


# --- Final Conclusions ---
st.header("💡 Conclusions")
st.markdown("""
Based on the analysis:

*   **High-ROI Positions**: Spending on **Quarterback (QB)** and **Defensive Line (DL)** shows strong positive correlations with winning percentage and related performance metrics. Statistical tests confirm that top spenders in these positions significantly outperform bottom spenders.
*   **Moderate-ROI Positions**: **Wide Receiver (WR)** and **Offensive Line (OL)** also show positive correlations, but often less pronounced than QB or DL, suggesting their impact is significant but perhaps not as singularly decisive.
*   **Lower-ROI Positions**: **Running Back (RB)** spending generally exhibits weaker correlations with overall team success, aligning with modern NFL trends that de-emphasize high investment in this position.
*   **Strategic Allocation**: Teams aiming for higher win percentages should prioritize investment in QB and DL. Balanced spending across other positions, or finding value, is also crucial for sustained success.
""")

st.subheader("Download Data & Run Locally")
st.markdown("""
To interact with this dashboard locally, follow these steps:

1.  **Download the data files** (ensure they are in the same directory as the `streamlit_dashboard.py` file):
    *   `df_combined.csv`
    *   `avg_wl_top_spenders.csv`
    *   `avg_wl_bottom_spenders.csv`
    *   `avg_wl_other_spenders.csv`
    *   `correlation_matrix.csv`
    *   `ttest_results.pkl`
    *   `avg_top_performers_spending.csv`
    *   `avg_cap_pct_trends.csv`

2.  **Save this script**:
    *   Copy the code from this cell and save it as `streamlit_dashboard.py`.

3.  **Install Streamlit** (if you haven't already) and other required libraries:
    *   `pip install streamlit pandas numpy plotly scipy`

4.  **Run the app** from your terminal in the directory where you saved the files:
    *   `streamlit run streamlit_dashboard.py`

This will open the interactive dashboard in your web browser!
""")
