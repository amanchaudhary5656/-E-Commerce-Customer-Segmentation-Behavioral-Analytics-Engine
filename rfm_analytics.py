import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. SYNTHETIC DATASET GENERATION (Pandas)
# ==========================================
np.random.seed(42)
n_orders = 1000

# Generating raw transaction logs
customer_ids = np.random.randint(1001, 1100, size=n_orders)
order_dates = pd.date_range(start='2025-01-01', end='2026-06-30', periods=n_orders)
order_values = np.random.exponential(scale=120, size=n_orders) + 10  # Purchase amounts ($)

df_transactions = pd.DataFrame({
    'CustomerID': customer_ids,
    'OrderDate': order_dates,
    'OrderValue': order_values
})

print("=== Raw Transaction Log Sample (Pandas) ===")
print(df_transactions.head())

# ==========================================
# 2. RFM AGGREGATION & CLEANING (Pandas)
# ==========================================
snapshot_date = pd.to_datetime('2026-07-01')

rfm_table = df_transactions.groupby('CustomerID').agg({
    'OrderDate': lambda x: (snapshot_date - x.max()).days, # Recency (Days since last purchase)
    'CustomerID': 'count',                                 # Frequency (Total order count)
    'OrderValue': 'sum'                                    # Monetary Value (Total amount spent)
}).rename(columns={
    'OrderDate': 'Recency',
    'CustomerID': 'Frequency',
    'OrderValue': 'Monetary'
})

print("\n=== Aggregated RFM Metrics ===")
print(rfm_table.head())

# ==========================================
# 3. QUANTILE SCORING FROM SCRATCH (NumPy)
# ==========================================
# Assigning RFM Scores (1-4) using pure NumPy quantile boundaries
def get_rfm_scores(df):
    recency_vals = df['Recency'].values
    freq_vals = df['Frequency'].values
    monetary_vals = df['Monetary'].values

    # Calculate 25th, 50th, 75th percentiles via NumPy
    r_q = np.percentile(recency_vals, [25, 50, 75])
    f_q = np.percentile(freq_vals, [25, 50, 75])
    m_q = np.percentile(monetary_vals, [25, 50, 75])

    # Recency Score: Lower days = Better score (4 to 1)
    r_score = np.select(
        [recency_vals <= r_q[0], recency_vals <= r_q[1], recency_vals <= r_q[2]],
        [4, 3, 2], default=1
    )

    # Frequency & Monetary Scores: Higher = Better score (1 to 4)
    f_score = np.select(
        [freq_vals <= f_q[0], freq_vals <= f_q[1], freq_vals <= f_q[2]],
        [1, 2, 3], default=4
    )
    
    m_score = np.select(
        [monetary_vals <= m_q[0], monetary_vals <= m_q[1], monetary_vals <= m_q[2]],
        [1, 2, 3], default=4
    )

    return r_score, f_score, m_score

rfm_table['R_Score'], rfm_table['F_Score'], rfm_table['M_Score'] = get_rfm_scores(rfm_table)
rfm_table['Total_RFM'] = rfm_table['R_Score'] + rfm_table['F_Score'] + rfm_table['M_Score']

# Assign Customer Segments based on total RFM Score
segment_conditions = [
    rfm_table['Total_RFM'] >= 10,
    (rfm_table['Total_RFM'] >= 7) & (rfm_table['Total_RFM'] < 10),
    (rfm_table['Total_RFM'] >= 5) & (rfm_table['Total_RFM'] < 7)
]
segment_labels = ['VIP Champions', 'Loyal Customers', 'At-Risk Customers']
rfm_table['Segment'] = np.select(segment_conditions, segment_labels, default='Lost Customers')

print("\n=== Segment Summary ===")
print(rfm_table['Segment'].value_counts())

# ==========================================
# 4. DASHBOARD VISUALIZATION (Matplotlib)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Customer Segment Breakdown (Bar Chart)
segment_counts = rfm_table['Segment'].value_counts()
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

axes[0].bar(segment_counts.index, segment_counts.values, color=colors, edgecolor='black', alpha=0.85)
axes[0].set_title('Customer Distribution by RFM Segment', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Customer Segment')
axes[0].set_ylabel('Number of Customers')
axes[0].tick_params(axis='x', rotation=15)

# Plot 2: Frequency vs Monetary Scatter Plot
axes[1].scatter(rfm_table['Frequency'], rfm_table['Monetary'], c=rfm_table['Total_RFM'], cmap='viridis', s=60, alpha=0.8)
axes[1].set_title('Frequency vs. Monetary Value (Colored by RFM Score)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Purchase Frequency (Order Count)')
axes[1].set_ylabel('Monetary Value ($)')

plt.tight_layout()
plt.show()