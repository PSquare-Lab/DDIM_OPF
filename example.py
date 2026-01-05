import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.metrics import kl_divergence, calculate_w1_distance_for_eta

COLOR_ACTUAL = "#2F4F4F"
COLOR_GEN = "#DC143C"


# For histograms, specify any 4 columns (e.g., 'P1', 'Q3', 'P4', 'V9')
histogram_cols = ['P1', 'Q3', 'P4', 'V1']
# For scatter plots, specify bus indices (0-based) for which to plot P vs Q and V vs Theta
scatter_buses = [0, 1, 2, 3]

# File paths
actual_csv = 'test_data.csv'
generated_csv = 'generated.csv'

actual = pd.read_csv(actual_csv)
generated = pd.read_csv(generated_csv)

p_cols = [c for c in actual.columns if c.startswith('P')]
q_cols = [c for c in actual.columns if c.startswith('Q')]
v_cols = [c for c in actual.columns if c.startswith('V')]
theta_cols = [c for c in actual.columns if c.startswith('Th') or c.startswith('Theta')]

selected_p = [p_cols[i] for i in scatter_buses]
selected_q = [q_cols[i] for i in scatter_buses]
selected_v = [v_cols[i] for i in scatter_buses]
selected_theta = [theta_cols[i] for i in scatter_buses]

# 1. Histogram comparison for selected features
plt.figure(figsize=(16, 8))
for idx, col in enumerate(histogram_cols):
	if idx >= 4: break
	plt.subplot(2, 2, idx+1)
	if col.startswith('P') or col.startswith('V'):
		plt.hist(actual[col], bins=40, density=True, alpha=0.5, label='Actual', color=COLOR_ACTUAL)
		plt.hist(generated[col], bins=40, density=True, alpha=0.5, label='Generated', color=COLOR_GEN)
	else:
		plt.hist(actual[col], bins=40, density=True, alpha=0.5, label='Actual', color=COLOR_GEN)
		plt.hist(generated[col], bins=40, density=True, alpha=0.5, label='Generated', color=COLOR_ACTUAL)
	plt.title(f'Histogram: {col}')
	plt.xlabel(col)
	plt.ylabel('Density')
	plt.legend()
plt.tight_layout()
plt.savefig('histogram_comparison_selected_features.png')
plt.close()

# 2. Scatter plots for P vs Q and V vs Theta for selected buses
plt.figure(figsize=(16, 8))
for idx, bus in enumerate(scatter_buses):
	if idx >= 4: break
	plt.subplot(2, 2, idx+1)
	plt.scatter(actual[selected_p[idx]], actual[selected_q[idx]], alpha=0.4, label='Actual P', s=10, color=COLOR_ACTUAL)
	plt.scatter(generated[selected_p[idx]], generated[selected_q[idx]], alpha=0.4, label='Generated Q', s=10, color=COLOR_GEN)
	plt.title(f'Bus {bus+1}: P vs Q')
	plt.xlabel(selected_p[idx])
	plt.ylabel(selected_q[idx])
	plt.legend()
plt.tight_layout()
plt.savefig('scatter_pq_selected_features.png')
plt.close()

plt.figure(figsize=(16, 8))
for idx, bus in enumerate(scatter_buses):
	if idx >= 4: break
	plt.subplot(2, 2, idx+1)
	plt.scatter(actual[selected_v[idx]], actual[selected_theta[idx]], alpha=0.4, label='Actual V', s=10, color=COLOR_ACTUAL)
	plt.scatter(generated[selected_v[idx]], generated[selected_theta[idx]], alpha=0.4, label='Generated Theta', s=10, color=COLOR_GEN)
	plt.title(f'Bus {bus+1}: V vs Theta')
	plt.xlabel(selected_v[idx])
	plt.ylabel(selected_theta[idx])
	plt.legend()
plt.tight_layout()
plt.savefig('scatter_vtheta_selected_features.png')
plt.close()

# 3. Mean KL divergence and W1 distance for selected features
kl_vals = []
for p_col, q_col, v_col, t_col in zip(p_cols, q_cols, v_cols, theta_cols):
	kl_vals.append(kl_divergence(actual[p_col], generated[p_col]))
	kl_vals.append(kl_divergence(actual[q_col], generated[q_col]))
	kl_vals.append(kl_divergence(actual[v_col], generated[ v_col]))
	kl_vals.append(kl_divergence(actual[t_col], generated[t_col]))
mean_kl = np.mean(kl_vals)
print(f"Mean KL divergence: {mean_kl:.4f}")

w1 = calculate_w1_distance_for_eta(actual, generated, p_cols, q_cols, v_cols, theta_cols)
print(f"W1 distance (selected features): {w1:.4f}")

# 4. Buswise KL divergence: one subplot per feature, bar plot per bus
sns.set_theme(style="whitegrid", context="talk", rc={"axes.grid": False})

bus_indices = np.arange(1, len(p_cols)+1)
buswise_kl = { 'P': [], 'Q': [], 'V': [], 'Theta': [] }
for i in range(len(p_cols)):
	buswise_kl['P'].append(kl_divergence(actual[p_cols[i]], generated[p_cols[i]]))
	buswise_kl['Q'].append(kl_divergence(actual[q_cols[i]], generated[q_cols[i]]))
	buswise_kl['V'].append(kl_divergence(actual[v_cols[i]], generated[v_cols[i]]))
	buswise_kl['Theta'].append(kl_divergence(actual[theta_cols[i]], generated[theta_cols[i]]))

feature_titles = [
	'Active Power (P)',
	'Reactive Power (Q)',
	'Voltage Magnitude (V)',
	'Voltage Angle (θ)'
]
feature_keys = ['P', 'Q', 'V', 'Theta']

fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
bar_color = '#C94A5E'
for idx, (ax, key, title) in enumerate(zip(axes, feature_keys, feature_titles)):
	ax.bar(bus_indices, buswise_kl[key], color=bar_color)
	ax.set_ylabel('KL Divergence', fontsize=13)
	ax.set_title(title, fontsize=16, fontweight='bold')
	ax.set_xticks(bus_indices)
	if idx < 3:
		ax.set_xlabel("")
	else:
		ax.set_xlabel('Bus Number', fontsize=13)
plt.tight_layout(h_pad=1.5)
plt.savefig('buswise_kl_divergence.png')
plt.close()
