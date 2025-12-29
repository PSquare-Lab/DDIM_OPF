# Data loading and preprocessing functions
import os
import pandas as pd
import numpy as np
import pandapower as pp

def load_ybus(net):
	nb = len(net.bus)
	Y = np.zeros((nb, nb), dtype=complex)
	for _, row in net.line.iterrows():
		i, j = int(row["from_bus"]), int(row["to_bus"])
		r, x = row["r_ohm_per_km"], row["x_ohm_per_km"]
		length = row.get("length_km", 1.0)
		y = 1 / complex(r * length, x * length)
		Y[i, i] += y; Y[j, j] += y; Y[i, j] -= y; Y[j, i] -= y
	return np.real(Y), np.imag(Y)

def load_data(data_csv, num_features):
	if not os.path.exists(data_csv):
		raise FileNotFoundError(f"Need {data_csv} generated from PF dataset code.")
	actual = pd.read_csv(data_csv)
	cols = actual.columns.tolist()
	if actual.shape[1] != num_features:
		raise ValueError(f"Dataset must have {num_features} columns.")
	data = actual.values.astype(np.float32)
	return data, cols
