import torch
import torch.nn as nn

class MLPDenoiser(nn.Module):
	def __init__(self, dim, neurons, num_layers=2, timesteps=1000):
		super().__init__()
		self.timesteps = timesteps
		layers = [
			nn.Linear(dim + 1, neurons),
			nn.LayerNorm(neurons),
			nn.SiLU()
		]
		for _ in range(num_layers):
			layers.extend([
				nn.Linear(neurons, neurons),
				nn.LayerNorm(neurons),
				nn.SiLU()
			])
		layers.append(nn.Linear(neurons, dim))
		self.net = nn.Sequential(*layers)

	def forward(self, x, t):
		t = t.float().unsqueeze(1) / self.timesteps
		return self.net(torch.cat([x, t], 1))
