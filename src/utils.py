import torch

def normalize(x, xmin, range_eps):
	"""Normalize data to [-1, 1] range."""
	return 2 * (x - xmin) / range_eps - 1

def denormalize(xn, xmin, range_eps):
	"""Denormalize data from [-1, 1] range back to original scale."""
	return ((xn + 1) / 2) * range_eps + xmin

def split_vec(x, NUM_BUS):
	p, q, v, th = x[:, :NUM_BUS], x[:, NUM_BUS:2*NUM_BUS], \
				  x[:, 2*NUM_BUS:3*NUM_BUS], x[:, 3*NUM_BUS:4*NUM_BUS]
	return torch.cat([p, th], 1), torch.cat([q, v], 1)

def concat_vec(x1, x2, NUM_BUS):
	p, th = x1[:, :NUM_BUS], x1[:, NUM_BUS:]
	q, v  = x2[:, :NUM_BUS], x2[:, NUM_BUS:]
	return torch.cat([p, q, v, th], 1)

def linear_beta_schedule(T, start=1e-4, end=0.02, device=None):
	return torch.linspace(start, end, T, device=device)
