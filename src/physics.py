import torch
import math

def power_residuals(x_den, NUM_BUS, G, B):
	p = x_den[:, :NUM_BUS]
	q = x_den[:, NUM_BUS:2 * NUM_BUS]
	v = x_den[:, 2 * NUM_BUS:3 * NUM_BUS]
	th = x_den[:, 3 * NUM_BUS:4 * NUM_BUS]

	th_r = th * math.pi / 180

	vb = v.unsqueeze(-1)
	vj = v.unsqueeze(-2)
	dth = th_r.unsqueeze(-1) - th_r.unsqueeze(-2)

	cosd = torch.cos(dth)
	sind = torch.sin(dth)

	G_b = G.unsqueeze(0)
	B_b = B.unsqueeze(0)

	P = (vb * vj * (G_b * cosd + B_b * sind)).sum(dim=-1)
	Q = (vb * vj * (G_b * sind - B_b * cosd)).sum(dim=-1)

	return torch.cat([p - P, q - Q], dim=1)

def RH_sq(x_den, NUM_BUS, G, B):
	return (power_residuals(x_den, NUM_BUS, G, B) ** 2).sum(dim=1)

def RG_sq(x_den, NUM_BUS, p_min, p_max, q_min, q_max, v_min, v_max):
	p = x_den[:, :NUM_BUS]
	q = x_den[:, NUM_BUS:2 * NUM_BUS]
	v = x_den[:, 2 * NUM_BUS:3 * NUM_BUS]

	viol_p_upper = torch.relu(p - p_max)
	viol_p_lower = torch.relu(p_min - p)
	viol_q_upper = torch.relu(q - q_max)
	viol_q_lower = torch.relu(q_min - q)
	viol_v_upper = torch.relu(v - v_max)
	viol_v_lower = torch.relu(v_min - v)

	viol = torch.cat([
		viol_p_upper, viol_p_lower,
		viol_q_upper, viol_q_lower,
		viol_v_upper, viol_v_lower,
	], dim=1)

	return (viol ** 2).sum(dim=1)
