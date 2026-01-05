import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from .utils import denormalize
from .physics import RH_sq, RG_sq


def apply_zero_threshold(data, threshold=1e-6):
	data[np.abs(data) <= threshold] = 0.0
	return data

def make_ddim_timesteps(TIMESTEPS, ddim_steps, device):
	step = TIMESTEPS // ddim_steps
	return torch.arange(TIMESTEPS - 1, -1, -step, device=device, dtype=torch.long)

def ddim_sample(n_samples, model, NUM_FEATURES, cols, range_eps, xmin, alphas_cumprod, G, B, p_min, p_max, q_min, q_max, v_min, v_max, TIMESTEPS, DDIM_STEPS, GUIDANCE_LAMBDA, ETA, batch_size=1000, OUT_CSV="ddim_stochastic_constrained_generated.csv", device="cpu", zero_threshold=1e-6):
	print(f"Sampling {n_samples} samples using stochastic DDIM in batches of {batch_size}...")
	total_start_time = time.time()
	total_generated = 0
	if os.path.exists(OUT_CSV):
		os.remove(OUT_CSV)
	while total_generated < n_samples:
		current_batch_size = min(batch_size, n_samples - total_generated)
		batch_start_time = time.time()
		x_t = torch.randn(current_batch_size, NUM_FEATURES, device=device)
		re = torch.tensor(range_eps, device=device, dtype=torch.float32)
		xm = torch.tensor(xmin, device=device, dtype=torch.float32)
		timesteps = make_ddim_timesteps(TIMESTEPS, DDIM_STEPS, device)
		for step_idx, t_idx in enumerate(tqdm(timesteps.tolist(), desc=f"Batch {total_generated//batch_size + 1} DDIM", leave=False)):
			t = torch.full((current_batch_size,), t_idx, device=device, dtype=torch.long)
			with torch.no_grad():
				eps_pred = model(x_t, t)
				sA = torch.sqrt(alphas_cumprod[t]).unsqueeze(1)
				sO = torch.sqrt(1 - alphas_cumprod[t]).unsqueeze(1)
				x0 = (x_t - sO * eps_pred) / sA
			x_den = ((x0 + 1.0) / 2.0) * re + xm
			x_den = x_den.detach().clone().requires_grad_(True)
			R_total = RH_sq(x_den, NUM_FEATURES//4, G, B) + RG_sq(x_den, NUM_FEATURES//4, p_min, p_max, q_min, q_max, v_min, v_max)
			g_phys = torch.autograd.grad(R_total.sum(), x_den)[0]
			grad_norm = g_phys * (re / 2.0)
			lam_t = GUIDANCE_LAMBDA * ((t.float() + 1.0) / TIMESTEPS).unsqueeze(1)
			x0_guided = x0 - lam_t * grad_norm
			x0_guided = torch.clamp(x0_guided, -1.0, 1.0)
			a_bar_t = alphas_cumprod[t].unsqueeze(1)
			if step_idx < len(timesteps) - 1:
				t_prev_idx = timesteps[step_idx + 1].item()
				t_prev = torch.full((current_batch_size,), t_prev_idx, device=device, dtype=torch.long)
				a_bar_prev = alphas_cumprod[t_prev].unsqueeze(1)
			else:
				a_bar_prev = torch.ones_like(a_bar_t)
				t_prev_idx = -1
			if t_prev_idx == -1:
				x_t = x0_guided.detach()
				continue
			denom = (1.0 - a_bar_t).clamp(min=1e-8)
			eps_pred = (x_t - torch.sqrt(a_bar_t) * x0_guided) / torch.sqrt(denom)
			frac = ((1.0 - a_bar_prev) / denom).clamp(min=0.0)
			inner = (1.0 - (a_bar_t / a_bar_prev)).clamp(min=0.0)
			sigma = ETA * torch.sqrt(frac * inner)
			x_prev_mean = torch.sqrt(a_bar_prev) * x0_guided + torch.sqrt((1.0 - a_bar_prev - sigma**2).clamp(min=0.0)) * eps_pred
			x_t = x_prev_mean + sigma * torch.randn_like(x_t)
			x_t = x_t.detach()
		gen_norm = x_t.cpu().numpy()
		gen_denorm = denormalize(gen_norm, xmin, range_eps)
		gen_denorm = np.nan_to_num(gen_denorm, nan=0.0, posinf=0.0, neginf=0.0)
		
		# Set very small values to zero (for zero-injection/slack buses)
		gen_denorm = apply_zero_threshold(gen_denorm, threshold=zero_threshold)
		
		df_batch = pd.DataFrame(gen_denorm, columns=cols)
		df_batch.to_csv(OUT_CSV, mode='a', index=False, header=not os.path.exists(OUT_CSV))
		batch_end_time = time.time()
		batch_duration = batch_end_time - batch_start_time
		total_generated += current_batch_size
		print(f"Saved {total_generated}/{n_samples} samples. (Batch time: {batch_duration:.2f}s)")
	total_duration = time.time() - total_start_time
	print(f"Sampling Complete. Total time: {total_duration:.2f} seconds.")

def ddpm_sample(n_samples, model, NUM_FEATURES, cols, range_eps, xmin, alphas, betas, alphas_cumprod, G, B, p_min, p_max, q_min, q_max, v_min, v_max, TIMESTEPS, GUIDANCE_LAMBDA, batch_size=1000, OUT_CSV="ddpm_constrained_generated.csv", device="cpu", zero_threshold=1e-6):
	print(f"Sampling {n_samples} samples using standard DDPM in batches of {batch_size}...")
	total_start_time = time.time()
	total_generated = 0
	model.eval()
	if os.path.exists(OUT_CSV):
		os.remove(OUT_CSV)
	while total_generated < n_samples:
		current_batch_size = min(batch_size, n_samples - total_generated)
		batch_start_time = time.time()
		x_t = torch.randn(current_batch_size, NUM_FEATURES, device=device)
		re = torch.tensor(range_eps, device=device, dtype=torch.float32)
		xm = torch.tensor(xmin, device=device, dtype=torch.float32)
		for t_idx in tqdm(list(reversed(range(TIMESTEPS))), desc=f"Batch {total_generated//batch_size + 1} DDPM", leave=False):
			t = torch.full((current_batch_size,), t_idx, device=device, dtype=torch.long)
			with torch.no_grad():
				eps = model(x_t, t)
			sA = torch.sqrt(alphas_cumprod[t]).unsqueeze(1)
			sO = torch.sqrt(1 - alphas_cumprod[t]).unsqueeze(1)
			x0 = (x_t - sO * eps) / sA
			x_den = ((x0 + 1.0) / 2.0) * re + xm
			x_den = x_den.detach().clone().requires_grad_(True)
			R_total = RH_sq(x_den, NUM_FEATURES//4, G, B) + RG_sq(x_den, NUM_FEATURES//4, p_min, p_max, q_min, q_max, v_min, v_max)
			g_phys = torch.autograd.grad(R_total.sum(), x_den)[0]
			grad_norm = g_phys * (re / 2.0)
			lam_t = GUIDANCE_LAMBDA * ((t.float() + 1.0) / TIMESTEPS).unsqueeze(1)
			x0_guided = torch.clamp(x0 - lam_t * grad_norm, -1.0, 1.0)
			a_t = alphas[t].unsqueeze(1)
			b_t = betas[t].unsqueeze(1)
			a_bar_t = alphas_cumprod[t].unsqueeze(1)
			a_bar_prev = alphas_cumprod[t-1].unsqueeze(1) if t_idx > 0 else torch.ones_like(a_bar_t)
			sigma_t = torch.clamp(b_t * (1 - a_bar_prev) / (1 - a_bar_t), min=0.0)
			coef_x = torch.sqrt(a_t) * (1 - a_bar_prev) / (1 - a_bar_t)
			coef_x0 = torch.sqrt(a_bar_prev) * b_t / (1 - a_bar_t)
			z = torch.randn_like(x_t) if t_idx > 0 else torch.zeros_like(x_t)
			x_t = coef_x * x_t + coef_x0 * x0_guided + torch.sqrt(sigma_t) * z
		gen_denorm = denormalize(x_t.detach().cpu().numpy(), xmin, range_eps)
		gen_denorm = np.nan_to_num(gen_denorm, nan=0.0, posinf=0.0, neginf=0.0)
		
		# Set very small values to zero (for zero-injection/slack buses)
		gen_denorm = apply_zero_threshold(gen_denorm, threshold=zero_threshold)
		
		df_batch = pd.DataFrame(gen_denorm, columns=cols)
		df_batch.to_csv(OUT_CSV, mode='a', index=False, header=not os.path.exists(OUT_CSV))
		batch_duration = time.time() - batch_start_time
		total_generated += current_batch_size
		print(f"Saved {total_generated}/{n_samples} samples. (Batch time: {batch_duration:.2f}s)")
	print(f"DDPM Sampling Complete. Total time: {time.time() - total_start_time:.2f}s")
