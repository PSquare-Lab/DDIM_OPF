import torch
import torch.optim as optim
import torch.nn as nn
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from .model import MLPDenoiser
from .utils import linear_beta_schedule

def train_denoiser(data_tensor, NUM_BUS, layer_num, neurons, EPOCHS, BATCH_SIZE, LR, TIMESTEPS, range_eps, xmin, p_min, p_max, q_min, q_max, v_min, v_max, G, B, GUIDANCE_LAMBDA, log_filename=None, device="cpu"):
	if not log_filename:
		log_filename = f"training_log_layers_{layer_num}_neurons_{neurons}_bus_{NUM_BUS}.csv"
	start_time = time.time()
	model_dim = 4 * NUM_BUS
	model = MLPDenoiser(model_dim, neurons, num_layers=layer_num, timesteps=TIMESTEPS).to(device)
	opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3, betas=(0.9, 0.999))
	mse = nn.MSELoss()
	train_data, val_data = train_test_split(data_tensor, test_size=0.1, random_state=42)
	Ntrain, Nval = train_data.shape[0], val_data.shape[0]
	print(f"Training denoiser with validation monitoring...")
	best_val = float("inf")
	history = []
	for ep in range(EPOCHS):
		model.train()
		perm = torch.randperm(Ntrain, device=device)
		tot_loss = 0.0
		for i in range(0, Ntrain, BATCH_SIZE):
			b = train_data[perm[i:i+BATCH_SIZE]]
			x = b
			Bc = x.size(0)
			if Bc == 0:
				continue
			t = torch.randint(0, TIMESTEPS, (Bc,), device=device)
			eps = torch.randn_like(x)
			betas = linear_beta_schedule(TIMESTEPS, device=device)
			alphas = 1 - betas
			alphas_cumprod = torch.cumprod(alphas, 0)
			sA = torch.sqrt(alphas_cumprod[t]).unsqueeze(1)
			sO = torch.sqrt(1 - alphas_cumprod[t]).unsqueeze(1)
			x_t = sA * x + sO * eps
			pred = model(x_t, t)
			loss = mse(pred, eps)
			opt.zero_grad()
			loss.backward()
			opt.step()
			tot_loss += loss.item() * Bc
		train_loss = tot_loss / Ntrain
		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for i in range(0, Nval, BATCH_SIZE):
				b = val_data[i:i+BATCH_SIZE]
				x = b
				Bc = x.size(0)
				if Bc == 0:
					continue
				t = torch.randint(0, TIMESTEPS, (Bc,), device=device)
				eps = torch.randn_like(x)
				betas = linear_beta_schedule(TIMESTEPS, device=device)
				alphas = 1 - betas
				alphas_cumprod = torch.cumprod(alphas, 0)
				sA = torch.sqrt(alphas_cumprod[t]).unsqueeze(1)
				sO = torch.sqrt(1 - alphas_cumprod[t]).unsqueeze(1)
				x_t = sA * x + sO * eps
				pred = model(x_t, t)
				val_loss += mse(pred, eps).item() * Bc
		val_loss /= Nval
		history.append({"Epoch": ep + 1, "Train_Loss": train_loss, "Val_Loss": val_loss})
		if (ep+1) % 100 == 0:
			print(f"Epoch {ep+1:4d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
		if val_loss < best_val:
			best_val = val_loss
			torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "val_loss": val_loss}, "best_ddim_denoiser.pth")
	log_df = pd.DataFrame(history)
	log_df.to_csv(log_filename, index=False)
	print(f"Training logs saved to {log_filename}")
	print("Loading best model from checkpoint...")
	checkpoint = torch.load("best_ddim_denoiser.pth")
	model.load_state_dict(checkpoint["model"])
	end_time = time.time()
	elapsed_time = end_time - start_time
	print(f"Training done. Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes).\n")
	return model
