import torch
from src.data import load_ybus, load_data
from src.utils import normalize, linear_beta_schedule
from src.model import MLPDenoiser
from src.train import train_denoiser
from src.sample import ddim_sample, ddpm_sample
from src.generate_dataset import generate_opf_dataset_pp

# --- Config ---
NUM_BUS = 6
BASE_MVA = 100
DATA_CSV = "actual_data_6.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FEATURES = 4 * NUM_BUS
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4
TIMESTEPS = 1000
DDIM_STEPS = 30
ETA = 0.2
SAMPLE_COUNT = 5000
GUIDANCE_LAMBDA = 1e-4



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train and sample with DDIM/DDPM.")
    parser.add_argument('--train', action='store_true', help='Train the denoiser model')
    parser.add_argument('--sample', action='store_true', help='Sample using trained model')
    parser.add_argument('--ddim', action='store_true', help='Use DDIM sampling (default: DDPM)')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--neurons', type=int, default=4096, help='Number of neurons in MLP')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers in MLP')
    parser.add_argument('--samples', type=int, default=SAMPLE_COUNT, help='Number of samples to generate')
    parser.add_argument('--ppcase', type=str, default='case24_ieee_rts', help='pandapower case name to use (e.g., case6ww, case24_ieee_rts, case118)')
    parser.add_argument('--generate-dataset', action='store_true', help='Generate AC-OPF dataset')
    parser.add_argument('--dataset-samples', type=int, default=5000, help='Number of samples for dataset generation')
    parser.add_argument('--dataset-network', type=str, default='case118', help='Network for dataset generation (e.g., case6ww, case24_ieee_rts, case118)')
    parser.add_argument('--dataset-save-path', type=str, default='generated_data.csv', help='Path to save generated dataset')
    args = parser.parse_args()


    if args.generate_dataset:
        generate_opf_dataset_pp(
            n_samples=args.dataset_samples,
            perturb_range=(0.8, 1.0),
            save_path=args.dataset_save_path,
            network=args.dataset_network
        )
        return

    # --- Data & Network ---
    import pandapower.networks as pn
    net = getattr(pn, args.ppcase)()
    G_np, B_np = load_ybus(net)
    G = torch.tensor(G_np, dtype=torch.float32, device=DEVICE)
    B = torch.tensor(B_np, dtype=torch.float32, device=DEVICE)

    data, cols = load_data(DATA_CSV, NUM_FEATURES)
    xmin, xmax = data.min(axis=0), data.max(axis=0)
    range_eps = (xmax - xmin).copy()
    range_eps[range_eps == 0] = 1e-6
    data_tensor = torch.tensor(normalize(data, xmin, range_eps), dtype=torch.float32, device=DEVICE)

    # --- Limits ---
    p_min_np = xmin[:NUM_BUS]
    p_max_np = xmax[:NUM_BUS]
    q_min_np = xmin[NUM_BUS:2 * NUM_BUS]
    q_max_np = xmax[NUM_BUS:2 * NUM_BUS]
    v_min_np = xmin[2 * NUM_BUS:3 * NUM_BUS]
    v_max_np = xmax[2 * NUM_BUS:3 * NUM_BUS]

    p_min = torch.tensor(p_min_np, device=DEVICE, dtype=torch.float32)
    p_max = torch.tensor(p_max_np, device=DEVICE, dtype=torch.float32)
    q_min = torch.tensor(q_min_np, device=DEVICE, dtype=torch.float32)
    q_max = torch.tensor(q_max_np, device=DEVICE, dtype=torch.float32)
    v_min = torch.tensor(v_min_np, device=DEVICE, dtype=torch.float32)
    v_max = torch.tensor(v_max_np, device=DEVICE, dtype=torch.float32)

    # --- Beta/Alpha Schedules ---
    betas = linear_beta_schedule(TIMESTEPS, device=DEVICE)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)

    model = None
    if args.train:
        model = train_denoiser(
            data_tensor, NUM_BUS, args.layers, args.neurons, args.epochs, args.batch_size, LR, TIMESTEPS, range_eps, xmin,
            p_min, p_max, q_min, q_max, v_min, v_max, G, B, GUIDANCE_LAMBDA, device=DEVICE
        )
    else:
        model = MLPDenoiser(4*NUM_BUS, args.neurons, num_layers=args.layers, timesteps=TIMESTEPS).to(DEVICE)
        checkpoint = torch.load("best_ddim_denoiser.pth", map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])

    if args.sample:
        if args.ddim:
            ddim_sample(
                args.samples, model, NUM_FEATURES, cols, range_eps, xmin, alphas_cumprod, G, B,
                p_min, p_max, q_min, q_max, v_min, v_max, TIMESTEPS, DDIM_STEPS, GUIDANCE_LAMBDA, ETA, device=DEVICE
            )
        else:
            ddpm_sample(
                args.samples, model, NUM_FEATURES, cols, range_eps, xmin, alphas, betas, alphas_cumprod, G, B,
                p_min, p_max, q_min, q_max, v_min, v_max, TIMESTEPS, GUIDANCE_LAMBDA, device=DEVICE
            )

if __name__ == "__main__":
    main()
