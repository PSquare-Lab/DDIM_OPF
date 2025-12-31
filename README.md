# DDIM_OPF

## Setup Instructions

1. **Clone the repository**

2. **Create and activate a Python virtual environment**

```sh
python3 -m venv venv
source venv/bin/activate
```

3. **Install the required packages**

```sh
pip install -r requirements.txt
```

## Usage

### 1. Generate Data for 6-Bus System

This will generate a dataset with 5000 samples for the 6-bus system and save it as `actual_data_6.csv`:

```sh
python main.py --generate-dataset --dataset-samples 5000 --dataset-network case6ww --dataset-save-path actual_data_6.csv
```

### 2. Train the Model for DDIM (6-Bus)

This will train the model using the generated 6-bus dataset and perform DDIM sampling:

```sh
python main.py --ppcase case6ww --train --sample --ddim --epochs 100 --batch_size 64 --neurons 4096 --layers 2 --samples 1000 --ddim --dataset-save-path actual_data_6.csv
```

- Adjust `--epochs`, `--batch_size`, `--neurons`, `--layers`, and `--samples` as needed.
- The `--ppcase` argument specifies the pandapower case to use (e.g., `case6ww`, `case24_ieee_rts`, `case118`).

### 3. Use an Already Trained Model for DDPM Generation (6-Bus)

This will use the best saved model checkpoint and perform DDPM sampling:

```sh
python main.py --ppcase case6ww --sample --samples 1000
```

- By default, this will use DDPM sampling. To use DDIM, add `--ddim`.
- Make sure the checkpoint file (`best_ddim_denoiser.pth`) exists from a previous training run.

### 4. Data Analysis and Visualization

The `example.py` script provides a quick way to analyze and visualize the results of your actual and generated datasets. It computes key metrics (KL divergence, Wasserstein distance) and generates comparison plots for selected features and buses.

- Histogram comparison for selected features (customizable in the script)
- Scatter plots for P vs Q and V vs Theta for selected buses
- Buswise KL divergence bar plots for all features

**How to use:**
1. Ensure you have your ground truth and generated CSV files (e.g., `test_data.csv` and `generated.csv`).
2. Edit the top of `example.py` to select which features/buses to plot, if desired.
3. Run:

```sh
python example.py
```

## Notes
- You can use any supported pandapower case by changing the `--ppcase` argument.
- For other bus systems, adjust the dataset generation and training commands accordingly.
- All outputs and logs will be saved in the current directory by default.
