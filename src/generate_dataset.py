import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

logging.getLogger("pandapower").setLevel(logging.ERROR)

BASE_MVA = 100

def generate_opf_dataset_pp(n_samples=200, perturb_range=(0.8, 1.0), save_path="test_data_118.csv", network="case118"):
    net = getattr(pn, network)()
    nominal_p = net.load.p_mw.values.copy()
    nominal_q = net.load.q_mvar.values.copy()
    data = []
    with tqdm(total=n_samples, desc="Generating AC-OPF dataset", ncols=100) as pbar:
        while len(data) < n_samples:
            scale = np.random.uniform(*perturb_range, size=len(net.load))
            net.load.p_mw = nominal_p * scale
            net.load.q_mvar = nominal_q * scale
            try:
                pp.runopp(net, verbose=False)
                if net.OPF_converged:
                    P = net.res_bus.p_mw.values / BASE_MVA
                    Q = net.res_bus.q_mvar.values / BASE_MVA
                    V = net.res_bus.vm_pu.values
                    theta = net.res_bus.va_degree.values
                    sample = np.concatenate([P, Q, V, theta]).astype(np.float32)
                    data.append(sample)
                    pbar.update(1)
            except KeyboardInterrupt:
                break
            except:
                continue
    n_buses = len(net.bus)
    columns = (
        [f"P{i}" for i in range(n_buses)] +
        [f"Q{i}" for i in range(n_buses)] +
        [f"V{i}" for i in range(n_buses)] +
        [f"Theta{i}" for i in range(n_buses)]
    )
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(save_path, index=False)
    print(f"Generated {len(df)} AC-OPF samples")
    print(f"Saved to '{save_path}'")
    return df
