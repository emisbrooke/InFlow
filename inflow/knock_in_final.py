import argparse
from importlib import reload
import numpy as np
import torch
import sys
import h5py
import os
import functools
from tqdm import tqdm
import helper_funcs as hf
import knock_in_funcs as ki
from pathlib import Path
import pandas as pd
reload(hf)
reload(ki)

print = functools.partial(print, flush=True)


def calc_ki(data_dir, model_dir, tissue, lam, thresh=0.1):
    t=tissue

    ki_path=Path(f'outputs/knock_in/tf_means_ki_{tissue}_lam_{lam}')
    if ki_path.exists():
        print(f'FILE EXISTS FOR {tissue}')
        return


    df = pd.read_csv('examples/ages_chosen.csv')

    age_t = df[df['tissue']==tissue]
    age_y = age_t['age_young'].item()
    age_o = age_t['age_old'].item()

    tf_data_y, data_path = hf.load_data(data_dir, tissue, age_y, 'tf') 
    tf_data_o, data_path = hf.load_data(data_dir, tissue, age_o, 'tf') 
    nTF, nC = tf_data_y.shape

    print(f'On tissues {tissue}, tf data shape is', tf_data_y.shape)
    theta, m = hf.load_pt_param(tissue, model_dir, age_o, lam, 'TF', threshold=thresh)
    l1_list = []
    tf_means=np.zeros((nTF, nTF))
    for i in range(nTF):
        print(f'Began TF {i}')
        samples=ki.knock_in(theta, tf_data_o, m, tf_idx=i, int_burn=10000, nSamples=nC, int_save=100)
        samples = samples.cpu().numpy()
        l1 = np.abs(samples.mean(axis=1)-tf_data_y.mean(axis=1)).sum()
        l1_list.append(l1)
        tf_means[i]=samples.mean(axis=1) 
        print(f'Completed TF {i}, l1={l1}')

    np.save(f'outputs/knock_in/tf_means_ki_{tissue}_lam_{lam}', tf_means)
    np.save(f'outputs/knock_in/tf_l1_ki_{tissue}_lam_{lam}', np.array(l1_list))


'''
for t in df['tissue'].unique():
    lam = df[df['tissue']==t]['lambda'].item()
    calc_ki(data_dir, model_dir, t, lam)
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', type=str, required=True)
    args = parser.parse_args()

    # Constants
    MODEL_DIR = Path('outputs/models/droplet')
    DATA_DIR = Path('data/droplet')
    
    # Get lambda for this specific tissue
    df_lams = pd.read_csv('best_lams_droplet_all.csv')
    lam_val = df_lams[df_lams['tissue'] == args.tissue]['lambda'].item()

    calc_ki(DATA_DIR, MODEL_DIR, args.tissue, lam_val)



