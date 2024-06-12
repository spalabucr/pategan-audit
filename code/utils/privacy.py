import math
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler

import torch


def ma_updated(num_teachers, votes, lamda, L):
    alpha_list = np.zeros([len(L)])

    for j in range(len(votes)):
        n0 = (num_teachers - votes[j, :]).squeeze().numpy()
        n1 = votes[j, :].squeeze().numpy()

        # Update moments accountant
        q = np.log(2 + lamda * abs(n0 - n1)) - np.log(4.0) - (lamda * abs(n0 - n1))
        q = np.exp(q)

        # Compute alpha
        for l in range(len(L)):
            temp1 = 2 * (lamda**2) * (l + 1) * (l + 2)
            temp2 = (1 - q) * (((1 - q) / (1 - q * np.exp(2 * lamda)))**(l + 1)) + q * np.exp(2 * lamda * (l + 1))
            alpha_list[l] = alpha_list[l] + np.min([temp1, np.log(temp2)])

    return alpha_list


def ma_synthcity(num_teachers, votes, lamda, alpha):
    n0 = (num_teachers - votes).numpy()
    n1 = votes.numpy()

    # Update moments accountant
    qbase = lamda * np.abs(n0 - n1)
    q = (2 + qbase) / (4 * np.exp(qbase))

    # Compute alpha
    alpha_list = []
    for lidx in range(len(alpha)):
        upper = 2 * lamda**2 * (lidx + 1) * (lidx + 2)
        t = (1 - q) * np.power((1 - q) / (1 - np.exp(2 * lamda) * q), lidx + 1)
        # BUG!!!
        t = np.log(t + q * np.exp(2 * lamda * lidx + 1))
        alpha_list.append(np.clip(t, a_min=0, a_max=upper).sum())
    return alpha_list


def ma_borai(num_teachers, clean_votes, lap_scale, l_list):
    q = (2 + lap_scale * torch.abs(2 * clean_votes - num_teachers)
         ) / (4 * torch.exp(lap_scale * torch.abs(2 * clean_votes - num_teachers)))

    update = []
    for l in l_list:
        a = 2 * lap_scale * lap_scale * l * (l + 1)
        t_one = (1 - q) * torch.pow((1 - q) / (1 - math.exp(2 * lap_scale) * q), l)
        t_two = q * torch.exp(2 * lap_scale * l)
        # BUG!!!
        t = t_one + t_two
        update.append(torch.clamp(t, max=a).sum())

    return update


def ma_smartnoise(num_teachers, votes, lap_scale, l_list, device="cpu"):
    q = (2 + lap_scale * torch.abs(2 * votes - num_teachers)) / (
        4 * torch.exp(lap_scale * torch.abs(2 * votes - num_teachers))).to(device)

    alpha = []
    for l_val in l_list:
        a = 2 * lap_scale ** 2 * l_val * (l_val + 1)
        t_one = (1 - q) * torch.pow((1 - q) / (1 - math.exp(2 * lap_scale) * q), l_val)
        t_two = q * torch.exp(2 * lap_scale * l_val)
        # BUG!!!
        t = t_one + t_two
        alpha.append(torch.clamp(t, max=a).sum())

    return alpha


def get_vuln_records_dists(df, k=5):
    vuln_dists = np.zeros(len(df))

    processor = MinMaxScaler()
    X_proc = processor.fit_transform(df)

    tree = KDTree(X_proc, metric="l2", leaf_size=3)

    for i in tqdm(range(len(X_proc)), desc="dists", leave=False):
        dist, _ = tree.query(X_proc[i].reshape(1, -1), k=k + 1)
        vuln_dists[i] = dist[0][1:].mean()
    return vuln_dists
