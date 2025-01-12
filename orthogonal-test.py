import numpy as np
from tqdm import tqdm
from lll import LLL_reduction as RED
from chol_diff import chol_rev
from numpy import linalg as la
from numba import prange, jit
import matplotlib.pyplot as plt
from schedulers import CosineAnnealingRestartLRScheduler, ExponentialLRScheduler, StepLRScheduler
from util import ORTH, URAN_matrix, GRAN, CLP, det, grader, Theta_Image_Drawer, Loss_Drawer
import time
import os

np.random.seed(114514)

def calc_B(G, L):
    return la.cholesky(
        np.mean(np.matmul(np.matmul(G, L), np.swapaxes(np.matmul(G, L), -1,
                                                       -2)),
                axis=0))


def calc_NSM(B, batch_size, n):
    z = URAN_matrix(batch_size, n)
    y = z - CLP(B, z @ B)
    e = y @ B
    e2 = la.norm(e, axis=-1)**2

    NSM = (det(B)**(-2 / n)) * e2 / n
    return y, e, e2, np.mean(NSM)


def calc_B_diff(y, e, e2, B, n):
    B_diff = np.tril(np.einsum('ij,ik->ijk', y, e))
    B_diff.transpose(1, 2, 0)[np.diag_indices(n)] -= np.outer(
        1 / np.diag(B), e2 / n)
    B_diff = np.mean(B_diff, axis=0)
    B_diff = B_diff * 2 * (det(B)**(-2 / n)) / n
    return B_diff


def calc_A_diff(B, B_diff, n):
    A_diff = chol_rev(B, B_diff)
    A_diff = (np.tril(A_diff) + np.tril(A_diff).T) / n
    return A_diff


def calc_L_diff(G, A_diff, L):
    return np.mean(np.matmul(np.swapaxes(G, -1, -2),
                             np.matmul(A_diff, np.matmul(G, L)) * 2),
                   axis=0)


def calc_diff(y, e, e2, G, L, B, n):
    B_diff = calc_B_diff(y, e, e2, B, n)
    A_diff = calc_A_diff(B, B_diff, n)
    L_diff = calc_L_diff(G, A_diff, L)
    return L_diff


def reduce_L(L, n):
    # L = ORTH(RED(L))
    L = L / (np.abs(det(L)) ** (1 / n))
    return L

def train(T, Tr, G, L, scheduler, n, batch_size):

    for t in tqdm(range(T)):
        mu = scheduler.step()

        B = calc_B(G, L)

        y, e, e2, NSM = calc_NSM(B, batch_size, n)

        L_diff = calc_diff(y, e, e2, G, L, B, n)

        L -= mu * L_diff

        if t % Tr == Tr - 1:
            L = reduce_L(L, n)
    return L


def solve(n, m):
    Tr = 100
    T = Tr * 1000
    mu0 = 10
    v = 1000
    batch_size = 128

    I = np.eye(n)
    I2 = np.eye(n)
    for i in range(0, m):
        I2[i,i] = -1
    G = [I, I2]
    G = np.array(G)
    L = GRAN(n, n)
    L = L / (np.abs(det(L))**(1 / n))

    # scheduler = CosineAnnealingRestartLRScheduler(initial_lr=mu0)
    scheduler = ExponentialLRScheduler(initial_lr=mu0, gamma=v**(-1 / T))

    L = train(T, Tr, G, L, scheduler, n, batch_size)

    B = calc_B(G, L)
    B = ORTH(RED(B))
    B = B / (det(B)**(1 / n))

    NSM, sigma = grader(B)
    data = {
        'B': B,
        'NSM': NSM,
        'G': G,
        'sigma': sigma,
        'n': n,
        'batch_size': batch_size,
        'T': T,
        'Tr': Tr,
        'mu0': mu0,
        'filename': __file__,
    }

    save_path = f"./data/{n}_dim/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    """
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)
    print("B: ", B)
    """
    
    filename = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())

    np.savez(
        save_path + "B" + filename,
        **data)


if __name__ == "__main__":
    solve(14,2)
    for i in range(2,23):
        for j in range(1,i//2+1):
            solve(i,j)