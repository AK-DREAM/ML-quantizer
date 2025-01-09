import numpy as np
from tqdm import tqdm
from lll import LLL_reduction as RED
from chol_diff import chol_rev
from numpy import linalg as la
from numba import prange, jit
import matplotlib.pyplot as plt
from schedulers import CosineAnnealingRestartLRScheduler, ExponentialLRScheduler, StepLRScheduler
from util import ORTH, URAN_matrix, GRAN, CLP, det, grader, theta_image
import time
import os

np.random.seed(114514)

#TODO:(perhaps) change numpy to cupy for GPU acceleration
#TODO: design G
#TODO: add covariance to error


def calc_B(G, L, n, m):
	B1 = la.cholesky(np.mean(np.matmul(np.matmul(G, L[:m,:m]), np.swapaxes(np.matmul(G, L[:m,:m]), -1, -2)), axis=0))
	return np.vstack([np.hstack([B1, np.zeros((m, n-m))]), L[m:]])


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


def calc_A_diff(B, B_diff, m):
	A_diff = chol_rev(B, B_diff)
	A_diff = (np.tril(A_diff) + np.tril(A_diff).T) / m
	return A_diff


def calc_L_diff(G, A_diff, L):
	return np.mean(np.matmul(np.swapaxes(G, -1, -2), np.matmul(A_diff, np.matmul(G, L)) * 2), axis=0)


def calc_diff(y, e, e2, G, L, B, n, m):
	B_diff = calc_B_diff(y, e, e2, B, n)
	A_diff = calc_A_diff(B[:m,:m], B_diff[:m,:m], m)
	L_diff = calc_L_diff(G, A_diff, L[:m,:m])
	L_diff = np.vstack([np.hstack([L_diff, np.zeros((m, n-m))]), B_diff[m:]])
	return L_diff


def reduce_L(L):
	L = ORTH(RED(L))
	L = L / (det(L)**(1 / n))
	return L


def train(T, G, L, scheduler, n, m, batch_size):

	for t in tqdm(range(T)):
		mu = scheduler.step()

		B = calc_B(G, L, n, m)

		y, e, e2, NSM = calc_NSM(B, batch_size, n)

		L_diff = calc_diff(y, e, e2, G, L, B, n, m)

		L -= mu * L_diff

		if t % Tr == Tr - 1:
			L = reduce_L(L)

	return L


if __name__ == "__main__":
	Tr = 100
	T = Tr * 1000
	mu0 = 0.5
	v = 1000
	n = 10
	m = 10  # restrictions only on first m vectors 
	batch_size = 128

	I = np.eye(m)
	G = [I]  # array of m*m matrices
	"""
	G = [
		np.array([[1,0],[0,1]]),
	    np.array([[0,-1],[1,1]]),
		np.array([[-1,-1],[1,0]]),
		np.array([[-1,0],[0,-1]]),
		np.array([[0,1],[-1,-1]]),
		np.array([[1,1],[-1,0]]),
		np.array([[1,0],[-1,-1]]),
	    np.array([[0,-1],[-1,0]]),
		np.array([[-1,-1],[0,1]]),
		np.array([[-1,0],[1,1]]),
		np.array([[0,1],[1,0]]),
		np.array([[1,1],[0,-1]]),
	]
	"""
	G = np.array(G)
	L = ORTH(RED(GRAN(n, n)))
	L = L / (det(L)**(1 / n))

	scheduler = CosineAnnealingRestartLRScheduler(initial_lr=mu0)
	# scheduler = ExponentialLRScheduler(initial_lr=mu0, gamma=v**(-1 / T))

	L = train(T, G, L, scheduler, n, m, batch_size)

	B = calc_B(G, L, n, m)
	B = ORTH(RED(B))
	B = B / (det(B)**(1 / n))

	NSM, sigma_squared = grader(B)
	sigma = np.sqrt(sigma_squared)
	data = {
	    'B': B,
	    'NSM': NSM,
	    'G': G,
	    'sigma': sigma,
	    'n': n,
	    'batch_size': batch_size,
	    'T': T,
	    'mu0': mu0
	}
	save_path = f"./data/{n}_dim/"
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	np.set_printoptions(suppress=True)
	np.set_printoptions(precision=4)
	print("B: ", B)

	np.savez(
	    save_path + "B" + time.strftime("%Y%m%d-%H-%M-%S", time.localtime()),
	    **data)

	theta_image(B)
