import numpy as np
from tqdm import tqdm
from lll import LLL_reduction as RED
from chol_diff import chol_rev
from numpy import linalg as la
import matplotlib.pyplot as plt
import torch
from schedulers import CosineAnnealingRestartLRScheduler, ExponentialLRScheduler, StepLRScheduler
from util import ORTH, URAN_matrix, GRAN, CLP, det, grader, Theta_Image_Drawer
import time
import os

np.random.seed(19260817)

def calc_NSM(B_t, batch_size, n):
	B = B_t.detach().numpy()
	z = URAN_matrix(batch_size, n)
	y = z - CLP(B, z @ B)
	e = torch.tensor(y) @ B_t
	e2 = torch.norm(e, dim=-1)**2

	NSM = (torch.prod(torch.diagonal(B_t))**(-2 / n)) * e2 / n
	return torch.mean(NSM)


def reduce_L(L):
	L = ORTH(RED(L))
	L = L / (det(L)**(1 / n))
	return L


def train(T, G, L, scheduler, n, batch_size):

	G = torch.tensor(G)

	for t in tqdm(range(T)):
		mu = scheduler.step()

		leaf_L = torch.tensor(L, requires_grad=True)
		B_t = torch.linalg.cholesky(
		    torch.mean((G @ leaf_L) @ (G @ leaf_L).transpose(-1, -2), dim=0))

		NSM = calc_NSM(B_t, batch_size, n)

		NSM.backward()

		L -= mu * leaf_L.grad.numpy()

		if t % Tr == Tr - 1:
			L = reduce_L(L)
	
	return L


if __name__ == "__main__":

	Tr = 100
	T = Tr * 1000
	mu0 = 0.5
	v = 1000
	n = 13
	batch_size = 128

	I = np.eye(n)
	I_13 = I.copy()
	I_13[12, 12] = -1
	G = [I, I_13]
	
	G = np.array(G)
	L = ORTH(RED(GRAN(n, n)))
	L = L / (det(L)**(1 / n))

	# scheduler = CosineAnnealingRestartLRScheduler(initial_lr=mu0)
	scheduler = ExponentialLRScheduler(initial_lr=mu0, gamma=v**(-1 / T))

	L = train(T, G, L, scheduler, n, batch_size)

	A = np.mean(np.matmul(np.matmul(G, L), np.swapaxes(np.matmul(G, L), -1,
	                                                   -2)),
	            axis=0)

	B = la.cholesky(A)
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
	

	filename = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
	
	np.savez(
	    save_path + "B" + filename,
	    **data)
