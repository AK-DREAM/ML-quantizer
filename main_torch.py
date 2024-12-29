import numpy as np
from tqdm import tqdm
from lll import LLL_reduction as RED
from chol_diff import chol_rev
from numpy import linalg as la
import matplotlib.pyplot as plt
import torch
from schedulers import CosineAnnealingRestartLRScheduler, ExponentialLRScheduler, StepLRScheduler
from util import ORTH, URAN_matrix, GRAN, CLP, det, grader
import time

np.random.seed(19260817)

#TODO:(perhaps) change numpy to cupy for GPU acceleration
#TODO: generate theta-image
#Done
#TODO: design G
#TODO: add covariance to error



def calc_B(G, L):
	return la.cholesky(
	    np.mean(np.matmul(np.matmul(G, L), np.swapaxes(np.matmul(G, L), -1,
	                                                   -2)),
	            axis=0))


def calc_NSM(B_t, batch_size, n):
	B = B_t.detach().numpy()
	z = URAN_matrix(batch_size, n)
	y = z - CLP(B, z @ B)
	e = torch.tensor(y) @ B_t
	e2 = torch.norm(e, dim=-1) ** 2

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

		leaf_L = torch.tensor(L, requires_grad = True)
		B_t = torch.linalg.cholesky(
	    torch.mean((G @ leaf_L) @ (G @ leaf_L).transpose(-1, -2),
	            dim=0))

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
	n = 10
	batch_size = 128

	I = np.eye(n)
	I_swapped = I.copy()
	I_swapped[[0, 1]] = I_swapped[[1, 0]]
	G = [I]
	# G = [
	#     np.diag([1, 1]),
	#     np.diag([-1, 1]),
	#     np.diag([1, -1]),
	#     np.diag([-1, -1])
	# ]
	G = np.array(G)
	L = ORTH(RED(GRAN(n, n)))
	L = L / (det(L)**(1 / n))

	scheduler = CosineAnnealingRestartLRScheduler(initial_lr=mu0)
	# scheduler = ExponentialLRScheduler(initial_lr=mu0, gamma=v**(-1 / T))

	L = train(T, G, L, scheduler, n, batch_size)

	A = np.mean(np.matmul(np.matmul(G, L), np.swapaxes(np.matmul(G, L), -1,
	                                                   -2)),
	            axis=0)

	B = la.cholesky(A)
	B = B / (det(B)**(1 / n))

	grader(B)

	# print("B: ", B)

	np.save("B"+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), B)