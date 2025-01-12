from numba import prange, jit, njit
import numpy as np
import math
from numpy import linalg as la
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.linalg as sl


def ORTH(B):
	return la.cholesky(B @ B.T)


def URAN(n):
	return np.random.rand(n)


# @njit
def URAN_matrix(n, m):
	return np.random.rand(n, m)


def GRAN(n, m):
	return np.random.normal(size=(n, m))


@jit(nopython=True, fastmath=True)
def CLP_single(G, r):
	n = G.shape[0]
	C = np.inf
	i = n
	d = np.array([n - 1] * n)
	Lambda = np.zeros(n + 1)
	F = np.zeros((n, n))
	F[n - 1] = r.copy()
	p = np.zeros(n)
	u = np.zeros(n)
	Delta = np.zeros(n)
	while True:
		while True:
			if i != 0:
				i = i - 1
				for j in range(d[i], i, -1):
					F[j - 1, i] = F[j, i] - u[j] * G[j, i]
				p[i] = F[i, i] / G[i, i]
				u[i] = np.round(p[i])
				y = (p[i] - u[i]) * G[i, i]
				Delta[i] = np.sign(y)
				Lambda[i] = Lambda[i + 1] + y**2
			else:
				uu = u.copy()
				C = Lambda[0]
			if Lambda[i] >= C:
				break
		m = i
		while True:
			if i == n - 1:
				return uu
			else:
				i = i + 1
				u[i] = u[i] + Delta[i]
				Delta[i] = -Delta[i] - np.sign(Delta[i])
				y = (p[i] - u[i]) * G[i][i]
				Lambda[i] = Lambda[i + 1] + y**2
			if Lambda[i] < C:
				break

		d[m:i] = i
		for j in range(m - 1, -1, -1):
			if d[j] < i:
				d[j] = i
			else:
				break


@jit(nopython=True, parallel=True, fastmath=True)
def CLP(G, r_batch):
	res = np.zeros_like(r_batch)
	for i in prange(r_batch.shape[0]):
		res[i] = CLP_single(G, r_batch[i])
	return res


det = la.det


def grader(B, test=100000, batchsize=128):
	n = B.shape[0]
	B = B / (det(B)**(1 / n))
	G = 0
	sigma = 0
	for i in tqdm(range(test)):
		z = URAN_matrix(batchsize, n)
		y = z - CLP(B, z @ B)
		e = y @ B
		e2 = la.norm(e, axis=-1)**2
		val = 1 / n * e2
		G += val.sum()
		sigma += (val**2).sum()
	T = test * batchsize
	G = G / T
	sigma = math.sqrt((sigma / T - G**2) / (T - 1))
	print(G, sigma)
	return (G, sigma)


class Theta_Image_Drawer:

	def __init__(self, UP=5):
		self.fig, self.ax = plt.subplots(figsize=(8, 6))
		self.UP = UP
		self.ax.set_xlabel('r^2')
		self.ax.set_xlim(0, UP)
		self.ax.set_ylabel('N(B,r)')
		self.ax.set_yscale('log')
		self.ax.grid(True)

	def add(self, B, label=None, style={}):
		n = B.shape[0]
		data = []

		def dfs(dep, sum, nowvec):
			if sum > self.UP:
				return
			if dep == -1:
				data.append(sum)
				return
			# sum + (now[dep] + i * B[dep][dep]) ** 2 <= UP
			# |now[dep] + i * B[dep][dep]| <= sqrt(UP - sum)
			# -now[dep] - sqrt(UP - sum) <= i * B[dep][dep] <= -now[dep] + sqrt(UP - sum)

			l = math.ceil(
			    (-nowvec[dep] - math.sqrt(self.UP - sum)) / B[dep, dep])
			r = math.floor(
			    (-nowvec[dep] + math.sqrt(self.UP - sum)) / B[dep, dep])

			for i in range(l, r + 1):
				newvec = nowvec + i * B[dep]
				dfs(dep - 1, sum + newvec[dep]**2, newvec)

		dfs(n - 1, 0, np.zeros(n))
		sorted_data = list(np.sort(data))
		cdf = [i for i in range(1, len(sorted_data) + 1)]
		cdf.append(len(sorted_data))
		sorted_data.append(self.UP)
		if "linestyle" in style:
			linestyle = style["linestyle"]
		else:
			linestyle = ""
		if "alpha" in style:
			alpha = style["alpha"]
		else:
			alpha = 1
		if "color" in style:
			color = style["color"]
		else:
			color = "blue"
		self.ax.step(sorted_data,
		             cdf,
		             where='post',
		             linestyle=linestyle,
		             color=color,
		             alpha=alpha,
		             markersize=0,
		             label=label)

	def show(self, path=None):
		self.ax.legend()
		if path == None:
			self.fig.show()
		else:
			self.fig.savefig(path)

	def __delete__(self):
		plt.close(self.fig)


def theta_image(B, path=None):
	A = Theta_Image_Drawer()
	A.add(B, label='Theta Image', style={"linestyle": '-', "alpha": 1})
	A.show(path=path)


class Loss_Drawer:

	def __init__(self, start):
		self.start = start
		self.cnt = 0
		self.fig, self.ax = plt.subplots(figsize=(8, 6))
		self.ax.set_xlabel('cnt')
		self.ax.set_ylabel('loss')
		self.ax.grid(True)
		self.loss = []

	def __deinit__(self):
		self.fig.close()

	def add(self, val):
		self.cnt += 1
		if self.cnt > self.start:
			self.loss.append(val)

	def show(self, path=None):
		self.ax.plot(np.arange(len(self.loss)) + self.start, self.loss)
		self.ax.legend()
		if path == None:
			self.fig.show()
		else:
			self.fig.savefig(path)


def to_exact(B, threshold, intervals):
	n = B.shape[0]
	A = B @ B.T
	data = []

	def to_coeff(vec):
		nvec = np.outer(vec, vec)
		nvec[np.diag_indices(n)] /= 2
		nvec = nvec + nvec.T
		return nvec[np.tril_indices(n)]

	def dfs(dep, sum, nowvec, intvec):
		if sum > threshold:
			return
		if dep == -1:
			# print(sum, nowvec, nowvec @ A @ nowvec)
			data.append([sum, to_coeff(intvec)])
			assert np.max(nowvec - (intvec @ B)) < 1e-3
			return
		# sum + (now[dep] + i * B[dep][dep]) ** 2 <= UP
		# |now[dep] + i * B[dep][dep]| <= sqrt(UP - sum)
		# -now[dep] - sqrt(UP - sum) <= i * B[dep][dep] <= -now[dep] + sqrt(UP - sum)

		l = math.ceil(
		    (-nowvec[dep] - math.sqrt(threshold - sum)) / B[dep, dep])
		r = math.floor(
		    (-nowvec[dep] + math.sqrt(threshold - sum)) / B[dep, dep])

		for i in range(l, r + 1):
			intvec[dep] = i
			newvec = nowvec + i * B[dep]
			dfs(dep - 1, sum + newvec[dep]**2, newvec, intvec)

	dfs(n - 1, 0, np.zeros(n), np.zeros(n))
	# sort data based on sum
	data.sort(key=lambda x: x[0])
	coeff = []
	for interval in intervals:
		selected = [
		    x[1] for x in data if interval[0] <= x[0] and x[0] <= interval[1]
		]
		selected = np.array(selected)
		# coeff = difference of selected adjacent elements
		tmp = np.diff(selected, axis=0)
		coeff.append(tmp)
	coeff = np.vstack(coeff)
	print(coeff.shape)
	res = sl.null_space(coeff)
	print(res.shape)
	dim = res.shape[1]
	if dim == 0:
		raise Exception("No exact solution! Try with smaller intervals")
	res = res / res[0]
	mat = np.zeros((dim, n, n))
	for i in range(dim):
		mat[i][np.tril_indices(n)] = res[:, i]
		mat[i] = mat[i] + mat[i].T
		mat[i][np.diag_indices(n)] /= 2
		mat[i] = la.cholesky(mat[i])
	return dim, mat
