import numpy as np
from tqdm import tqdm
from lll import LLL_reduction as RED
import torch

def ORTH(B):
    return np.linalg.cholesky(B @ B.T)
def URAN(n):
    return np.random.randn(n)
def GRAN(n, m):
    return np.random.normal(size = (n, m))
def CLP(G, r):
    n = G.shape[0]
    C = np.float64("inf")
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
                Lambda[i] = Lambda[i + 1] + y ** 2
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
                Lambda[i] = Lambda[i + 1] + y ** 2
            if Lambda[i] < C:
                break
        for j in range(m, i):
            d[j] = i
        for j in range(m - 1, -1, -1):
            if d[j] < i:
                d[j] = i
            else:
                break
def det(B):
    res = 1
    for i in range(n):
        res = res * B[i, i]
    return res

Tr = 100
T = Tr * 10000
batchSize = 1
mu0 = 0.001
v = 250
n = 10

I = np.eye(n)
#I_swapped = I.copy() 
#I_swapped[[0, 1]] = I_swapped[[1, 0]]  
#G = [torch.tensor(I), torch.tensor(I_swapped)]
G = [torch.tensor(I)]

L = ORTH(RED(GRAN(n, n)))
L = L / (det(L) ** (1 / n))
with tqdm(range(T)) as te:
    for t in te:
        mu = mu0 * (v ** (-t / (T - 1)))
        
        leaf_L = torch.tensor(L, requires_grad = True)
        A = torch.zeros(n, n, dtype=torch.float64)
        for g in G:

            A += (g @ leaf_L) @ (g @ leaf_L).T
        A /= len(G)
        B_t = torch.linalg.cholesky(A)
        B = B_t.detach().numpy()
        Loss = torch.tensor(0.)
        for i in range(batchSize):
            z = URAN(n)
            y = z - CLP(B, z @ B)
            e = torch.tensor(y) @ B_t
            Loss += torch.norm(e) ** 2
        Loss /= B_t.diagonal().prod() ** (2 / n)
        Loss /= batchSize
        # te.set_postfix(loss = Loss.item())
        Loss.backward()
        
        L = L - mu * leaf_L.grad.numpy()
        
        if t % Tr == Tr - 1:
            L = ORTH(RED(L))
            L = L / (det(L) ** (1 / n))

B = L
test = 10000
G = 0
sigma = 0
for i in tqdm(range(test)):
    z = URAN(n)
    y = z - CLP(B, z @ B)
    e = y @ B
    e2 = np.linalg.norm(e) ** 2
    val = 1 / n * e2
    G += val
    sigma += val ** 2

G = G / test
sigma = (sigma / test - G ** 2) / (test - 1)

print("G:", G, " sigma:", sigma)
