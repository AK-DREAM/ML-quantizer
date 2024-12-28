import numpy as np
from tqdm import tqdm
from lll import LLL_reduction as RED
from chol_diff import chol_rev
import torch

def ORTH(B):
    return torch.linalg.cholesky(B @ B.T) 
def URAN(n):
    return torch.randn(n) 
def GRAN(n, m):
    return torch.normal(mean=torch.zeros(n, m), std=torch.ones(n, m))

# def URAN_TEST(n): 
#     return torch.tensor([0.342,0.46556,0.8])


def CLP(GG, rr):
    G = GG.numpy()
    r = rr.numpy()

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
                ret = torch.from_numpy(uu).float()
                return ret
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

def mydet(B):
    res = 1
    for i in range(n):
        res = res * B[i, i]
    return res


Tr = 100
T = Tr * 100
mu0 = 0.1
v = 500
n = 10

G = [np.eye(n)]
torch_G = [torch.from_numpy(g).float() for g in G]

L = ORTH(RED(GRAN(n, n)))
L = L / (torch.linalg.det(L) ** (1 / n))
L.requires_grad_()

for t in tqdm(range(T)):
    mu = mu0 * (v ** (-t / (T - 1)))

    A = torch.zeros(n, n)
    for g in torch_G: 
        A = A + torch.matmul(g, L).matmul(torch.matmul(g, L).T)
    A = A / len(G)

    B = torch.linalg.cholesky(A)

    with torch.no_grad():
        z = URAN(n)
        y = z - CLP(B, z @ B)
    e = y @ B
    V = torch.linalg.det(B)

    e2 = torch.linalg.norm(e) ** 2
    NSM = (V**(-2/n))*e2/n

    # B.retain_grad()
    # A.retain_grad()
    L.grad = None
    # B.grad = None
    # A.grad = None
    NSM.backward()

    # print(B.grad)
    # print(A.grad)
    # print(L.grad)

    with torch.no_grad():
        L -= mu * L.grad
        if t % Tr == Tr - 1:
            L.data = ORTH(RED(L.data))
            L.data = L.data / (torch.linalg.det(L.data) ** (1 / n))



with torch.no_grad():
    A = torch.zeros(n, n)
    for g in torch_G: 
        A = A + torch.matmul(g, L).matmul(torch.matmul(g, L).T)
    A = A / len(G)
    B = torch.linalg.cholesky(A)
    B = ORTH(RED(B))
    B = B / (torch.linalg.det(B) ** (1 / n))

    test = 10000
    NSM = 0
    sigma = 0
    for i in tqdm(range(test)):
        z = URAN(n)
        y = z - CLP(B, z @ B)
        e = y @ B
        e2 = torch.linalg.norm(e) ** 2
        val = 1 / n * e2
        NSM += val
        sigma += val * val

    NSM = NSM / test
    sigma = (sigma / test - NSM ** 2) / (test - 1)

    print("G:", NSM, " sigma:", sigma)
    print("B: ", B)