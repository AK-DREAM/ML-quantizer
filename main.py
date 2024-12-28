import numpy as np
from tqdm import tqdm
from lll import LLL_reduction as RED
from chol_diff import chol_rev


def ORTH(B):
    return np.linalg.cholesky(B @ B.T)
def URAN(n):
    return np.random.randn(n)
def GRAN(n, m):
    return np.random.normal(size = (n, m))

# def URAN_TEST(n): 
#     return np.array([0.342,0.46556,0.8])

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
T = Tr * 1000
mu0 = 0.1
v = 500
n = 2

G = [np.array([[1,0],[0,1]]), np.array([[-1,0],[0,-1]]), np.array([[1,0],[0,-1]]), np.array([[-1,0],[0,1]])]

L = ORTH(RED(GRAN(n, n)))
# L = np.array([[1,3,5],[2,4,3],[6,6,6]], dtype = float)
# L = ORTH(RED(L))
L = L / (det(L) ** (1 / n))

for t in tqdm(range(T)):
    mu = mu0 * (v ** (-t / (T - 1)))
    # mu = 0.1

    A = np.zeros((n, n))
    for g in G: 
        A += g@L@((g@L).T)
    A /= len(G)

    B = np.linalg.cholesky(A)
    B_diff = np.zeros((n, n))

    z = URAN(n)
    y = z - CLP(B, z @ B)
    e = y @ B
    e2 = np.linalg.norm(e) ** 2

    for i in range(n):
        for j in range(i):
            B_diff[i, j] = y[i] * e[j]
        B_diff[i, i] = (y[i] * e[i] - e2 / (n * B[i, i]))

    A_diff = chol_rev(B, B_diff) 
    A_diff = (np.tril(A_diff) + np.tril(A_diff).T) / n

    L_diff = np.zeros((n, n))
    for g in G:
        gL_diff = A_diff @ g @ L * 2
        L_diff += g.T @ gL_diff
    L_diff /= len(G)

    # print(B_diff)
    # print(A_diff)
    # print(L_diff)

    L -= mu * L_diff

    if t % Tr == Tr - 1:
        L = ORTH(RED(L))
        L = L / (det(L) ** (1 / n))



A = np.zeros((n, n))
for g in G: 
    A += g@L@((g@L).T)
A /= len(G)

B = np.linalg.cholesky(A)
B = ORTH(RED(B))
B = B / (det(B) ** (1 / n))

test = 100000
NSM = 0
sigma = 0
for i in tqdm(range(test)):
    z = URAN(n)
    y = z - CLP(B, z @ B)
    e = y @ B
    e2 = np.linalg.norm(e) ** 2
    val = 1 / n * e2
    NSM += val
    sigma += val * val

NSM = NSM / test
sigma = (sigma / test - NSM ** 2) / (test - 1)

print("NSM:", NSM, " sigma:", sigma)
print("B: ", B)