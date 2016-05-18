#!/usr/bin/env python3
import numpy as np

# Load the term-document matrix
print("Loading")
termdoc = A = np.load("termdoc.npy")
m, n = A.shape
k = 50
q = num_iterations = 2

# Stage A
print("Stage A")
omega = np.random.normal(0, 1, (n, 2*k))
Y = np.dot(A, omega)
for i in range(num_iterations):
  Y = np.dot(A.T, Y)
  Y = np.dot(A, Y)
Q, _ = np.linalg.qr(Y)

# Stage B
print("Stage B")
B = np.dot(Q.T, A)
Uhat, sigma, Vt = np.linalg.svd(B, full_matrices=False)
U = np.dot(Q, Uhat)

print("Checking")
Ahat = np.dot(U, np.dot(np.diag(sigma), Vt))

print(((A-Ahat)**2).mean())
