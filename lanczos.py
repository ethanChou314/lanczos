import numpy as np
from qr import approx_until_converged


DEBUG = False


def normalize(v, eps=1e-32):
	norm = np.linalg.norm(v)
	if norm < eps:
		raise ValueError("Normalization breakdown: near-zero vector.")
	return v / norm


def tridiag_from_coeffs(alpha, beta):
	alpha = np.asarray(alpha)
	beta = np.asarray(beta)

	if alpha.ndim != 1 or beta.ndim != 1:
		raise ValueError("alpha and beta must be 1D arrays")
	if len(beta) != len(alpha) - 1:
		raise ValueError("beta must be one element shorter than alpha")

	n = len(alpha)
	T = np.zeros((n, n))
	for i in range(n):
		T[i, i] = alpha[i]
		if i > 0:
			T[i, i - 1] = beta[i - 1]
		if i < n - 1:
			T[i, i + 1] = beta[i]
	return T


def lanczos(A, b, n, m, reorthogonalize=True):
	if b.shape[0] != n:
		raise ValueError(f"Expected b of length {n}, got {len(b)}")

	Q = np.zeros((n, m))
	alpha = np.zeros(m)
	beta = np.zeros(m - 1)

	Q[:, 0] = normalize(b)
	q_im1 = np.zeros((n, 1))

	for i in range(m):
		q_i = Q[:, i].reshape(-1, 1)
		w = A @ q_i
		alpha[i] = float(q_i.T @ w)
		v = w - alpha[i] * q_i - (beta[i - 1] * q_im1 if i > 0 else 0)

		if reorthogonalize:
			# Full modified Gram–Schmidt against all previous q_j
			for j in range(i + 1):
				q_j = Q[:, j:j+1]              # shape (n, 1)
				coeff = float(q_j.T @ v)       # scalar
				v = v - coeff * q_j           # (n,1) - scalar*(n,1)

			# second pass (helps with roundoff)
			for j in range(i + 1):
				q_j = Q[:, j:j+1]              # shape (n, 1)
				coeff = float(q_j.T @ v)       # scalar
				v = v - coeff * q_j

		if i < m - 1:
			beta[i] = np.linalg.norm(v)
			q_ip1 = normalize(v)
			Q[:, i + 1] = q_ip1.flatten()
			q_im1 = q_i

	return Q, alpha, beta


def exact_diagonalization(A, m):
	A = np.asarray(A, dtype=float)
	n = A.shape[0]

	if A.ndim != 2 or n != A.shape[1]:
		raise ValueError("A must be square")
	if not (0 < m <= n):
		raise ValueError("Invalid value of m")

	b = normalize(np.random.rand(n))
	Q, alpha, beta = lanczos(A, b, n, m)
	T = tridiag_from_coeffs(alpha, beta)

	return Q, T, Q.T


def lanczos_with_qr(A, m):
    Q, T, _ = exact_diagonalization(A, m)
    return np.linalg.eigvals(T)

# ----------- newly implemented code -----------
def _rel_err(new, old, eps=1e-15):
	if old is None or new.shape != old.shape:
		return np.inf
	return np.linalg.norm(new - old) / np.linalg.norm(new)


def _select_evals(evals, k, mode):
	if evals is None:
		return None

	if mode == "arbitrary":
		return evals[:k]
	elif mode == "smallest":
		return np.sort(evals)[:k]
	elif mode == "largest":
		return np.sort(evals)[-k:]
	elif mode == "smallest abs":
		idx = np.argsort(np.abs(evals))
		return evals[idx][:k]
	elif mode == "largest abs":
		idx = np.argsort(np.abs(evals))
		return evals[idx][-k:]


def diagonalize_until_converged(A, k, max_iter=None, criterion=1e-3,
								mode="smallest", reorthogonalize=True):
	"""
	A: matrix to be diagonalized
	k: number of eigenvalues that will be estimated
	criterion: convergence criterion

	"""
	available_modes = ("arbitrary", "smallest", "largest", "smallest abs", "largest abs")
	if mode not in available_modes:
		raise ValueError(f"Mode not available: {mode}. " + \
						 f"Available modes: {', '.join(available_modes)}.")

	A = np.asarray(A, dtype=float)
	n = A.shape[0]
	if A.ndim != 2 or n != A.shape[1]:
		raise ValueError("A must be square and 2D.")
	if not (1 <= k <= n):
		raise ValueError("k must be in [1, n].")

	if max_iter is None:
		max_iter = n + 1

	prev_evals = None
	b = normalize(np.random.rand(n))
	data = []
	for m in range(k, max_iter):
		Q, alpha, beta = lanczos(A, b, n, m, reorthogonalize=reorthogonalize)
		T = tridiag_from_coeffs(alpha, beta)
		evals, evecs = np.linalg.eigh(T)

		evals = _select_evals(evals, k, mode)
		data.append((m, evals))

		# if m == 10:
		# 	return T

		if _rel_err(evals, prev_evals) < criterion:
			print(f"m = {m}")
			return evals, Q, T, data
		prev_evals = evals

	print("Max iters reached")
	return evals, Q, T, data
