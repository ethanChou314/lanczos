This Python module provides tools for performing Lanczos iterations. It includes a high-level function to expand the Krylov subspace until the desired eigenvalues have converged.

**Key Features**

\begin{itemize}

\item Lanczos Iteration: Efficiently generates a Krylov subspace and reduces a symmetric matrix to tridiagonal form.

\item Re-orthogonalization: Uses a double-pass modified Gram-Schmidt process to prevent the "ghost eigenvalues" caused by loss of orthogonality in floating-point arithmetic.

\item Determining Convergence: Iteratively increases the subspace size $m$ until the requested $k$ eigenvalues stabilize.

\item Multiple Selection Modes: Find eigenvalues based on: smallest or largest algebraic value, smallest abs or largest abs magnitude, or arbitrary (the first $k$ found).

\end{itemize}
