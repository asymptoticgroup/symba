# *Sym*metric, *Ba*nded Matrix Library (SYMBA)

This is a minimal library for the definition and solution of linear systems
involving symmetric, banded matrices. In particular, an in-place Cholesky
decomposition is provided.

Refer to the generated documentation for usage; primarily you will allocate
the matrix, then define it either via `set(i, j, x)` or `band(k)[...] = ...`.

Call `factor()` once (and only once!), then `solve(x)` as many times as you
need.
