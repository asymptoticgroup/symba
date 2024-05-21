import { assert } from "jsr:@std/assert";
import Matrix from "./mod.ts";

Deno.test("Test Heat Equation", () => {
  const n = 101;
  const matrix = new Matrix(n, 3);

  // Tridiagonal matrix with (1, -2, 1) pattern
  matrix.band(0).fill(-2);
  matrix.band(1).fill(+1);

  // Setup the Cholesky factorization
  matrix.factor();

  const rhs = new Float64Array(n);
  rhs.fill(1);

  // Solve it
  matrix.solve(rhs);

  // Now we check our work:
  let error = 0;
  for (let i = 0; i < n; i++) {
    error = Math.max(
      error,
      Math.abs(
        (i === 0 ? 0 : rhs[i - 1]) -
          2 * rhs[i] +
          (i + 1 === n ? 0 : rhs[i + 1]) -
          1
      )
    );
  }

  // We're not in exact arithmetic, but the error should be small anyway
  assert(error < 1e-12);
});
