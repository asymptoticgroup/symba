export default class Matrix {
  // The dimensions of the (square) matrix; equivalently the maximum number of elements in a band
  #dim: number;

  // The raw backing storage of the matrix
  #raw: Float64Array;

  /**
   * Construct a new matrix of size dim x dim.
   * @param dim The dimensions of the matrix
   * @param bands The number of bands; must be a positive, odd number
   */
  constructor(dim: number, bands: number) {
    if (bands & 1) {
      this.#dim = dim;
      this.#raw = new Float64Array(dim * ((bands + 1) >> 1));
    } else {
      throw new Error("bands must be a positive, odd number");
    }
  }

  /**
   * The dimensions of the (square) matrix
   */
  get dim(): number {
    return this.#dim;
  }

  /**
   * Obtain a view on the kth band.
   *
   * It is represented in the following manner:
   * Suppose we want element (i, j). This corresponds to band k = |i - j|.
   * Then, band(|i - j|)[min(i, j)] is the desired element.
   *
   * @param k The desired band (>= 0), no bounds checking performed
   * @returns A read-write view of the band, starting at its first element
   */
  band(k: number): Float64Array {
    return this.#raw.subarray(k * this.#dim, (k + 1) * this.#dim - k);
  }

  // Helper method for individual element indexing. No bounds checking is performed.
  #index(i: number, j: number): number {
    return Math.abs(i - j) * this.#dim + Math.min(i, j);
  }

  /**
   * Retrieve the value at (i, j) = (j, i). Bounds checking is the responsibility of the caller.
   * @param i
   * @param j
   * @returns The value at the specified index
   */
  get(i: number, j: number): number {
    return this.#raw[this.#index(i, j)];
  }

  /**
   * Sets the value at (i, j) = (j, i). Bounds checking is the responsibility of the caller.
   * @param i
   * @param j
   * @param x The value to set to
   */
  set(i: number, j: number, x: number) {
    this.#raw[this.#index(i, j)] = x;
  }

  /**
   * Factors the matrix in-place.
   *
   * The user is responsible for correct usage. This usage is as follows:
   * 1. Call `factor()` once (and only once) before any calls to `solve(x)`.
   * 2. Once `factor()` has been called, it is undefined behavior to access or modify elements or bands of the matrix.
   *    (This is because these values have been destructively updated in-place with the factorization.)
   * 3. Once `factor()` has been called, the user is able to call `solve(x)` to update x *in-place* with the solution.
   */
  factor() {
    // There is nothing special here, just a textbook implementation of a Cholesky LDLT decomposition.
    // See https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition_2 for a useful reference.

    // The bandwidth of the matrix (used to keep us within the valid indices)
    const b = this.#raw.length / this.#dim - 1;

    for (let j = 0; j < this.#dim; j++) {
      // Compute the diagonal entry
      let d = this.#raw[j];
      for (let k = Math.max(0, j - b); k < j; k++) {
        d -= Math.pow(this.get(j, k), 2) * this.#raw[k];
      }
      this.#raw[j] = d;

      // Compute the rest of the column
      for (let i = Math.min(this.#dim - 1, j + b); i > j; i--) {
        let value = this.get(i, j);
        for (let k = Math.max(0, i - b); k < j; k++) {
          value -= this.get(i, k) * this.get(j, k) * this.#raw[k];
        }
        value /= d;
        this.set(i, j, value);
      }
    }
  }

  /**
   * Solves the corresponding linear system with the given right-hand side, updating it in-place.
   *
   * It is essential that `factor()` is called exactly once before calling this method.
   *
   * @param x The RHS to solve for
   */
  solve(x: Float64Array) {
    // Our matrix is of the form L*D*L^T, where L and D are stored in our raw buffer.
    // 1. Solve L*x1 = x0. (x1 = D*L^T*x)
    // 2. Solve D*x2 = x1. (x2 = L^T*x)
    // 3. Solve L^T*x3 = x2. (x3 = x)

    // The bandwidth of the matrix (used to keep us within the valid indices)
    const b = this.#raw.length / this.#dim - 1;

    // Step 1.
    for (let i = 1; i < this.#dim; i++) {
      // Note that we only need to "go back" so far in j, due to the bandedness
      for (let j = Math.max(0, i - b); j < i; j++) {
        x[i] -= this.get(i, j) * x[j];
      }
    }

    // Step 2.
    for (let i = 0; i < this.#dim; i++) {
      // Since `this.band(0)` is the same as the first n elements of `this.#raw`,
      // we can skip the getter for a small boost in performance.
      x[i] /= this.#raw[i];
    }

    // Step 3.
    for (let i = this.#dim - 2; i >= 0; i--) {
      // Same observation here with the bandedness.
      for (let j = Math.min(this.#dim - 1, i + b); j > i; j--) {
        x[i] -= this.get(i, j) * x[j];
      }
    }

    // All done!
  }
}
