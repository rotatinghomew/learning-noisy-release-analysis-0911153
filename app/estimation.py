"""Linear latent-count estimation under correlated release noise.

Every published measurement is modelled as a linear function of latent cell
counts plus a linear combination of shared and measurement-specific noise
variables::

    y = A x + L e,        e ~ N(0, diag(noise_variance)),   S = L diag(v) L^T

``x`` is the unknown non-negative latent count table.  ``A`` comes from the
published ``coefficients`` and ``L`` from the published ``loading`` entries, so
the full covariance ``S`` is used throughout.  Consequences that matter for
this data:

* a release's total and its sub-items that load on the same variable are not
  independent observations;
* a later release that shares a noise variable with an earlier one is not an
  independent replicate, so averaging them does not divide the variance;
* directions of ``S`` with zero variance are exact linear relations on ``x``
  and are enforced as such instead of being silently dropped;
* two published rows with identical design, value and loading add no
  information, which falls out of the rank of the whitened design rather than
  being special-cased.

The module is deterministic.  There is no random number generation, so results
are reproducible from the packet, the options and the stated tolerances.
Intervals are profile-likelihood intervals for known covariance, which is the
uncertainty of the *release mechanism*, not sampling uncertainty about a
population.  Hard relations published by the packet (period totals, mutual
exclusivity of the cell partition, non-negativity, cross-table relations) are
linear constraints that are never relaxed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog, minimize
from scipy.stats import chi2

_EIGEN_TOLERANCE = 1e-10
_RANK_TOLERANCE = 1e-9
_FEASIBILITY_TOLERANCE = 1e-7


class InconsistentEvidence(RuntimeError):
    """The zero-noise part of the evidence contradicts the hard constraints."""


@dataclass
class Fit:
    """Outcome of one constrained generalised-least-squares fit."""

    feasible: bool
    counts: np.ndarray | None
    objective: float | None
    dof: int
    p_value: float | None
    converged: bool
    constraint_violation: float
    notes: list = field(default_factory=list)


def _rank(matrix, tolerance=_RANK_TOLERANCE):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return 0
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0:
        return 0
    return int(np.sum(singular > tolerance * max(float(singular[0]), 1.0)))


def _rows(values, width):
    if values is None or len(values) == 0:
        return np.zeros((0, width))
    return np.atleast_2d(np.asarray(values, dtype=float))


class NoiseModel:
    """Whitened constrained least-squares problem over latent counts."""

    def __init__(self, design, released, loading, noise_variance, equal, equal_values,
                 upper=None, upper_values=None, identifiers=None):
        self.design = np.asarray(design, dtype=float)
        self.released = np.asarray(released, dtype=float)
        self.loading = np.asarray(loading, dtype=float)
        self.noise_variance = np.asarray(noise_variance, dtype=float)
        if self.design.ndim != 2:
            raise ValueError("design must be a matrix")
        self.n, self.d = self.design.shape
        if self.released.shape != (self.n,):
            raise ValueError("released values do not match design rows")
        if self.loading.shape[0] != self.n:
            raise ValueError("loading rows do not match measurements")
        self.identifiers = list(identifiers) if identifiers is not None else list(range(self.n))

        covariance = self.loading @ np.diag(self.noise_variance) @ self.loading.T
        self.covariance = (covariance + covariance.T) / 2.0
        scale = max(float(np.abs(self.covariance).max()) if self.covariance.size else 0.0, 1.0)
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        if (eigenvalues < -1e-8 * scale).any():
            raise ValueError("release noise covariance is not positive semidefinite")

        self.conflicting_duplicates = []
        self.redundant_rows = []
        self.kept_indices = list(range(self.n))
        self.original_design = self.design.copy()
        self.original_released = self.released.copy()
        self._canonicalise_rows()

        values, vectors = np.linalg.eigh(self.covariance)
        values = np.clip(values, 0.0, None)
        tolerance = _EIGEN_TOLERANCE * max(float(values.max()) if values.size else 0.0, 1.0)
        noisy = values > tolerance
        self.whitener = (vectors[:, noisy] / np.sqrt(values[noisy])).T
        self.exact_directions = vectors[:, ~noisy].T

        whitened_design = self.whitener @ self.design
        whitened_released = self.whitener @ self.released

        self.equal = _rows(equal, self.d)
        self.equal_values = (np.zeros(0) if equal_values is None
                             else np.atleast_1d(np.asarray(equal_values, dtype=float)))
        if self.equal.shape[0] != self.equal_values.shape[0]:
            raise ValueError("equality rows and values differ in length")
        self.upper = _rows(upper, self.d)
        self.upper_values = (np.zeros(0) if upper_values is None
                             else np.atleast_1d(np.asarray(upper_values, dtype=float)))
        if self.upper.shape[0] != self.upper_values.shape[0]:
            raise ValueError("inequality rows and values differ in length")

        # Directions of S with zero variance are exact observations on x: the
        # released combination has no noise, so it must hold exactly.  A
        # combination that is identically zero in x but non-zero in the data is
        # a flat contradiction, kept as a row so that feasibility reports it
        # instead of the row being silently dropped.
        self.exact_relation_rows = []
        self.exact_relation_values = []
        self.exact_relation_directions = []
        if self.exact_directions.size:
            rows = self.exact_directions @ self.design
            values_exact = self.exact_directions @ self.released
            for direction, row, value in zip(self.exact_directions, rows, values_exact):
                self.exact_relation_rows.append(np.asarray(row, dtype=float))
                self.exact_relation_values.append(float(value))
                self.exact_relation_directions.append(np.asarray(direction, dtype=float))
        self.exact_matrix = (np.array(self.exact_relation_rows) if self.exact_relation_rows
                             else np.zeros((0, self.d)))
        self.exact_values = (np.array(self.exact_relation_values) if self.exact_relation_values
                             else np.zeros(0))

        self._eliminate()

    # -- geometry ---------------------------------------------------------
    def _canonicalise_rows(self):
        """Drop rows that add no information, and detect conflicting ones.

        Two measurements with the same design row, the same released value and
        proportional loading vectors carry one observation, not two.  Keeping
        both would leave a zero-variance direction whose *mean* is also zero; a
        pseudo-inverse then gives it zero weight, which is correct, but any
        numerical noise in the near-zero eigenvalue can leak weight into it.
        Removing the redundant row is exact and keeps the fit invariant to
        re-publishing the same numbers.
        """
        keep = []
        groups = {}
        for index in range(self.n):
            key = tuple(np.round(self.design[index], 12))
            groups.setdefault(key, []).append(index)
        dropped = []
        for key, members in groups.items():
            representative = None
            for index in members:
                loading = self.loading[index]
                if representative is None:
                    representative = index
                    keep.append(index)
                    continue
                reference = self.loading[representative]
                if np.abs(reference).max(initial=0.0) <= 1e-12:
                    same_noise = np.abs(loading).max(initial=0.0) <= 1e-12
                    factor = 1.0
                else:
                    nonzero = np.abs(reference) > 1e-12
                    factor = float(loading[nonzero][0] / reference[nonzero][0])
                    same_noise = np.abs(loading - factor * reference).max(initial=0.0) <= 1e-9 * max(
                        float(np.abs(reference).max()), 1.0)
                if not same_noise:
                    continue  # different noise structure: a genuinely different observation
                predicted = factor * self.released[representative]
                if abs(self.released[index] - predicted) <= 1e-9 * max(abs(predicted), 1.0):
                    dropped.append({"index": index, "duplicate_of": representative,
                                    "scaled_by": factor})
                else:
                    self.conflicting_duplicates.append({
                        "index": index, "of": representative,
                        "released": float(self.released[index]),
                        "implied": float(predicted)})
        self.redundant_rows = dropped
        if not dropped:
            return
        keep = sorted(keep)
        self.original_released = self.released.copy()
        self.original_design = self.design.copy()
        self.design = self.design[keep]
        self.released = self.released[keep]
        self.loading = self.loading[keep]
        self.identifiers = [self.identifiers[index] for index in keep]
        self.n = self.design.shape[0]
        covariance = self.loading @ np.diag(self.noise_variance) @ self.loading.T
        self.covariance = (covariance + covariance.T) / 2.0
        self.dropped_covariance = None
        self.kept_indices = keep

    def _eliminate(self):
        d = self.d
        if self.equal.shape[0] or self.exact_matrix.shape[0]:
            stacked = np.vstack([self.equal, self.exact_matrix]) if self.equal.shape[0] else self.exact_matrix
            values = (np.concatenate([self.equal_values, self.exact_values]) if self.equal.shape[0]
                      else self.exact_values)
            particular, *_ = np.linalg.lstsq(stacked, values, rcond=None)
            residual = stacked @ particular - values
            reference = max(float(np.abs(values).max(initial=0.0)), 1.0)
            self.equal_consistent = bool(np.abs(residual).max(initial=0.0) <= _FEASIBILITY_TOLERANCE * reference)
            self.equality_matrix = stacked
            self.equality_values = values
            basis = null_space(stacked)
        else:
            particular = np.zeros(d)
            self.equal_consistent = True
            self.equality_matrix = np.zeros((0, d))
            self.equality_values = np.zeros(0)
            basis = np.eye(d)
        self.particular = particular
        self.basis = basis
        self.free_dimension = basis.shape[1]

        self.whitened_design = self.whitener @ self.design
        self.whitened_released = self.whitener @ self.released
        if self.whitened_design.size and self.free_dimension:
            reduced = self.whitened_design @ self.basis
            _, singular, vectors_t = np.linalg.svd(reduced, full_matrices=True)
            self.free_rank = int(np.sum(singular > _RANK_TOLERANCE * max(float(singular[0]), 1.0)))
            self.row_space = vectors_t[: self.free_rank].T
            self.flat_basis = vectors_t[self.free_rank:].T
        else:
            self.free_rank = 0
            self.row_space = np.zeros((self.free_dimension, 0))
            self.flat_basis = (np.eye(self.free_dimension) if self.free_dimension
                               else np.zeros((0, 0)))
        self.information_rank = _rank(self.whitener)
        # Residual degrees of freedom of the goodness-of-fit test: the number of
        # independent noisy observations minus the number of free directions
        # they actually inform.
        self.dof = max(self.information_rank - self.free_rank, 0)
        self.feasible_point = self._feasible_point()
        self.feasible = self.feasible_point is not None and self.equal_consistent

    def _lp(self, objective, equality=True):
        return linprog(
            objective,
            A_eq=self.equality_matrix if (equality and self.equality_matrix.shape[0]) else None,
            b_eq=self.equality_values if (equality and self.equality_matrix.shape[0]) else None,
            A_ub=self.upper if self.upper.shape[0] else None,
            b_ub=self.upper_values if self.upper.shape[0] else None,
            bounds=[(0, None)] * self.d,
            method="highs",
        )

    def _feasible_point(self):
        if not self.equal_consistent:
            return None
        result = self._lp(np.zeros(self.d))
        return result.x if result.status == 0 else None

    # -- fit coordinates --------------------------------------------------
    def _pack(self, x):
        return self.basis.T @ (np.asarray(x, dtype=float) - self.particular)

    def _unpack(self, z):
        return self.particular + self.basis @ np.asarray(z, dtype=float)

    def _flat_part(self, z):
        if self.flat_basis.size == 0:
            return np.zeros_like(z)
        return self.flat_basis @ (self.flat_basis.T @ z)

    def _violation(self, x):
        parts = [max(0.0, -float(np.min(x))) if x.size else 0.0]
        if self.equality_matrix.shape[0]:
            parts.append(float(np.abs(self.equality_matrix @ x - self.equality_values).max(initial=0.0)))
        if self.upper.shape[0]:
            parts.append(max(0.0, float(np.max(self.upper @ x - self.upper_values, initial=0.0))))
        return max(parts)

    def _q_at_x(self, x):
        residual = self.whitened_released - self.whitened_design @ x
        return float(residual @ residual)

    def _gradient_at_x(self, x):
        residual = self.whitened_released - self.whitened_design @ x
        return -2.0 * (self.whitened_design.T @ residual)

    def _constraints(self):
        constraints = []
        if self.equality_matrix.shape[0]:
            matrix = self.equality_matrix
            values = self.equality_values
            constraints.append({
                "type": "eq",
                "fun": lambda x, m=matrix, v=values: m @ x - v,
                "jac": lambda x, m=matrix: m,
            })
        if self.upper.shape[0]:
            matrix = self.upper
            values = self.upper_values
            constraints.append({
                "type": "ineq",
                "fun": lambda x, m=matrix, v=values: v - m @ x,
                "jac": lambda x, m=matrix: -m,
            })
        return constraints

    # -- fit --------------------------------------------------------------
    def solve(self):
        if not self.feasible:
            return Fit(False, None, None, self.dof, None, False, math.inf,
                       ["hard constraints and exact relations admit no latent table"])
        if self.free_rank == 0:
            x = np.clip(self.feasible_point, 0.0, None)
            q = self._q_at_x(x)
            # No noisy observations remain, so there is nothing to test: the
            # table either satisfies every exact relation (q == 0) or
            # feasibility would have failed.  A p-value of 1 is honest here.
            notes = ["noiseless evidence determines the table exactly"]
            return Fit(True, x, q, 0, 1.0, True, self._violation(x), notes)

        starts = [np.clip(self.feasible_point, 0.0, None)]
        unconstrained = self._unconstrained()
        if unconstrained is not None:
            starts.append(np.clip(unconstrained, 0.0, None))

        constraints = self._constraints()
        best = None
        for start in starts:
            result = minimize(self._q_at_x, start, jac=self._gradient_at_x,
                              constraints=constraints, bounds=[(0.0, None)] * self.d,
                              method="SLSQP", options={"ftol": 1e-13, "maxiter": 1000})
            x = np.clip(result.x, 0.0, None)
            violation = self._violation(x)
            if violation > _FEASIBILITY_TOLERANCE:
                continue
            candidate = (self._q_at_x(x), x, bool(result.success))
            if best is None or candidate[0] < best[0] - 1e-9:
                best = candidate
        if best is None:
            return Fit(True, None, None, self.dof, None, False, math.inf,
                       ["constrained optimiser did not return a feasible point"])

        q, x, converged = best
        notes = [] if converged else ["optimiser reported non-convergence but the point is feasible"]
        return Fit(True, x, q, self.dof, chi_square_p(q, self.dof), converged,
                   self._violation(x), notes)

    def _unconstrained(self):
        """Generalised least squares ignoring non-negativity, as an extra start."""
        matrix = self.whitened_design @ self.basis
        target = self.whitened_released - self.whitened_design @ self.particular
        if matrix.size == 0:
            return None
        solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        return self._unpack(solution)

    # -- functionals ------------------------------------------------------
    def pinned_by_equalities(self, coefficients):
        """True when declared equalities alone fix the functional exactly."""
        if self.equality_matrix.shape[0] == 0:
            return False
        coefficients = np.asarray(coefficients, dtype=float)
        reference = max(float(np.abs(coefficients).max(initial=0.0)), 1.0)
        least, *_ = np.linalg.lstsq(self.equality_matrix.T, coefficients, rcond=None)
        residue = coefficients - self.equality_matrix.T @ least
        return float(np.abs(residue).max(initial=0.0)) <= 1e-8 * reference

    def identifiable(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=float)
        if self.basis.shape[1] == 0:
            return True
        projected = self.basis.T @ coefficients
        if self.flat_basis.size == 0:
            return True
        residue = self.flat_basis.T @ projected
        reference = max(float(np.abs(coefficients).max(initial=0.0)), 1.0)
        return float(np.abs(residue).max(initial=0.0)) <= 1e-8 * reference

    def structural_bounds(self, coefficients):
        """Non-negative attainable range under the hard linear relations only.

        Returns ``None`` only when the hard system has no non-negative point at
        all.  An unbounded direction is reported with ``None`` for that side
        rather than a fabricated finite number.
        """
        coefficients = np.asarray(coefficients, dtype=float)
        lower = self._lp(coefficients)
        upper = self._lp(-coefficients)
        low = float(lower.fun) if lower.status == 0 else None
        high = float(-upper.fun) if upper.status == 0 else None
        if low is None and high is None:
            return None
        return low, high

    def profile_bounds(self, coefficients, level, q_reference):
        coefficients = np.asarray(coefficients, dtype=float)
        structural = self.structural_bounds(coefficients)
        if structural is None:
            return None
        if self.free_rank == 0 or not self.identifiable(coefficients):
            return structural
        base = self.solve()
        if base.counts is None:
            return structural
        limit = float(q_reference) + float(chi2.ppf(level, 1))

        def q_constraint(x):
            return limit - self._q_at_x(x)

        constraints = self._constraints()
        constraints.append({"type": "ineq", "fun": q_constraint, "jac": lambda x: -self._gradient_at_x(x)})
        starts = [base.counts]
        if self.feasible_point is not None:
            starts.append(np.clip(self.feasible_point, 0.0, None))

        results = []
        for sign in (1.0, -1.0):
            best = None
            for start in starts:
                result = minimize(lambda x: sign * float(coefficients @ x), start,
                                  jac=lambda x: sign * coefficients, constraints=constraints,
                                  bounds=[(0.0, None)] * self.d,
                                  method="SLSQP", options={"ftol": 1e-14, "maxiter": 1500})
                x = np.clip(result.x, 0.0, None)
                if self._q_at_x(x) > limit + 1e-5 or self._violation(x) > _FEASIBILITY_TOLERANCE:
                    continue
                value = float(coefficients @ x)
                best = value if best is None else (min(best, value) if sign > 0 else max(best, value))
            results.append(best)
        if results[0] is None or results[1] is None:
            return structural
        low_struct, high_struct = structural
        lower = results[0] if low_struct is None else max(results[0], low_struct)
        upper = results[1] if high_struct is None else min(results[1], high_struct)
        if upper < lower - 1e-9:
            return structural
        return float(lower), float(upper)

    # -- noise accounting -------------------------------------------------
    def least_squares_weights(self, coefficients):
        """Weights ``a`` with ``a'A = c`` minimising ``a' S a``.

        Returns ``None`` when ``c`` is not in the row space of the design, in
        which case the estimand is not a linear combination of releases.
        """
        coefficients = np.asarray(coefficients, dtype=float)
        reference = max(float(np.abs(coefficients).max(initial=0.0)), 1.0)
        values, vectors = np.linalg.eigh(self.covariance)
        values = np.clip(values, 0.0, None)
        if values.size == 0 or float(values.max()) <= 0:
            if np.abs(coefficients).max(initial=0.0) <= 1e-12:
                return np.zeros(self.n)
            return None
        keep = values > _EIGEN_TOLERANCE * float(values.max())
        inverse = (vectors[:, keep] / values[keep]) @ vectors[:, keep].T
        gram = self.design.T @ inverse @ self.design
        gram_values, gram_vectors = np.linalg.eigh((gram + gram.T) / 2.0)
        support = gram_values > _RANK_TOLERANCE * max(float(np.abs(gram_values).max()), 1.0)
        if not support.any():
            return np.zeros(self.n) if np.abs(coefficients).max(initial=0.0) <= 1e-12 else None
        basis = gram_vectors[:, support]
        projection = basis.T @ coefficients
        residue = coefficients - basis @ projection
        if np.abs(residue).max(initial=0.0) > 1e-7 * reference:
            return None
        return inverse @ self.design @ basis @ (projection / gram_values[support])

    def noise_decomposition(self, coefficients):
        """Split the variance of a linear estimand by noise variable."""
        weights = self.least_squares_weights(coefficients)
        if weights is None:
            return None
        contributions = self.loading.T @ weights
        variances = contributions ** 2 * self.noise_variance
        return {"weights": weights, "contributions": contributions, "variances": variances}

def chi_square_p(q, dof):
    if dof <= 0:
        return 1.0
    return float(chi2.sf(max(float(q), 0.0), dof))
