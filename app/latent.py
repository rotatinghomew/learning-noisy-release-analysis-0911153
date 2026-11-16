"""Latent participation and change analysis for noisy release exchanges.

The partner school only exchanges anonymised aggregates.  Each release is a
linear function of latent cell counts plus correlated noise.  This analysis
turns the exchanged material into latent participation counts, participation
changes between periods, and an explicit account of what the releases can and
cannot establish.

Three things are kept strictly apart in the output:

``fit`` / ``periods`` / ``changes``
    Estimates and profile intervals.  Their uncertainty is *release noise*
    under the published covariance, not sampling uncertainty about residents.
``coverage``
    Cells or populations the exchanged material never measured.  Noise cannot
    recover them, so they are never folded into an interval.
``contradictions``
    Release evidence that cannot be explained together.  It is reported as a
    contradiction with a witness, never repaired into one tidy table.

The analysis is deterministic: no random number generation and no Monte Carlo
error.  Everything is reproducible from the packet, the options and the
tolerances recorded in the result.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from scipy.optimize import linprog

from app.estimation import NoiseModel
from app.protocol import finite

ANALYSIS_VERSION = 1

DEFAULT_OPTIONS = {
    "confidence_level": 0.90,
    "consistency_level": 0.05,
    "release_scope": "current",
    "release_ids": None,
    "include_superseded": False,
    "subset_limit": 12,
}


class OptionError(ValueError):
    """Raised for options that would silently change the meaning of the analysis."""


# ---------------------------------------------------------------------------
# option handling


def _options(raw):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise OptionError("options must be an object")
    unknown = sorted(set(raw) - set(DEFAULT_OPTIONS))
    if unknown:
        raise OptionError(f"unknown option(s): {', '.join(unknown)}")
    resolved = dict(DEFAULT_OPTIONS)
    resolved.update(raw)

    level = resolved["confidence_level"]
    if isinstance(level, bool) or not isinstance(level, (int, float)) or not (0.0 < level < 1.0):
        raise OptionError("confidence_level must be a number in (0, 1)")
    alpha = resolved["consistency_level"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not (0.0 < alpha < 1.0):
        raise OptionError("consistency_level must be a number in (0, 1)")
    scope = resolved["release_scope"]
    if scope not in ("current", "all", "ids"):
        raise OptionError("release_scope must be 'current', 'all' or 'ids'")
    identifiers = resolved["release_ids"]
    if scope == "ids":
        if not isinstance(identifiers, list) or not identifiers or not all(isinstance(i, str) for i in identifiers):
            raise OptionError("release_scope 'ids' requires a non-empty release_ids list of strings")
    elif identifiers is not None:
        raise OptionError("release_ids is only meaningful with release_scope 'ids'")
    if not isinstance(resolved["include_superseded"], bool):
        raise OptionError("include_superseded must be a boolean")
    limit = resolved["subset_limit"]
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise OptionError("subset_limit must be a positive integer")
    return resolved


# ---------------------------------------------------------------------------
# packet structure


def _periods(packet):
    """Order periods by the declared chronology, independent of row order.

    The order of the periods carries meaning: reversing it flips the sign of
    every reported change.  It therefore must not depend on how the ``cells``
    or ``releases`` arrays happen to be ordered.  The packet's ``constraints``
    are the only place that states a chronology explicitly, so their period
    order is used first; periods not declared there are appended in cell order.
    """
    order = []
    for constraint in packet.get("constraints", []):
        period = constraint.get("period")
        if period is not None and period not in order:
            order.append(period)
    for cell in packet["cells"]:
        if cell["period"] not in order:
            order.append(cell["period"])
    return order


def _class_programs(classes):
    """Map class labels to the set of programs each label represents.

    ``both`` and ``neither`` are treated as wildcards; any other token names a
    program directly.  Derived from the packet vocabulary so no program names
    are hard-coded here.
    """
    programs = []
    for label in classes:
        for token in label.split("_"):
            if token not in ("both", "neither", "none", "only", "part") and token not in programs:
                programs.append(token)
    mapping = {}
    for label in classes:
        tokens = label.split("_")
        if "neither" in tokens or "none" in tokens:
            mapping[label] = frozenset()
        elif "both" in tokens:
            mapping[label] = frozenset(programs)
        else:
            mapping[label] = frozenset(tokens)
    return programs, mapping


def _structure(packet):
    cells = packet["cells"]
    cell_ids = [cell["cell_id"] for cell in cells]
    by_id = {cell["cell_id"]: cell for cell in cells}
    index = {cell_id: position for position, cell_id in enumerate(cell_ids)}
    periods = _periods(packet)
    classes = []
    for cell in cells:
        if cell["class"] not in classes:
            classes.append(cell["class"])
    programs, mapping = _class_programs(classes)
    pairs = [(cell["period"], cell["class"]) for cell in cells]
    duplicates = sorted({pair for pair in pairs if pairs.count(pair) > 1})
    aggregates = {}
    for program in programs:
        aggregates[program] = [cell_id for cell_id in cell_ids
                               if program in mapping[by_id[cell_id]["class"]]]
    aggregates["participating"] = [cell_id for cell_id in cell_ids
                                   if mapping[by_id[cell_id]["class"]]]
    return {
        "cell_ids": cell_ids,
        "index": index,
        "periods": periods,
        "classes": classes,
        "programs": programs,
        "class_programs": mapping,
        "aggregates": aggregates,
        "duplicate_cells": [list(pair) for pair in duplicates],
        "by_id": by_id,
    }


def _cell_lookup(packet):
    return {cell["cell_id"]: cell for cell in packet["cells"]}


def _period_sets(packet, structure):
    lookup = _cell_lookup(packet)
    membership = {}
    for period in structure["periods"]:
        membership[period] = [cell_id for cell_id in structure["cell_ids"]
                              if lookup[cell_id]["period"] == period]
    return membership


def _aggregate_vector(structure, period_cells, cell_ids_in_aggregate):
    vector = np.zeros(len(structure["cell_ids"]))
    allowed = set(period_cells)
    for cell_id in cell_ids_in_aggregate:
        if cell_id in allowed:
            vector[structure["index"][cell_id]] = 1.0
    return vector


# ---------------------------------------------------------------------------
# release selection


def _select_releases(packet, options):
    releases = packet["releases"]
    known = {release["release_id"] for release in releases}
    # A release is superseded when another release declares ``replaces`` naming
    # it.  The replacing release is the current one.
    superseded = sorted({release["replaces"] for release in releases
                         if release["replaces"] is not None})

    used, excluded = [], []
    if options["release_scope"] == "ids":
        requested = list(options["release_ids"])
        missing = [identifier for identifier in requested if identifier not in known]
        if missing:
            raise OptionError(f"unknown release_id(s): {', '.join(sorted(missing))}")
        chosen = [release for release in releases if release["release_id"] in requested]
        for release in releases:
            if release["release_id"] not in requested:
                excluded.append({"release_id": release["release_id"],
                                 "reason": "not requested by release_ids"})
    elif options["release_scope"] == "all":
        # Every release is used, including superseded ones.  Nothing is
        # excluded; the revision chain is reported separately so the reader can
        # see that corrections are not independent replicates.
        chosen = list(releases)
    else:  # current
        current = [release for release in releases if release["release_id"] not in superseded]
        if options["include_superseded"]:
            chosen = list(releases)
        else:
            chosen = current
            for release in releases:
                if release["release_id"] in superseded:
                    excluded.append({"release_id": release["release_id"],
                                     "reason": "superseded by a later release of the same period"})

    used = [release["release_id"] for release in chosen]
    if not chosen:
        raise OptionError("no releases selected")
    return chosen, used, excluded, sorted(superseded)


# ---------------------------------------------------------------------------
# model construction


def _constraint_rows(packet):
    """Translate declared packet constraints into linear rows.

    Supported forms, all treated as facts about the population rather than as
    targets to fit towards:

    ``fixed_total``
        ``{"kind": "fixed_total", "period": p, "total": n}``
    ``linear`` / ``relation``
        ``{"kind": "linear", "coefficients": {cell_id: a}, "value": v}``,
        used for cross-table intersections or any other declared identity.

    Returns applied rows, the declarations that were *not* applied (reported,
    never silently dropped) and the period totals implied by ``fixed_total``.
    """
    cells = packet["cells"]
    cell_ids = [cell["cell_id"] for cell in cells]
    index = {cell_id: position for position, cell_id in enumerate(cell_ids)}
    periods = _periods(packet)
    applied, unhandled = [], []
    totals = {}
    for constraint in packet.get("constraints", []):
        kind = constraint.get("kind")
        if kind == "fixed_total":
            period = constraint.get("period")
            if period not in periods:
                unhandled.append({"constraint": constraint, "reason": "unknown period"})
                continue
            row = np.zeros(len(cell_ids))
            for cell in cells:
                if cell["period"] == period:
                    row[index[cell["cell_id"]]] = 1.0
            applied.append({"kind": "fixed_total", "period": period,
                            "total": float(constraint["total"]), "row": row})
            totals[period] = float(constraint["total"])
        elif kind in ("linear", "relation"):
            coefficients = constraint.get("coefficients")
            if not isinstance(coefficients, dict) or not coefficients:
                unhandled.append({"constraint": constraint, "reason": "missing coefficients"})
                continue
            unknown = sorted(set(coefficients) - set(index))
            if unknown:
                unhandled.append({"constraint": constraint,
                                  "reason": f"unknown cell_id(s): {', '.join(unknown)}"})
                continue
            row = np.zeros(len(cell_ids))
            for cell_id, coefficient in coefficients.items():
                row[index[cell_id]] = float(coefficient)
            applied.append({"kind": kind, "period": constraint.get("period"),
                            "total": float(constraint["value"]), "row": row})
        else:
            unhandled.append({"constraint": constraint, "reason": "unsupported constraint kind"})
    return applied, unhandled, totals


def _measurement_table(packet, releases):
    noise_names = list(packet["noise_variance"])
    rows = []
    for release in releases:
        for measurement in release["measurements"]:
            rows.append({
                "release_id": release["release_id"],
                "measurement": measurement["measurement"],
                "value": float(measurement["value"]),
                "coefficients": dict(measurement["coefficients"]),
                "loading": dict(measurement["loading"]),
                "mechanism": release.get("mechanism"),
            })
    return rows, noise_names


def _build_model(packet, structure, period_cells, releases, totals, constraints=()):
    cell_ids = structure["cell_ids"]
    rows, noise_names = _measurement_table(packet, releases)
    design = np.zeros((len(rows), len(cell_ids)))
    released = np.zeros(len(rows))
    loading = np.zeros((len(rows), len(noise_names)))
    for position, row in enumerate(rows):
        for cell_id, coefficient in row["coefficients"].items():
            design[position, structure["index"][cell_id]] = float(coefficient)
        released[position] = row["value"]
        for noise, coefficient in row["loading"].items():
            loading[position, noise_names.index(noise)] = float(coefficient)
    variance = np.array([float(packet["noise_variance"][name]) for name in noise_names])

    equality, values = [], []
    for period in structure["periods"]:
        if period in totals:
            vector = np.zeros(len(cell_ids))
            for cell_id in period_cells[period]:
                vector[structure["index"][cell_id]] = 1.0
            equality.append(vector)
            values.append(totals[period])
    for entry in constraints:
        if entry["kind"] != "fixed_total":
            equality.append(np.asarray(entry["row"], dtype=float))
            values.append(float(entry["total"]))
    model = NoiseModel(design, released, loading, variance, equality, values,
                       identifiers=[row["measurement"] for row in rows])
    return model, rows


# ---------------------------------------------------------------------------
# reporting helpers


def _clean(value, digits=10):
    if value is None:
        return None
    if isinstance(value, (bool, str)):
        return value
    number = float(value)
    if not math.isfinite(number):
        return None
    rounded = round(number, digits)
    return 0.0 if rounded == 0 else rounded


def _interval(bounds, level):
    if bounds is None:
        return None
    lower, upper = bounds
    width = (upper - lower) if (lower is not None and upper is not None) else None
    return {"lower": _clean(lower), "upper": _clean(upper), "level": level,
            "width": _clean(width)}


def _pinned_reason(model, coefficients):
    """Why a functional has zero structural width, if it does."""
    if model.information_rank == 0:
        return "noiseless_evidence"
    pinned = model.pinned_by_equalities(coefficients)
    if pinned:
        return "declared_constraints"
    return None


def _functional_report(model, coefficients, level, fit):
    structural = model.structural_bounds(coefficients)
    report = {
        "estimate": None,
        "interval": None,
        "structural_bounds": None,
        "identified": model.identifiable(coefficients),
    }
    if fit.counts is not None:
        report["estimate"] = _clean(float(coefficients @ fit.counts))
    pinned_reason = None
    if structural is not None:
        report["structural_bounds"] = {"lower": _clean(structural[0]), "upper": _clean(structural[1])}
        if structural[0] is not None and structural[1] is not None and structural[1] - structural[0] <= 1e-7:
            pinned_reason = _pinned_reason(model, coefficients)
            report["pinned_by"] = pinned_reason
    if not report["identified"]:
        report["estimate"] = None
        report["estimate_note"] = ("该量在声明的约束与发布证据下不可识别；"
                                   "点估计只代表一个等价解，不应作为人数使用。")
        # A profile interval for an unidentified quantity would look like
        # inference while merely repeating the structural bounds.
        report["interval"] = None
        report["interval_note"] = "不可识别的量不给出噪声区间；只有结构性范围。"
    elif fit.counts is not None and fit.objective is not None:
        report["interval"] = _interval(model.profile_bounds(coefficients, level, fit.objective), level)
    if fit.counts is not None:
        noise = model.noise_decomposition(coefficients)
        if noise is None:
            report["linear_in_releases"] = False
            report["release_noise_standard_deviation"] = None
        else:
            report["linear_in_releases"] = True
            if pinned_reason == "declared_constraints":
                # The declared total pins this exactly; a release-noise standard
                # deviation would describe an estimator that is not used.
                report["release_noise_standard_deviation"] = None
                report["noise_sd_note"] = "由声明的人数约束精确确定，发布噪声不改变该值。"
            else:
                report["release_noise_standard_deviation"] = _clean(
                    math.sqrt(float(noise["variances"].sum())))
    return report


# ---------------------------------------------------------------------------
# subset consistency


def _subset_signature(fit, alpha):
    if not fit.feasible:
        return "structurally_inconsistent", None
    if fit.counts is None:
        return "not_estimated", None
    if fit.dof > 0 and fit.p_value is not None and fit.p_value < alpha:
        return "statistically_tension", fit.p_value
    return "consistent", fit.p_value


def _maximal_consistent_subsets(packet, structure, period_cells, releases, totals, constraints, alpha, limit):
    identifiers = [release["release_id"] for release in releases]
    by_id = {release["release_id"]: release for release in releases}
    if len(identifiers) > limit:
        return None
    table = {}
    for size in range(1, len(identifiers) + 1):
        for combination in itertools.combinations(identifiers, size):
            subset = [by_id[identifier] for identifier in combination]
            model, _ = _build_model(packet, structure, period_cells, subset, totals, constraints)
            fit = model.solve()
            verdict, p_value = _subset_signature(fit, alpha)
            table[combination] = {"verdict": verdict, "p_value": p_value,
                                  "dof": fit.dof if fit.feasible else None,
                                  "objective": fit.objective}
    consistent = [set(combination) for combination, info in table.items()
                  if info["verdict"] == "consistent"]
    maximal = []
    for candidate in consistent:
        if not any(candidate < other for other in consistent):
            maximal.append(candidate)
    return table, maximal


def _infeasibility_witness(packet, structure, period_cells, releases, totals, constraints):
    """Smallest exact-relation set that clashes with the hard constraints.

    Exact relations are zero-variance combinations of releases.  When the full
    set is infeasible, rows are removed greedily while the remainder stays
    infeasible, leaving a small witness that operations can inspect.  The
    witness is small, not guaranteed minimum-cardinality.
    """
    model, rows = _build_model(packet, structure, period_cells, releases, totals, constraints)
    if model.feasible or model.exact_matrix.shape[0] == 0:
        return None

    def feasible(selected):
        exact_rows = model.exact_matrix[selected]
        exact_values = model.exact_values[selected]
        if model.equal.shape[0]:
            system = np.vstack([model.equal, exact_rows])
            rhs = np.concatenate([model.equal_values, exact_values])
        else:
            system, rhs = exact_rows, exact_values
        result = linprog(np.zeros(model.d), A_eq=system, b_eq=rhs,
                         A_ub=model.upper if model.upper.shape[0] else None,
                         b_ub=model.upper_values if model.upper.shape[0] else None,
                         bounds=[(0, None)] * model.d, method="highs")
        return result.status == 0

    selected = list(range(model.exact_matrix.shape[0]))
    shrunk = True
    while shrunk:
        shrunk = False
        for position in range(len(selected) - 1, -1, -1):
            trial = selected[:position] + selected[position + 1:]
            if trial and not feasible(trial):
                selected = trial
                shrunk = True
                break

    relations = []
    for position in selected:
        direction = model.exact_relation_directions[position]
        relations.append({
            "exact_combination": [
                {"release_id": rows[i]["release_id"], "measurement": rows[i]["measurement"],
                 "weight": _clean(direction[i])}
                for i in range(len(rows)) if abs(direction[i]) > 1e-9],
            "implies_sum_of_cells": {
                structure["cell_ids"][j]: _clean(model.exact_matrix[position][j])
                for j in range(len(structure["cell_ids"]))
                if abs(model.exact_matrix[position][j]) > 1e-9},
            "released_value": _clean(model.exact_values[position]),
        })
    return {
        "relations": relations,
        "conflicts_with": [
            {"kind": entry["kind"], "period": entry["period"], "total": _clean(entry["total"])}
            for entry in getattr(model, "applied_constraints", [])],
        "explanation": ("这些组合的噪声方差为零，因此发布值必须精确成立；"
                        "它们与声明的人数约束或互斥分类同时成立时无解。"),
    }


# ---------------------------------------------------------------------------
# coverage


def _coverage(packet, structure, period_cells, used_rows, model, fit, level):
    covered = set()
    for row in used_rows:
        for cell_id, coefficient in row["coefficients"].items():
            if abs(float(coefficient)) > 0:
                covered.add(cell_id)
    report = {"covered_cells": sorted(covered),
              "uncovered_cells": [], "measured_classes": sorted({packet["cells"][structure["index"][c]]["class"]
                                                                  for c in covered}),
              "all_classes": list(structure["classes"])}
    for cell_id in structure["cell_ids"]:
        if cell_id in covered:
            continue
        entry = {"cell_id": cell_id,
                 "period": packet["cells"][structure["index"][cell_id]]["period"],
                 "class": packet["cells"][structure["index"][cell_id]]["class"]}
        coefficients = np.zeros(len(structure["cell_ids"]))
        coefficients[structure["index"][cell_id]] = 1.0
        bounds = model.structural_bounds(coefficients)
        if (bounds is not None and bounds[0] is not None and bounds[1] is not None
                and bounds[1] - bounds[0] <= 1e-7):
            entry["implied_by_constraints"] = _clean(bounds[0])
            entry["status"] = "determined_by_declared_constraints"
            entry["note"] = "该人群未被直接交换测量，但由声明的人数约束唯一确定。"
        else:
            entry["implied_by_constraints"] = None
            entry["structural_bounds"] = (
                {"lower": _clean(bounds[0]), "upper": _clean(bounds[1])} if bounds else None)
            entry["status"] = "outside_exchanged_material"
            entry["note"] = "未交换测量覆盖该人群；发布噪声无法恢复，不能当作 0。"
        report["uncovered_cells"].append(entry)
    return report


# ---------------------------------------------------------------------------
# interpretation text


def _interpretation(result):
    lines = []
    fit = result["fit"]
    contradictions = result["contradictions"]
    if not contradictions["structurally_feasible"]:
        lines.append("发布证据与声明的人数约束无法同时成立；未生成插补后的完整人数表。")
    elif not result["fit"]["estimated"]:
        lines.append("没有可用于估计的发布噪声证据；未生成人数表。")
    elif contradictions["jointly_consumable"] is False:
        lines.append("发布证据彼此存在统计张力，点估计仅供参考，不能当作一致口径。")
    elif not result["fit"]["independent_noisy_observations"]:
        lines.append("所用发布没有随机噪声项，人数由确定性关系给出，不涉及发布噪声区间。")
    else:
        lines.append("在当前声明的线性机制、已知方差与人数约束下，发布证据可以联合解释。")
    if result["coverage"]["uncovered_cells"]:
        lines.append("存在未交换人群，其人数不能由发布噪声恢复；相关区间只覆盖已交换口径。")
    if result["noise"]["equal_weight_average"]["understates_uncertainty"]:
        lines.append("若把重复发布当作独立观测做平均，会低估不确定性；结果已按相关噪声计算。")
    lines.append("所有数值是发布噪声意义下的估计或可识别范围，不是居民抽样推断。")
    return lines


def _business(result):
    summary = []
    blocked = None
    if not result["contradictions"]["structurally_feasible"]:
        blocked = "所用发布与声明约束互相矛盾，无法形成可解释的人数表"
    elif not result["fit"]["estimated"]:
        blocked = "证据不足，无法得到可行的人数表"
    for change in result["changes"]:
        interval = change["interval"]
        if change.get("estimate") is None:
            reason = blocked or change.get("note") or change.get("estimate_note") or "证据不足"
            summary.append(f"{change['label']}：{change['from_period']}→{change['to_period']} "
                           f"无法估计（{reason}）")
            continue
        direction = "上升" if change["estimate"] > 0 else "下降"
        line = f"{change['label']}：{change['from_period']}→{change['to_period']} 估计变化 {change['estimate']:+.2f}"
        if change.get("pinned_by") == "declared_constraints":
            line += "；由声明的人数约束精确确定，不涉及发布噪声，两期总人数相同"
        elif change.get("pinned_by") == "noiseless_evidence":
            line += "；无噪声证据下精确确定，无发布噪声区间"
        else:
            has_interval = bool(interval and interval["lower"] is not None and interval["upper"] is not None)
            if has_interval:
                line += f"（{int(interval['level'] * 100)}% 发布噪声区间 {interval['lower']:+.2f} 至 {interval['upper']:+.2f}）"
                if interval["lower"] <= 0 <= interval["upper"]:
                    line += "；区间跨 0，方向不能确认"
                else:
                    line += f"；区间不含 0，方向为{direction}"
            else:
                line += "；区间不可计算，不能据此确认方向"
        summary.append(line)
    return summary


# ---------------------------------------------------------------------------
# main entry point


def latent_participation(packet, options):
    options = _options(options)
    level = float(options["confidence_level"])
    alpha = float(options["consistency_level"])

    structure = _structure(packet)
    period_cells = _period_sets(packet, structure)
    chosen, used, excluded, superseded = _select_releases(packet, options)
    applied, unhandled, totals = _constraint_rows(packet)

    model, used_rows = _build_model(packet, structure, period_cells, chosen, totals, applied)
    fit = model.solve()

    if fit.counts is not None:
        for cell_id in structure["cell_ids"]:
            finite(float(fit.counts[structure["index"][cell_id]]), f"latent count {cell_id}")

    # ---- fit diagnostics
    fit_report = {
        "feasible": fit.feasible,
        "estimated": fit.counts is not None,
        "converged": fit.converged,
        "objective": _clean(fit.objective),
        "degrees_of_freedom": int(fit.dof),
        "p_value": _clean(fit.p_value),
        "constraint_violation": _clean(fit.constraint_violation),
        "independent_noisy_observations": int(model.information_rank),
        "free_informed_directions": int(model.free_rank),
        "notes": fit.notes,
    }
    if fit.counts is not None:
        fit_report["residuals"] = [_clean(value) for value in
                                   (model.released - model.design @ fit.counts)]

    # ---- per-period counts and aggregates
    periods_report = {}
    for period in structure["periods"]:
        cells_report = {}
        for cell_id in period_cells[period]:
            coefficients = np.zeros(len(structure["cell_ids"]))
            coefficients[structure["index"][cell_id]] = 1.0
            cells_report[cell_id] = _functional_report(model, coefficients, level, fit)
        aggregates = {}
        for name, members in structure["aggregates"].items():
            vector = _aggregate_vector(structure, period_cells[period], members)
            if not vector.any():
                continue
            aggregates[name] = _functional_report(model, vector, level, fit)
        total_vector = np.zeros(len(structure["cell_ids"]))
        for cell_id in period_cells[period]:
            total_vector[structure["index"][cell_id]] = 1.0
        aggregates["total"] = _functional_report(model, total_vector, level, fit)
        periods_report[period] = {"cells": cells_report, "aggregates": aggregates}

    # ---- changes between consecutive periods
    changes = []
    for earlier, later in zip(structure["periods"], structure["periods"][1:]):
        for name, members in list(structure["aggregates"].items()) + [("total", structure["cell_ids"])]:
            after = _aggregate_vector(structure, period_cells[later], members)
            before = _aggregate_vector(structure, period_cells[earlier], members)
            if not after.any() or not before.any():
                continue
            coefficients = after - before
            if not coefficients.any():
                continue
            report = _functional_report(model, coefficients, level, fit)
            entry = {"measure": name, "label": _label(name, structure),
                     "from_period": earlier, "to_period": later}
            entry.update(report)
            if report.get("pinned_by") == "declared_constraints":
                entry["direction_resolved"] = False
                entry["note"] = "由声明的各期总人数约束精确确定，不存在发布噪声导致的年度变化。"
            interval = entry.get("interval")
            entry["direction_resolved"] = bool(
                entry.get("direction_resolved") or (
                    interval and interval["lower"] is not None and interval["upper"] is not None
                    and (interval["lower"] > 0 or interval["upper"] < 0)))
            changes.append(entry)

    # ---- noise accounting
    noise_report = _noise_accounting(packet, model, used_rows, changes)

    # ---- contradictions
    contradictions = _contradictions(packet, structure, period_cells, chosen, totals, applied, fit,
                                     alpha, options["subset_limit"], used)

    # ---- coverage
    coverage = _coverage(packet, structure, period_cells, used_rows, model, fit, level)

    # ---- identities that must hold by construction
    identities = _identities(packet, structure, period_cells, periods_report, level)

    conditions = [
        "发布机制按资料声明的线性可加模型与已知噪声方差解释；未声明的机制差异不被假设为不存在。",
        "同键噪声变量跨发布共享，区间按完整协方差计算，不把重复发布当作独立观测。",
        "未覆盖人群不进入区间；结构性零与未观测在结果中分开列示。",
        "区间是发布噪声下的 profile 区间，不是对居民或学员总体的抽样推断。",
    ]
    mechanisms = packet.get("mechanisms", {})
    for name, declaration in mechanisms.items():
        distribution = declaration.get("distribution")
        conditions.append(f"资料声明机制 {name} 为 {distribution}；该分布假设未被本分析独立验证。")
    if fit.dof > 0:
        conditions.append("拟合优度卡方检验假设约束未在边界处生效；若估计落在 0 边界，p 值应视为近似。")

    result = {
        "method": {
            "name": "constrained_generalised_least_squares",
            "analysis_version": ANALYSIS_VERSION,
            "random_seed": None,
            "monte_carlo": False,
            "numerical_note": "确定性求解，无蒙特卡洛误差；区间为 profile likelihood 区间。",
            "tolerances": {"eigenvalue": 1e-10, "rank": 1e-9, "feasibility": 1e-7, "profile": 1e-5},
        },
        "estimand": {
            "quantity": "latent_participation_counts",
            "periods": structure["periods"],
            "units": "persons (as defined by the exchanged aggregates)",
            "population": "只有在交换资料中被测量或由声明约束确定的人群",
            "programs": structure["programs"],
            "note": "被测量的是各期互斥分类组合的人数；总量与交集共享同一噪声结构。",
        },
        "scenario": {
            "options": options,
            "description": _scenario_text(options),
        },
        "data": {
            "revision": packet["revision"],
            "synthetic": bool(packet.get("synthetic", False)),
            "releases_used": used,
            "releases_excluded": excluded,
            "superseded_releases": superseded,
            "measurements_used": [
                {"release_id": row["release_id"], "measurement": row["measurement"],
                 "released_value": _clean(row["value"]),
                 "coefficients": row["coefficients"], "loading": row["loading"]}
                for row in used_rows],
            "raw_release_values": [
                {"release_id": release["release_id"], "replaces": release["replaces"],
                 "measurement": measurement["measurement"],
                 "released_value": _clean(float(measurement["value"])),
                 "coefficients": dict(measurement["coefficients"]),
                 "loading": dict(measurement["loading"]),
                 "marginal_noise_variance": _clean(sum(
                     float(packet["noise_variance"][noise]) * float(coefficient) ** 2
                     for noise, coefficient in measurement["loading"].items()))}
                for release in packet["releases"] for measurement in release["measurements"]],
            "noise_variance": {name: _clean(float(value))
                               for name, value in packet["noise_variance"].items()},
            "constraints_applied": [{"kind": entry["kind"], "period": entry["period"],
                                     "total": _clean(entry["total"])} for entry in applied],
            "constraints_not_applied": unhandled,
            "malformed_cells": structure["duplicate_cells"],
        },
        "fit": fit_report,
        "periods": periods_report,
        "changes": changes,
        "identities": identities,
        "noise": noise_report,
        "coverage": coverage,
        "contradictions": contradictions,
        "conditions": conditions,
        "interpretation": {
            "headline": _interpretation({"fit": fit_report, "coverage": coverage,
                                         "noise": noise_report, "contradictions": contradictions}),
            "business": _business({"changes": changes, "fit": fit_report,
                                   "contradictions": contradictions}),
        },
    }
    _assert_finite(result)
    return result


def _label(name, structure):
    if name == "participating":
        return "参与人数"
    if name == "total":
        return "各期总人数"
    return f"{name} 参与人数"


def _scenario_text(options):
    if options["release_scope"] == "ids":
        return f"仅使用指定发布 {', '.join(options['release_ids'])}。"
    if options["release_scope"] == "all" or options["include_superseded"]:
        return "使用全部发布，包括被更正版本替代的旧发布；相关噪声已计入。"
    return "默认只使用未被替代的当前发布；被替代的旧发布不重复计入。"


def _noise_accounting(packet, model, used_rows, changes):
    """How much of the uncertainty comes from the correlated design.

    Two shortcuts are quantified for comparison, both computed with the same
    full covariance so only the *weighting* differs:

    ``independent_republication``
        Weight every release equally within its period, i.e. average a
        correction as if it were an independent second measurement.  This is
        the shortcut the operational discussion used.
    ``measurement_independence``
        Pretend the noise variables are uncorrelated across rows (keep each
        row's marginal variance).  This ignores that a total and its sub-items
        share a perturbation.

    Neither shortcut is used for the reported estimate; both are diagnostics.
    """
    ratios = []
    for entry in changes:
        if entry.get("release_noise_standard_deviation") and entry.get("independent_release_standard_deviation"):
            if entry["independent_release_standard_deviation"] > 0:
                ratios.append(entry["release_noise_standard_deviation"]
                              / entry["independent_release_standard_deviation"])

    averaging = _equal_weight_diagnostic(packet, model, changes)
    return {
        "model": "full_covariance_from_shared_loadings",
        "released_measurements": int(model.n),
        "independent_noisy_observations": int(model.information_rank),
        "noise_variables": [
            {"noise_variable": name, "variance": _clean(float(variance)),
             "shared_across_releases": _shared_variable(packet, name)}
            for name, variance in packet["noise_variance"].items()],
        "redundant_measurement_rows_removed": len(model.redundant_rows),
        "redundant_republished_measurements": [
            {"measurement": used_rows[entry["index"]]["measurement"],
             "release_id": used_rows[entry["index"]]["release_id"],
             "duplicate_of": used_rows[entry["duplicate_of"]]["measurement"],
             "scaled_by": _clean(entry["scaled_by"])}
            for entry in model.redundant_rows],
        "conflicting_republished_measurements": [
            {"measurement": used_rows[entry["index"]]["measurement"],
             "release_id": used_rows[entry["index"]]["release_id"],
             "of": used_rows[entry["of"]]["measurement"],
             "released": _clean(entry["released"]),
             "implied_by_other_release": _clean(entry["implied"])}
            for entry in model.conflicting_duplicates],
        "equal_weight_average": averaging,
        "max_change_uncertainty_ratio": _clean(max(ratios) if ratios else None),
    }


def _equal_weight_diagnostic(packet, model, changes):
    """Variance of averaging two re-publications as if they were independent.

    For each pair of measurements from *different* releases that target the
    same latent combination, the shortcut summary is their equal-weight mean.
    Treating the two as independent gives ``(v1 + v2) / 4``; the correct
    variance uses the published covariance, ``(v1 + v2 + 2 c12) / 4``.  The
    ratio shows how much false precision the shortcut buys.  Nothing here is
    used for the reported estimate.
    """
    rows = []
    for first in range(model.n):
        for second in range(first + 1, model.n):
            if model.identifiers[first] == model.identifiers[second]:
                continue
            if not np.allclose(model.design[first], model.design[second]):
                continue
            variance_first = float(model.covariance[first, first])
            variance_second = float(model.covariance[second, second])
            covariance = float(model.covariance[first, second])
            independent = (variance_first + variance_second) / 4.0
            correlated = (variance_first + variance_second + 2.0 * covariance) / 4.0
            rows.append({
                "first_measurement": model.identifiers[first],
                "second_measurement": model.identifiers[second],
                "shared_noise_covariance": _clean(covariance),
                "independent_average_standard_deviation": _clean(math.sqrt(max(independent, 0.0))),
                "correlated_average_standard_deviation": _clean(math.sqrt(max(correlated, 0.0))),
                "understatement_factor": _clean(
                    math.sqrt(correlated / independent) if independent > 0 else None),
            })
    understated = any(row["understatement_factor"] is not None and row["understatement_factor"] > 1.0001
                      for row in rows)
    return {
        "assumption": "把同一目标量的两次发布当作独立观测后等权平均",
        "used_for_reporting": False,
        "understates_uncertainty": understated,
        "pairs": rows[:20],
        "note": ("两次发布共享噪声变量时，平均不会按 1/sqrt(2) 降低方差；"
                 "understatement_factor > 1 表示独立口径把不确定性低估了该倍数。"
                 if understated else
                 "所用发布之间没有直接的等权平均口径冲突；相关结构仍按全协方差计入。"),
    }


def _shared_variable(packet, name):
    releases = {release["release_id"] for release in packet["releases"]
                for measurement in release["measurements"]
                if name in measurement["loading"]}
    return len(releases) > 1


def _contradictions(packet, structure, period_cells, releases, totals, constraints, fit, alpha, limit, used):
    if fit.feasible:
        witness = None
    else:
        witness = _infeasibility_witness(packet, structure, period_cells, releases, totals, constraints)

    statistically_rejected = bool(fit.feasible and fit.dof > 0 and fit.p_value is not None
                                  and fit.p_value < alpha)
    joint = bool(fit.feasible and not statistically_rejected)

    detail = None
    if not joint:
        detail = _maximal_consistent_subsets(packet, structure, period_cells, releases, totals, constraints, alpha, limit)
    maximal = []
    conflicts = []
    if detail is not None:
        table, maximal_sets = detail
        maximal = [sorted(combination) for combination in maximal_sets]
        all_identifiers = set(used)
        for combination, info in table.items():
            if info["verdict"] != "consistent" and set(combination) <= all_identifiers:
                if len(combination) > 1 or _single_release_bad(packet, structure, period_cells,
                                                               releases, totals, constraints, combination, alpha):
                    conflicts.append({"releases": sorted(combination),
                                      "verdict": info["verdict"],
                                      "p_value": _clean(info["p_value"]),
                                      "dof": info["dof"]})
        conflicts.sort(key=lambda entry: (len(entry["releases"]), entry["releases"]))
        conflicts = conflicts[:32]

    return {
        "structurally_feasible": fit.feasible,
        "jointly_consumable": joint,
        "statistically_rejected": statistically_rejected,
        "consistency_level": alpha,
        "p_value": _clean(fit.p_value),
        "infeasibility_witness": witness,
        "maximal_consistent_release_sets": maximal,
        "conflicting_groups": conflicts,
        "explanation": _contradiction_text(fit, witness, maximal, used, alpha),
    }


def _single_release_bad(packet, structure, period_cells, releases, totals, constraints, combination, alpha):
    by_id = {release["release_id"]: release for release in releases}
    subset = [by_id[identifier] for identifier in combination]
    model, _ = _build_model(packet, structure, period_cells, subset, totals, constraints)
    fit = model.solve()
    return not fit.feasible or (fit.dof > 0 and fit.p_value is not None and fit.p_value < alpha)


def _contradiction_text(fit, witness, maximal, used, alpha):
    if witness is not None:
        return ("这些发布的零噪声关系与声明的人数约束互相排斥：无法找到任何非负人数表同时满足它们。"
                "已保留原始发布值与冲突证据，未做静默修正。")
    if fit.feasible and fit.dof > 0 and fit.p_value is not None and fit.p_value < alpha:
        return ("各发布单独看可以解释，但放在同一协方差模型下不能同时解释；"
                "以下最大一致集合列出可以一起使用的发布组合。")
    return "在声明的机制与约束下，所用发布可以联合解释。"


def _identities(packet, structure, period_cells, periods_report, level):
    rows = []
    for period in structure["periods"]:
        aggregates = periods_report[period]["aggregates"]
        programs = [name for name in structure["programs"] if name in aggregates]
        relation = []
        if len(programs) == 2:
            relation = {
                "period": period,
                "kind": "inclusion_exclusion",
                "statement": f"{programs[0]} + {programs[1]} - 交集 = 参与人数",
                "note": "交集由与两个项目都共现的分类承担，约束系统内自动满足。",
            }
        rows.append({"period": period,
                     "total": aggregates["total"]["estimate"],
                     "participating": aggregates["participating"]["estimate"] if "participating" in aggregates else None,
                     "aggregate_estimates": {name: aggregates[name]["estimate"] for name in aggregates},
                     "relation": relation})
    return rows


def _assert_finite(value, path="result"):
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for position, item in enumerate(value):
            _assert_finite(item, f"{path}[{position}]")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite number in result")
