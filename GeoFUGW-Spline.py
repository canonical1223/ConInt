"""GeoFUGW-Spline: событийная корреляция двух скважин без DTW.

Скрипт последовательно сопоставляет A-B, B-C, ... по устойчивым
многомасштабным событиям ГИС с помощью
энтропийного unbalanced fused Gromov-Wasserstein приближения, явно
регуляризует стратиграфический порядок и восстанавливает непрерывную
строго монотонную cubic-Hermite spline-карту глубин.

Пример:

    python GeoFUGW-Spline.py A.csv B.csv C.csv \
        --curves GR RHOB NPHI --md MD -o correlation.csv

Результат начинается с тех же колонок, что minimal_dtw.py, и читается
plot_correlation.py; GeoFUGW-QC добавляется справа. Опциональный CSV
мягких маркеров ``marker,left_md,right_md`` передаётся отдельным
--anchors для каждой соседней пары. Скрипт зависит только от numpy и pandas.

Это практическая событийная аппроксимация FUGW, а не точная реализация
конкретного библиотечного solver. Энтропия transport plan является QC,
но не калиброванной байесовской вероятностью. Для эмпирических полос
используется ансамбль возмущений событий (--bootstrap).
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import warnings
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd


EPS = np.finfo(float).eps
STANDARD_MISSING = (-999.25, -999.0, 999.25)


@dataclass(slots=True)
class PreparedWell:
    name: str
    md: np.ndarray
    raw: np.ndarray
    normalized: np.ndarray
    curves: tuple[str, ...]
    step: float


@dataclass(slots=True)
class EventSet:
    depth: np.ndarray
    u: np.ndarray
    features: np.ndarray
    mass: np.ndarray
    score: np.ndarray
    anchor_id: np.ndarray


@dataclass(slots=True)
class TransportResult:
    plan: np.ndarray
    feature_cost: np.ndarray
    iterations: int
    relative_change: float
    transported_mass: float
    crossing_mass: float
    gamma: np.ndarray


@dataclass(slots=True)
class MonotoneSpline:
    x: np.ndarray
    y: np.ndarray
    slopes: np.ndarray

    def evaluate(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query, dtype=float)
        values = np.full(query.shape, np.nan, dtype=float)
        derivative = np.full(query.shape, np.nan, dtype=float)
        valid = np.isfinite(query) & (query >= self.x[0]) & (query <= self.x[-1])
        if not np.any(valid):
            return values, derivative

        q = query[valid]
        index = np.searchsorted(self.x, q, side="right") - 1
        index = np.clip(index, 0, len(self.x) - 2)
        x0, x1 = self.x[index], self.x[index + 1]
        y0, y1 = self.y[index], self.y[index + 1]
        m0, m1 = self.slopes[index], self.slopes[index + 1]
        h = x1 - x0
        t = (q - x0) / h

        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        values[valid] = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1

        dh00 = 6 * t**2 - 6 * t
        dh10 = 3 * t**2 - 4 * t + 1
        dh01 = -6 * t**2 + 6 * t
        dh11 = 3 * t**2 - 2 * t
        derivative[valid] = (
            dh00 * y0 + dh10 * h * m0 + dh01 * y1 + dh11 * h * m1
        ) / h
        return values, derivative


@dataclass(slots=True)
class Config:
    curves: tuple[str, ...]
    md_column: str = "MD"
    step: float | None = None
    max_interpolation_gap: float | None = None
    scales: tuple[float, ...] | None = None
    min_event_distance: float | None = None
    max_events: int = 120
    event_percentile: float = 65.0
    alpha: float = 0.55
    epsilon: float = 0.06
    mass_penalty: float = 0.8
    order_strength: float = 0.6
    order_sigma: float = 0.22
    fugw_iterations: int = 24
    sinkhorn_iterations: int = 250
    damping: float = 0.65
    tolerance: float = 1e-6
    anchor_weight: float = 8.0
    anchor_penalty: float = 12.0
    spline_knots: int = 18
    spline_smoothness: float = 1.0
    min_stretch: float = 0.05
    max_stretch: float = 20.0
    min_row_transport: float = 0.02
    bootstrap: int = 0
    bootstrap_jitter: float = 0.04
    seed: int = 42

    def validate(self) -> None:
        if not self.curves or any(not str(curve).strip() for curve in self.curves):
            raise ValueError("Нужно указать хотя бы одну непустую кривую")
        if self.step is not None and (not np.isfinite(self.step) or self.step <= 0):
            raise ValueError("--step должен быть положительным")
        if self.max_interpolation_gap is not None and self.max_interpolation_gap <= 0:
            raise ValueError("--max-interpolation-gap должен быть положительным")
        if self.scales is not None and (
            not self.scales or any(not np.isfinite(v) or v <= 0 for v in self.scales)
        ):
            raise ValueError("Все --scales должны быть положительными")
        if self.min_event_distance is not None and self.min_event_distance <= 0:
            raise ValueError("--min-event-distance должен быть положительным")
        if self.max_events < 8:
            raise ValueError("--max-events должен быть не меньше 8")
        if not 0 <= self.event_percentile < 100:
            raise ValueError("--event-percentile должен лежать в [0, 100)")
        if not 0 <= self.alpha <= 1:
            raise ValueError("--alpha должен лежать в [0, 1]")
        for name, value in (
            ("epsilon", self.epsilon),
            ("mass-penalty", self.mass_penalty),
            ("order-sigma", self.order_sigma),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"--{name} должен быть положительным")
        if self.order_strength < 0 or self.anchor_penalty < 0:
            raise ValueError("Штрафы должны быть неотрицательными")
        if self.fugw_iterations < 1 or self.sinkhorn_iterations < 1:
            raise ValueError("Число итераций должно быть положительным")
        if not 0 < self.damping <= 1:
            raise ValueError("--damping должен лежать в (0, 1]")
        if self.spline_knots < 2:
            raise ValueError("--spline-knots должен быть не меньше 2")
        if self.spline_smoothness < 0:
            raise ValueError("--spline-smoothness должен быть неотрицательным")
        if not 0 < self.min_stretch < self.max_stretch:
            raise ValueError("Нужны 0 < --min-stretch < --max-stretch")
        if not 0 <= self.min_row_transport <= 1:
            raise ValueError("--min-row-transport должен лежать в [0, 1]")
        if self.bootstrap < 0 or self.bootstrap_jitter < 0:
            raise ValueError("Параметры bootstrap должны быть неотрицательными")


def _numeric(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    text = text.str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
    text = text.str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce")


def _read_csv(path: str | Path, required: Sequence[str]) -> pd.DataFrame:
    source = Path(path)
    failures: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1251", "latin1"):
        try:
            frame = pd.read_csv(source, sep=None, engine="python", encoding=encoding, dtype=str)
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            failures.append(f"{encoding}: {exc}")
            continue
        frame.columns = [str(column).strip().replace("\ufeff", "") for column in frame]
        if all(column in frame.columns for column in required):
            return frame
        failures.append(f"{encoding}: столбцы {list(frame.columns)}")
    raise ValueError(
        f"{source}: не найдены обязательные столбцы {list(required)}; "
        + "; ".join(failures[-2:])
    )


def _load_raw_well(path: str | Path, md_column: str, curves: tuple[str, ...]) -> pd.DataFrame:
    frame = _read_csv(path, (md_column, *curves))
    numeric = pd.DataFrame({md_column: _numeric(frame[md_column])})
    for curve in curves:
        numeric[curve] = _numeric(frame[curve])
        numeric.loc[numeric[curve].isin(STANDARD_MISSING), curve] = np.nan
    numeric = numeric.dropna(subset=[md_column])
    if np.any(np.isinf(numeric.to_numpy(float))):
        raise ValueError(f"{path}: обнаружены бесконечные значения")
    numeric = numeric.groupby(md_column, as_index=False, sort=True)[list(curves)].mean()
    numeric = numeric.sort_values(md_column).reset_index(drop=True)
    if len(numeric) < 5:
        raise ValueError(f"{path}: недостаточно точек")
    md = numeric[md_column].to_numpy(float)
    if np.any(np.diff(md) <= 0):
        raise ValueError(f"{path}: MD должен строго возрастать")
    if not any(np.isfinite(numeric[curve].to_numpy(float)).sum() >= 5 for curve in curves):
        raise ValueError(f"{path}: нет кривой с пятью валидными значениями")
    return numeric


def _positive_steps(frame: pd.DataFrame, md_column: str) -> np.ndarray:
    values = frame[md_column].to_numpy(float)
    delta = np.diff(values)
    return delta[np.isfinite(delta) & (delta > 0)]


def _grid(start: float, stop: float, step: float) -> np.ndarray:
    count = int(math.floor((stop - start) / step)) + 1
    if count < 2 or count > 500_000:
        raise ValueError("Некорректный размер регулярной сетки; измените --step")
    result = start + np.arange(count, dtype=float) * step
    if result[-1] < stop and not np.isclose(result[-1], stop):
        result = np.append(result, stop)
    else:
        result[-1] = stop
    return result


def _interpolate_segments(
    source_md: np.ndarray,
    source_values: np.ndarray,
    target_md: np.ndarray,
    max_gap: float,
) -> np.ndarray:
    result = np.full(target_md.shape, np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(source_values))
    if not len(valid_indices):
        return result
    starts = [0]
    for position in range(1, len(valid_indices)):
        previous = valid_indices[position - 1]
        current = valid_indices[position]
        if source_md[current] - source_md[previous] > max_gap:
            starts.append(position)
    starts.append(len(valid_indices))
    for lo, hi in zip(starts[:-1], starts[1:]):
        indices = valid_indices[lo:hi]
        if len(indices) == 1:
            nearest = int(np.argmin(np.abs(target_md - source_md[indices[0]])))
            if abs(target_md[nearest] - source_md[indices[0]]) <= max_gap / 2:
                result[nearest] = source_values[indices[0]]
            continue
        x, y = source_md[indices], source_values[indices]
        inside = (target_md >= x[0]) & (target_md <= x[-1])
        result[inside] = np.interp(target_md[inside], x, y)
    return result


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=float)
    for column in range(values.shape[1]):
        data = values[:, column]
        valid = np.isfinite(data)
        if valid.sum() < 3:
            continue
        median = float(np.median(data[valid]))
        mad = float(np.median(np.abs(data[valid] - median)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= EPS:
            q25, q75 = np.percentile(data[valid], [25, 75])
            scale = float((q75 - q25) / 1.349)
        if not np.isfinite(scale) or scale <= EPS:
            continue
        normalized = (data - median) / scale
        result[:, column] = np.clip(normalized, -8.0, 8.0)
    return result


def _prepare_well(
    frame: pd.DataFrame,
    name: str,
    md_column: str,
    curves: tuple[str, ...],
    step: float,
    max_gap: float,
) -> PreparedWell:
    source_md = frame[md_column].to_numpy(float)
    target_md = _grid(float(source_md[0]), float(source_md[-1]), step)
    raw = np.column_stack(
        [
            _interpolate_segments(
                source_md,
                frame[curve].to_numpy(float),
                target_md,
                max_gap,
            )
            for curve in curves
        ]
    )
    normalized = _robust_normalize(raw)
    if not np.any(np.isfinite(normalized)):
        raise ValueError(f"{name}: выбранные кривые постоянны или полностью отсутствуют")
    return PreparedWell(name, target_md, raw, normalized, curves, step)


def _finite_runs(values: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(np.isfinite(values))
    if not len(indices):
        return []
    split = np.flatnonzero(np.diff(indices) > 1) + 1
    return [part for part in np.split(indices, split) if len(part)]


def _convolve_same(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    full = np.convolve(values, kernel, mode="full")
    start = (len(kernel) - 1) // 2
    return full[start : start + len(values)]


def _smooth_segments(values: np.ndarray, sigma_samples: float) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=float)
    for indices in _finite_runs(values):
        segment = values[indices]
        if len(segment) < 3 or sigma_samples <= 0.35:
            result[indices] = segment
            continue
        radius = min(int(math.ceil(3 * sigma_samples)), max(1, len(segment) - 1))
        positions = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (positions / max(sigma_samples, 0.35)) ** 2)
        kernel /= kernel.sum()
        result[indices] = _convolve_same(segment, kernel)
    return result


def _gradient_segments(values: np.ndarray, md: np.ndarray) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=float)
    for indices in _finite_runs(values):
        if len(indices) < 3:
            result[indices] = 0.0
        else:
            result[indices] = np.gradient(values[indices], md[indices], edge_order=1)
    return result


def _nearest_index(md: np.ndarray, depth: float) -> int:
    position = int(np.searchsorted(md, depth))
    candidates = [max(0, min(len(md) - 1, position))]
    if position:
        candidates.append(position - 1)
    return min(candidates, key=lambda index: abs(md[index] - depth))


def _extract_events(
    well: PreparedWell,
    scales: tuple[float, ...],
    min_distance: float,
    max_events: int,
    percentile: float,
    anchors: list[tuple[int, float]],
    anchor_weight: float,
) -> EventSet:
    feature_columns: list[np.ndarray] = []
    score_components: list[np.ndarray] = []
    for column in range(well.normalized.shape[1]):
        base = well.normalized[:, column]
        feature_columns.append(base)
        for scale in scales:
            smooth = _smooth_segments(base, scale / well.step)
            first = _gradient_segments(smooth, well.md)
            second = _gradient_segments(first, well.md)
            feature_columns.extend((smooth, first * scale, second * scale * scale))
            score_components.extend((np.abs(first) * math.sqrt(scale), np.abs(second) * scale))

    all_features = np.column_stack(feature_columns)
    score_matrix = np.column_stack(score_components)
    score = np.nanmax(score_matrix, axis=1)
    score[~np.any(np.isfinite(all_features), axis=1)] = np.nan
    finite_score = score[np.isfinite(score)]
    if not len(finite_score):
        raise ValueError(f"{well.name}: не удалось выделить события")
    threshold = float(np.percentile(finite_score, percentile))

    local = np.zeros(len(score), dtype=bool)
    if len(score) >= 3:
        local[1:-1] = (
            np.isfinite(score[1:-1])
            & (score[1:-1] >= score[:-2])
            & (score[1:-1] >= score[2:])
            & (score[1:-1] >= threshold)
        )
    candidates = np.flatnonzero(local)
    if len(candidates) < 6:
        finite = np.flatnonzero(np.isfinite(score))
        candidates = finite[np.argsort(score[finite])[::-1][: max(6, max_events)]]

    ordered = candidates[np.argsort(score[candidates])[::-1]]
    chosen: list[int] = []
    for index in ordered:
        if all(abs(well.md[index] - well.md[other]) >= min_distance for other in chosen):
            chosen.append(int(index))
        if len(chosen) >= max_events:
            break

    valid_rows = np.flatnonzero(np.any(np.isfinite(all_features), axis=1))
    for endpoint in (int(valid_rows[0]), int(valid_rows[-1])):
        if endpoint not in chosen:
            chosen.append(endpoint)

    anchor_entries: list[tuple[int, int]] = []
    for anchor_id, depth in anchors:
        anchor_entries.append((_nearest_index(well.md, depth), int(anchor_id)))

    entries = [(index, -1) for index in chosen] + anchor_entries
    entries.sort(key=lambda item: (well.md[item[0]], item[1]))
    indices = np.asarray([item[0] for item in entries], dtype=int)
    anchor_ids = np.asarray([item[1] for item in entries], dtype=int)
    event_score = np.nan_to_num(score[indices], nan=0.0)
    positive = event_score[event_score > 0]
    reference = float(np.median(positive)) if len(positive) else 1.0
    mass = np.clip(0.15 + event_score / max(reference, EPS), 0.15, 12.0)
    mass[anchor_ids >= 0] *= anchor_weight
    mass /= mass.sum()
    depth = well.md[indices]
    span = float(well.md[-1] - well.md[0])
    u = (depth - well.md[0]) / span
    return EventSet(depth, u, all_features[indices], mass, event_score, anchor_ids)


def _load_anchors(path: str | Path | None) -> list[tuple[str, float, float]]:
    if path is None:
        return []
    frame = _read_csv(path, ("marker", "left_md", "right_md"))
    result: list[tuple[str, float, float]] = []
    for _, row in frame.iterrows():
        marker = str(row["marker"]).strip()
        left = _numeric(pd.Series([row["left_md"]])).iloc[0]
        right = _numeric(pd.Series([row["right_md"]])).iloc[0]
        if marker and np.isfinite(left) and np.isfinite(right):
            result.append((marker, float(left), float(right)))
    if len({name.casefold() for name, _, _ in result}) != len(result):
        raise ValueError("В --anchors имена marker должны быть уникальными")
    left_order = np.argsort([left for _, left, _ in result])
    ordered_right = np.asarray([result[index][2] for index in left_order], dtype=float)
    if np.any(np.diff(ordered_right) <= 0):
        raise ValueError("Мягкие маркеры должны иметь одинаковый строгий порядок")
    return result


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    maximum[~np.isfinite(maximum)] = 0.0
    total = np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    answer = maximum + np.log(np.maximum(total, EPS))
    return np.squeeze(answer, axis=axis)


def _feature_cost(left: EventSet, right: EventSet, anchor_penalty: float) -> np.ndarray:
    """Robust multichannel event dissimilarity with missing-channel support."""
    n_left, n_right = len(left.depth), len(right.depth)
    accumulated = np.zeros((n_left, n_right), dtype=float)
    available = np.zeros((n_left, n_right), dtype=float)
    dimensions = min(left.features.shape[1], right.features.shape[1])
    for column in range(dimensions):
        a = left.features[:, column]
        b = right.features[:, column]
        pooled = np.concatenate((a[np.isfinite(a)], b[np.isfinite(b)]))
        if len(pooled) < 4:
            continue
        centre = float(np.median(pooled))
        scale = 1.4826 * float(np.median(np.abs(pooled - centre)))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(np.std(pooled))
        scale = max(scale, 1e-6)
        difference = np.abs((a[:, None] - b[None, :]) / scale)
        valid = np.isfinite(difference)
        huber = np.where(difference <= 1.5, 0.5 * difference**2, 1.5 * (difference - 0.75))
        accumulated[valid] += huber[valid]
        available[valid] += 1.0
    cost = accumulated / np.maximum(available, 1.0)
    fallback = float(np.nanpercentile(cost[available > 0], 90)) if np.any(available > 0) else 10.0
    cost[available == 0] = max(fallback, 1.0) * 2.0

    left_anchor = left.anchor_id[:, None]
    right_anchor = right.anchor_id[None, :]
    both = (left_anchor >= 0) & (right_anchor >= 0)
    same = both & (left_anchor == right_anchor)
    mismatch = both & ~same
    only_one = (left_anchor >= 0) ^ (right_anchor >= 0)
    cost[same] *= 0.05
    cost[mismatch] += anchor_penalty
    cost[only_one] += 0.35 * anchor_penalty
    reference_values = cost[np.isfinite(cost) & (cost > 0)]
    reference = float(np.median(reference_values)) if len(reference_values) else 1.0
    return np.nan_to_num(cost / max(reference, EPS), nan=10.0, posinf=10.0, neginf=0.0)


def _weighted_isotonic(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted PAVA projection onto the non-decreasing cone."""
    y = np.asarray(values, dtype=float)
    w = np.maximum(np.asarray(weights, dtype=float), EPS)
    level: list[float] = []
    mass: list[float] = []
    start: list[int] = []
    end: list[int] = []
    for index, (value, weight) in enumerate(zip(y, w, strict=True)):
        level.append(float(value))
        mass.append(float(weight))
        start.append(index)
        end.append(index + 1)
        while len(level) >= 2 and level[-2] > level[-1]:
            combined = mass[-2] + mass[-1]
            merged = (mass[-2] * level[-2] + mass[-1] * level[-1]) / combined
            level[-2:] = [merged]
            mass[-2:] = [combined]
            end[-2:] = [end[-1]]
            start.pop()
    result = np.empty_like(y)
    for value, first, last in zip(level, start, end, strict=True):
        result[first:last] = value
    return result


def _unbalanced_sinkhorn(
    cost: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    mass_penalty: float,
    iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, float]:
    """Entropic KL-unbalanced transport; returns a normalized plan and raw mass."""
    log_a = np.log(np.maximum(a, EPS))
    log_b = np.log(np.maximum(b, EPS))
    log_kernel = -np.asarray(cost, dtype=float) / epsilon
    log_u = np.zeros_like(a)
    log_v = np.zeros_like(b)
    tau = mass_penalty / (mass_penalty + epsilon)
    for _ in range(iterations):
        previous = log_u.copy()
        log_u = tau * (log_a - _logsumexp(log_kernel + log_v[None, :], axis=1))
        log_v = tau * (log_b - _logsumexp(log_kernel + log_u[:, None], axis=0))
        if np.max(np.abs(log_u - previous)) < tolerance:
            break
    log_plan = log_kernel + log_u[:, None] + log_v[None, :]
    plan = np.exp(np.clip(log_plan, -745.0, 60.0))
    raw_mass = float(plan.sum())
    if not np.isfinite(raw_mass) or raw_mass <= EPS:
        raise RuntimeError("Транспортный план выродился; увеличьте --epsilon или --order-sigma")
    return plan / raw_mass, raw_mass


def _initial_order_map(
    left: EventSet,
    right: EventSet,
    anchor_pairs: Sequence[tuple[float, float]],
    left_range: tuple[float, float],
    right_range: tuple[float, float],
) -> np.ndarray:
    if not anchor_pairs:
        return left.u.copy()
    left_min, left_max = left_range
    right_min, right_max = right_range
    source = [0.0]
    target = [0.0]
    for left_depth, right_depth in sorted(anchor_pairs):
        source.append((left_depth - left_min) / (left_max - left_min))
        target.append((right_depth - right_min) / (right_max - right_min))
    source.append(1.0)
    target.append(1.0)
    source_array = np.asarray(source, dtype=float)
    target_array = np.asarray(target, dtype=float)
    keep = np.r_[True, np.diff(source_array) > 1e-10]
    return np.interp(left.u, source_array[keep], target_array[keep])


def _crossing_mass(plan: np.ndarray) -> float:
    previous_rows = np.zeros(plan.shape[1], dtype=float)
    crossing = 0.0
    for row in plan:
        greater = np.cumsum(previous_rows[::-1])[::-1] - previous_rows
        crossing += float(np.dot(row, greater))
        previous_rows += row
    return max(0.0, min(1.0, 2.0 * crossing))


def _solve_transport(
    left: EventSet,
    right: EventSet,
    config: Config,
    anchor_pairs: Sequence[tuple[float, float]],
    left_range: tuple[float, float],
    right_range: tuple[float, float],
) -> TransportResult:
    feature = _feature_cost(left, right, config.anchor_penalty)
    distance_left = np.abs(left.u[:, None] - left.u[None, :])
    distance_right = np.abs(right.u[:, None] - right.u[None, :])
    plan = np.outer(left.mass, right.mass)
    gamma = _initial_order_map(left, right, anchor_pairs, left_range, right_range)
    relative_change = math.inf
    raw_mass = 1.0
    completed = 0
    for outer in range(config.fugw_iterations):
        row_mass = plan.sum(axis=1)
        column_mass = plan.sum(axis=0)
        gw = (
            (distance_left**2 @ row_mass)[:, None]
            + (distance_right**2 @ column_mass)[None, :]
            - 2.0 * distance_left @ plan @ distance_right.T
        )
        gw -= np.min(gw)
        gw_reference = float(np.median(gw[gw > 0])) if np.any(gw > 0) else 1.0
        gw /= max(gw_reference, EPS)
        order = (right.u[None, :] - gamma[:, None]) ** 2 / (2.0 * config.order_sigma**2)
        joint_cost = (1.0 - config.alpha) * feature + config.alpha * gw + config.order_strength * order
        proposed, raw_mass = _unbalanced_sinkhorn(
            joint_cost,
            left.mass,
            right.mass,
            config.epsilon,
            config.mass_penalty,
            config.sinkhorn_iterations,
            config.tolerance,
        )
        updated = config.damping * plan + (1.0 - config.damping) * proposed
        updated /= updated.sum()
        relative_change = float(np.linalg.norm(updated - plan) / max(np.linalg.norm(plan), EPS))
        plan = updated
        barycentric = (plan @ right.u) / np.maximum(plan.sum(axis=1), EPS)
        gamma = _weighted_isotonic(barycentric, np.maximum(plan.sum(axis=1), EPS))
        completed = outer + 1
        if relative_change < config.tolerance * 10.0:
            break
    return TransportResult(
        plan=plan,
        feature_cost=feature,
        iterations=completed,
        relative_change=relative_change,
        transported_mass=raw_mass,
        crossing_mass=_crossing_mass(plan),
        gamma=gamma,
    )


def _bounded_monotone(values: np.ndarray, x: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    y = np.asarray(values, dtype=float).copy()
    y = _weighted_isotonic(y, np.ones_like(y))
    for _ in range(4):
        for index in range(1, len(y)):
            dx = x[index] - x[index - 1]
            y[index] = np.clip(y[index], y[index - 1] + minimum * dx, y[index - 1] + maximum * dx)
        for index in range(len(y) - 2, -1, -1):
            dx = x[index + 1] - x[index]
            y[index] = np.clip(y[index], y[index + 1] - maximum * dx, y[index + 1] - minimum * dx)
    return y


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson derivatives for a shape-preserving cubic Hermite spline."""
    if len(x) == 2:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return np.asarray([slope, slope], dtype=float)
    h = np.diff(x)
    delta = np.diff(y) / h
    derivative = np.zeros_like(y)
    for index in range(1, len(y) - 1):
        if delta[index - 1] <= 0 or delta[index] <= 0:
            derivative[index] = 0.0
        else:
            w1 = 2.0 * h[index] + h[index - 1]
            w2 = h[index] + 2.0 * h[index - 1]
            derivative[index] = (w1 + w2) / (w1 / delta[index - 1] + w2 / delta[index])
    derivative[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    derivative[-1] = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if derivative[0] * delta[0] <= 0:
        derivative[0] = 0.0
    elif derivative[0] > 3.0 * delta[0]:
        derivative[0] = 3.0 * delta[0]
    if derivative[-1] * delta[-1] <= 0:
        derivative[-1] = 0.0
    elif derivative[-1] > 3.0 * delta[-1]:
        derivative[-1] = 3.0 * delta[-1]
    return derivative


def _unique_weighted_points(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, inverse = np.unique(x, return_inverse=True)
    y_sum = np.zeros(len(unique), dtype=float)
    w_sum = np.zeros(len(unique), dtype=float)
    for index, group in enumerate(inverse):
        y_sum[group] += weights[index] * y[index]
        w_sum[group] += weights[index]
    return unique, y_sum / np.maximum(w_sum, EPS), w_sum


def _compress_points(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, maximum: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(x) <= maximum:
        return x, y, weights
    groups = np.array_split(np.arange(len(x)), maximum)
    compressed_x = []
    compressed_y = []
    compressed_w = []
    for group in groups:
        mass = weights[group].sum()
        compressed_x.append(float(np.average(x[group], weights=weights[group])))
        compressed_y.append(float(np.average(y[group], weights=weights[group])))
        compressed_w.append(float(mass))
    return np.asarray(compressed_x), np.asarray(compressed_y), np.asarray(compressed_w)


def _fit_spline(
    left: EventSet,
    right: EventSet,
    transport: TransportResult,
    config: Config,
) -> tuple[MonotoneSpline, np.ndarray, np.ndarray, np.ndarray]:
    row_mass = transport.plan.sum(axis=1)
    conditional = transport.plan / np.maximum(row_mass[:, None], EPS)
    mapped = conditional @ right.depth
    if transport.plan.shape[1] > 1:
        entropy = -np.sum(conditional * np.log(np.maximum(conditional, EPS)), axis=1) / math.log(
            transport.plan.shape[1]
        )
    else:
        entropy = np.zeros(len(left.depth), dtype=float)
    row_coverage = np.minimum(row_mass / np.maximum(left.mass, EPS), 1.0)
    confidence = np.clip(row_coverage * (1.0 - entropy), 0.0, 1.0)
    weights = np.maximum(left.mass * (0.05 + confidence), EPS)
    valid = np.isfinite(mapped) & (row_coverage >= config.min_row_transport)
    if np.count_nonzero(valid) < 4:
        valid = np.isfinite(mapped)
    x, y, w = _unique_weighted_points(left.depth[valid], mapped[valid], weights[valid])
    x, y, w = _compress_points(x, y, w, config.spline_knots)
    if len(x) < 2:
        raise RuntimeError("Недостаточно транспортных соответствий для сплайна")
    if len(x) >= 3 and config.spline_smoothness > 0:
        difference = np.zeros((len(x) - 2, len(x)), dtype=float)
        for index in range(len(x) - 2):
            difference[index, index : index + 3] = (1.0, -2.0, 1.0)
        system = np.diag(w / max(np.median(w), EPS)) + config.spline_smoothness * difference.T @ difference
        y = np.linalg.solve(system, (w / max(np.median(w), EPS)) * y)
    y = _bounded_monotone(y, x, config.min_stretch, config.max_stretch)
    spline = MonotoneSpline(x=x, y=y, slopes=_pchip_slopes(x, y))
    return spline, confidence, entropy, mapped


def _interpolate_event_metric(event_depth: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    order = np.argsort(event_depth)
    x = event_depth[order]
    y = values[order]
    unique, inverse = np.unique(x, return_inverse=True)
    total = np.zeros(len(unique), dtype=float)
    count = np.zeros(len(unique), dtype=float)
    for index, group in enumerate(inverse):
        if np.isfinite(y[index]):
            total[group] += y[index]
            count[group] += 1.0
    averaged = total / np.maximum(count, 1.0)
    return np.interp(query, unique, averaged, left=averaged[0], right=averaged[-1])


def _validate_anchor_ranges(
    anchors: Sequence[tuple[str, float, float]], left: PreparedWell, right: PreparedWell
) -> None:
    for marker, left_depth, right_depth in anchors:
        if not left.md[0] <= left_depth <= left.md[-1]:
            raise ValueError(f"Маркер {marker!r}: left_md вне диапазона левой скважины")
        if not right.md[0] <= right_depth <= right.md[-1]:
            raise ValueError(f"Маркер {marker!r}: right_md вне диапазона правой скважины")


def _perturb_events(events: EventSet, rng: np.random.Generator, jitter: float) -> EventSet:
    features = events.features.copy()
    finite = np.isfinite(features)
    features[finite] += rng.normal(0.0, jitter, size=np.count_nonzero(finite))
    mass = events.mass * np.exp(rng.normal(0.0, jitter, size=len(events.mass)))
    mass[events.anchor_id >= 0] *= np.exp(rng.normal(0.0, 0.25 * jitter, size=np.count_nonzero(events.anchor_id >= 0)))
    mass /= mass.sum()
    return replace(events, features=features, mass=mass)


def _bootstrap_maps(
    left_events: EventSet,
    right_events: EventSet,
    config: Config,
    anchor_pairs: Sequence[tuple[float, float]],
    left_range: tuple[float, float],
    right_range: tuple[float, float],
    query: np.ndarray,
) -> np.ndarray:
    if config.bootstrap <= 0:
        return np.empty((0, len(query)), dtype=float)
    rng = np.random.default_rng(config.seed)
    maps: list[np.ndarray] = []
    bootstrap_config = replace(
        config,
        bootstrap=0,
        fugw_iterations=max(8, int(math.ceil(0.65 * config.fugw_iterations))),
        sinkhorn_iterations=max(80, int(math.ceil(0.65 * config.sinkhorn_iterations))),
    )
    for sample in range(config.bootstrap):
        try:
            perturbed_left = _perturb_events(left_events, rng, config.bootstrap_jitter)
            perturbed_right = _perturb_events(right_events, rng, config.bootstrap_jitter)
            transport = _solve_transport(
                perturbed_left,
                perturbed_right,
                bootstrap_config,
                anchor_pairs,
                left_range,
                right_range,
            )
            spline, _, _, _ = _fit_spline(
                perturbed_left, perturbed_right, transport, bootstrap_config
            )
            values, _ = spline.evaluate(query)
            if np.count_nonzero(np.isfinite(values)) >= max(2, len(values) // 2):
                maps.append(values)
        except (RuntimeError, np.linalg.LinAlgError, ValueError) as error:
            warnings.warn(f"Bootstrap {sample + 1} пропущен: {error}", RuntimeWarning, stacklevel=2)
    return np.asarray(maps, dtype=float) if maps else np.empty((0, len(query)), dtype=float)


def correlate_wells(
    left_path: str | Path,
    right_path: str | Path,
    config: Config,
    anchors_path: str | Path | None = None,
    *,
    pair_index: int = 0,
    left_name: str | None = None,
    right_name: str | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Run the complete GeoFUGW-Spline workflow and return mapping plus QC."""
    config.validate()
    left_raw = _load_raw_well(left_path, config.md_column, config.curves)
    right_raw = _load_raw_well(right_path, config.md_column, config.curves)
    native_step = float(
        min(
            np.median(_positive_steps(left_raw, config.md_column)),
            np.median(_positive_steps(right_raw, config.md_column)),
        )
    )
    step = float(config.step if config.step is not None else native_step)
    max_gap = float(
        config.max_interpolation_gap
        if config.max_interpolation_gap is not None
        else 8.0 * max(step, native_step)
    )
    scales = tuple(config.scales) if config.scales is not None else (2.0 * step, 6.0 * step, 18.0 * step)
    min_event_distance = float(
        config.min_event_distance if config.min_event_distance is not None else 2.0 * step
    )
    effective = replace(
        config,
        step=step,
        max_interpolation_gap=max_gap,
        scales=scales,
        min_event_distance=min_event_distance,
    )
    left = _prepare_well(
        left_raw,
        left_name or Path(left_path).stem,
        effective.md_column,
        effective.curves,
        step,
        max_gap,
    )
    right = _prepare_well(
        right_raw,
        right_name or Path(right_path).stem,
        effective.md_column,
        effective.curves,
        step,
        max_gap,
    )
    anchors = _load_anchors(anchors_path)
    _validate_anchor_ranges(anchors, left, right)
    left_anchor_events = [(index, item[1]) for index, item in enumerate(anchors)]
    right_anchor_events = [(index, item[2]) for index, item in enumerate(anchors)]
    anchor_pairs = [(item[1], item[2]) for item in anchors]
    left_events = _extract_events(
        left,
        effective.scales,
        effective.min_event_distance,
        effective.max_events,
        effective.event_percentile,
        left_anchor_events,
        effective.anchor_weight,
    )
    right_events = _extract_events(
        right,
        effective.scales,
        effective.min_event_distance,
        effective.max_events,
        effective.event_percentile,
        right_anchor_events,
        effective.anchor_weight,
    )
    left_range = (float(left.md[0]), float(left.md[-1]))
    right_range = (float(right.md[0]), float(right.md[-1]))
    transport = _solve_transport(
        left_events, right_events, effective, anchor_pairs, left_range, right_range
    )
    spline, event_confidence, event_entropy, _ = _fit_spline(
        left_events, right_events, transport, effective
    )
    mapped, local_stretch = spline.evaluate(left.md)
    confidence = _interpolate_event_metric(left_events.depth, event_confidence, left.md)
    entropy = _interpolate_event_metric(left_events.depth, event_entropy, left.md)

    ensemble = _bootstrap_maps(
        left_events,
        right_events,
        effective,
        anchor_pairs,
        left_range,
        right_range,
        left.md,
    )
    if len(ensemble):
        p05, p50, p95 = np.nanpercentile(ensemble, (5.0, 50.0, 95.0), axis=0)
        p50 = np.where(np.isfinite(p50), p50, mapped)
    else:
        p05 = mapped.copy()
        p50 = mapped.copy()
        p95 = mapped.copy()

    output: dict[str, object] = {
        "left_well": np.repeat(left.name, len(left.md)),
        "right_well": np.repeat(right.name, len(left.md)),
        "left_md": left.md,
        "mapped_right_md": mapped,
        "map_p05": p05,
        "map_p50": p50,
        "map_p95": p95,
        "local_stretch": local_stretch,
        "confidence": confidence,
        "transport_entropy": entropy,
    }
    residual_columns: list[np.ndarray] = []
    mapped_normalized_columns: list[np.ndarray] = []
    matched_channels = np.zeros(len(left.md), dtype=int)
    for column, curve in enumerate(config.curves):
        mapped_raw = _interpolate_segments(
            right.md, right.raw[:, column], mapped, effective.max_interpolation_gap
        )
        mapped_normalized = _interpolate_segments(
            right.md, right.normalized[:, column], mapped, effective.max_interpolation_gap
        )
        residual = np.abs(left.normalized[:, column] - mapped_normalized)
        valid = np.isfinite(residual)
        matched_channels += valid.astype(int)
        residual_columns.append(residual)
        mapped_normalized_columns.append(mapped_normalized)
        output[f"left_{curve}"] = left.raw[:, column]
        output[f"mapped_right_{curve}"] = mapped_raw
        output[f"normalized_residual_{curve}"] = residual
    residual_matrix = np.column_stack(residual_columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        normalized_residual = np.nanmean(residual_matrix, axis=1)
    output["matched_channels"] = matched_channels
    output["normalized_residual"] = normalized_residual
    status = np.full(len(left.md), "accepted", dtype=object)
    status[(confidence < 0.25) | (matched_channels == 0)] = "low_confidence"
    status[~np.isfinite(mapped)] = "unmapped"
    output["status"] = status

    expected_feature_cost = float(np.sum(transport.plan * transport.feature_cost))
    anchor_residuals: list[float] = []
    for _, left_depth, right_depth in anchors:
        estimate, _ = spline.evaluate(np.asarray([left_depth]))
        if np.isfinite(estimate[0]):
            anchor_residuals.append(abs(float(estimate[0]) - right_depth))
    diagnostics: dict[str, float | int | str] = {
        "method": "GeoFUGW-Spline",
        "left_events": len(left_events.depth),
        "right_events": len(right_events.depth),
        "fugw_iterations": transport.iterations,
        "relative_plan_change": transport.relative_change,
        "transported_mass_raw": transport.transported_mass,
        "crossing_mass": transport.crossing_mass,
        "expected_feature_cost": expected_feature_cost,
        "spline_knots_used": len(spline.x),
        "bootstrap_success": len(ensemble),
        "median_confidence": float(np.nanmedian(confidence)),
        "median_normalized_residual": float(np.nanmedian(normalized_residual)),
        "median_anchor_residual": float(np.median(anchor_residuals)) if anchor_residuals else math.nan,
    }
    for key, value in diagnostics.items():
        output[f"qc_{key}"] = np.repeat(value, len(left.md))
    details = pd.DataFrame(output)
    left_z = left.normalized[:, 0]
    right_z = mapped_normalized_columns[0]
    compatible = np.isfinite(left.md) & np.isfinite(mapped) & np.isfinite(left_z) & np.isfinite(right_z)
    if np.count_nonzero(compatible) < 2:
        raise RuntimeError(
            f"{left.name} -> {right.name}: недостаточно валидных соответствий основной кривой "
            f"{effective.curves[0]!r}"
        )
    details = details.loc[compatible].reset_index(drop=True)
    left_z = left_z[compatible]
    right_z = right_z[compatible]
    local_cost = np.abs(left_z - right_z)
    mean_cost = float(np.mean(local_cost))
    count = len(details)
    legacy = pd.DataFrame(
        {
            "pair": np.repeat(int(pair_index), count),
            "left_well": np.repeat(left.name, count),
            "right_well": np.repeat(right.name, count),
            "left_md": details["left_md"].to_numpy(float),
            "right_md": details["mapped_right_md"].to_numpy(float),
            "left_z": left_z,
            "right_z": right_z,
            "local_cost": local_cost,
            "mean_cost": np.repeat(mean_cost, count),
            "path_index": np.arange(count, dtype=int),
            "move": np.concatenate((np.asarray(["start"], dtype=object), np.repeat("diagonal", count - 1))),
            "cumulative_cost": np.cumsum(local_cost),
            "total_cost": np.repeat(float(np.sum(local_cost)), count),
            "normalized_cost": np.repeat(mean_cost, count),
            "warp_penalty": np.zeros(count, dtype=float),
            "diagonal_steps": np.repeat(count - 1, count),
            "up_steps": np.zeros(count, dtype=int),
            "left_steps": np.zeros(count, dtype=int),
            "max_warp_run": np.zeros(count, dtype=int),
            "boundary_hit": np.zeros(count, dtype=bool),
            "boundary_fraction": np.zeros(count, dtype=float),
        }
    )
    duplicate = {"left_well", "right_well", "left_md"}
    extras = details[[column for column in details.columns if column not in duplicate]]
    return pd.concat((legacy, extras), axis=1), diagnostics


def _well_name(path: str | Path) -> str:
    source = Path(path)
    if source.stem.casefold() == "well_gis":
        return source.resolve().parent.name.strip() or source.stem
    return source.stem.strip()


def _resolve_well_names(
    files: Sequence[Path], well_names: Sequence[str] | None
) -> list[str]:
    if well_names is not None:
        if isinstance(well_names, (str, bytes)) or len(well_names) != len(files):
            raise ValueError("--well-name нужно указать ровно по одному разу для каждого файла")
        names = [str(name).strip() for name in well_names]
    else:
        names = [_well_name(path) for path in files]
        counts = Counter(name.casefold() for name in names)
        for index, name in enumerate(names):
            if counts[name.casefold()] > 1:
                parent = files[index].resolve().parent.name.strip()
                if parent:
                    names[index] = parent
    if any(not name for name in names):
        raise ValueError("Имя скважины не может быть пустым")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError(
            "Не удалось однозначно определить имена скважин; задайте --well-name для каждого файла"
        )
    return names


def correlate(
    files: Sequence[str | Path],
    config: Config,
    *,
    well_names: Sequence[str] | None = None,
    anchors: Sequence[str | Path] | str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Последовательно скоррелировать A-B, B-C, ... в формате minimal_dtw."""
    if isinstance(files, (str, bytes, Path)):
        raise ValueError("files должен быть последовательностью файлов, а не одной строкой")
    paths = [Path(path) for path in files]
    if len(paths) < 2:
        raise ValueError("Нужно минимум два файла скважин")
    config.validate()
    names = _resolve_well_names(paths, well_names)

    if config.step is None:
        well_steps: list[float] = []
        for path in paths:
            raw = _load_raw_well(path, config.md_column, config.curves)
            steps = _positive_steps(raw, config.md_column)
            if len(steps):
                well_steps.append(float(np.median(steps)))
        if not well_steps:
            raise ValueError("Невозможно определить общий шаг MD")
        effective = replace(config, step=float(np.median(well_steps)))
    else:
        effective = config

    pair_count = len(paths) - 1
    if anchors is None:
        anchor_files: list[str | Path | None] = [None] * pair_count
    else:
        supplied = [anchors] if isinstance(anchors, (str, Path)) else list(anchors)
        if len(supplied) != pair_count:
            raise ValueError(
                "--anchors нужно указать один раз для каждой соседней пары "
                f"({pair_count} файлов anchors для {len(paths)} скважин)"
            )
        anchor_files = list(supplied)

    tables: list[pd.DataFrame] = []
    for pair_index, (left_path, right_path, anchor_path) in enumerate(
        zip(paths[:-1], paths[1:], anchor_files, strict=True)
    ):
        pair_config = replace(effective, seed=effective.seed + pair_index)
        table, diagnostics = correlate_wells(
            left_path,
            right_path,
            pair_config,
            anchor_path,
            pair_index=pair_index,
            left_name=names[pair_index],
            right_name=names[pair_index + 1],
        )
        tables.append(table)
        if progress is not None:
            progress(
                f"{names[pair_index]} -> {names[pair_index + 1]}: {len(table)} точек, "
                f"mean|Δz|={float(table['mean_cost'].iloc[0]):.5g}, "
                f"confidence={diagnostics['median_confidence']:.3f}"
            )
    return pd.concat(tables, ignore_index=True)


def _atomic_write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix=f".{destination.stem}.", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GeoFUGW-Spline: последовательная корреляция A-B, B-C, ... с выходом, "
            "совместимым с minimal_dtw"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("files", nargs="*", help="CSV/TSV скважин в порядке корреляции")
    parser.add_argument("--curves", nargs="+", help="Общие числовые кривые, например GR RHOB NPHI")
    parser.add_argument(
        "--curve",
        action="append",
        help="Одна общая кривая; параметр можно повторить. По умолчанию GR",
    )
    parser.add_argument(
        "--md", "--md-column", default="MD", dest="md_column", help="Столбец измеренной глубины"
    )
    parser.add_argument("-o", "--output", default="correlation.csv", help="Выходной CSV")
    parser.add_argument(
        "--anchors",
        action="append",
        help="CSV marker,left_md,right_md; повторить для каждой соседней пары",
    )
    parser.add_argument(
        "--well-name",
        action="append",
        help="Явное имя; повторить по одному разу для каждого входного файла",
    )
    parser.add_argument(
        "--step", "--sampling-step", type=float, help="Единый шаг регулярной сетки всех скважин"
    )
    parser.add_argument("--max-interpolation-gap", type=float, help="Максимальный зазор для интерполяции")
    parser.add_argument("--scales", type=float, nargs="+", help="Физические масштабы событий в единицах MD")
    parser.add_argument("--min-event-distance", type=float, help="Минимальное расстояние между событиями")
    parser.add_argument("--max-events", type=int, default=120)
    parser.add_argument("--event-percentile", type=float, default=65.0)
    parser.add_argument("--alpha", type=float, default=0.55, help="Вес геометрии FUGW [0,1]")
    parser.add_argument("--epsilon", type=float, default=0.06, help="Энтропийная регуляризация")
    parser.add_argument("--mass-penalty", type=float, default=0.8, help="KL-штраф unbalanced transport")
    parser.add_argument("--order-strength", type=float, default=0.6, help="Вес стратиграфического порядка")
    parser.add_argument("--order-sigma", type=float, default=0.22, help="Допуск порядка в долях интервала")
    parser.add_argument("--fugw-iterations", type=int, default=24)
    parser.add_argument("--sinkhorn-iterations", type=int, default=250)
    parser.add_argument("--damping", type=float, default=0.65)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--anchor-weight", type=float, default=8.0)
    parser.add_argument("--anchor-penalty", type=float, default=12.0)
    parser.add_argument("--spline-knots", type=int, default=18)
    parser.add_argument("--spline-smoothness", type=float, default=1.0)
    parser.add_argument("--min-stretch", type=float, default=0.05)
    parser.add_argument("--max-stretch", type=float, default=20.0)
    parser.add_argument("--min-row-transport", type=float, default=0.02)
    parser.add_argument("--bootstrap", type=int, default=0, help="Число эмпирических реализаций")
    parser.add_argument("--bootstrap-jitter", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Разрешить замену выходного CSV")
    parser.add_argument("--quiet", action="store_true", help="Не печатать прогресс по парам")
    parser.add_argument("--self-test", action="store_true", help="Запустить встроенный синтетический тест")
    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    selected_curves = args.curves if args.curves else args.curve if args.curve else ["GR"]
    return Config(
        curves=tuple(selected_curves),
        md_column=args.md_column,
        step=args.step,
        max_interpolation_gap=args.max_interpolation_gap,
        scales=tuple(args.scales) if args.scales else None,
        min_event_distance=args.min_event_distance,
        max_events=args.max_events,
        event_percentile=args.event_percentile,
        alpha=args.alpha,
        epsilon=args.epsilon,
        mass_penalty=args.mass_penalty,
        order_strength=args.order_strength,
        order_sigma=args.order_sigma,
        fugw_iterations=args.fugw_iterations,
        sinkhorn_iterations=args.sinkhorn_iterations,
        damping=args.damping,
        tolerance=args.tolerance,
        anchor_weight=args.anchor_weight,
        anchor_penalty=args.anchor_penalty,
        spline_knots=args.spline_knots,
        spline_smoothness=args.spline_smoothness,
        min_stretch=args.min_stretch,
        max_stretch=args.max_stretch,
        min_row_transport=args.min_row_transport,
        bootstrap=args.bootstrap,
        bootstrap_jitter=args.bootstrap_jitter,
        seed=args.seed,
    )


def _synthetic_signal(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    peak_a = np.exp(-0.5 * ((depth - 31.0) / 2.4) ** 2)
    peak_b = np.exp(-0.5 * ((depth - 78.0) / 4.2) ** 2)
    peak_c = np.exp(-0.5 * ((depth - 103.0) / 1.8) ** 2)
    gr = 72.0 + 17.0 * np.sin(depth / 7.2) + 9.0 * np.sin(depth / 2.7) + 33.0 * peak_a - 24.0 * peak_b + 18.0 * peak_c
    rhob = 2.42 - 0.12 * np.sin(depth / 7.2 + 0.35) - 0.09 * peak_a + 0.11 * peak_b
    nphi = 0.22 + 0.065 * np.sin(depth / 7.2 - 0.45) + 0.055 * peak_a - 0.045 * peak_c
    return gr, rhob, nphi


def _run_self_test() -> None:
    rng = np.random.default_rng(1701)
    left_md = np.arange(0.0, 120.0 + 0.5, 0.5)
    right_md = np.arange(8.0, 170.0 + 0.5, 0.5)
    left_curves = _synthetic_signal(left_md)
    right_stratigraphic = (right_md - 8.0) / 1.35
    right_curves = _synthetic_signal(right_stratigraphic)
    left_frame = pd.DataFrame(
        {
            "MD": left_md,
            "GR": left_curves[0] + rng.normal(0.0, 0.7, len(left_md)),
            "RHOB": left_curves[1] + rng.normal(0.0, 0.004, len(left_md)),
            "NPHI": left_curves[2] + rng.normal(0.0, 0.003, len(left_md)),
        }
    )
    right_frame = pd.DataFrame(
        {
            "MD": right_md,
            "GR": right_curves[0] + rng.normal(0.0, 0.8, len(right_md)),
            "RHOB": right_curves[1] + rng.normal(0.0, 0.005, len(right_md)),
            "NPHI": right_curves[2] + rng.normal(0.0, 0.004, len(right_md)),
        }
    )
    third_md = np.arange(-5.0, 103.0 + 0.5, 0.5)
    third_stratigraphic = (third_md + 5.0) / 0.9
    third_curves = _synthetic_signal(third_stratigraphic)
    third_frame = pd.DataFrame(
        {
            "MD": third_md,
            "GR": third_curves[0] + rng.normal(0.0, 0.75, len(third_md)),
            "RHOB": third_curves[1] + rng.normal(0.0, 0.004, len(third_md)),
            "NPHI": third_curves[2] + rng.normal(0.0, 0.003, len(third_md)),
        }
    )
    with tempfile.TemporaryDirectory(prefix="geofugw_selftest_") as directory:
        left_path = Path(directory) / "left.csv"
        right_path = Path(directory) / "right.csv"
        third_path = Path(directory) / "third.csv"
        left_frame.to_csv(left_path, index=False)
        right_frame.to_csv(right_path, index=False)
        third_frame.to_csv(third_path, index=False)
        config = Config(
            curves=("GR", "RHOB", "NPHI"),
            step=0.5,
            max_interpolation_gap=4.0,
            scales=(1.0, 3.0, 8.0),
            min_event_distance=1.5,
            max_events=80,
            event_percentile=55.0,
            alpha=0.5,
            epsilon=0.08,
            order_strength=0.75,
            order_sigma=0.18,
            fugw_iterations=16,
            sinkhorn_iterations=180,
            spline_knots=16,
            spline_smoothness=0.25,
            min_stretch=0.2,
            max_stretch=3.0,
            bootstrap=2,
        )
        result = correlate(
            [left_path, right_path, third_path],
            config,
            well_names=["A", "B", "C"],
        )
        output_path = _atomic_write_csv(result, Path(directory) / "mapping.csv")
        round_trip = pd.read_csv(output_path)
        try:
            from plot_correlation import load_report_data
        except ImportError:
            compatible_report = None
        else:
            compatible_report = load_report_data(output_path)
    first_pair = result.loc[result["pair"] == 0].reset_index(drop=True)
    estimated = first_pair["right_md"].to_numpy(float)
    expected = 8.0 + 1.35 * first_pair["left_md"].to_numpy(float)
    error = np.abs(estimated - expected)
    median_error = float(np.nanmedian(error))
    p95_error = float(np.nanpercentile(error, 95))
    monotone = bool(np.all(np.diff(estimated[np.isfinite(estimated)]) > 0))
    legacy_columns = [
        "pair",
        "left_well",
        "right_well",
        "left_md",
        "right_md",
        "left_z",
        "right_z",
        "local_cost",
        "mean_cost",
    ]
    required_columns = {*legacy_columns, "confidence", "status", "qc_method"}
    if (
        not monotone
        or median_error > 5.0
        or p95_error > 12.0
        or not required_columns.issubset(round_trip.columns)
        or list(round_trip.columns[: len(legacy_columns)]) != legacy_columns
        or round_trip["pair"].drop_duplicates().tolist() != [0, 1]
        or result[["left_well", "right_well"]].drop_duplicates().to_records(index=False).tolist()
        != [("A", "B"), ("B", "C")]
        or not np.allclose(result["local_cost"], np.abs(result["left_z"] - result["right_z"]))
        or not np.all(result["qc_bootstrap_success"] == 2)
        or (compatible_report is not None and len(compatible_report["pairs"]) != 2)
    ):
        raise AssertionError(
            f"self-test failed: monotone={monotone}, median_error={median_error:.3f}, p95={p95_error:.3f}"
        )
    print(
        "SELF-TEST OK: "
        f"median mapping error={median_error:.3f}, p95={p95_error:.3f}, "
        f"pairs={result['pair'].nunique()}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        _run_self_test()
        return 0
    if len(args.files) < 2:
        parser.error("укажите минимум два файла скважин")
    output = Path(args.output)
    resolved_output = output.resolve()
    if any(Path(path).resolve() == resolved_output for path in args.files):
        parser.error("выходной CSV не может совпадать с входным файлом")
    if output.exists() and not args.overwrite:
        parser.error(f"выходной файл уже существует: {output}; добавьте --overwrite")
    config = _config_from_args(args)
    try:
        frame = correlate(
            args.files,
            config,
            well_names=args.well_name,
            anchors=args.anchors,
            progress=None if args.quiet else print,
        )
        destination = _atomic_write_csv(frame, output)
    except (OSError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        print(f"GeoFUGW-Spline: ошибка: {error}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(f"Результат: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
