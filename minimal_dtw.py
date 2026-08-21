"""Минимальная корреляция скважин методом DTW.

Каждый CSV содержит MD и одну кривую. Порядок файлов задаёт пары A-B, B-C, ...

    python minimal_dtw.py A.csv B.csv C.csv --curve GR -o correlation.csv

Используется устойчивый точный DTW внутри симметричного коридора вокруг
масштабированной диагонали. Стоимости хранятся только для двух строк, а
указатели пути — плотно, по одному байту на разрешённую клетку.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd


START, DIAGONAL, UP, LEFT = 0, 1, 2, 3
STANDARD_MISSING_VALUES = (-999.25, -999.0, 999.25)
DEFAULT_MAX_RESAMPLED_POINTS = 200_000
DEFAULT_MAX_DTW_CELLS = 50_000_000
DEFAULT_MAX_MEMORY_MB = 512.0

FilePath = str | PathLike[str]


@dataclass(slots=True)
class DtwResult:
    """Полный результат DTW и основные показатели деформации."""

    path: np.ndarray
    local_cost: np.ndarray
    total_cost: float
    normalized_cost: float
    mean_local_cost: float
    warp_penalty: float
    diagonal_steps: int
    up_steps: int
    left_steps: int
    max_warp_run: int
    boundary_fraction: float

    @property
    def non_diagonal_steps(self) -> int:
        return self.up_steps + self.left_steps


def _read_csv(file: FilePath, required_columns: Sequence[str]) -> pd.DataFrame:
    """Прочитать CSV с распространёнными кодировками и автоопределением разделителя."""
    path = Path(file)
    errors: list[str] = []
    seen_columns: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1251", "latin1"):
        try:
            frame = pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding=encoding,
                dtype=str,
            )
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            errors.append(f"{encoding}: {exc}")
            continue
        frame.columns = [str(column).strip().replace("\ufeff", "") for column in frame]
        seen_columns = list(frame.columns)
        if all(column in frame.columns for column in required_columns):
            return frame
        errors.append(f"{encoding}: столбцы {seen_columns}")
    required = " и ".join(str(column) for column in required_columns)
    details = "; ".join(errors[-2:])
    raise ValueError(f"{path}: нужны столбцы {required}; {details}")


def _numeric(series: pd.Series) -> pd.Series:
    """Преобразовать числа, включая десятичную запятую и пробелы-разделители."""
    text = series.astype("string").str.strip()
    text = text.str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
    text = text.str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce")


def _load_log_data(
    file: FilePath,
    curve: str,
    md_column: str,
    missing_values: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    curve, md_column = str(curve).strip(), str(md_column).strip()
    if not curve or not md_column or curve == md_column:
        raise ValueError("Имена столбцов MD и кривой должны быть непустыми и различаться")

    frame = _read_csv(file, (md_column, curve))
    md = _numeric(frame[md_column]).to_numpy(float)
    values = _numeric(frame[curve]).to_numpy(float)
    if np.any(np.isinf(md)) or np.any(np.isinf(values)):
        raise ValueError(f"{file}: найдены бесконечные значения")

    sentinels = np.asarray(tuple(float(value) for value in missing_values), dtype=float)
    if sentinels.size and not np.all(np.isfinite(sentinels)):
        raise ValueError("Значения пропусков должны быть конечными числами")
    if sentinels.size:
        values[np.isin(values, sentinels)] = np.nan

    data = pd.DataFrame({md_column: md, curve: values}).dropna(subset=[md_column])
    data = (
        data.groupby(md_column, as_index=False, sort=True)[curve]
        .mean()
        .dropna(subset=[curve])
    )
    if len(data) < 2:
        raise ValueError(f"{file}: недостаточно конечных данных кривой {curve}")

    md = data[md_column].to_numpy(float)
    values = data[curve].to_numpy(float)
    if not np.all(np.isfinite(md)) or not np.all(np.isfinite(values)):
        raise ValueError(f"{file}: после очистки остались неконечные значения")
    if np.any(np.diff(md) <= 0):
        raise ValueError(f"{file}: MD должен строго возрастать после очистки")
    return md, values


def _regular_grid(
    start: float,
    end: float,
    step: float,
    max_points: int,
    file: FilePath,
) -> np.ndarray:
    if not math.isfinite(step) or step <= 0:
        raise ValueError("Шаг MD должен быть конечным положительным числом")
    try:
        integer_max_points = int(max_points)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_resampled_points должен быть целым числом") from exc
    if integer_max_points != max_points or integer_max_points < 2:
        raise ValueError("max_resampled_points должен быть не меньше двух")
    max_points = integer_max_points

    span = end - start
    ratio = span / step
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"{file}: шаг MD некорректен для выбранного интервала")
    if ratio > max_points:
        approximate_count = int(math.ceil(ratio)) + 1
        raise MemoryError(
            f"{file}: регулярная сетка содержит около {approximate_count:,} точек, "
            f"выше max_resampled_points={max_points:,}; увеличьте --step"
        )

    intervals = int(math.floor(ratio))
    magnitude = max(abs(start), abs(end), abs(step), 1.0)
    ulp = abs(float(np.spacing(magnitude)))
    if step <= ulp * 8:
        raise ValueError(
            f"{file}: шаг MD слишком мал относительно абсолютных значений глубины"
        )
    tolerance = max(abs(step) * 1e-10, ulp * 4)
    candidate_end = start + intervals * step
    if candidate_end > end + tolerance:
        intervals -= 1
        candidate_end = start + intervals * step
    append_end = not math.isclose(candidate_end, end, rel_tol=0.0, abs_tol=tolerance)
    point_count = intervals + 1 + int(append_end)
    if point_count > max_points:
        raise MemoryError(
            f"{file}: регулярная сетка содержит около {point_count:,} точек, "
            f"выше max_resampled_points={max_points:,}; увеличьте --step"
        )

    grid = start + np.arange(intervals + 1, dtype=float) * step
    if append_end:
        grid = np.append(grid, end)
    else:
        grid[-1] = end
    if len(grid) < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError(
            f"{file}: шаг MD слишком мал относительно абсолютных значений глубины"
        )
    return grid


def _normalize(
    values: np.ndarray,
    file: FilePath,
    curve: str,
    clip_percentiles: tuple[float, float] | None,
) -> np.ndarray:
    minimum, maximum = float(np.min(values)), float(np.max(values))
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum == maximum:
        raise ValueError(f"{file}: кривая {curve} постоянна")
    # Деление обеих границ до сложения не даёт переполниться midpoint,
    # а вычитание midpoint сохраняет малые вариации на большом фоне.
    midpoint = minimum / 2.0 + maximum / 2.0
    shifted = values - midpoint
    magnitude = float(np.max(np.abs(shifted)))
    if not math.isfinite(magnitude) or magnitude == 0:
        raise ValueError(f"{file}: кривая {curve} постоянна")
    scaled = shifted / magnitude

    if clip_percentiles is not None:
        if len(clip_percentiles) != 2:
            raise ValueError("clip_percentiles должен содержать две границы")
        low_percentile, high_percentile = map(float, clip_percentiles)
        if not (
            math.isfinite(low_percentile)
            and math.isfinite(high_percentile)
            and 0 <= low_percentile < high_percentile <= 100
        ):
            raise ValueError("Процентили clipping должны удовлетворять 0 <= low < high <= 100")
        low, high = np.percentile(scaled, [low_percentile, high_percentile])
        scaled = np.clip(scaled, low, high)

    center = math.fsum(float(value) for value in scaled) / len(scaled)
    centered = scaled - center
    variance = math.fsum(float(value) * float(value) for value in centered) / len(centered)
    scale = math.sqrt(variance)
    if not math.isfinite(scale) or scale == 0:
        raise ValueError(f"{file}: кривая {curve} постоянна после подготовки")
    normalized = centered / scale
    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"{file}: нормализация кривой {curve} дала неконечные значения")
    return normalized


def _prepare_log(
    md: np.ndarray,
    values: np.ndarray,
    *,
    file: FilePath,
    curve: str,
    step: float,
    max_interpolation_gap: float | None,
    interpolate_all_gaps: bool,
    clip_percentiles: tuple[float, float] | None,
    max_resampled_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if interpolate_all_gaps and max_interpolation_gap is not None:
        raise ValueError(
            "Нельзя одновременно задавать max_interpolation_gap и interpolate_all_gaps"
        )
    grid = _regular_grid(md[0], md[-1], step, max_resampled_points, file)

    if not interpolate_all_gaps:
        gap_limit = step * 10 if max_interpolation_gap is None else float(max_interpolation_gap)
        if not math.isfinite(gap_limit) or gap_limit <= 0:
            raise ValueError("max_interpolation_gap должен быть конечным положительным числом")
        gaps = np.diff(md)
        tolerance = max(
            abs(step) * 1e-9,
            abs(float(np.spacing(max(abs(md[0]), abs(md[-1]), 1.0)))) * 8,
        )
        bad = np.flatnonzero(gaps > gap_limit + tolerance)
        if len(bad):
            index = int(bad[np.argmax(gaps[bad])])
            raise ValueError(
                f"{file}: разрыв MD {gaps[index]:.12g} между {md[index]:.12g} и "
                f"{md[index + 1]:.12g} превышает max_interpolation_gap={gap_limit:.12g}; "
                "увеличьте предел или используйте --interpolate-all-gaps"
            )

    interpolated = np.interp(grid, md, values)
    return grid, _normalize(interpolated, file, curve, clip_percentiles)


def read_log(
    file: FilePath,
    curve: str = "GR",
    md_column: str = "MD",
    step: float | None = None,
    *,
    missing_values: Sequence[float] = STANDARD_MISSING_VALUES,
    max_interpolation_gap: float | None = None,
    interpolate_all_gaps: bool = False,
    clip_percentiles: tuple[float, float] | None = None,
    max_resampled_points: int = DEFAULT_MAX_RESAMPLED_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Прочитать, очистить, интерполировать и нормализовать одну кривую ГИС."""
    md, values = _load_log_data(file, curve, md_column, missing_values)
    resolved_step = float(np.median(np.diff(md)) if step is None else step)
    return _prepare_log(
        md,
        values,
        file=file,
        curve=curve,
        step=resolved_step,
        max_interpolation_gap=max_interpolation_gap,
        interpolate_all_gaps=interpolate_all_gaps,
        clip_percentiles=clip_percentiles,
        max_resampled_points=max_resampled_points,
    )


def _corridor(n: int, m: int, window: float) -> tuple[np.ndarray, np.ndarray]:
    """Симметричный связный коридор в относительных координатах."""
    n, m = int(n), int(m)
    window = float(window)
    if n < 2 or m < 2:
        raise ValueError("Для коридора нужны длины не меньше двух")
    if not math.isfinite(window) or not 0 <= window <= 1:
        raise ValueError("window должен быть конечным числом от 0 до 1")

    left_scale, right_scale = n - 1, m - 1
    # Половина шага более грубой относительной сетки — минимальный симметричный
    # дискретный запас, при котором коридор остаётся связным. Для равных длин
    # window=0 разрешает только главную диагональ.
    band = int(
        math.floor(
            window * (left_scale * right_scale)
            + 0.5 * max(left_scale, right_scale)
        )
    )
    # Арифметика in-place удерживает пиковую память построения на двух массивах
    # int64 вместо цепочки временных массивов для каждого выражения.
    ends = np.arange(n, dtype=np.int64)
    ends *= right_scale
    starts = ends.copy()
    starts -= band
    starts += left_scale - 1
    starts //= left_scale
    np.maximum(starts, 0, out=starts)
    ends += band
    ends //= left_scale
    np.minimum(ends, m - 1, out=ends)
    return starts, ends


def _maximum_warp_run(differences: np.ndarray) -> int:
    maximum = current = 0
    previous: tuple[int, int] | None = None
    for difference in differences:
        kind = (int(difference[0]), int(difference[1]))
        if kind in {(1, 0), (0, 1)}:
            current = current + 1 if kind == previous else 1
            maximum = max(maximum, current)
        else:
            current = 0
        previous = kind
    return maximum


def _use_transposed_orientation(left: np.ndarray, right: np.ndarray) -> bool:
    """Выбрать одинаковую внутреннюю ориентацию для (a, b) и (b, a)."""
    if len(left) != len(right):
        # Более длинный ряд по строкам уменьшает максимальную рабочую ширину.
        return len(left) < len(right)
    # Ищем первое различие блоками, не создавая временную маску длиной со вход.
    for start in range(0, len(left), 1_000_000):
        stop = min(start + 1_000_000, len(left))
        unequal = left[start:stop] != right[start:stop]
        if np.any(unequal):
            first = start + int(np.argmax(unequal))
            return bool(left[first] < right[first])
    return False


def _validate_dtw_inputs(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    warp_penalty: float,
    max_warp_run: int | None,
    max_dtw_cells: int,
    max_memory_mb: float,
) -> tuple[np.ndarray, np.ndarray, float, int | None, int, float]:
    try:
        left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("DTW нужны два числовых одномерных ряда") from exc
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("DTW нужны два одномерных ряда")
    if len(left) < 2 or len(right) < 2:
        raise ValueError("DTW нужны два ряда длиной не менее двух")
    for values in (left, right):
        for start in range(0, len(values), 1_000_000):
            if not np.all(np.isfinite(values[start : start + 1_000_000])):
                raise ValueError("DTW нужны два конечных ряда")

    try:
        penalty = float(warp_penalty)
        memory_mb = float(max_memory_mb)
        cell_limit = int(max_dtw_cells)
        run_limit = None if max_warp_run is None else int(max_warp_run)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Параметры DTW имеют неверный числовой формат") from exc
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("warp_penalty должен быть конечным неотрицательным числом")
    if run_limit is not None and (run_limit != max_warp_run or run_limit < 1):
        raise ValueError("max_warp_run должен быть положительным целым числом или None")
    if cell_limit != max_dtw_cells or cell_limit < 1:
        raise ValueError("max_dtw_cells должен быть положительным целым числом")
    if not math.isfinite(memory_mb) or memory_mb <= 0:
        raise ValueError("max_memory_mb должен быть конечным положительным числом")
    if len(left) + len(right) - 1 > np.iinfo(np.int32).max:
        raise MemoryError("DTW-путь слишком длинный для компактного массива индексов")
    return left, right, penalty, run_limit, cell_limit, memory_mb


def dtw_details(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    window: float = 0.10,
    *,
    warp_penalty: float = 0.0,
    max_warp_run: int | None = None,
    max_dtw_cells: int = DEFAULT_MAX_DTW_CELLS,
    max_memory_mb: float = DEFAULT_MAX_MEMORY_MB,
) -> DtwResult:
    """Вернуть устойчивый точный DTW-путь и расширенные QC-метрики."""
    (
        input_left,
        input_right,
        penalty,
        max_warp_run,
        max_dtw_cells,
        max_memory_mb,
    ) = _validate_dtw_inputs(a, b, warp_penalty, max_warp_run, max_dtw_cells, max_memory_mb)
    memory_limit = int(max_memory_mb * 1024 * 1024)
    # В каждой строке выбранной (более длинной) ориентации коридора есть хотя
    # бы одна клетка. Проверяем этот нижний предел и неизбежные O(n + m)
    # массивы ещё до сравнения рядов и построения коридора.
    maximum_length = max(len(input_left), len(input_right))
    minimum_cell_count = maximum_length
    if minimum_cell_count > max_dtw_cells:
        raise MemoryError(
            f"DTW-коридор содержит не менее {minimum_cell_count:,} клеток, выше "
            f"max_dtw_cells={int(max_dtw_cells):,}; увеличьте step"
        )
    base_estimated_bytes = maximum_length * 32 + (
        len(input_left) + len(input_right)
    ) * 48
    if base_estimated_bytes > memory_limit:
        raise MemoryError(
            f"DTW требует не менее {base_estimated_bytes / 1024 / 1024:.1f} МБ, "
            f"выше max_memory_mb={max_memory_mb:g}"
        )

    transposed = _use_transposed_orientation(input_left, input_right)
    left, right = (
        (input_right, input_left) if transposed else (input_left, input_right)
    )
    starts, ends = _corridor(len(left), len(right), window)
    sizes = ends - starts + 1
    offsets = np.empty(len(left) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(sizes, dtype=np.int64, out=offsets[1:])
    cell_count = int(offsets[-1])
    if cell_count > max_dtw_cells:
        raise MemoryError(
            f"DTW-коридор содержит {cell_count:,} клеток, выше "
            f"max_dtw_cells={int(max_dtw_cells):,}; уменьшите window или увеличьте step"
        )

    width = int(np.max(sizes))
    # Помимо backpointer-ов учитываем обе строки cost/length, границы строк,
    # максимальный путь, local_cost, differences и временные булевы маски QC.
    estimated_bytes = base_estimated_bytes + cell_count + width * 56
    if estimated_bytes > memory_limit:
        raise MemoryError(
            f"DTW требует ориентировочно {estimated_bytes / 1024 / 1024:.1f} МБ, "
            f"выше max_memory_mb={max_memory_mb:g}"
        )

    back = np.full(cell_count, START, dtype=np.uint8)
    previous = np.empty(0, dtype=float)
    previous_length = np.empty(0, dtype=np.int64)
    previous_start = 0
    max_length = np.iinfo(np.int64).max

    for i in range(len(left)):
        start, end = int(starts[i]), int(ends[i])
        size = end - start + 1
        with np.errstate(over="ignore", invalid="ignore"):
            local = np.abs(left[i] - right[start : end + 1])
        if not np.all(np.isfinite(local)):
            raise FloatingPointError(
                "Локальная стоимость DTW вышла за численный диапазон float64"
            )
        current = np.full(size, np.inf, dtype=float)
        current_length = np.full(size, max_length, dtype=np.int64)
        row_offset = int(offsets[i])

        for column, j in enumerate(range(start, end + 1)):
            if i == 0 and j == 0:
                current[column] = float(local[column])
                current_length[column] = 1
                continue

            best_cost = math.inf
            best_length = max_length
            move = START

            diagonal_index = j - 1 - previous_start
            if i > 0 and 0 <= diagonal_index < len(previous):
                best_cost = float(previous[diagonal_index])
                best_length = int(previous_length[diagonal_index])
                move = DIAGONAL

            up_index = j - previous_start
            if i > 0 and 0 <= up_index < len(previous):
                previous_cost = float(previous[up_index])
                candidate = previous_cost + penalty
                if math.isfinite(previous_cost) and not math.isfinite(candidate):
                    raise FloatingPointError("Накопленная стоимость DTW переполнила float64")
                candidate_length = int(previous_length[up_index])
                if candidate < best_cost or (
                    candidate == best_cost and candidate_length < best_length
                ):
                    best_cost = candidate
                    best_length = candidate_length
                    move = UP

            if column > 0:
                previous_cost = float(current[column - 1])
                candidate = previous_cost + penalty
                if math.isfinite(previous_cost) and not math.isfinite(candidate):
                    raise FloatingPointError("Накопленная стоимость DTW переполнила float64")
                candidate_length = int(current_length[column - 1])
                if candidate < best_cost or (
                    candidate == best_cost and candidate_length < best_length
                ):
                    best_cost = candidate
                    best_length = candidate_length
                    move = LEFT

            if move == START or not math.isfinite(best_cost):
                continue
            value = best_cost + float(local[column])
            if not math.isfinite(value):
                raise FloatingPointError("Накопленная стоимость DTW переполнила float64")
            current[column] = value
            current_length[column] = best_length + 1
            back[row_offset + column] = move

        previous = current
        previous_length = current_length
        previous_start = start

    total_index = len(right) - 1 - previous_start
    if not 0 <= total_index < len(previous) or not math.isfinite(previous[total_index]):
        raise ValueError("DTW-путь не найден: увеличьте window")

    path = np.empty((len(left) + len(right) - 1, 2), dtype=np.int32)
    k, i, j = len(path), len(left) - 1, len(right) - 1
    while True:
        k -= 1
        path[k] = i, j
        move = int(back[int(offsets[i]) + j - int(starts[i])])
        if move == START:
            break
        if move == DIAGONAL:
            i, j = i - 1, j - 1
        elif move == UP:
            i -= 1
        elif move == LEFT:
            j -= 1
        else:  # pragma: no cover
            raise RuntimeError("Неизвестный указатель DTW-пути")
        if i < 0 or j < 0 or not (starts[i] <= j <= ends[i]):
            raise RuntimeError("Повреждён DTW-путь")

    path = path[k:]
    if tuple(path[0]) != (0, 0) or tuple(path[-1]) != (len(left) - 1, len(right) - 1):
        raise RuntimeError("Повреждён DTW-путь")
    if transposed:
        path = path[:, ::-1].copy()

    local_cost = np.abs(input_left[path[:, 0]] - input_right[path[:, 1]])
    differences = np.diff(path, axis=0)
    diagonal_steps = int(np.sum(np.all(differences == (1, 1), axis=1)))
    up_steps = int(np.sum(np.all(differences == (1, 0), axis=1)))
    left_steps = int(np.sum(np.all(differences == (0, 1), axis=1)))
    maximum_run = _maximum_warp_run(differences)
    if max_warp_run is not None and maximum_run > max_warp_run:
        raise ValueError(
            f"DTW-путь требует {maximum_run} последовательных горизонтальных/вертикальных "
            f"шагов, выше max_warp_run={max_warp_run}"
        )

    local_total = math.fsum(float(value) for value in local_cost)
    total_cost = local_total + penalty * (up_steps + left_steps)
    mean_local_cost = local_total / len(path)
    result_starts, result_ends = _corridor(len(input_left), len(input_right), window)
    boundary_hits = (
        (path[:, 1] == result_starts[path[:, 0]])
        | (path[:, 1] == result_ends[path[:, 0]])
    )
    # Начало и конец обязаны лежать на границе по определению и не являются
    # признаком слишком узкого коридора.
    boundary_hits[[0, -1]] = False
    boundary_fraction = float(np.mean(boundary_hits))

    return DtwResult(
        path=path,
        local_cost=local_cost,
        total_cost=float(total_cost),
        normalized_cost=float(total_cost / len(path)),
        mean_local_cost=float(mean_local_cost),
        warp_penalty=penalty,
        diagonal_steps=diagonal_steps,
        up_steps=up_steps,
        left_steps=left_steps,
        max_warp_run=maximum_run,
        boundary_fraction=boundary_fraction,
    )


def dtw(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    window: float = 0.10,
    *,
    warp_penalty: float = 0.0,
    max_warp_run: int | None = None,
    max_dtw_cells: int = DEFAULT_MAX_DTW_CELLS,
    max_memory_mb: float = DEFAULT_MAX_MEMORY_MB,
) -> tuple[np.ndarray, float]:
    """Вернуть DTW-путь и среднюю стоимость оптимизируемой цели."""
    result = dtw_details(
        a,
        b,
        window,
        warp_penalty=warp_penalty,
        max_warp_run=max_warp_run,
        max_dtw_cells=max_dtw_cells,
        max_memory_mb=max_memory_mb,
    )
    return result.path, result.normalized_cost


def _well_name(file: FilePath) -> str:
    path = Path(file)
    if path.stem.casefold() == "well_gis":
        return path.resolve().parent.name.strip() or path.stem
    return path.stem.strip()


def _resolve_well_names(
    files: Sequence[Path],
    well_names: Sequence[str] | None,
) -> list[str]:
    if well_names is not None:
        if isinstance(well_names, (str, bytes)) or len(well_names) != len(files):
            raise ValueError("well_names должен содержать ровно одно имя на каждый CSV")
        names = [str(name).strip() for name in well_names]
    else:
        names = [_well_name(file) for file in files]
        counts = Counter(name.casefold() for name in names)
        duplicate_indices = [
            index for index, name in enumerate(names) if counts[name.casefold()] > 1
        ]
        for index in duplicate_indices:
            parent_name = files[index].resolve().parent.name.strip()
            if parent_name:
                names[index] = parent_name

    if any(not name for name in names):
        raise ValueError("Имя скважины не может быть пустым")
    normalized = [name.casefold() for name in names]
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "Не удалось однозначно определить имена скважин; задайте --well-name для каждого CSV"
        )
    return names


def correlate(
    files: Sequence[FilePath],
    curve: str = "GR",
    md_column: str = "MD",
    step: float | None = None,
    window: float = 0.10,
    *,
    well_names: Sequence[str] | None = None,
    missing_values: Sequence[float] = STANDARD_MISSING_VALUES,
    max_interpolation_gap: float | None = None,
    interpolate_all_gaps: bool = False,
    clip_percentiles: tuple[float, float] | None = None,
    warp_penalty: float = 0.05,
    max_warp_run: int | None = 25,
    max_dtw_cells: int = DEFAULT_MAX_DTW_CELLS,
    max_resampled_points: int = DEFAULT_MAX_RESAMPLED_POINTS,
    max_memory_mb: float = DEFAULT_MAX_MEMORY_MB,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Последовательно скоррелировать A-B, B-C и вернуть один DataFrame.

    ``max_memory_mb`` ограничивает оценённую рабочую память DTW одной пары;
    входные журналы и итоговый DataFrame в эту оценку не входят.
    """
    if isinstance(files, (str, bytes, PathLike)):
        raise ValueError("files должен быть последовательностью CSV-файлов, а не одной строкой")
    paths = [Path(file) for file in files]
    if len(paths) < 2:
        raise ValueError("Нужно минимум два CSV-файла")
    names = _resolve_well_names(paths, well_names)

    if step is None:
        raw_logs = [_load_log_data(file, curve, md_column, missing_values) for file in paths]
        # Каждая скважина имеет одинаковый вес независимо от длины журнала.
        well_steps = [float(np.median(np.diff(md))) for md, _values in raw_logs]
        well_steps = [value for value in well_steps if math.isfinite(value) and value > 0]
        if not well_steps:
            raise ValueError("Невозможно определить общий шаг MD")
        resolved_step = float(np.median(well_steps))
    else:
        raw_logs = None
        resolved_step = float(step)

    logs: list[tuple[np.ndarray, np.ndarray]] = []
    for index, file in enumerate(paths):
        if raw_logs is None:
            md, values = _load_log_data(file, curve, md_column, missing_values)
        else:
            md, values = raw_logs[index]
        logs.append(
            _prepare_log(
                md,
                values,
                file=file,
                curve=curve,
                step=resolved_step,
                max_interpolation_gap=max_interpolation_gap,
                interpolate_all_gaps=interpolate_all_gaps,
                clip_percentiles=clip_percentiles,
                max_resampled_points=max_resampled_points,
            )
        )
        if raw_logs is not None:
            # Освободить исходный журнал сразу после ресэмплинга, а не держать
            # одновременно полные raw_logs и logs до конца подготовки.
            raw_logs[index] = (np.empty(0), np.empty(0))

    tables: list[pd.DataFrame] = []
    for pair, ((left_md, left), (right_md, right)) in enumerate(zip(logs, logs[1:])):
        result = dtw_details(
            left,
            right,
            window,
            warp_penalty=warp_penalty,
            max_warp_run=max_warp_run,
            max_dtw_cells=max_dtw_cells,
            max_memory_mb=max_memory_mb,
        )
        path = result.path
        transitions = np.r_[False, np.any(np.diff(path, axis=0) == 0, axis=1)]
        cumulative_cost = np.cumsum(
            result.local_cost + transitions.astype(float) * result.warp_penalty
        )
        pair_starts, pair_ends = _corridor(len(left), len(right), window)
        boundary_hit = (
            (path[:, 1] == pair_starts[path[:, 0]])
            | (path[:, 1] == pair_ends[path[:, 0]])
        )
        boundary_hit[[0, -1]] = False
        moves = np.full(len(path), "start", dtype=object)
        if len(path) > 1:
            move_lookup = {(1, 1): "diagonal", (1, 0): "up", (0, 1): "left"}
            moves[1:] = [move_lookup[tuple(map(int, value))] for value in np.diff(path, axis=0)]

        left_name, right_name = names[pair], names[pair + 1]
        tables.append(
            pd.DataFrame(
                {
                    "pair": pair,
                    "left_well": left_name,
                    "right_well": right_name,
                    "left_md": left_md[path[:, 0]],
                    "right_md": right_md[path[:, 1]],
                    "left_z": left[path[:, 0]],
                    "right_z": right[path[:, 1]],
                    "local_cost": result.local_cost,
                    "mean_cost": result.mean_local_cost,
                    "path_index": np.arange(len(path), dtype=int),
                    "move": moves,
                    "cumulative_cost": cumulative_cost,
                    "total_cost": result.total_cost,
                    "normalized_cost": result.normalized_cost,
                    "warp_penalty": result.warp_penalty,
                    "diagonal_steps": result.diagonal_steps,
                    "up_steps": result.up_steps,
                    "left_steps": result.left_steps,
                    "max_warp_run": result.max_warp_run,
                    "boundary_hit": boundary_hit,
                    "boundary_fraction": result.boundary_fraction,
                }
            )
        )
        if progress is not None:
            progress(
                f"{left_name} -> {right_name}: {len(path)} точек, "
                f"mean|Δz|={result.mean_local_cost:.5g}, DTW={result.normalized_cost:.5g}"
            )
    return pd.concat(tables, ignore_index=True)


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("нужно конечное положительное число")
    return number


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("нужно конечное неотрицательное число")
    return number


def _window_ratio(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("нужно число от 0 до 1")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("нужно положительное целое число")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="CSV-файлы в порядке корреляции")
    parser.add_argument("--curve", default="GR")
    parser.add_argument(
        "--md", "--md-column", dest="md", default="MD", help="Имя столбца глубины"
    )
    parser.add_argument(
        "--step",
        "--sampling-step",
        dest="step",
        type=_positive_float,
        help="Шаг регулярной сетки MD",
    )
    parser.add_argument(
        "--window",
        "--window-ratio",
        dest="window",
        type=_window_ratio,
        default=0.10,
        help="Относительная полуширина симметричного DTW-коридора, 0..1",
    )
    parser.add_argument("--warp-penalty", type=_nonnegative_float, default=0.05)
    parser.add_argument(
        "--max-warp-run",
        type=int,
        default=25,
        help="QC-предел серии однотипных warp-шагов после DTW; 0 отключает проверку",
    )
    parser.add_argument("--max-interpolation-gap", type=_positive_float)
    parser.add_argument(
        "--interpolate-all-gaps",
        action="store_true",
        help="Разрешить интерполяцию разрывов любой длины",
    )
    parser.add_argument(
        "--clip-percentiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Необязательный clipping перед нормализацией, например 1 99",
    )
    parser.add_argument(
        "--missing-value",
        type=float,
        action="append",
        default=[],
        help="Дополнительное точное значение пропуска; параметр можно повторять",
    )
    parser.add_argument(
        "--no-standard-missing-values",
        action="store_true",
        help="Не считать -999.25, -999 и 999.25 пропусками",
    )
    parser.add_argument(
        "--well-name",
        action="append",
        help="Явное имя скважины; укажите по одному разу для каждого CSV",
    )
    parser.add_argument("--max-dtw-cells", type=_positive_int, default=DEFAULT_MAX_DTW_CELLS)
    parser.add_argument(
        "--max-resampled-points", type=_positive_int, default=DEFAULT_MAX_RESAMPLED_POINTS
    )
    parser.add_argument(
        "--max-memory-mb",
        type=_positive_float,
        default=DEFAULT_MAX_MEMORY_MB,
        help="Оценочный лимит рабочей памяти DTW для одной пары",
    )
    parser.add_argument("-o", "--output", default="correlation.csv")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _write_csv_atomic(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        frame.to_csv(temporary, index=False)
        temporary.replace(output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.files) < 2:
        parser.error("укажите минимум два CSV-файла")
    if args.max_warp_run < 0:
        parser.error("--max-warp-run должен быть неотрицательным")
    if args.max_interpolation_gap is not None and args.interpolate_all_gaps:
        parser.error("выберите либо --max-interpolation-gap, либо --interpolate-all-gaps")

    output = Path(args.output)
    resolved_output = output.resolve()
    if any(Path(file).resolve() == resolved_output for file in args.files):
        parser.error("выходной CSV не может совпадать с входным файлом")
    if output.exists() and not args.overwrite:
        parser.error(f"выходной файл уже существует: {output}; добавьте --overwrite")

    standard = () if args.no_standard_missing_values else STANDARD_MISSING_VALUES
    missing_values = (*standard, *args.missing_value)
    clip_percentiles = (
        tuple(args.clip_percentiles) if args.clip_percentiles is not None else None
    )
    try:
        result = correlate(
            args.files,
            args.curve,
            args.md,
            args.step,
            args.window,
            well_names=args.well_name,
            missing_values=missing_values,
            max_interpolation_gap=args.max_interpolation_gap,
            interpolate_all_gaps=args.interpolate_all_gaps,
            clip_percentiles=clip_percentiles,
            warp_penalty=args.warp_penalty,
            max_warp_run=None if args.max_warp_run == 0 else args.max_warp_run,
            max_dtw_cells=args.max_dtw_cells,
            max_resampled_points=args.max_resampled_points,
            max_memory_mb=args.max_memory_mb,
            progress=None if args.quiet else print,
        )
        _write_csv_atomic(result, output)
    except (ValueError, MemoryError, OSError, FloatingPointError) as exc:
        parser.error(str(exc))
    if not args.quiet:
        print(f"Результат: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
