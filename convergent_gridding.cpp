#include "convergent_gridding.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace convergent {
namespace {

// Внутренний pipeline реализации:
//   1. проверить и канонизировать входные точки;
//   2. построить иерархию сеток от грубой к исходному разрешению;
//   3. на каждом уровне интерполировать prior, построить мягкие Snap/Taylor
//      штрафы и решить minimum-curvature задачу методом PCG;
//   4. на финальном уровне дополнительно учесть мягкие билинейные ограничения;
//   5. при необходимости отдельной проекцией точно выполнить C*u = d;
//   6. проверить точность после преобразования в qreal и атомарно записать grid.
//
// Gaussian Snap, билинейный перенос между уровнями, свободное поведение границ
// и евклидова exact-проекция — численные решения этой реализации. Они не
// выдаются за неизвестные внутренние детали алгоритма Petrel.

using Scalar = double;

// Канонический внутренний формат получается после сортировки, удаления точек
// с нулевым весом и объединения точек с одинаковыми x/y.
struct CanonicalPoint {
    Scalar x{};
    Scalar y{};
    Scalar value{};
    Scalar weight{};
};

struct Grid {
    std::size_t nx{};
    std::size_t ny{};
    Scalar minx{};
    Scalar maxx{};
    Scalar miny{};
    Scalar maxy{};
    // Тот же порядок, что у Surface::grid: iy == 0 соответствует y == miny,
    // а index(ix, iy) идет слева направо и затем снизу вверх.
    std::vector<Scalar> values;

    std::size_t index(std::size_t ix, std::size_t iy) const noexcept
    {
        return iy * nx + ix;
    }

    Scalar dx() const noexcept
    {
        return (maxx - minx) / static_cast<Scalar>(nx - 1);
    }

    Scalar dy() const noexcept
    {
        return (maxy - miny) / static_cast<Scalar>(ny - 1);
    }
};

struct Derivatives {
    Scalar gx{};
    Scalar gy{};
    Scalar gxx{};
    Scalar gxy{};
    Scalar gyy{};
};

struct StencilTerm {
    std::size_t index{};
    Scalar coefficient{};
};

struct PointConstraint {
    std::array<StencilTerm, 4> terms{};
    std::size_t count{};
    Scalar value{};
    Scalar weight{};
};

struct SolverResult {
    std::vector<Scalar> values;
    std::size_t iterations{};
    Scalar relativeResidual{};
    bool converged{};
};

struct ProjectionResult {
    std::size_t iterations{};
    Scalar maxError{};
    bool converged{};
};

bool finite(Scalar value) noexcept
{
    return std::isfinite(value);
}

std::size_t checkedNodeCount(std::size_t nx, std::size_t ny)
{
    if (nx == 0 || ny > std::numeric_limits<std::size_t>::max() / nx) {
        throw std::invalid_argument("surface dimensions overflow size_t");
    }
    return nx * ny;
}

void validateOptions(const ConvergentGriddingOptions& options)
{
    const auto positiveFinite = [](qreal value) {
        return finite(static_cast<Scalar>(value)) && value > qreal(0);
    };
    const auto nonNegativeFinite = [](qreal value) {
        return finite(static_cast<Scalar>(value)) && value >= qreal(0);
    };

    if (options.initialSnapNodes == 0 || options.coarsestIntervals == 0
        || options.maxLevels == 0 || options.maxSolverIterations == 0) {
        throw std::invalid_argument("gridding counts must be greater than zero");
    }
    if (options.enforceExactControls
        && options.maxControlProjectionIterations == 0) {
        throw std::invalid_argument(
            "maxControlProjectionIterations must be positive in exact-control mode");
    }
    if (!nonNegativeFinite(options.smoothness)
        || !positiveFinite(options.priorWeight)
        || !nonNegativeFinite(options.snapStrength)
        || !nonNegativeFinite(options.finalPointStrength)
        || !positiveFinite(options.gaussianSigma)
        || !positiveFinite(options.relativeTolerance)
        || !nonNegativeFinite(options.absoluteTolerance)
        || !nonNegativeFinite(options.controlTolerance)) {
        throw std::invalid_argument("invalid convergent gridding numeric option");
    }
    const Scalar sigma = static_cast<Scalar>(options.gaussianSigma);
    if (!finite(sigma * sigma) || !(sigma * sigma > 0)) {
        throw std::invalid_argument("gaussianSigma squared must be finite and positive");
    }
    if (options.taylorOrder < 0 || options.taylorOrder > 2) {
        throw std::invalid_argument("taylorOrder must be 0, 1, or 2");
    }
}

std::vector<CanonicalPoint> validateInput(
    const Surface& surface,
    const std::vector<Point>& points,
    const ConvergentGriddingOptions& options)
{
    validateOptions(options);

    if (surface.nx < 2 || surface.ny < 2) {
        throw std::invalid_argument("surface.nx and surface.ny must both be at least 2");
    }

    const Scalar minx = static_cast<Scalar>(surface.minx);
    const Scalar maxx = static_cast<Scalar>(surface.maxx);
    const Scalar miny = static_cast<Scalar>(surface.miny);
    const Scalar maxy = static_cast<Scalar>(surface.maxy);
    if (!finite(minx) || !finite(maxx) || !finite(miny) || !finite(maxy)
        || !(minx < maxx) || !(miny < maxy)) {
        throw std::invalid_argument("surface bounds must be finite and strictly increasing");
    }

    const Scalar spanX = maxx - minx;
    const Scalar spanY = maxy - miny;
    const Scalar dx = spanX / static_cast<Scalar>(surface.nx - 1);
    const Scalar dy = spanY / static_cast<Scalar>(surface.ny - 1);
    if (!finite(spanX) || !finite(spanY) || !finite(dx) || !finite(dy)
        || !(spanX > 0) || !(spanY > 0) || !(dx > 0) || !(dy > 0)
        || !finite(dx / dy) || !finite(dy / dx)) {
        throw std::invalid_argument("surface bounds produce an unsafe grid spacing");
    }

    const std::size_t expected = checkedNodeCount(surface.nx, surface.ny);
    if (surface.grid.size() != expected) {
        throw std::invalid_argument("surface.grid.size() must equal surface.nx * surface.ny");
    }
    for (qreal value : surface.grid) {
        if (!finite(static_cast<Scalar>(value))) {
            throw std::invalid_argument("surface.grid must contain only finite values");
        }
    }

    std::vector<CanonicalPoint> result;
    result.reserve(points.size());
    for (const Point& point : points) {
        const CanonicalPoint p{
            static_cast<Scalar>(point.x),
            static_cast<Scalar>(point.y),
            static_cast<Scalar>(point.value),
            static_cast<Scalar>(point.weight)};

        if (!finite(p.x) || !finite(p.y) || !finite(p.value) || !finite(p.weight)) {
            throw std::invalid_argument("control point fields must be finite");
        }
        if (p.weight < 0) {
            throw std::invalid_argument("control point weight must not be negative");
        }
        if (p.x < minx || p.x > maxx || p.y < miny || p.y > maxy) {
            throw std::invalid_argument("control point lies outside the surface bounds");
        }
        if (p.weight == 0) {
            continue;
        }
        result.push_back(p);
    }

    // Канонический порядок делает накопление воспроизводимым и не зависящим от
    // перестановки одного и того же набора входных точек.
    std::sort(result.begin(), result.end(), [](const CanonicalPoint& a, const CanonicalPoint& b) {
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        if (a.value != b.value) return a.value < b.value;
        return a.weight < b.weight;
    });

    // Совпадающие координаты задают одно физическое ограничение. Объединение до
    // нормализации делает {(z=10,w=1),(z=30,w=3)} эквивалентным
    // {(z=25,w=4)} и не создает повторяющиеся строки ограничений.
    std::vector<CanonicalPoint> grouped;
    grouped.reserve(result.size());
    for (std::size_t begin = 0; begin < result.size();) {
        std::size_t end = begin + 1;
        while (end < result.size()
               && result[end].x == result[begin].x
               && result[end].y == result[begin].y) {
            ++end;
        }
        long double weightSum = 0;
        long double weightedValue = 0;
        for (std::size_t i = begin; i < end; ++i) {
            weightSum += static_cast<long double>(result[i].weight);
            weightedValue += static_cast<long double>(result[i].weight)
                * static_cast<long double>(result[i].value);
        }
        const long double combinedValue = weightedValue / weightSum;
        if (!std::isfinite(weightSum) || !std::isfinite(combinedValue)
            || weightSum > static_cast<long double>(std::numeric_limits<Scalar>::max())
            || std::abs(combinedValue)
                > static_cast<long double>(std::numeric_limits<Scalar>::max())) {
            throw std::invalid_argument("co-located control point weights overflow");
        }
        grouped.push_back({result[begin].x, result[begin].y,
                           static_cast<Scalar>(combinedValue),
                           static_cast<Scalar>(weightSum)});
        begin = end;
    }
    result = std::move(grouped);

    Scalar maximumWeight = 0;
    for (const CanonicalPoint& point : result) {
        maximumWeight = std::max(maximumWeight, point.weight);
    }
    if (options.normalizePointWeights && maximumWeight > 0) {
        // Деление на общий максимум сохраняет относительные веса и делает
        // результат инвариантным к умножению всех исходных весов на константу.
        for (CanonicalPoint& point : result) {
            point.weight /= maximumWeight;
        }
    }
    return result;
}

Grid makeGridGeometry(const Surface& surface, std::size_t nx, std::size_t ny)
{
    Grid result;
    result.nx = nx;
    result.ny = ny;
    result.minx = static_cast<Scalar>(surface.minx);
    result.maxx = static_cast<Scalar>(surface.maxx);
    result.miny = static_cast<Scalar>(surface.miny);
    result.maxy = static_cast<Scalar>(surface.maxy);
    result.values.resize(checkedNodeCount(nx, ny));
    return result;
}

Scalar bilinearSample(const Grid& grid, Scalar x, Scalar y)
{
    Scalar fx = (x - grid.minx) / grid.dx();
    Scalar fy = (y - grid.miny) / grid.dy();
    fx = std::clamp(fx, Scalar(0), static_cast<Scalar>(grid.nx - 1));
    fy = std::clamp(fy, Scalar(0), static_cast<Scalar>(grid.ny - 1));

    const std::size_t ix0 = std::min(
        static_cast<std::size_t>(std::floor(fx)), grid.nx - 1);
    const std::size_t iy0 = std::min(
        static_cast<std::size_t>(std::floor(fy)), grid.ny - 1);
    const std::size_t ix1 = std::min(ix0 + 1, grid.nx - 1);
    const std::size_t iy1 = std::min(iy0 + 1, grid.ny - 1);
    const Scalar tx = fx - static_cast<Scalar>(ix0);
    const Scalar ty = fy - static_cast<Scalar>(iy0);

    const Scalar v00 = grid.values[grid.index(ix0, iy0)];
    const Scalar v10 = grid.values[grid.index(ix1, iy0)];
    const Scalar v01 = grid.values[grid.index(ix0, iy1)];
    const Scalar v11 = grid.values[grid.index(ix1, iy1)];
    return (Scalar(1) - tx) * (Scalar(1) - ty) * v00
         + tx * (Scalar(1) - ty) * v10
         + (Scalar(1) - tx) * ty * v01
         + tx * ty * v11;
}

// Билинейный restriction/prolongation между сетками с одинаковыми min/max.
// Эта функция используется и для перехода входной поверхности на грубую
// сетку, и для уточнения решения предыдущего уровня. Последний узел каждой оси
// явно ставится в maxx/maxy, чтобы не накопить ошибку округления координаты.
// Производные отдельно не переносятся: на новом уровне они будут снова
// вычислены из полученных значений.
Grid resample(const Grid& source, std::size_t nx, std::size_t ny)
{
    Grid result = source;
    result.nx = nx;
    result.ny = ny;
    result.values.assign(checkedNodeCount(nx, ny), Scalar(0));

    for (std::size_t iy = 0; iy < ny; ++iy) {
        const Scalar ty = static_cast<Scalar>(iy) / static_cast<Scalar>(ny - 1);
        const Scalar y = iy + 1 == ny
            ? result.maxy
            : result.miny + ty * (result.maxy - result.miny);
        for (std::size_t ix = 0; ix < nx; ++ix) {
            const Scalar tx = static_cast<Scalar>(ix) / static_cast<Scalar>(nx - 1);
            const Scalar x = ix + 1 == nx
                ? result.maxx
                : result.minx + tx * (result.maxx - result.minx);
            result.values[result.index(ix, iy)] = bilinearSample(source, x, y);
        }
    }
    return result;
}

Grid surfaceAsGrid(const Surface& surface)
{
    Grid result = makeGridGeometry(surface, surface.nx, surface.ny);
    std::transform(surface.grid.begin(), surface.grid.end(), result.values.begin(),
        [](qreal value) { return static_cast<Scalar>(value); });
    return result;
}

std::size_t ceilDivide(std::size_t value, std::size_t divisor)
{
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

// Иерархия строится по числу ИНТЕРВАЛОВ (nx-1, ny-1). На каждом шаге фактор
// увеличивается вдвое, затем список разворачивается в порядок coarse-to-fine.
// ceilDivide(...)+1 сохраняет общие физические границы для размеров, которые
// не имеют вида 2^k+1; поэтому узлы соседних уровней не обязаны точно совпадать.
// maxLevels является жестким пределом, а исходный finalNx*finalNy в любом
// случае добавляется последним уровнем.
std::vector<std::pair<std::size_t, std::size_t>> buildHierarchy(
    std::size_t finalNx,
    std::size_t finalNy,
    const ConvergentGriddingOptions& options)
{
    std::vector<std::size_t> factors{1};
    std::size_t factor = 1;
    while (factors.size() < options.maxLevels) {
        const std::size_t intervalsX = ceilDivide(finalNx - 1, factor);
        const std::size_t intervalsY = ceilDivide(finalNy - 1, factor);
        if (intervalsX <= options.coarsestIntervals
            && intervalsY <= options.coarsestIntervals) {
            break;
        }
        if (factor > std::numeric_limits<std::size_t>::max() / 2) {
            break;
        }
        factor *= 2;
        factors.push_back(factor);
    }

    std::vector<std::pair<std::size_t, std::size_t>> hierarchy;
    hierarchy.reserve(factors.size());
    for (auto it = factors.rbegin(); it != factors.rend(); ++it) {
        const std::size_t nx = ceilDivide(finalNx - 1, *it) + 1;
        const std::size_t ny = ceilDivide(finalNy - 1, *it) + 1;
        if (hierarchy.empty() || hierarchy.back() != std::make_pair(nx, ny)) {
            hierarchy.emplace_back(nx, ny);
        }
    }
    if (hierarchy.empty() || hierarchy.back() != std::make_pair(finalNx, finalNy)) {
        hierarchy.emplace_back(finalNx, finalNy);
    }
    return hierarchy;
}

Scalar valueAt(const Grid& grid, std::size_t ix, std::size_t iy)
{
    return grid.values[grid.index(ix, iy)];
}

// Производные нужны для Taylor-переноса значения контрольной точки на Snap-
// узлы. Внутри сетки применяются центральные разности в физических единицах
// x/y; у границы — односторонние разности или уменьшенный прямоугольник для
// смешанной производной. Это не задает граничное условие основной задачи.
Derivatives derivativesAtNode(const Grid& grid, std::size_t ix, std::size_t iy)
{
    const Scalar hx = grid.dx();
    const Scalar hy = grid.dy();
    Derivatives d;

    if (ix == 0) {
        d.gx = (valueAt(grid, 1, iy) - valueAt(grid, 0, iy)) / hx;
    } else if (ix + 1 == grid.nx) {
        d.gx = (valueAt(grid, ix, iy) - valueAt(grid, ix - 1, iy)) / hx;
    } else {
        d.gx = (valueAt(grid, ix + 1, iy) - valueAt(grid, ix - 1, iy)) / (Scalar(2) * hx);
    }

    if (iy == 0) {
        d.gy = (valueAt(grid, ix, 1) - valueAt(grid, ix, 0)) / hy;
    } else if (iy + 1 == grid.ny) {
        d.gy = (valueAt(grid, ix, iy) - valueAt(grid, ix, iy - 1)) / hy;
    } else {
        d.gy = (valueAt(grid, ix, iy + 1) - valueAt(grid, ix, iy - 1)) / (Scalar(2) * hy);
    }

    if (grid.nx >= 3) {
        if (ix == 0) {
            d.gxx = (valueAt(grid, 0, iy) - Scalar(2) * valueAt(grid, 1, iy)
                     + valueAt(grid, 2, iy)) / (hx * hx);
        } else if (ix + 1 == grid.nx) {
            d.gxx = (valueAt(grid, ix, iy) - Scalar(2) * valueAt(grid, ix - 1, iy)
                     + valueAt(grid, ix - 2, iy)) / (hx * hx);
        } else {
            d.gxx = (valueAt(grid, ix - 1, iy) - Scalar(2) * valueAt(grid, ix, iy)
                     + valueAt(grid, ix + 1, iy)) / (hx * hx);
        }
    }

    if (grid.ny >= 3) {
        if (iy == 0) {
            d.gyy = (valueAt(grid, ix, 0) - Scalar(2) * valueAt(grid, ix, 1)
                     + valueAt(grid, ix, 2)) / (hy * hy);
        } else if (iy + 1 == grid.ny) {
            d.gyy = (valueAt(grid, ix, iy) - Scalar(2) * valueAt(grid, ix, iy - 1)
                     + valueAt(grid, ix, iy - 2)) / (hy * hy);
        } else {
            d.gyy = (valueAt(grid, ix, iy - 1) - Scalar(2) * valueAt(grid, ix, iy)
                     + valueAt(grid, ix, iy + 1)) / (hy * hy);
        }
    }

    const std::size_t ix0 = ix == 0 ? 0 : ix - 1;
    const std::size_t ix1 = ix + 1 < grid.nx ? ix + 1 : ix;
    const std::size_t iy0 = iy == 0 ? 0 : iy - 1;
    const std::size_t iy1 = iy + 1 < grid.ny ? iy + 1 : iy;
    const Scalar spanX = static_cast<Scalar>(ix1 - ix0) * hx;
    const Scalar spanY = static_cast<Scalar>(iy1 - iy0) * hy;
    if (spanX > 0 && spanY > 0) {
        d.gxy = (valueAt(grid, ix1, iy1) - valueAt(grid, ix1, iy0)
                 - valueAt(grid, ix0, iy1) + valueAt(grid, ix0, iy0))
            / (spanX * spanY);
    }
    return d;
}

// Сначала вычисляются пять полей производных в четырех углах содержащей ячейки,
// затем каждое поле билинейно интерполируется в фактическую координату точки.
// Таким образом, Taylor-модель описывает prior текущего уровня, а не сама по
// себе восстанавливает производные из контрольных данных.
Derivatives derivativesAt(const Grid& grid, Scalar x, Scalar y)
{
    Scalar fx = std::clamp((x - grid.minx) / grid.dx(), Scalar(0),
                           static_cast<Scalar>(grid.nx - 1));
    Scalar fy = std::clamp((y - grid.miny) / grid.dy(), Scalar(0),
                           static_cast<Scalar>(grid.ny - 1));
    const std::size_t ix0 = std::min(static_cast<std::size_t>(std::floor(fx)), grid.nx - 1);
    const std::size_t iy0 = std::min(static_cast<std::size_t>(std::floor(fy)), grid.ny - 1);
    const std::size_t ix1 = std::min(ix0 + 1, grid.nx - 1);
    const std::size_t iy1 = std::min(iy0 + 1, grid.ny - 1);
    const Scalar tx = fx - static_cast<Scalar>(ix0);
    const Scalar ty = fy - static_cast<Scalar>(iy0);

    const Derivatives d00 = derivativesAtNode(grid, ix0, iy0);
    const Derivatives d10 = derivativesAtNode(grid, ix1, iy0);
    const Derivatives d01 = derivativesAtNode(grid, ix0, iy1);
    const Derivatives d11 = derivativesAtNode(grid, ix1, iy1);
    const auto blend = [&](Scalar Derivatives::*member) {
        return (Scalar(1) - tx) * (Scalar(1) - ty) * d00.*member
             + tx * (Scalar(1) - ty) * d10.*member
             + (Scalar(1) - tx) * ty * d01.*member
             + tx * ty * d11.*member;
    };
    return {blend(&Derivatives::gx), blend(&Derivatives::gy),
            blend(&Derivatives::gxx), blend(&Derivatives::gxy),
            blend(&Derivatives::gyy)};
}

struct CandidateNode {
    std::size_t index{};
    std::size_t ix{};
    std::size_t iy{};
    Scalar distanceSquared{};
};

// Выбирает requested ближайших узлов по евклидову расстоянию в КООРДИНАТАХ
// ЯЧЕЕК: (ix-fx)^2 + (iy-fy)^2. При dx != dy это не совпадает с расстоянием в
// физических координатах. Равные расстояния разрешаются плоским индексом, что
// сохраняет детерминированный результат.
std::vector<CandidateNode> nearestNodes(
    const Grid& grid, Scalar x, Scalar y, std::size_t requested)
{
    requested = std::min(requested, grid.values.size());
    const Scalar fx = (x - grid.minx) / grid.dx();
    const Scalar fy = (y - grid.miny) / grid.dy();
    const std::size_t centerX = std::min(
        static_cast<std::size_t>(std::floor(std::clamp(
            fx, Scalar(0), static_cast<Scalar>(grid.nx - 1)))), grid.nx - 1);
    const std::size_t centerY = std::min(
        static_cast<std::size_t>(std::floor(std::clamp(
            fy, Scalar(0), static_cast<Scalar>(grid.ny - 1)))), grid.ny - 1);
    const std::size_t radius = static_cast<std::size_t>(
        std::ceil(std::sqrt(static_cast<Scalar>(requested)))) + 3;
    const std::size_t minX = centerX > radius ? centerX - radius : 0;
    const std::size_t minY = centerY > radius ? centerY - radius : 0;
    const std::size_t maxX = std::min(centerX + radius, grid.nx - 1);
    const std::size_t maxY = std::min(centerY + radius, grid.ny - 1);

    std::vector<CandidateNode> candidates;
    candidates.reserve((maxX - minX + 1) * (maxY - minY + 1));
    for (std::size_t iy = minY; iy <= maxY; ++iy) {
        for (std::size_t ix = minX; ix <= maxX; ++ix) {
            const Scalar dxCells = static_cast<Scalar>(ix) - fx;
            const Scalar dyCells = static_cast<Scalar>(iy) - fy;
            candidates.push_back({grid.index(ix, iy), ix, iy,
                                  dxCells * dxCells + dyCells * dyCells});
        }
    }

    // Для очень тонкой сетки или точки у края локального квадрата может быть
    // недостаточно. Редкий полный просмотр сохраняет точный результат именно
    // поиска ближайших узлов (это не относится к exact-интерполяции значений).
    if (candidates.size() < requested) {
        candidates.clear();
        candidates.reserve(grid.values.size());
        for (std::size_t iy = 0; iy < grid.ny; ++iy) {
            for (std::size_t ix = 0; ix < grid.nx; ++ix) {
                const Scalar dxCells = static_cast<Scalar>(ix) - fx;
                const Scalar dyCells = static_cast<Scalar>(iy) - fy;
                candidates.push_back({grid.index(ix, iy), ix, iy,
                                      dxCells * dxCells + dyCells * dyCells});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const CandidateNode& a,
                                                       const CandidateNode& b) {
        if (a.distanceSquared != b.distanceSquared) {
            return a.distanceSquared < b.distanceSquared;
        }
        return a.index < b.index;
    });
    candidates.resize(requested);
    return candidates;
}

// Число Snap-узлов уменьшается вместе с геометрическим средним шага сетки:
//   N = round(initialN * sqrt((dx/dx0) * (dy/dy0))).
// При двукратном изотропном уточнении это дает 16 -> 8 -> 4 -> 2 -> 1.
// На финальном уровне одна точка всегда воздействует ровно на один Snap-узел.
std::size_t snapNodeCount(
    const Grid& grid,
    const Grid& coarsest,
    bool finalLevel,
    const ConvergentGriddingOptions& options)
{
    if (finalLevel) return 1;
    const std::size_t cappedInitial = std::min(
        options.initialSnapNodes, grid.values.size());
    const long double scaleSquared =
        static_cast<long double>(grid.dx() / coarsest.dx())
        * static_cast<long double>(grid.dy() / coarsest.dy());
    const long double scaled = static_cast<long double>(cappedInitial)
        * std::sqrt(scaleSquared);
    if (!std::isfinite(scaled) || scaled < 0
        || scaled > static_cast<long double>(grid.values.size())) {
        throw std::invalid_argument("initialSnapNodes cannot be scaled safely");
    }
    return std::max<std::size_t>(1,
        static_cast<std::size_t>(std::floor(scaled + 0.5L)));
}

// Формирует мягкую Snap-часть квадратичного функционала.
//
// Для точки p и соседнего узла n строится Taylor-прогноз
//   T_p(n) = value_p + grad(u_prior)_p * delta
//          + 1/2 * delta^T * Hessian(u_prior)_p * delta.
// Порядок ограничивается options.taylorOrder. Gaussian-коэффициенты
//   k_pn = exp(-distanceCells^2 / (2*sigma^2))
// нормируются ОТДЕЛЬНО для каждой точки, поэтому сумма ее коэффициентов по
// выбранным узлам равна point.weight. Если в узел попали несколько точек, их
// вклады складываются и образуют взвешенное смешивание прогнозов.
//
// Полученные diagonal/rhs кодируют
//   snapStrength * sum_pn w_pn * (u_n - T_p(n))^2.
// Поэтому Snap здесь является мягким штрафом, а не непосредственной записью
// контрольного значения в grid. Gaussian-ядро — выбор данной реализации.
void addSnapConstraints(
    const Grid& prior,
    const std::vector<CanonicalPoint>& points,
    std::size_t count,
    const ConvergentGriddingOptions& options,
    std::vector<Scalar>& diagonal,
    std::vector<Scalar>& rhs)
{
    std::vector<Scalar> accumulatedWeight(prior.values.size(), Scalar(0));
    std::vector<Scalar> accumulatedValue(prior.values.size(), Scalar(0));
    const Scalar sigma2 = static_cast<Scalar>(options.gaussianSigma)
        * static_cast<Scalar>(options.gaussianSigma);

    for (const CanonicalPoint& point : points) {
        const Derivatives d = derivativesAt(prior, point.x, point.y);
        const std::vector<CandidateNode> nodes = nearestNodes(prior, point.x, point.y, count);
        std::vector<Scalar> kernels(nodes.size());
        Scalar kernelSum = 0;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            kernels[i] = std::exp(-nodes[i].distanceSquared / (Scalar(2) * sigma2));
            kernelSum += kernels[i];
        }
        if (!(kernelSum > 0) || !finite(kernelSum)) {
            throw std::runtime_error("failed to normalize the Snap distance kernel");
        }

        for (std::size_t i = 0; i < nodes.size(); ++i) {
            const CandidateNode& node = nodes[i];
            const Scalar nodeX = prior.minx + static_cast<Scalar>(node.ix) * prior.dx();
            const Scalar nodeY = prior.miny + static_cast<Scalar>(node.iy) * prior.dy();
            const Scalar deltaX = nodeX - point.x;
            const Scalar deltaY = nodeY - point.y;
            Scalar projected = point.value;
            if (options.taylorOrder >= 1) {
                projected += d.gx * deltaX + d.gy * deltaY;
            }
            if (options.taylorOrder >= 2) {
                projected += Scalar(0.5) * (d.gxx * deltaX * deltaX
                    + Scalar(2) * d.gxy * deltaX * deltaY
                    + d.gyy * deltaY * deltaY);
            }
            if (!finite(projected)) {
                throw std::runtime_error("Taylor projection produced a non-finite value");
            }

            const Scalar weight = point.weight * kernels[i] / kernelSum;
            const Scalar nextWeight = accumulatedWeight[node.index] + weight;
            const Scalar nextValue = accumulatedValue[node.index] + weight * projected;
            if (!finite(weight) || !finite(nextWeight) || !finite(nextValue)) {
                throw std::runtime_error("Snap constraint accumulation overflowed");
            }
            accumulatedWeight[node.index] = nextWeight;
            accumulatedValue[node.index] = nextValue;
        }
    }

    const Scalar strength = static_cast<Scalar>(options.snapStrength);
    for (std::size_t i = 0; i < diagonal.size(); ++i) {
        diagonal[i] += strength * accumulatedWeight[i];
        rhs[i] += strength * accumulatedValue[i];
        if (!finite(diagonal[i]) || !finite(rhs[i])) {
            throw std::runtime_error("scaled Snap constraint overflowed");
        }
    }
}

// Добавляет коэффициент в разреженную строку ограничения. Совпавшие индексы
// объединяются; это происходит, например, на maxx/maxy, где два угла
// билинейной ячейки могут обозначать один и тот же граничный узел.
void addTerm(PointConstraint& constraint, std::size_t index, Scalar coefficient)
{
    if (coefficient == 0) return;
    for (std::size_t i = 0; i < constraint.count; ++i) {
        if (constraint.terms[i].index == index) {
            constraint.terms[i].coefficient += coefficient;
            return;
        }
    }
    constraint.terms[constraint.count++] = {index, coefficient};
}

// Строит строку C_p билинейного оператора для каждой точки:
//   C_p * u = sum_{k=1..4} c_k * u_k.
// В строке остается от одного до четырех уникальных узлов, коэффициенты
// неотрицательны и в точной арифметике суммируются в единицу.
std::vector<PointConstraint> makePointConstraints(
    const Grid& grid, const std::vector<CanonicalPoint>& points)
{
    std::vector<PointConstraint> result;
    result.reserve(points.size());
    for (const CanonicalPoint& point : points) {
        Scalar fx = std::clamp((point.x - grid.minx) / grid.dx(), Scalar(0),
                               static_cast<Scalar>(grid.nx - 1));
        Scalar fy = std::clamp((point.y - grid.miny) / grid.dy(), Scalar(0),
                               static_cast<Scalar>(grid.ny - 1));
        const std::size_t ix0 = std::min(static_cast<std::size_t>(std::floor(fx)), grid.nx - 1);
        const std::size_t iy0 = std::min(static_cast<std::size_t>(std::floor(fy)), grid.ny - 1);
        const std::size_t ix1 = std::min(ix0 + 1, grid.nx - 1);
        const std::size_t iy1 = std::min(iy0 + 1, grid.ny - 1);
        const Scalar tx = fx - static_cast<Scalar>(ix0);
        const Scalar ty = fy - static_cast<Scalar>(iy0);

        PointConstraint constraint;
        constraint.value = point.value;
        constraint.weight = point.weight;
        addTerm(constraint, grid.index(ix0, iy0), (Scalar(1) - tx) * (Scalar(1) - ty));
        addTerm(constraint, grid.index(ix1, iy0), tx * (Scalar(1) - ty));
        addTerm(constraint, grid.index(ix0, iy1), (Scalar(1) - tx) * ty);
        addTerm(constraint, grid.index(ix1, iy1), tx * ty);
        result.push_back(constraint);
    }
    return result;
}

// Накапливает K*input в output, где K = smoothness * B^T*B — нормальный
// оператор дискретной thin-plate/Hessian энергии
//
//   E(u) = smoothness * [
//       sum ((dy/dx) * Dxx(u))^2
//     + sum ((dx/dy) * Dyy(u))^2
//     + 2 * sum (1/4 * Dxy4(u))^2 ].
//
// Отношения шагов учитывают форму прямоугольной ячейки до общего множителя.
// Dxy4 использует четыре диагональных узла вокруг центра. Это cell-scaled
// Hessian energy данной реализации, а не буквальная матрица Laplacian^2.
//
// Используются только шаблоны, целиком лежащие внутри grid: ghost-узлов и явно
// заданных Dirichlet/Neumann условий нет. Поэтому граница является свободной
// (natural), а любая аффинная плоскость принадлежит nullspace K. Функция
// ДОБАВЛЯЕТ вклад в output и не очищает его.
void applyCurvature(
    const Grid& grid,
    const std::vector<Scalar>& input,
    std::vector<Scalar>& output,
    Scalar smoothness)
{
    if (smoothness == 0) return;
    const Scalar ax = grid.dy() / grid.dx();
    const Scalar ay = grid.dx() / grid.dy();
    const auto accumulate3 = [&](std::size_t a, std::size_t b, std::size_t c,
                                 Scalar scale) {
        const Scalar d = scale * (input[a] - Scalar(2) * input[b] + input[c]);
        const Scalar weighted = smoothness * scale * d;
        output[a] += weighted;
        output[b] -= Scalar(2) * weighted;
        output[c] += weighted;
    };

    if (grid.nx >= 3) {
        for (std::size_t iy = 0; iy < grid.ny; ++iy) {
            for (std::size_t ix = 1; ix + 1 < grid.nx; ++ix) {
                accumulate3(grid.index(ix - 1, iy), grid.index(ix, iy),
                            grid.index(ix + 1, iy), ax);
            }
        }
    }
    if (grid.ny >= 3) {
        for (std::size_t iy = 1; iy + 1 < grid.ny; ++iy) {
            for (std::size_t ix = 0; ix < grid.nx; ++ix) {
                accumulate3(grid.index(ix, iy - 1), grid.index(ix, iy),
                            grid.index(ix, iy + 1), ay);
            }
        }
    }
    if (grid.nx >= 3 && grid.ny >= 3) {
        constexpr Scalar c = Scalar(0.25);
        for (std::size_t iy = 1; iy + 1 < grid.ny; ++iy) {
            for (std::size_t ix = 1; ix + 1 < grid.nx; ++ix) {
                const std::size_t pp = grid.index(ix + 1, iy + 1);
                const std::size_t pm = grid.index(ix + 1, iy - 1);
                const std::size_t mp = grid.index(ix - 1, iy + 1);
                const std::size_t mm = grid.index(ix - 1, iy - 1);
                const Scalar d = c * (input[pp] - input[pm] - input[mp] + input[mm]);
                const Scalar weighted = Scalar(2) * smoothness * c * d;
                output[pp] += weighted;
                output[pm] -= weighted;
                output[mp] -= weighted;
                output[mm] += weighted;
            }
        }
    }
}

// Вычисляет точную диагональ того же K для Jacobi-предобуславливателя PCG.
// При изменении любого stencil или масштаба эту функцию необходимо менять
// синхронно с applyCurvature(), иначе предобуславливатель станет несогласованным.
std::vector<Scalar> curvatureDiagonal(const Grid& grid, Scalar smoothness)
{
    std::vector<Scalar> diagonal(grid.values.size(), Scalar(0));
    if (smoothness == 0) return diagonal;
    const Scalar ax = grid.dy() / grid.dx();
    const Scalar ay = grid.dx() / grid.dy();
    const auto accumulate3 = [&](std::size_t a, std::size_t b, std::size_t c,
                                 Scalar scale) {
        const Scalar factor = smoothness * scale * scale;
        diagonal[a] += factor;
        diagonal[b] += Scalar(4) * factor;
        diagonal[c] += factor;
    };
    if (grid.nx >= 3) {
        for (std::size_t iy = 0; iy < grid.ny; ++iy) {
            for (std::size_t ix = 1; ix + 1 < grid.nx; ++ix) {
                accumulate3(grid.index(ix - 1, iy), grid.index(ix, iy),
                            grid.index(ix + 1, iy), ax);
            }
        }
    }
    if (grid.ny >= 3) {
        for (std::size_t iy = 1; iy + 1 < grid.ny; ++iy) {
            for (std::size_t ix = 0; ix < grid.nx; ++ix) {
                accumulate3(grid.index(ix, iy - 1), grid.index(ix, iy),
                            grid.index(ix, iy + 1), ay);
            }
        }
    }
    if (grid.nx >= 3 && grid.ny >= 3) {
        constexpr Scalar coefficientSquared = Scalar(1) / Scalar(16);
        const Scalar factor = Scalar(2) * smoothness * coefficientSquared;
        for (std::size_t iy = 1; iy + 1 < grid.ny; ++iy) {
            for (std::size_t ix = 1; ix + 1 < grid.nx; ++ix) {
                diagonal[grid.index(ix + 1, iy + 1)] += factor;
                diagonal[grid.index(ix + 1, iy - 1)] += factor;
                diagonal[grid.index(ix - 1, iy + 1)] += factor;
                diagonal[grid.index(ix - 1, iy - 1)] += factor;
            }
        }
    }
    return diagonal;
}

Scalar constraintValue(const PointConstraint& constraint,
                       const std::vector<Scalar>& values)
{
    Scalar result = 0;
    for (std::size_t i = 0; i < constraint.count; ++i) {
        result += constraint.terms[i].coefficient * values[constraint.terms[i].index];
    }
    return result;
}

Scalar dot(const std::vector<Scalar>& a, const std::vector<Scalar>& b)
{
    // Более широкий аккумулятор уменьшает потерю точности в нормах и скалярных
    // произведениях PCG; наружу все равно возвращается внутренний Scalar.
    long double result = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result += static_cast<long double>(a[i]) * static_cast<long double>(b[i]);
    }
    return static_cast<Scalar>(result);
}

// Решает на одном уровне неявную симметричную положительно определенную систему
// A*u = b, соответствующую сумме квадратичных штрафов:
//
//   A = K + priorWeight*I + diag(snapDiagonal)
//         + sum_p finalStrength*w_p*c_p*c_p^T,
//   b = priorWeight*prior + snapRhs
//         + sum_p finalStrength*w_p*c_p*value_p.
//
// c_p — строка билинейного оператора точки. Финальная сумма присутствует только
// на последнем уровне. Положительный priorWeight устраняет аффинное nullspace K
// и гарантирует SPD даже там, где нет точечных штрафов. Матрица целиком не
// хранится: applyA последовательно применяет все ее части. prior используется
// как теплое начальное приближение x0.
SolverResult solveLevel(
    const Grid& geometry,
    const std::vector<Scalar>& prior,
    const std::vector<Scalar>& extraDiagonal,
    const std::vector<Scalar>& extraRhs,
    const std::vector<PointConstraint>& finalConstraints,
    const ConvergentGriddingOptions& options)
{
    const std::size_t size = prior.size();
    const Scalar priorWeight = static_cast<Scalar>(options.priorWeight);
    const Scalar smoothness = static_cast<Scalar>(options.smoothness);
    const Scalar finalStrength = static_cast<Scalar>(options.finalPointStrength);

    std::vector<Scalar> rhs(size);
    std::vector<Scalar> diagonal = curvatureDiagonal(geometry, smoothness);
    for (std::size_t i = 0; i < size; ++i) {
        diagonal[i] += priorWeight + extraDiagonal[i];
        rhs[i] = priorWeight * prior[i] + extraRhs[i];
        if (!finite(diagonal[i]) || !(diagonal[i] > 0) || !finite(rhs[i])) {
            throw std::runtime_error("level-system coefficients are not finite");
        }
    }
    for (const PointConstraint& constraint : finalConstraints) {
        const Scalar strength = finalStrength * constraint.weight;
        if (!finite(strength)) {
            throw std::runtime_error("point-constraint strength overflowed");
        }
        for (std::size_t a = 0; a < constraint.count; ++a) {
            const StencilTerm& term = constraint.terms[a];
            rhs[term.index] += strength * term.coefficient * constraint.value;
            diagonal[term.index] += strength * term.coefficient * term.coefficient;
            if (!finite(rhs[term.index]) || !finite(diagonal[term.index])) {
                throw std::runtime_error("point-constraint coefficients overflowed");
            }
        }
    }

    const auto applyA = [&](const std::vector<Scalar>& input,
                            std::vector<Scalar>& output) {
        std::fill(output.begin(), output.end(), Scalar(0));
        applyCurvature(geometry, input, output, smoothness);
        for (std::size_t i = 0; i < size; ++i) {
            output[i] += (priorWeight + extraDiagonal[i]) * input[i];
        }
        for (const PointConstraint& constraint : finalConstraints) {
            const Scalar strength = finalStrength * constraint.weight;
            const Scalar projected = constraintValue(constraint, input);
            for (std::size_t i = 0; i < constraint.count; ++i) {
                const StencilTerm& term = constraint.terms[i];
                output[term.index] += strength * term.coefficient * projected;
            }
        }
    };

    std::vector<Scalar> x = prior;
    std::vector<Scalar> ax(size), residual(size), z(size), direction(size), ad(size);
    applyA(x, ax);
    for (std::size_t i = 0; i < size; ++i) {
        residual[i] = rhs[i] - ax[i];
        z[i] = residual[i] / diagonal[i];
    }
    direction = z;
    Scalar rz = dot(residual, z);
    const Scalar rhsNorm = std::sqrt(std::max(Scalar(0), dot(rhs, rhs)));
    const Scalar initialAxNorm = std::sqrt(std::max(Scalar(0), dot(ax, ax)));
    const Scalar normalizer = std::max(rhsNorm, initialAxNorm);
    const Scalar tolerance = static_cast<Scalar>(options.relativeTolerance);
    const Scalar absoluteTolerance = static_cast<Scalar>(options.absoluteTolerance);
    // Нормировка фиксируется по начальному состоянию и не меняется во время
    // итераций: ||r|| <= absTol + relTol*max(||b||, ||A*x0||).
    const Scalar threshold = absoluteTolerance + tolerance * normalizer;
    if (!finite(normalizer) || !finite(threshold)) {
        throw std::runtime_error("PCG residual scale is not finite");
    }
    Scalar residualNorm = std::sqrt(std::max(Scalar(0), dot(residual, residual)));
    Scalar relativeResidual = normalizer > 0 ? residualNorm / normalizer : Scalar(0);

    SolverResult result;
    result.values = x;
    result.relativeResidual = relativeResidual;
    result.converged = residualNorm <= threshold;
    if (result.converged) return result;

    for (std::size_t iteration = 0; iteration < options.maxSolverIterations; ++iteration) {
        applyA(direction, ad);
        const Scalar denominator = dot(direction, ad);
        if (!(denominator > 0) || !finite(denominator) || !finite(rz)) {
            throw std::runtime_error("PCG breakdown while solving the biharmonic system");
        }
        const Scalar alpha = rz / denominator;
        if (!finite(alpha)) {
            throw std::runtime_error("PCG produced a non-finite step");
        }
        for (std::size_t i = 0; i < size; ++i) {
            x[i] += alpha * direction[i];
            residual[i] -= alpha * ad[i];
        }

        // Периодически пересчитываем r=b-A*x и перезапускаем сопряженное
        // направление, устраняя накопившийся дрейф рекурсивной невязки.
        const bool restartDirection = (iteration + 1) % 50 == 0;
        if (restartDirection) {
            applyA(x, ax);
            for (std::size_t i = 0; i < size; ++i) residual[i] = rhs[i] - ax[i];
        }
        residualNorm = std::sqrt(std::max(Scalar(0), dot(residual, residual)));
        relativeResidual = normalizer > 0 ? residualNorm / normalizer : Scalar(0);
        result.iterations = iteration + 1;
        if (residualNorm <= threshold) {
            result.values = std::move(x);
            result.relativeResidual = relativeResidual;
            result.converged = true;
            return result;
        }

        for (std::size_t i = 0; i < size; ++i) z[i] = residual[i] / diagonal[i];
        const Scalar nextRz = dot(residual, z);
        if (!finite(nextRz)) {
            throw std::runtime_error("PCG produced a non-finite residual");
        }
        if (restartDirection) {
            direction = z;
        } else {
            const Scalar beta = nextRz / rz;
            for (std::size_t i = 0; i < size; ++i) {
                direction[i] = z[i] + beta * direction[i];
            }
        }
        rz = nextRz;
    }

    result.values = std::move(x);
    result.relativeResidual = relativeResidual;
    result.converged = false;
    return result;
}

Scalar maximumAbsolute(const std::vector<Scalar>& values)
{
    Scalar result = 0;
    for (Scalar value : values) {
        if (!finite(value)) {
            throw std::runtime_error("point-projection residual is not finite");
        }
        result = std::max(result, std::abs(value));
    }
    return result;
}

Scalar effectiveControlTolerance(
    const std::vector<PointConstraint>& constraints,
    const ConvergentGriddingOptions& options)
{
    Scalar valueScale = 1;
    for (const PointConstraint& constraint : constraints) {
        valueScale = std::max(valueScale, std::abs(constraint.value));
    }
    // controlTolerance — абсолютный пользовательский допуск. Нижняя граница
    // учитывает точность публичного qreal, потому что именно в него в конце
    // преобразуется grid; внутренняя точность double сама по себе недостаточна.
    const Scalar roundingFloor = Scalar(64)
        * static_cast<Scalar>(std::numeric_limits<qreal>::epsilon()) * valueScale;
    return std::max(static_cast<Scalar>(options.controlTolerance), roundingFloor);
}

// Матрично-свободное умножение (C*C^T)*v в пространстве контрольных точек:
// сначала scatter nodeWork=C^T*v, затем gather output=C*nodeWork.
void applyConstraintGram(
    const std::vector<PointConstraint>& constraints,
    const std::vector<Scalar>& input,
    std::vector<Scalar>& output,
    std::vector<Scalar>& nodeWork)
{
    std::fill(nodeWork.begin(), nodeWork.end(), Scalar(0));
    for (std::size_t i = 0; i < constraints.size(); ++i) {
        for (std::size_t termIndex = 0;
             termIndex < constraints[i].count; ++termIndex) {
            const StencilTerm& term = constraints[i].terms[termIndex];
            nodeWork[term.index] += term.coefficient * input[i];
        }
    }
    for (std::size_t i = 0; i < constraints.size(); ++i) {
        output[i] = constraintValue(constraints[i], nodeWork);
        if (!finite(output[i])) {
            throw std::runtime_error("point-constraint Gram product overflowed");
        }
    }
}

// После гладкого решения ищет минимальную по узловой L2-норме поправку delta:
//
//   C*(u + delta) = d,       min ||delta||_2,
//   delta = C^T*lambda,
//   (C*C^T)*lambda = d - C*u.
//
// Последняя система решается Jacobi-PCG в пространстве точек; диагональ
// предобуславливателя равна квадрату нормы строки C_p. В hard-фазе weight не
// масштабирует отдельное равенство: после объединения совпадающих точек все
// оставшиеся C_p*u=value_p обязательны одинаково.
//
// Это евклидова проекция, НЕ equality-constrained minimum-curvature решение:
// после поправки повторного сглаживания нет, а ее поддержка ограничена узлами
// билинейных строк C. Зависимые либо несовместимые строки приводят к ошибке.
ProjectionResult projectOntoPointConstraints(
    Grid& grid,
    const std::vector<PointConstraint>& constraints,
    const ConvergentGriddingOptions& options)
{
    ProjectionResult result;
    result.converged = true;
    if (constraints.empty()) return result;

    const std::size_t count = constraints.size();
    const Scalar tolerance = effectiveControlTolerance(constraints, options);
    std::vector<Scalar> rhs(count), residual(count), inverseDiagonal(count);
    for (std::size_t i = 0; i < count; ++i) {
        rhs[i] = constraints[i].value
            - constraintValue(constraints[i], grid.values);
        Scalar diagonal = 0;
        for (std::size_t termIndex = 0;
             termIndex < constraints[i].count; ++termIndex) {
            const Scalar coefficient = constraints[i].terms[termIndex].coefficient;
            diagonal += coefficient * coefficient;
        }
        if (!finite(rhs[i]) || !finite(diagonal) || !(diagonal > 0)) {
            throw std::runtime_error("invalid exact point constraint");
        }
        inverseDiagonal[i] = Scalar(1) / diagonal;
    }

    result.maxError = maximumAbsolute(rhs);
    if (result.maxError <= tolerance) return result;

    std::vector<Scalar> multiplier(count, Scalar(0));
    residual = rhs;
    std::vector<Scalar> preconditioned(count), direction(count), gramDirection(count);
    std::vector<Scalar> gramMultiplier(count), nodeWork(grid.values.size(), Scalar(0));
    for (std::size_t i = 0; i < count; ++i) {
        preconditioned[i] = residual[i] * inverseDiagonal[i];
    }
    direction = preconditioned;
    Scalar residualPreconditioned = dot(residual, preconditioned);

    result.converged = false;
    for (std::size_t iteration = 0;
         iteration < options.maxControlProjectionIterations; ++iteration) {
        applyConstraintGram(constraints, direction, gramDirection, nodeWork);
        const Scalar denominator = dot(direction, gramDirection);
        if (!(denominator > 0) || !finite(denominator)
            || !(residualPreconditioned > 0)
            || !finite(residualPreconditioned)) {
            throw std::runtime_error(
                "exact control equations are dependent or incompatible at this grid resolution");
        }
        const Scalar alpha = residualPreconditioned / denominator;
        if (!finite(alpha)) {
            throw std::runtime_error("exact-control projection produced a non-finite step");
        }
        for (std::size_t i = 0; i < count; ++i) {
            multiplier[i] += alpha * direction[i];
            residual[i] -= alpha * gramDirection[i];
        }

        const bool restartDirection = (iteration + 1) % 25 == 0;
        if (restartDirection || maximumAbsolute(residual) <= tolerance) {
            applyConstraintGram(constraints, multiplier, gramMultiplier, nodeWork);
            for (std::size_t i = 0; i < count; ++i) {
                residual[i] = rhs[i] - gramMultiplier[i];
            }
        }

        result.iterations = iteration + 1;
        result.maxError = maximumAbsolute(residual);
        if (result.maxError <= tolerance) {
            result.converged = true;
            break;
        }

        for (std::size_t i = 0; i < count; ++i) {
            preconditioned[i] = residual[i] * inverseDiagonal[i];
        }
        const Scalar nextResidualPreconditioned = dot(residual, preconditioned);
        if (!(nextResidualPreconditioned > 0)
            || !finite(nextResidualPreconditioned)) {
            throw std::runtime_error(
                "exact control equations are dependent or incompatible at this grid resolution");
        }
        if (restartDirection) {
            direction = preconditioned;
        } else {
            const Scalar beta = nextResidualPreconditioned / residualPreconditioned;
            for (std::size_t i = 0; i < count; ++i) {
                direction[i] = preconditioned[i] + beta * direction[i];
            }
        }
        residualPreconditioned = nextResidualPreconditioned;
    }

    if (!result.converged) {
        throw std::runtime_error(
            "exact control projection did not converge; controls may be incompatible "
            "at this grid resolution");
    }

    std::fill(nodeWork.begin(), nodeWork.end(), Scalar(0));
    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t termIndex = 0;
             termIndex < constraints[i].count; ++termIndex) {
            const StencilTerm& term = constraints[i].terms[termIndex];
            nodeWork[term.index] += term.coefficient * multiplier[i];
        }
    }
    for (std::size_t i = 0; i < grid.values.size(); ++i) {
        grid.values[i] += nodeWork[i];
        if (!finite(grid.values[i])) {
            throw std::runtime_error("exact-control correction produced a non-finite grid value");
        }
    }

    result.maxError = 0;
    for (const PointConstraint& constraint : constraints) {
        result.maxError = std::max(result.maxError,
            std::abs(constraintValue(constraint, grid.values) - constraint.value));
    }
    if (result.maxError > tolerance) {
        throw std::runtime_error(
            "exact-control correction lost accuracy while updating the grid");
    }
    return result;
}

Scalar maxControlError(const Grid& grid, const std::vector<CanonicalPoint>& points)
{
    Scalar result = 0;
    for (const CanonicalPoint& point : points) {
        result = std::max(result, std::abs(bilinearSample(grid, point.x, point.y) - point.value));
    }
    return result;
}

} // namespace

ConvergentGriddingReport convergentGridding(
    Surface& surface,
    const std::vector<Point>& points,
    const ConvergentGriddingOptions& options)
{
    const std::vector<CanonicalPoint> controls = validateInput(surface, points, options);
    ConvergentGriddingReport report;
    if (controls.empty()) {
        // Валидная поверхность без ненулевых контрольных точек не изменяется.
        return report;
    }

    // Самый грубый prior получается ресемплированием входной surface.grid.
    // Отдельный тренд по точкам здесь не строится. Иерархия содержит общие
    // min/max и заканчивается строго исходными surface.nx/surface.ny.
    const Grid input = surfaceAsGrid(surface);
    const auto hierarchy = buildHierarchy(surface.nx, surface.ny, options);
    Grid coarsest = resample(input, hierarchy.front().first, hierarchy.front().second);
    Grid solved = coarsest;

    for (std::size_t levelIndex = 0; levelIndex < hierarchy.size(); ++levelIndex) {
        const auto [nx, ny] = hierarchy[levelIndex];
        // Refine: первый prior уже имеет нужный грубый размер; далее решение
        // предыдущего уровня билинейно переносится на более частую сетку.
        Grid prior = levelIndex == 0 ? coarsest : resample(solved, nx, ny);
        const bool finalLevel = levelIndex + 1 == hierarchy.size();
        const std::size_t snapNodes = snapNodeCount(prior, coarsest, finalLevel, options);

        // Snap: контрольные значения Taylor-проецируются в ближайшие узлы и
        // превращаются в диагональные мягкие штрафы текущего уровня.
        std::vector<Scalar> snapDiagonal(prior.values.size(), Scalar(0));
        std::vector<Scalar> snapRhs(prior.values.size(), Scalar(0));
        addSnapConstraints(prior, controls, snapNodes, options, snapDiagonal, snapRhs);

        // На финальном уровне добавляется мягкая билинейная привязка C_p*u=d_p.
        // Она уменьшает величину последующей exact-поправки, но не заменяет ее.
        const std::vector<PointConstraint> finalConstraints = finalLevel
            ? makePointConstraints(prior, controls)
            : std::vector<PointConstraint>{};

        // Smooth: PCG балансирует кривизну, сохранение prior, Snap и (только на
        // последнем уровне) билинейные ограничения. При разрешенной
        // несходимости последний iterate все равно передается на следующий этап.
        SolverResult level = solveLevel(prior, prior.values, snapDiagonal, snapRhs,
                                        finalConstraints, options);
        report.levels.push_back({nx, ny, snapNodes, level.iterations,
                                 static_cast<qreal>(level.relativeResidual), level.converged});
        if (!level.converged && options.throwOnNonConvergence) {
            throw std::runtime_error("biharmonic PCG did not converge at grid level "
                + std::to_string(nx) + "x" + std::to_string(ny));
        }
        solved = prior;
        solved.values = std::move(level.values);
    }

    // Soft-решение обычно лишь приближенно выполняет C*u=d. Опциональная
    // проекция доводит билинейные значения до exact-допуска отдельным решением
    // в пространстве контрольных точек.
    const std::vector<PointConstraint> outputConstraints =
        makePointConstraints(solved, controls);
    if (options.enforceExactControls) {
        const ProjectionResult projection = projectOntoPointConstraints(
            solved, outputConstraints, options);
        report.controlProjectionIterations = projection.iterations;
    }

    // Публичный qreal может быть уже внутреннего double, поэтому преобразование
    // выполняется во временный буфер и проверяется до изменения surface.grid.
    std::vector<qreal> output(solved.values.size());
    std::transform(solved.values.begin(), solved.values.end(), output.begin(),
        [](Scalar value) {
            if (!finite(value)) {
                throw std::runtime_error("convergent gridding produced a non-finite result");
            }
            const qreal converted = static_cast<qreal>(value);
            if (!finite(static_cast<Scalar>(converted))) {
                throw std::runtime_error("convergent gridding result does not fit in qreal");
            }
            return converted;
        });

    // Отчет измеряет невязку именно возвращаемых значений, включая возможную
    // потерю точности, когда qreal в Qt-сборке уже внутреннего Scalar.
    Grid returned = solved;
    std::transform(output.begin(), output.end(), returned.values.begin(),
        [](qreal value) { return static_cast<Scalar>(value); });
    report.maxControlError = static_cast<qreal>(maxControlError(returned, controls));
    const Scalar acceptedControlError = effectiveControlTolerance(
        outputConstraints, options);
    report.controlsSatisfied = static_cast<Scalar>(report.maxControlError)
        <= acceptedControlError;
    if (options.enforceExactControls && !report.controlsSatisfied) {
        throw std::runtime_error(
            "qreal precision is insufficient to retain the exact control-point correction");
    }

    // Единственная запись в объект пользователя: все вычисления, exact-проекция,
    // преобразование и проверки уже завершены (strong exception guarantee).
    surface.grid = std::move(output);
    return report;
}

Surface convergentGriddedSurface(
    const Surface& surface,
    const std::vector<Point>& points,
    const ConvergentGriddingOptions& options,
    ConvergentGriddingReport* report)
{
    Surface result = surface;
    ConvergentGriddingReport localReport = convergentGridding(result, points, options);
    if (report != nullptr) *report = std::move(localReport);
    return result;
}

} // namespace convergent
