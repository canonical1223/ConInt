#include "convergent_interpolation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace geo::convergent {
namespace {

using Vector = std::vector<qreal>;

struct GridGeometry {
    qreal minx = 0;
    qreal maxx = 0;
    qreal miny = 0;
    qreal maxy = 0;
    std::size_t nx = 0;
    std::size_t ny = 0;

    [[nodiscard]] qreal dx() const
    {
        return (maxx - minx) / static_cast<qreal>(nx - 1);
    }

    [[nodiscard]] qreal dy() const
    {
        return (maxy - miny) / static_cast<qreal>(ny - 1);
    }
};

struct PointStencil {
    std::array<std::size_t, 4> index{};
    std::array<qreal, 4> coefficient{};
    qreal value = 0;
    qreal weight = 0;
};

struct Plane {
    qreal a = 0;
    qreal bx = 0;
    qreal by = 0;
    qreal cx = 0;
    qreal cy = 0;
    qreal sx = 1;
    qreal sy = 1;

    [[nodiscard]] qreal evaluate(qreal x, qreal y) const noexcept
    {
        return a + bx * ((x - cx) / sx) + by * ((y - cy) / sy);
    }
};

[[nodiscard]] bool finite(qreal value)
{
    return std::isfinite(static_cast<double>(value));
}

void validateSurface(const GridGeometry& surface)
{
    if (surface.nx < 2 || surface.ny < 2) {
        throw std::invalid_argument(
            "convergent::interpolate: nx and ny must be at least 2 (node counts)");
    }
    if (!finite(surface.minx) || !finite(surface.maxx)
        || !finite(surface.miny) || !finite(surface.maxy)
        || !(surface.maxx > surface.minx) || !(surface.maxy > surface.miny)) {
        throw std::invalid_argument(
            "convergent::interpolate: surface bounds must be finite and non-degenerate");
    }
    if (!finite(surface.maxx - surface.minx)
        || !finite(surface.maxy - surface.miny)) {
        throw std::invalid_argument(
            "convergent::interpolate: surface coordinate span is too large");
    }
    if (surface.nx > std::numeric_limits<std::size_t>::max() / surface.ny) {
        throw std::overflow_error("convergent::interpolate: surface is too large");
    }
}

void validateOptions(const Options& options)
{
    if (options.coarsestGridSize < 2) {
        throw std::invalid_argument(
            "convergent::interpolate: coarsestGridSize must be at least 2");
    }
    if (options.maxIterationsPerLevel == 0) {
        throw std::invalid_argument(
            "convergent::interpolate: maxIterationsPerLevel must be positive");
    }
    if (options.enforcePointConstraints && options.maxConstraintIterations == 0) {
        throw std::invalid_argument(
            "convergent::interpolate: maxConstraintIterations must be positive");
    }
    if (options.enforcePointConstraints
        && (options.maxExactProjectionIterations == 0
            || options.maxExactProjectionPasses == 0)) {
        throw std::invalid_argument(
            "convergent::interpolate: exact projection limits must be positive");
    }
    if (!finite(options.relativeTolerance) || options.relativeTolerance <= 0
        || !finite(options.smoothness) || options.smoothness <= 0
        || !finite(options.dataWeight) || options.dataWeight <= 0
        || !finite(options.regularization) || options.regularization <= 0
        || !finite(options.constraintPenaltyGrowth)
        || options.constraintPenaltyGrowth < 1
        || !finite(options.pointConstraintRelativeTolerance)
        || options.pointConstraintRelativeTolerance <= 0) {
        throw std::invalid_argument(
            "convergent::interpolate: numerical options must be finite and positive");
    }
}

[[nodiscard]] qreal coordinateTolerance(qreal lo, qreal hi)
{
    const qreal scale = std::max<qreal>(
        1, std::max(std::abs(lo), std::max(std::abs(hi), std::abs(hi - lo))));
    return 64 * std::numeric_limits<qreal>::epsilon() * scale;
}

[[nodiscard]] std::vector<Sample> validateAndFilterPoints(
    const GridGeometry& surface,
    const std::vector<Sample>& points,
    bool ignoreOutsidePoints)
{
    std::vector<Sample> result;
    result.reserve(points.size());

    const qreal xtol = coordinateTolerance(surface.minx, surface.maxx);
    const qreal ytol = coordinateTolerance(surface.miny, surface.maxy);

    for (const Sample& point : points) {
        if (!finite(point.x) || !finite(point.y) || !finite(point.value)
            || !finite(point.weight)) {
            throw std::invalid_argument(
                "convergent::interpolate: point coordinates, value and weight must be finite");
        }
        if (point.weight < 0) {
            throw std::invalid_argument(
                "convergent::interpolate: point weight must not be negative");
        }
        if (point.weight == 0) {
            continue;
        }

        const bool outside = point.x < surface.minx - xtol
            || point.x > surface.maxx + xtol
            || point.y < surface.miny - ytol
            || point.y > surface.maxy + ytol;
        if (outside) {
            if (ignoreOutsidePoints) {
                continue;
            }
            throw std::out_of_range(
                "convergent::interpolate: a conditioning point lies outside the surface");
        }

        Sample clamped = point;
        clamped.x = std::clamp(clamped.x, surface.minx, surface.maxx);
        clamped.y = std::clamp(clamped.y, surface.miny, surface.maxy);
        result.push_back(clamped);
    }

    if (result.empty()) {
        throw std::invalid_argument(
            "convergent::interpolate: no positive-weight points inside the surface");
    }

    // All reductions below are intentionally performed in a canonical order.
    // This makes the result independent of the caller's point order (apart
    // from the unavoidable last bits of platform floating-point arithmetic).
    std::sort(result.begin(), result.end(), [](const Sample& left, const Sample& right) {
        if (left.x != right.x) {
            return left.x < right.x;
        }
        if (left.y != right.y) {
            return left.y < right.y;
        }
        if (left.value != right.value) {
            return left.value < right.value;
        }
        return left.weight < right.weight;
    });
    qreal maximumWeight = 0;
    for (const Sample& point : result) {
        maximumWeight = std::max(maximumWeight, point.weight);
    }
    for (Sample& point : result) {
        point.weight /= maximumWeight;
    }
    return result;
}

[[nodiscard]] bool solve3x3(
    std::array<std::array<qreal, 3>, 3> matrix,
    std::array<qreal, 3> rhs,
    std::array<qreal, 3>& solution)
{
    qreal largest = 0;
    for (const auto& row : matrix) {
        for (qreal value : row) {
            largest = std::max(largest, std::abs(value));
        }
    }
    const qreal pivotTolerance = std::max<qreal>(1, largest)
        * 128 * std::numeric_limits<qreal>::epsilon();

    for (std::size_t column = 0; column < 3; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < 3; ++row) {
            if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) {
                pivot = row;
            }
        }
        if (std::abs(matrix[pivot][column]) <= pivotTolerance) {
            return false;
        }
        if (pivot != column) {
            std::swap(matrix[pivot], matrix[column]);
            std::swap(rhs[pivot], rhs[column]);
        }

        for (std::size_t row = column + 1; row < 3; ++row) {
            const qreal factor = matrix[row][column] / matrix[column][column];
            matrix[row][column] = 0;
            for (std::size_t k = column + 1; k < 3; ++k) {
                matrix[row][k] -= factor * matrix[column][k];
            }
            rhs[row] -= factor * rhs[column];
        }
    }

    for (std::size_t reverse = 0; reverse < 3; ++reverse) {
        const std::size_t row = 2 - reverse;
        qreal value = rhs[row];
        for (std::size_t column = row + 1; column < 3; ++column) {
            value -= matrix[row][column] * solution[column];
        }
        solution[row] = value / matrix[row][row];
    }
    return true;
}

[[nodiscard]] Plane fitPlane(
    const GridGeometry& surface,
    const std::vector<Sample>& points)
{
    Plane plane;
    plane.cx = surface.minx + (surface.maxx - surface.minx) / 2;
    plane.cy = surface.miny + (surface.maxy - surface.miny) / 2;
    plane.sx = surface.maxx - surface.minx;
    plane.sy = surface.maxy - surface.miny;

    std::array<std::array<qreal, 3>, 3> normal{};
    std::array<qreal, 3> rhs{};
    qreal totalWeight = 0;
    qreal weightedValue = 0;

    for (const Sample& point : points) {
        const std::array<qreal, 3> row{
            1, (point.x - plane.cx) / plane.sx, (point.y - plane.cy) / plane.sy};
        totalWeight += point.weight;
        weightedValue += point.weight * point.value;
        for (std::size_t i = 0; i < 3; ++i) {
            rhs[i] += point.weight * row[i] * point.value;
            for (std::size_t j = 0; j < 3; ++j) {
                normal[i][j] += point.weight * row[i] * row[j];
            }
        }
    }

    // Регуляризуются только уклоны. Свободный член остаётся взвешенным
    // средним даже для единственной точки.
    const qreal slopeRidge = std::max<qreal>(1, totalWeight) * 1e-12;
    normal[1][1] += slopeRidge;
    normal[2][2] += slopeRidge;

    std::array<qreal, 3> parameters{};
    if (solve3x3(normal, rhs, parameters)) {
        plane.a = parameters[0];
        plane.bx = parameters[1];
        plane.by = parameters[2];
    } else {
        plane.a = weightedValue / totalWeight;
    }
    return plane;
}

[[nodiscard]] Vector evaluatePlane(const GridGeometry& grid, const Plane& plane)
{
    Vector values(grid.nx * grid.ny);
    const qreal dx = grid.dx();
    const qreal dy = grid.dy();
    for (std::size_t iy = 0; iy < grid.ny; ++iy) {
        const qreal y = grid.miny + static_cast<qreal>(iy) * dy;
        for (std::size_t ix = 0; ix < grid.nx; ++ix) {
            const qreal x = grid.minx + static_cast<qreal>(ix) * dx;
            values[surfaceIndex(ix, iy, grid.nx)] = plane.evaluate(x, y);
        }
    }
    return values;
}

[[nodiscard]] std::size_t coarsenNodeCount(std::size_t count)
{
    return count <= 2 ? count : count / 2 + 1;
}

[[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> buildLevels(
    std::size_t nx,
    std::size_t ny,
    const Options& options)
{
    std::vector<std::pair<std::size_t, std::size_t>> levels;
    levels.emplace_back(nx, ny);

    while ((levels.back().first > options.coarsestGridSize
               || levels.back().second > options.coarsestGridSize)
           && (options.maxLevels == 0 || levels.size() < options.maxLevels)) {
        const auto [currentNx, currentNy] = levels.back();
        const std::size_t coarseNx = coarsenNodeCount(currentNx);
        const std::size_t coarseNy = coarsenNodeCount(currentNy);
        if (coarseNx == currentNx && coarseNy == currentNy) {
            break;
        }
        levels.emplace_back(coarseNx, coarseNy);
    }

    std::reverse(levels.begin(), levels.end());
    return levels;
}

[[nodiscard]] Vector resample(
    const Vector& source,
    std::size_t sourceNx,
    std::size_t sourceNy,
    std::size_t targetNx,
    std::size_t targetNy)
{
    const auto hermite = [](qreal left,
                             qreal right,
                             qreal leftSlope,
                             qreal rightSlope,
                             qreal fraction) {
        const qreal fraction2 = fraction * fraction;
        const qreal fraction3 = fraction2 * fraction;
        return (2 * fraction3 - 3 * fraction2 + 1) * left
            + (fraction3 - 2 * fraction2 + fraction) * leftSlope
            + (-2 * fraction3 + 3 * fraction2) * right
            + (fraction3 - fraction2) * rightSlope;
    };

    // Cubic Hermite prolongation transfers the first derivatives of the
    // previous level instead of recreating a piecewise-linear surface.  The
    // one-sided boundary slopes make affine trends exactly reproducible.
    const auto rowSlope = [&](std::size_t x, std::size_t y) {
        if (sourceNx == 2) {
            return source[surfaceIndex(1, y, sourceNx)]
                - source[surfaceIndex(0, y, sourceNx)];
        }
        if (x == 0) {
            return (-3 * source[surfaceIndex(0, y, sourceNx)]
                       + 4 * source[surfaceIndex(1, y, sourceNx)]
                       - source[surfaceIndex(2, y, sourceNx)])
                / 2;
        }
        if (x + 1 == sourceNx) {
            return (3 * source[surfaceIndex(x, y, sourceNx)]
                       - 4 * source[surfaceIndex(x - 1, y, sourceNx)]
                       + source[surfaceIndex(x - 2, y, sourceNx)])
                / 2;
        }
        return (source[surfaceIndex(x + 1, y, sourceNx)]
                   - source[surfaceIndex(x - 1, y, sourceNx)])
            / 2;
    };

    Vector refinedX(targetNx * sourceNy);
    for (std::size_t y = 0; y < sourceNy; ++y) {
        for (std::size_t targetX = 0; targetX < targetNx; ++targetX) {
            const qreal sourceX = static_cast<qreal>(targetX)
                * static_cast<qreal>(sourceNx - 1)
                / static_cast<qreal>(targetNx - 1);
            const std::size_t x0 = std::min<std::size_t>(
                static_cast<std::size_t>(std::floor(sourceX)), sourceNx - 2);
            const std::size_t x1 = x0 + 1;
            const qreal fraction = sourceX - static_cast<qreal>(x0);
            refinedX[surfaceIndex(targetX, y, targetNx)] = hermite(
                source[surfaceIndex(x0, y, sourceNx)],
                source[surfaceIndex(x1, y, sourceNx)],
                rowSlope(x0, y),
                rowSlope(x1, y),
                fraction);
        }
    }

    const auto columnSlope = [&](std::size_t x, std::size_t y) {
        if (sourceNy == 2) {
            return refinedX[surfaceIndex(x, 1, targetNx)]
                - refinedX[surfaceIndex(x, 0, targetNx)];
        }
        if (y == 0) {
            return (-3 * refinedX[surfaceIndex(x, 0, targetNx)]
                       + 4 * refinedX[surfaceIndex(x, 1, targetNx)]
                       - refinedX[surfaceIndex(x, 2, targetNx)])
                / 2;
        }
        if (y + 1 == sourceNy) {
            return (3 * refinedX[surfaceIndex(x, y, targetNx)]
                       - 4 * refinedX[surfaceIndex(x, y - 1, targetNx)]
                       + refinedX[surfaceIndex(x, y - 2, targetNx)])
                / 2;
        }
        return (refinedX[surfaceIndex(x, y + 1, targetNx)]
                   - refinedX[surfaceIndex(x, y - 1, targetNx)])
            / 2;
    };

    Vector target(targetNx * targetNy);
    for (std::size_t targetY = 0; targetY < targetNy; ++targetY) {
        const qreal sourceY = static_cast<qreal>(targetY)
            * static_cast<qreal>(sourceNy - 1)
            / static_cast<qreal>(targetNy - 1);
        const std::size_t y0 = std::min<std::size_t>(
            static_cast<std::size_t>(std::floor(sourceY)), sourceNy - 2);
        const std::size_t y1 = y0 + 1;
        const qreal fraction = sourceY - static_cast<qreal>(y0);
        for (std::size_t x = 0; x < targetNx; ++x) {
            target[surfaceIndex(x, targetY, targetNx)] = hermite(
                refinedX[surfaceIndex(x, y0, targetNx)],
                refinedX[surfaceIndex(x, y1, targetNx)],
                columnSlope(x, y0),
                columnSlope(x, y1),
                fraction);
        }
    }
    return target;
}

[[nodiscard]] std::array<qreal, 4> interpolationCoefficients(
    qreal fx,
    qreal fy,
    CellInterpolation interpolation)
{
    switch (interpolation) {
    case CellInterpolation::Bilinear:
        return {
            (1 - fx) * (1 - fy),
            fx * (1 - fy),
            (1 - fx) * fy,
            fx * fy,
        };

    case CellInterpolation::TriangleBottomLeftToTopRight:
        if (fy <= fx) {
            // Треугольник: bottom-left, bottom-right, top-right.
            return {1 - fx, fx - fy, 0, fy};
        }
        // Треугольник: bottom-left, top-left, top-right.
        return {1 - fy, 0, fy - fx, fx};

    case CellInterpolation::TriangleBottomRightToTopLeft:
        if (fx + fy <= 1) {
            // Треугольник: bottom-left, bottom-right, top-left.
            return {1 - fx - fy, fx, fy, 0};
        }
        // Треугольник: bottom-right, top-left, top-right.
        return {0, 1 - fy, 1 - fx, fx + fy - 1};
    }
    throw std::invalid_argument("convergent::interpolate: unknown cell interpolation");
}

[[nodiscard]] PointStencil makePointStencil(
    const GridGeometry& grid,
    const Sample& point,
    CellInterpolation interpolation)
{
    const qreal dx = grid.dx();
    const qreal dy = grid.dy();
    const qreal gx = (point.x - grid.minx) / dx;
    const qreal gy = (point.y - grid.miny) / dy;
    const std::size_t ix = std::min<std::size_t>(
        static_cast<std::size_t>(std::floor(gx)), grid.nx - 2);
    const std::size_t iy = std::min<std::size_t>(
        static_cast<std::size_t>(std::floor(gy)), grid.ny - 2);
    const qreal fx = std::clamp(gx - static_cast<qreal>(ix), qreal(0), qreal(1));
    const qreal fy = std::clamp(gy - static_cast<qreal>(iy), qreal(0), qreal(1));

    PointStencil stencil;
    stencil.index = {
        surfaceIndex(ix, iy, grid.nx),
        surfaceIndex(ix + 1, iy, grid.nx),
        surfaceIndex(ix, iy + 1, grid.nx),
        surfaceIndex(ix + 1, iy + 1, grid.nx),
    };
    stencil.coefficient = interpolationCoefficients(fx, fy, interpolation);
    stencil.value = point.value;
    stencil.weight = point.weight;
    return stencil;
}

[[nodiscard]] std::vector<PointStencil> makePointStencils(
    const GridGeometry& grid,
    const std::vector<Sample>& points,
    CellInterpolation interpolation)
{
    std::vector<PointStencil> stencils;
    stencils.reserve(points.size());
    for (const Sample& point : points) {
        stencils.push_back(makePointStencil(grid, point, interpolation));
    }
    return stencils;
}

struct SurfaceDerivatives {
    Vector x;
    Vector y;
    Vector xx;
    Vector xy;
    Vector yy;
};

[[nodiscard]] SurfaceDerivatives surfaceDerivatives(
    const GridGeometry& grid,
    const Vector& values)
{
    SurfaceDerivatives result{
        Vector(values.size(), 0),
        Vector(values.size(), 0),
        Vector(values.size(), 0),
        Vector(values.size(), 0),
        Vector(values.size(), 0),
    };

    const auto value = [&](std::size_t x, std::size_t y) -> qreal {
        return values[surfaceIndex(x, y, grid.nx)];
    };
    for (std::size_t y = 0; y < grid.ny; ++y) {
        for (std::size_t x = 0; x < grid.nx; ++x) {
            const std::size_t index = surfaceIndex(x, y, grid.nx);
            if (grid.nx == 2) {
                result.x[index] = value(1, y) - value(0, y);
            } else if (x == 0) {
                result.x[index] = (-3 * value(0, y) + 4 * value(1, y)
                                      - value(2, y))
                    / 2;
                result.xx[index] = value(0, y) - 2 * value(1, y) + value(2, y);
            } else if (x + 1 == grid.nx) {
                result.x[index] = (3 * value(x, y) - 4 * value(x - 1, y)
                                      + value(x - 2, y))
                    / 2;
                result.xx[index] = value(x, y) - 2 * value(x - 1, y)
                    + value(x - 2, y);
            } else {
                result.x[index] = (value(x + 1, y) - value(x - 1, y)) / 2;
                result.xx[index] = value(x - 1, y) - 2 * value(x, y)
                    + value(x + 1, y);
            }

            if (grid.ny == 2) {
                result.y[index] = value(x, 1) - value(x, 0);
            } else if (y == 0) {
                result.y[index] = (-3 * value(x, 0) + 4 * value(x, 1)
                                      - value(x, 2))
                    / 2;
                result.yy[index] = value(x, 0) - 2 * value(x, 1) + value(x, 2);
            } else if (y + 1 == grid.ny) {
                result.y[index] = (3 * value(x, y) - 4 * value(x, y - 1)
                                      + value(x, y - 2))
                    / 2;
                result.yy[index] = value(x, y) - 2 * value(x, y - 1)
                    + value(x, y - 2);
            } else {
                result.y[index] = (value(x, y + 1) - value(x, y - 1)) / 2;
                result.yy[index] = value(x, y - 1) - 2 * value(x, y)
                    + value(x, y + 1);
            }
        }
    }

    // Differentiating the already computed x-slope is more stable at the
    // boundary than a separately special-cased four-corner expression.
    for (std::size_t y = 0; y < grid.ny; ++y) {
        for (std::size_t x = 0; x < grid.nx; ++x) {
            const std::size_t index = surfaceIndex(x, y, grid.nx);
            if (grid.ny == 2) {
                result.xy[index] = result.x[surfaceIndex(x, 1, grid.nx)]
                    - result.x[surfaceIndex(x, 0, grid.nx)];
            } else if (y == 0) {
                result.xy[index] =
                    (-3 * result.x[surfaceIndex(x, 0, grid.nx)]
                        + 4 * result.x[surfaceIndex(x, 1, grid.nx)]
                        - result.x[surfaceIndex(x, 2, grid.nx)])
                    / 2;
            } else if (y + 1 == grid.ny) {
                result.xy[index] =
                    (3 * result.x[surfaceIndex(x, y, grid.nx)]
                        - 4 * result.x[surfaceIndex(x, y - 1, grid.nx)]
                        + result.x[surfaceIndex(x, y - 2, grid.nx)])
                    / 2;
            } else {
                result.xy[index] =
                    (result.x[surfaceIndex(x, y + 1, grid.nx)]
                        - result.x[surfaceIndex(x, y - 1, grid.nx)])
                    / 2;
            }
        }
    }
    return result;
}

[[nodiscard]] qreal evaluateArray(
    const PointStencil& stencil,
    const Vector& values)
{
    qreal result = 0;
    for (std::size_t local = 0; local < 4; ++local) {
        result += stencil.coefficient[local] * values[stencil.index[local]];
    }
    return result;
}

[[nodiscard]] std::size_t snapNodeCount(
    std::size_t levelIndex,
    std::size_t levelCount)
{
    const std::size_t remaining = levelCount - levelIndex - 1;
    if (remaining == 0) {
        return 1;
    }
    // Historical convergent gridders reduce the support approximately as
    // 16 -> 8 -> 4 -> 1 while the grid interval is refined.
    const std::size_t exponent = std::min<std::size_t>(4, remaining + 1);
    return std::size_t(1) << exponent;
}

[[nodiscard]] std::vector<PointStencil> makeSnapStencils(
    const GridGeometry& grid,
    const std::vector<Sample>& points,
    CellInterpolation interpolation,
    const Vector& predicted,
    std::size_t levelIndex,
    std::size_t levelCount)
{
    struct Candidate {
        std::size_t index = 0;
        qreal offsetX = 0;
        qreal offsetY = 0;
        qreal squaredDistance = 0;
    };

    const SurfaceDerivatives derivatives = surfaceDerivatives(grid, predicted);
    Vector weightedTarget(predicted.size(), 0);
    Vector totalWeight(predicted.size(), 0);
    const std::size_t requestedNodeCount = snapNodeCount(levelIndex, levelCount);

    for (const Sample& point : points) {
        const qreal gridX = (point.x - grid.minx) / grid.dx();
        const qreal gridY = (point.y - grid.miny) / grid.dy();
        const auto baseX = static_cast<std::ptrdiff_t>(std::floor(gridX));
        const auto baseY = static_cast<std::ptrdiff_t>(std::floor(gridY));

        std::vector<Candidate> candidates;
        candidates.reserve(49);
        for (std::ptrdiff_t offsetY = -3; offsetY <= 3; ++offsetY) {
            const std::ptrdiff_t nodeY = baseY + offsetY;
            if (nodeY < 0 || nodeY >= static_cast<std::ptrdiff_t>(grid.ny)) {
                continue;
            }
            for (std::ptrdiff_t offsetX = -3; offsetX <= 3; ++offsetX) {
                const std::ptrdiff_t nodeX = baseX + offsetX;
                if (nodeX < 0 || nodeX >= static_cast<std::ptrdiff_t>(grid.nx)) {
                    continue;
                }
                const qreal dx = static_cast<qreal>(nodeX) - gridX;
                const qreal dy = static_cast<qreal>(nodeY) - gridY;
                candidates.push_back(Candidate{
                    surfaceIndex(
                        static_cast<std::size_t>(nodeX),
                        static_cast<std::size_t>(nodeY),
                        grid.nx),
                    dx,
                    dy,
                    dx * dx + dy * dy,
                });
            }
        }
        std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& left, const Candidate& right) {
                if (left.squaredDistance != right.squaredDistance) {
                    return left.squaredDistance < right.squaredDistance;
                }
                return left.index < right.index;
            });
        if (candidates.size() > requestedNodeCount) {
            candidates.resize(requestedNodeCount);
        }

        const PointStencil displayedStencil = makePointStencil(
            grid, point, interpolation);
        const PointStencil derivativeStencil = makePointStencil(
            grid, point, CellInterpolation::Bilinear);
        const qreal predictedAtPoint = evaluateArray(displayedStencil, predicted);
        const qreal slopeX = evaluateArray(derivativeStencil, derivatives.x);
        const qreal slopeY = evaluateArray(derivativeStencil, derivatives.y);
        const qreal curvatureX = evaluateArray(derivativeStencil, derivatives.xx);
        const qreal curvatureXY = evaluateArray(derivativeStencil, derivatives.xy);
        const qreal curvatureY = evaluateArray(derivativeStencil, derivatives.yy);

        qreal radialWeightSum = 0;
        for (const Candidate& candidate : candidates) {
            radialWeightSum += 1 / (qreal(0.25) + candidate.squaredDistance);
        }
        for (const Candidate& candidate : candidates) {
            const qreal radialWeight =
                (1 / (qreal(0.25) + candidate.squaredDistance))
                / radialWeightSum;
            const qreal taylorDifference = slopeX * candidate.offsetX
                + slopeY * candidate.offsetY
                + qreal(0.5)
                    * (curvatureX * candidate.offsetX * candidate.offsetX
                        + 2 * curvatureXY * candidate.offsetX * candidate.offsetY
                        + curvatureY * candidate.offsetY * candidate.offsetY);
            qreal projectedValue = point.value + taylorDifference;

            // When finite-difference curvature is locally noisy, the Taylor
            // term is allowed to change the existing point-to-node difference
            // by at most the current point residual.  This keeps extrapolation
            // stable without clipping genuine input values.
            const qreal currentDifference =
                predicted[candidate.index] - predictedAtPoint;
            const qreal pointResidual = point.value - predictedAtPoint;
            const qreal taylorAdjustment = projectedValue
                - (point.value + currentDifference);
            const qreal adjustmentLimit = std::max<qreal>(
                std::abs(pointResidual),
                64 * std::numeric_limits<qreal>::epsilon()
                    * std::max<qreal>(1, std::abs(point.value)));
            projectedValue = point.value + currentDifference
                + std::clamp(taylorAdjustment, -adjustmentLimit, adjustmentLimit);

            const qreal contributionWeight = point.weight * radialWeight;
            weightedTarget[candidate.index] += contributionWeight * projectedValue;
            totalWeight[candidate.index] += contributionWeight;
        }
    }

    std::vector<PointStencil> stencils;
    for (std::size_t node = 0; node < predicted.size(); ++node) {
        if (!(totalWeight[node] > 0)) {
            continue;
        }
        PointStencil stencil;
        stencil.index = {node, node, node, node};
        stencil.coefficient = {1, 0, 0, 0};
        stencil.value = weightedTarget[node] / totalWeight[node];
        stencil.weight = totalWeight[node];
        stencils.push_back(stencil);
    }
    return stencils;
}

void validateActivePointSupport(
    const std::vector<PointStencil>& stencils,
    const std::vector<std::uint8_t>* activeNodeMask,
    std::size_t expectedNodeCount)
{
    if (activeNodeMask == nullptr) {
        return;
    }
    if (activeNodeMask->size() != expectedNodeCount) {
        throw std::invalid_argument(
            "convergent::interpolate: activeNodeMask size must be nx * ny");
    }

    const qreal coefficientTolerance =
        64 * std::numeric_limits<qreal>::epsilon();
    for (const PointStencil& stencil : stencils) {
        for (std::size_t i = 0; i < 4; ++i) {
            if (stencil.coefficient[i] > coefficientTolerance
                && (*activeNodeMask)[stencil.index[i]] == 0) {
                throw std::out_of_range(
                    "convergent::interpolate: a point is located on an inactive "
                    "or clipped mesh cell");
            }
        }
    }
}

[[nodiscard]] std::size_t matrixRank(
    std::vector<std::vector<qreal>> matrix,
    qreal relativeTolerance)
{
    if (matrix.empty() || matrix.front().empty()) {
        return 0;
    }
    const std::size_t rowCount = matrix.size();
    const std::size_t columnCount = matrix.front().size();

    // Масштабирование столбцов делает сравнение коэффициентов формы и
    // геологических отметок независимым от единиц Z.
    for (std::size_t column = 0; column < columnCount; ++column) {
        qreal scale = 0;
        for (std::size_t row = 0; row < rowCount; ++row) {
            scale = std::max(scale, std::abs(matrix[row][column]));
        }
        if (scale > 0) {
            for (std::size_t row = 0; row < rowCount; ++row) {
                matrix[row][column] /= scale;
            }
        }
    }

    const qreal tolerance = std::max<qreal>(
        256 * std::numeric_limits<qreal>::epsilon(), relativeTolerance);
    std::size_t rank = 0;
    for (std::size_t column = 0;
         column < columnCount && rank < rowCount;
         ++column) {
        std::size_t pivot = rank;
        for (std::size_t row = rank + 1; row < rowCount; ++row) {
            if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) {
                pivot = row;
            }
        }
        if (std::abs(matrix[pivot][column]) <= tolerance) {
            continue;
        }
        std::swap(matrix[pivot], matrix[rank]);
        const qreal pivotValue = matrix[rank][column];
        for (std::size_t j = column; j < columnCount; ++j) {
            matrix[rank][j] /= pivotValue;
        }
        for (std::size_t row = 0; row < rowCount; ++row) {
            if (row == rank) {
                continue;
            }
            const qreal factor = matrix[row][column];
            for (std::size_t j = column; j < columnCount; ++j) {
                matrix[row][j] -= factor * matrix[rank][j];
            }
        }
        ++rank;
    }
    return rank;
}

void validateConstraintCompatibility(
    const std::vector<PointStencil>& stencils,
    qreal relativeTolerance)
{
    const qreal coefficientTolerance =
        64 * std::numeric_limits<qreal>::epsilon();

    // Constraints can be dependent even when their four-node supports are not
    // identical (two end points and a contradictory point on their common
    // edge are the smallest example).  Build connected components of the
    // point/node bipartite graph and compare rank(B) with rank([B|value]).
    std::vector<std::size_t> parent(stencils.size());
    for (std::size_t row = 0; row < parent.size(); ++row) {
        parent[row] = row;
    }
    const auto findRoot = [&](std::size_t row) {
        std::size_t root = row;
        while (parent[root] != root) {
            root = parent[root];
        }
        while (parent[row] != row) {
            const std::size_t next = parent[row];
            parent[row] = root;
            row = next;
        }
        return root;
    };
    const auto join = [&](std::size_t left, std::size_t right) {
        const std::size_t leftRoot = findRoot(left);
        const std::size_t rightRoot = findRoot(right);
        if (leftRoot != rightRoot) {
            parent[rightRoot] = leftRoot;
        }
    };

    std::map<std::size_t, std::size_t> firstRowByNode;
    for (std::size_t row = 0; row < stencils.size(); ++row) {
        for (std::size_t local = 0; local < 4; ++local) {
            if (std::abs(stencils[row].coefficient[local])
                <= coefficientTolerance) {
                continue;
            }
            const std::size_t node = stencils[row].index[local];
            const auto [position, inserted] = firstRowByNode.emplace(node, row);
            if (!inserted) {
                join(row, position->second);
            }
        }
    }

    std::map<std::size_t, std::vector<std::size_t>> components;
    for (std::size_t row = 0; row < stencils.size(); ++row) {
        components[findRoot(row)].push_back(row);
    }
    for (const auto& [root, rows] : components) {
        (void)root;
        std::map<std::size_t, std::size_t> columnByNode;
        for (std::size_t row : rows) {
            for (std::size_t local = 0; local < 4; ++local) {
                if (std::abs(stencils[row].coefficient[local])
                    > coefficientTolerance) {
                    columnByNode.emplace(
                        stencils[row].index[local], columnByNode.size());
                }
            }
        }

        // Dense elimination is deliberately limited to local components.  A
        // large connected seismic data set is normally full-row-rank; if it
        // is not, the metric hard-constraint solver below still diagnoses the
        // inconsistency without allocating a quadratic global matrix.
        if (rows.size() > 512 || columnByNode.size() > 512) {
            continue;
        }
        std::vector<std::vector<qreal>> coefficients(
            rows.size(), std::vector<qreal>(columnByNode.size()));
        std::vector<std::vector<qreal>> augmented(
            rows.size(), std::vector<qreal>(columnByNode.size() + 1));
        for (std::size_t localRow = 0; localRow < rows.size(); ++localRow) {
            const PointStencil& stencil = stencils[rows[localRow]];
            for (std::size_t local = 0; local < 4; ++local) {
                if (std::abs(stencil.coefficient[local])
                    <= coefficientTolerance) {
                    continue;
                }
                const std::size_t column = columnByNode.at(stencil.index[local]);
                coefficients[localRow][column] += stencil.coefficient[local];
                augmented[localRow][column] += stencil.coefficient[local];
            }
            augmented[localRow][columnByNode.size()] = stencil.value;
        }
        if (matrixRank(augmented, relativeTolerance)
            > matrixRank(coefficients, relativeTolerance)) {
            throw std::invalid_argument(
                "convergent::interpolate: point data cannot be represented by "
                "the selected grid resolution and triangle topology; refine the "
                "grid or use a mesh containing the control points as vertices");
        }
    }
}

template <std::size_t Count>
void addQuadraticStencil(
    const std::array<std::size_t, Count>& indices,
    const std::array<qreal, Count>& coefficients,
    qreal scale,
    const Vector& input,
    Vector& output)
{
    qreal residual = 0;
    for (std::size_t i = 0; i < Count; ++i) {
        residual += coefficients[i] * input[indices[i]];
    }
    const qreal scaledResidual = scale * residual;
    for (std::size_t i = 0; i < Count; ++i) {
        output[indices[i]] += coefficients[i] * scaledResidual;
    }
}

template <std::size_t Count>
void addQuadraticStencilDiagonal(
    const std::array<std::size_t, Count>& indices,
    const std::array<qreal, Count>& coefficients,
    qreal scale,
    Vector& diagonal)
{
    for (std::size_t i = 0; i < Count; ++i) {
        diagonal[indices[i]] += scale * coefficients[i] * coefficients[i];
    }
}

void addCurvatureOperator(
    const GridGeometry& grid,
    qreal smoothness,
    const Vector& input,
    Vector& output)
{
    // Нормировка общей длиной делает систему инвариантной к смене единиц
    // координат, но сохраняет отношение масштабов X/Y.
    const qreal referenceLength = std::max(grid.maxx - grid.minx, grid.maxy - grid.miny);
    const qreal dx = grid.dx() / referenceLength;
    const qreal dy = grid.dy() / referenceLength;
    const qreal area = dx * dy;
    const qreal invDx2 = 1 / (dx * dx);
    const qreal invDy2 = 1 / (dy * dy);
    const qreal invDxDy = 1 / (dx * dy);

    const std::array<qreal, 3> dxx{invDx2, -2 * invDx2, invDx2};
    const std::array<qreal, 3> dyy{invDy2, -2 * invDy2, invDy2};
    const std::array<qreal, 4> dxy{invDxDy, -invDxDy, -invDxDy, invDxDy};

    const qreal pureScale = smoothness * area;
    const qreal mixedScale = 2 * smoothness * area;

    for (std::size_t iy = 0; iy < grid.ny; ++iy) {
        for (std::size_t ix = 1; ix + 1 < grid.nx; ++ix) {
            addQuadraticStencil(
                std::array<std::size_t, 3>{
                    surfaceIndex(ix - 1, iy, grid.nx),
                    surfaceIndex(ix, iy, grid.nx),
                    surfaceIndex(ix + 1, iy, grid.nx)},
                dxx,
                pureScale,
                input,
                output);
        }
    }
    for (std::size_t iy = 1; iy + 1 < grid.ny; ++iy) {
        for (std::size_t ix = 0; ix < grid.nx; ++ix) {
            addQuadraticStencil(
                std::array<std::size_t, 3>{
                    surfaceIndex(ix, iy - 1, grid.nx),
                    surfaceIndex(ix, iy, grid.nx),
                    surfaceIndex(ix, iy + 1, grid.nx)},
                dyy,
                pureScale,
                input,
                output);
        }
    }
    for (std::size_t iy = 0; iy + 1 < grid.ny; ++iy) {
        for (std::size_t ix = 0; ix + 1 < grid.nx; ++ix) {
            addQuadraticStencil(
                std::array<std::size_t, 4>{
                    surfaceIndex(ix, iy, grid.nx),
                    surfaceIndex(ix + 1, iy, grid.nx),
                    surfaceIndex(ix, iy + 1, grid.nx),
                    surfaceIndex(ix + 1, iy + 1, grid.nx)},
                dxy,
                mixedScale,
                input,
                output);
        }
    }

    // Screen the biharmonic correction over approximately four cells.  A
    // pure biharmonic residual has an affine null space, so a fine-level point
    // correction can otherwise become an almost constant far-field shift.
    // The screened operator retains the regional model from coarse levels and
    // lets progressively finer corrections decay away from their data.
    const qreal cellScale = std::max(dx, dy);
    const qreal kappa = 1 / (4 * cellScale);
    const qreal gradientXScale =
        2 * smoothness * kappa * kappa * area / (dx * dx);
    const qreal gradientYScale =
        2 * smoothness * kappa * kappa * area / (dy * dy);
    const std::array<qreal, 2> firstDifference{-1, 1};
    for (std::size_t iy = 0; iy < grid.ny; ++iy) {
        const qreal boundaryWeight =
            (iy == 0 || iy + 1 == grid.ny) ? qreal(0.5) : qreal(1);
        for (std::size_t ix = 0; ix + 1 < grid.nx; ++ix) {
            addQuadraticStencil(
                std::array<std::size_t, 2>{
                    surfaceIndex(ix, iy, grid.nx),
                    surfaceIndex(ix + 1, iy, grid.nx)},
                firstDifference,
                gradientXScale * boundaryWeight,
                input,
                output);
        }
    }
    for (std::size_t iy = 0; iy + 1 < grid.ny; ++iy) {
        for (std::size_t ix = 0; ix < grid.nx; ++ix) {
            const qreal boundaryWeight =
                (ix == 0 || ix + 1 == grid.nx) ? qreal(0.5) : qreal(1);
            addQuadraticStencil(
                std::array<std::size_t, 2>{
                    surfaceIndex(ix, iy, grid.nx),
                    surfaceIndex(ix, iy + 1, grid.nx)},
                firstDifference,
                gradientYScale * boundaryWeight,
                input,
                output);
        }
    }
    const qreal massScale = smoothness * kappa * kappa * kappa * kappa * area;
    for (std::size_t iy = 0; iy < grid.ny; ++iy) {
        const qreal yWeight =
            (iy == 0 || iy + 1 == grid.ny) ? qreal(0.5) : qreal(1);
        for (std::size_t ix = 0; ix < grid.nx; ++ix) {
            const qreal xWeight =
                (ix == 0 || ix + 1 == grid.nx) ? qreal(0.5) : qreal(1);
            output[surfaceIndex(ix, iy, grid.nx)] +=
                massScale * xWeight * yWeight
                * input[surfaceIndex(ix, iy, grid.nx)];
        }
    }
}

[[nodiscard]] Vector curvatureDiagonal(
    const GridGeometry& grid,
    qreal smoothness)
{
    Vector diagonal(grid.nx * grid.ny, 0);
    const qreal referenceLength = std::max(grid.maxx - grid.minx, grid.maxy - grid.miny);
    const qreal dx = grid.dx() / referenceLength;
    const qreal dy = grid.dy() / referenceLength;
    const qreal area = dx * dy;
    const qreal invDx2 = 1 / (dx * dx);
    const qreal invDy2 = 1 / (dy * dy);
    const qreal invDxDy = 1 / (dx * dy);

    const std::array<qreal, 3> dxx{invDx2, -2 * invDx2, invDx2};
    const std::array<qreal, 3> dyy{invDy2, -2 * invDy2, invDy2};
    const std::array<qreal, 4> dxy{invDxDy, -invDxDy, -invDxDy, invDxDy};
    const qreal pureScale = smoothness * area;
    const qreal mixedScale = 2 * smoothness * area;

    for (std::size_t iy = 0; iy < grid.ny; ++iy) {
        for (std::size_t ix = 1; ix + 1 < grid.nx; ++ix) {
            addQuadraticStencilDiagonal(
                std::array<std::size_t, 3>{
                    surfaceIndex(ix - 1, iy, grid.nx),
                    surfaceIndex(ix, iy, grid.nx),
                    surfaceIndex(ix + 1, iy, grid.nx)},
                dxx,
                pureScale,
                diagonal);
        }
    }
    for (std::size_t iy = 1; iy + 1 < grid.ny; ++iy) {
        for (std::size_t ix = 0; ix < grid.nx; ++ix) {
            addQuadraticStencilDiagonal(
                std::array<std::size_t, 3>{
                    surfaceIndex(ix, iy - 1, grid.nx),
                    surfaceIndex(ix, iy, grid.nx),
                    surfaceIndex(ix, iy + 1, grid.nx)},
                dyy,
                pureScale,
                diagonal);
        }
    }
    for (std::size_t iy = 0; iy + 1 < grid.ny; ++iy) {
        for (std::size_t ix = 0; ix + 1 < grid.nx; ++ix) {
            addQuadraticStencilDiagonal(
                std::array<std::size_t, 4>{
                    surfaceIndex(ix, iy, grid.nx),
                    surfaceIndex(ix + 1, iy, grid.nx),
                    surfaceIndex(ix, iy + 1, grid.nx),
                    surfaceIndex(ix + 1, iy + 1, grid.nx)},
                dxy,
                mixedScale,
                diagonal);
        }
    }


    const qreal cellScale = std::max(dx, dy);
    const qreal kappa = 1 / (4 * cellScale);
    const qreal gradientXScale =
        2 * smoothness * kappa * kappa * area / (dx * dx);
    const qreal gradientYScale =
        2 * smoothness * kappa * kappa * area / (dy * dy);
    const std::array<qreal, 2> firstDifference{-1, 1};
    for (std::size_t iy = 0; iy < grid.ny; ++iy) {
        const qreal boundaryWeight =
            (iy == 0 || iy + 1 == grid.ny) ? qreal(0.5) : qreal(1);
        for (std::size_t ix = 0; ix + 1 < grid.nx; ++ix) {
            addQuadraticStencilDiagonal(
                std::array<std::size_t, 2>{
                    surfaceIndex(ix, iy, grid.nx),
                    surfaceIndex(ix + 1, iy, grid.nx)},
                firstDifference,
                gradientXScale * boundaryWeight,
                diagonal);
        }
    }
    for (std::size_t iy = 0; iy + 1 < grid.ny; ++iy) {
        for (std::size_t ix = 0; ix < grid.nx; ++ix) {
            const qreal boundaryWeight =
                (ix == 0 || ix + 1 == grid.nx) ? qreal(0.5) : qreal(1);
            addQuadraticStencilDiagonal(
                std::array<std::size_t, 2>{
                    surfaceIndex(ix, iy, grid.nx),
                    surfaceIndex(ix, iy + 1, grid.nx)},
                firstDifference,
                gradientYScale * boundaryWeight,
                diagonal);
        }
    }
    const qreal massScale = smoothness * kappa * kappa * kappa * kappa * area;
    for (std::size_t iy = 0; iy < grid.ny; ++iy) {
        const qreal yWeight =
            (iy == 0 || iy + 1 == grid.ny) ? qreal(0.5) : qreal(1);
        for (std::size_t ix = 0; ix < grid.nx; ++ix) {
            const qreal xWeight =
                (ix == 0 || ix + 1 == grid.nx) ? qreal(0.5) : qreal(1);
            diagonal[surfaceIndex(ix, iy, grid.nx)] +=
                massScale * xWeight * yWeight;
        }
    }
    return diagonal;
}

[[nodiscard]] qreal meanPositive(const Vector& values)
{
    qreal sum = 0;
    std::size_t count = 0;
    for (qreal value : values) {
        if (value > 0 && finite(value)) {
            sum += value;
            ++count;
        }
    }
    return count == 0 ? 1 : sum / static_cast<qreal>(count);
}

void addPointOperator(
    const std::vector<PointStencil>& stencils,
    qreal dataScale,
    const Vector& input,
    Vector& output)
{
    for (const PointStencil& stencil : stencils) {
        qreal interpolated = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            interpolated += stencil.coefficient[i] * input[stencil.index[i]];
        }
        const qreal scaled = dataScale * stencil.weight * interpolated;
        for (std::size_t i = 0; i < 4; ++i) {
            output[stencil.index[i]] += stencil.coefficient[i] * scaled;
        }
    }
}

void addPointDiagonal(
    const std::vector<PointStencil>& stencils,
    qreal dataScale,
    Vector& diagonal)
{
    for (const PointStencil& stencil : stencils) {
        const qreal scale = dataScale * stencil.weight;
        for (std::size_t i = 0; i < 4; ++i) {
            diagonal[stencil.index[i]] += scale
                * stencil.coefficient[i] * stencil.coefficient[i];
        }
    }
}

[[nodiscard]] Vector buildRightHandSide(
    std::size_t size,
    const std::vector<PointStencil>& stencils,
    qreal dataScale,
    const Vector& trend,
    qreal ridge)
{
    Vector rhs(size, 0);
    for (const PointStencil& stencil : stencils) {
        const qreal scaled = dataScale * stencil.weight * stencil.value;
        for (std::size_t i = 0; i < 4; ++i) {
            rhs[stencil.index[i]] += stencil.coefficient[i] * scaled;
        }
    }
    for (std::size_t i = 0; i < size; ++i) {
        rhs[i] += ridge * trend[i];
    }
    return rhs;
}

[[nodiscard]] Vector buildAugmentedRightHandSide(
    std::size_t size,
    const std::vector<PointStencil>& stencils,
    qreal penalty,
    const Vector& lagrangeMultiplier,
    const Vector& trend,
    qreal ridge)
{
    Vector rhs(size, 0);
    for (std::size_t pointIndex = 0; pointIndex < stencils.size(); ++pointIndex) {
        const PointStencil& stencil = stencils[pointIndex];
        // L(z, lambda) = E(z) + lambda*c + penalty*w*c^2/2,
        // где c = B*z - value.
        const qreal scaled = penalty * stencil.weight * stencil.value
            - lagrangeMultiplier[pointIndex];
        for (std::size_t i = 0; i < 4; ++i) {
            rhs[stencil.index[i]] += stencil.coefficient[i] * scaled;
        }
    }
    for (std::size_t i = 0; i < size; ++i) {
        rhs[i] += ridge * trend[i];
    }
    return rhs;
}

[[nodiscard]] qreal dot(const Vector& left, const Vector& right)
{
    qreal result = 0;
    qreal compensation = 0;
    for (std::size_t i = 0; i < left.size(); ++i) {
        const qreal product = left[i] * right[i];
        const qreal corrected = product - compensation;
        const qreal next = result + corrected;
        compensation = (next - result) - corrected;
        result = next;
    }
    return result;
}

[[nodiscard]] qreal norm(const Vector& vector)
{
    qreal scale = 0;
    qreal sumSquares = 1;
    for (qreal value : vector) {
        const qreal magnitude = std::abs(value);
        if (magnitude == 0) {
            continue;
        }
        if (scale < magnitude) {
            const qreal ratio = scale / magnitude;
            sumSquares = 1 + sumSquares * ratio * ratio;
            scale = magnitude;
        } else {
            const qreal ratio = magnitude / scale;
            sumSquares += ratio * ratio;
        }
    }
    return scale == 0 ? qreal(0) : scale * std::sqrt(sumSquares);
}

struct SolverResult {
    std::size_t iterations = 0;
    bool converged = false;
};

template <typename ApplyOperator>
[[nodiscard]] SolverResult preconditionedConjugateGradient(
    Vector& solution,
    const Vector& rhs,
    const Vector& diagonal,
    ApplyOperator&& applyOperator,
    std::size_t maxIterations,
    qreal relativeTolerance)
{
    const std::size_t size = rhs.size();
    Vector applied(size, 0);
    applyOperator(solution, applied);

    Vector residual(size);
    Vector preconditioned(size);
    Vector direction(size);
    Vector operatorDirection(size);
    for (std::size_t i = 0; i < size; ++i) {
        residual[i] = rhs[i] - applied[i];
        preconditioned[i] = residual[i] / diagonal[i];
        direction[i] = preconditioned[i];
    }

    const qreal rhsNorm = norm(rhs);
    const qreal initialResidualNorm = norm(residual);
    const qreal target = relativeTolerance
        * std::max<qreal>({1, rhsNorm, initialResidualNorm});
    if (initialResidualNorm <= target) {
        return {0, true};
    }

    qreal residualDotPreconditioned = dot(residual, preconditioned);
    for (std::size_t iteration = 1; iteration <= maxIterations; ++iteration) {
        std::fill(operatorDirection.begin(), operatorDirection.end(), 0);
        applyOperator(direction, operatorDirection);
        const qreal denominator = dot(direction, operatorDirection);
        if (!(denominator > 0)
            || !finite(denominator) || !finite(residualDotPreconditioned)) {
            return {iteration - 1, false};
        }

        const qreal alpha = residualDotPreconditioned / denominator;
        for (std::size_t i = 0; i < size; ++i) {
            solution[i] += alpha * direction[i];
            residual[i] -= alpha * operatorDirection[i];
        }

        const bool verifyConvergence = norm(residual) <= target;
        if (verifyConvergence) {
            std::fill(applied.begin(), applied.end(), 0);
            applyOperator(solution, applied);
            for (std::size_t i = 0; i < size; ++i) {
                residual[i] = rhs[i] - applied[i];
            }
        }
        if (norm(residual) <= target) {
            return {iteration, true};
        }

        for (std::size_t i = 0; i < size; ++i) {
            preconditioned[i] = residual[i] / diagonal[i];
        }
        const qreal nextResidualDotPreconditioned = dot(residual, preconditioned);
        if (!finite(nextResidualDotPreconditioned)) {
            return {iteration, false};
        }
        if (verifyConvergence) {
            direction = preconditioned;
            residualDotPreconditioned = nextResidualDotPreconditioned;
            continue;
        }
        const qreal beta = nextResidualDotPreconditioned / residualDotPreconditioned;
        for (std::size_t i = 0; i < size; ++i) {
            direction[i] = preconditioned[i] + beta * direction[i];
        }
        residualDotPreconditioned = nextResidualDotPreconditioned;
    }
    return {maxIterations, false};
}

[[nodiscard]] qreal evaluateStencil(
    const PointStencil& stencil,
    const Vector& values)
{
    qreal result = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        result += stencil.coefficient[i] * values[stencil.index[i]];
    }
    return result;
}

struct ConstraintStatistics {
    qreal maxAbsoluteResidual = 0;
    qreal maxRelativeResidual = 0;
    qreal weightedSquaredResidual = 0;
    qreal totalWeight = 0;
};

[[nodiscard]] ConstraintStatistics constraintStatistics(
    const std::vector<PointStencil>& stencils,
    const Vector& values)
{
    ConstraintStatistics statistics;
    for (const PointStencil& stencil : stencils) {
        const qreal residual = evaluateStencil(stencil, values) - stencil.value;
        statistics.maxAbsoluteResidual = std::max(
            statistics.maxAbsoluteResidual, std::abs(residual));
        statistics.maxRelativeResidual = std::max(
            statistics.maxRelativeResidual,
            std::abs(residual) / std::max<qreal>(1, std::abs(stencil.value)));
        statistics.weightedSquaredResidual += stencil.weight * residual * residual;
        statistics.totalWeight += stencil.weight;
    }
    return statistics;
}

struct ConstrainedSolverResult {
    std::size_t iterations = 0;
    std::size_t linearIterations = 0;
    qreal finalPenalty = 0;
    bool linearSolvesConverged = true;
    bool constraintsSatisfied = false;
};

struct ExactProjectionResult {
    std::size_t iterations = 0;
    std::size_t passes = 0;
    bool linearSolvesConverged = true;
    bool constraintsSatisfied = false;
};

// Simultaneous screened-biharmonic defect corrections.  Unlike a Euclidean
// B^T(BB^T)^-1 projection, each pass solves for a spatially smooth global
// correction.  Increasing the penalty converges to the minimum-energy
// correction subject to all point equations, without order-dependent local
// snapping or spikes on the four nodes around a point.
[[nodiscard]] ExactProjectionResult projectExactlyOntoConstraints(
    const GridGeometry& grid,
    const std::vector<PointStencil>& stencils,
    const Options& options,
    qreal initialPenalty,
    Vector& values)
{
    ExactProjectionResult result;
    const qreal curvatureScale = meanPositive(curvatureDiagonal(grid, 1));
    const Vector curvature = curvatureDiagonal(grid, options.smoothness);
    const qreal ridge = options.regularization * curvatureScale;
    qreal penalty = std::max(
        initialPenalty, options.dataWeight * curvatureScale);
    const Vector zero(values.size(), 0);

    for (std::size_t pass = 1;
         pass <= options.maxExactProjectionPasses;
         ++pass) {
        result.passes = pass;
        const ConstraintStatistics before = constraintStatistics(stencils, values);
        if (before.maxRelativeResidual
            <= options.pointConstraintRelativeTolerance) {
            result.constraintsSatisfied = true;
            return result;
        }

        std::vector<PointStencil> defects = stencils;
        for (PointStencil& defect : defects) {
            defect.value -= evaluateStencil(defect, values);
        }

        Vector diagonal = curvature;
        addPointDiagonal(defects, penalty, diagonal);
        for (qreal& value : diagonal) {
            value += ridge;
        }
        const Vector rhs = buildRightHandSide(
            values.size(), defects, penalty, zero, ridge);
        Vector correction(values.size(), 0);
        const auto applyOperator = [&](const Vector& input, Vector& output) {
            std::fill(output.begin(), output.end(), 0);
            addCurvatureOperator(grid, options.smoothness, input, output);
            addPointOperator(defects, penalty, input, output);
            for (std::size_t node = 0; node < input.size(); ++node) {
                output[node] += ridge * input[node];
            }
        };
        const SolverResult solver = preconditionedConjugateGradient(
            correction,
            rhs,
            diagonal,
            applyOperator,
            options.maxExactProjectionIterations,
            std::min(options.relativeTolerance,
                options.pointConstraintRelativeTolerance * qreal(0.1)));
        result.iterations += solver.iterations;
        result.linearSolvesConverged =
            result.linearSolvesConverged && solver.converged;
        for (std::size_t node = 0; node < values.size(); ++node) {
            values[node] += correction[node];
        }

        const ConstraintStatistics after = constraintStatistics(stencils, values);
        if (after.maxRelativeResidual
            <= options.pointConstraintRelativeTolerance) {
            result.constraintsSatisfied = true;
            return result;
        }

        const qreal growth = std::max<qreal>(
            options.constraintPenaltyGrowth * options.constraintPenaltyGrowth,
            qreal(2));
        if (penalty <= std::numeric_limits<qreal>::max() / growth) {
            penalty *= growth;
        }
    }
    return result;
}

[[nodiscard]] ConstrainedSolverResult solveConstrainedMinimumCurvature(
    const GridGeometry& grid,
    const Vector& trend,
    const std::vector<PointStencil>& stencils,
    const Options& options,
    Vector& values)
{
    const qreal curvatureScale = meanPositive(curvatureDiagonal(grid, 1));
    const Vector curvature = curvatureDiagonal(grid, options.smoothness);
    const qreal ridge = options.regularization * curvatureScale;
    qreal penalty = options.dataWeight * curvatureScale;
    Vector lagrangeMultiplier(stencils.size(), 0);

    ConstrainedSolverResult result;
    result.finalPenalty = penalty;
    qreal previousResidual = std::numeric_limits<qreal>::infinity();

    for (std::size_t iteration = 1;
         iteration <= options.maxConstraintIterations;
         ++iteration) {
        Vector diagonal = curvature;
        addPointDiagonal(stencils, penalty, diagonal);
        for (qreal& value : diagonal) {
            value += ridge;
        }

        const Vector rhs = buildAugmentedRightHandSide(
            values.size(), stencils, penalty, lagrangeMultiplier, trend, ridge);
        const auto applyOperator = [&](const Vector& input, Vector& output) {
            std::fill(output.begin(), output.end(), 0);
            addCurvatureOperator(grid, options.smoothness, input, output);
            addPointOperator(stencils, penalty, input, output);
            for (std::size_t i = 0; i < input.size(); ++i) {
                output[i] += ridge * input[i];
            }
        };

        const SolverResult linearSolve = preconditionedConjugateGradient(
            values,
            rhs,
            diagonal,
            applyOperator,
            options.maxIterationsPerLevel,
            std::min(options.relativeTolerance,
                options.pointConstraintRelativeTolerance * qreal(0.1)));
        result.iterations = iteration;
        result.linearIterations += linearSolve.iterations;
        result.linearSolvesConverged =
            result.linearSolvesConverged && linearSolve.converged;

        const ConstraintStatistics statistics = constraintStatistics(stencils, values);
        if (statistics.maxRelativeResidual
            <= options.pointConstraintRelativeTolerance) {
            result.finalPenalty = penalty;
            result.constraintsSatisfied = true;
            return result;
        }

        for (std::size_t pointIndex = 0;
             pointIndex < stencils.size();
             ++pointIndex) {
            const PointStencil& stencil = stencils[pointIndex];
            const qreal residual = evaluateStencil(stencil, values) - stencil.value;
            lagrangeMultiplier[pointIndex] += penalty * stencil.weight * residual;
        }

        // При стагнации усиливаем штраф. Множители остаются в немасштабной
        // форме и поэтому не требуют пересчёта при изменении penalty.
        if (statistics.maxRelativeResidual > previousResidual * qreal(0.7)
            && options.constraintPenaltyGrowth > 1) {
            const qreal safeLimit = std::numeric_limits<qreal>::max()
                / options.constraintPenaltyGrowth;
            if (penalty <= safeLimit) {
                penalty *= options.constraintPenaltyGrowth;
            }
        }
        result.finalPenalty = penalty;
        previousResidual = statistics.maxRelativeResidual;
    }
    return result;
}

} // namespace

Report interpolate(
    std::vector<qreal>& surfaceValues,
    qreal minx,
    qreal maxx,
    qreal miny,
    qreal maxy,
    std::size_t nx,
    std::size_t ny,
    const std::vector<Sample>& points,
    const Options& options)
{
    const GridGeometry surface{minx, maxx, miny, maxy, nx, ny};
    validateSurface(surface);
    validateOptions(options);
    const std::vector<Sample> validPoints = validateAndFilterPoints(
        surface, points, options.ignoreOutsidePoints);
    const Plane plane = fitPlane(surface, validPoints);
    const auto levels = buildLevels(surface.nx, surface.ny, options);
    const GridGeometry finalGrid = surface;
    const std::vector<PointStencil> finalStencils = makePointStencils(
        finalGrid, validPoints, options.cellInterpolation);
    if (options.enforcePointConstraints) {
        validateConstraintCompatibility(
            finalStencils, options.pointConstraintRelativeTolerance);
    }
    validateActivePointSupport(
        finalStencils, options.activeNodeMask, finalGrid.nx * finalGrid.ny);

    Report report;
    report.levelsUsed = levels.size();
    report.converged = true;

    Vector values;
    std::size_t previousNx = 0;
    std::size_t previousNy = 0;

    for (std::size_t levelIndex = 0; levelIndex < levels.size(); ++levelIndex) {
        const std::size_t levelNx = levels[levelIndex].first;
        const std::size_t levelNy = levels[levelIndex].second;
        const GridGeometry grid{
            surface.minx,
            surface.maxx,
            surface.miny,
            surface.maxy,
            levelNx,
            levelNy};
        const Vector trend = evaluatePlane(grid, plane);
        Vector predicted;
        if (values.empty()) {
            predicted = trend;
        } else {
            predicted = resample(
                values, previousNx, previousNy, levelNx, levelNy);
        }

        std::vector<PointStencil> stencils = makeSnapStencils(
            grid,
            validPoints,
            options.cellInterpolation,
            predicted,
            levelIndex,
            levels.size());
        for (PointStencil& stencil : stencils) {
            stencil.value -= evaluateStencil(stencil, predicted);
        }
        const qreal curvatureScale = meanPositive(curvatureDiagonal(grid, 1));
        Vector diagonal = curvatureDiagonal(grid, options.smoothness);
        const qreal dataScale = options.dataWeight * curvatureScale;
        const qreal ridge = options.regularization * curvatureScale;

        addPointDiagonal(stencils, dataScale, diagonal);
        for (qreal& value : diagonal) {
            value += ridge;
        }
        const Vector zero(predicted.size(), 0);
        const Vector rhs = buildRightHandSide(
            predicted.size(), stencils, dataScale, zero, ridge);

        const auto applyOperator = [&](const Vector& input, Vector& output) {
            std::fill(output.begin(), output.end(), 0);
            addCurvatureOperator(grid, options.smoothness, input, output);
            addPointOperator(stencils, dataScale, input, output);
            for (std::size_t i = 0; i < input.size(); ++i) {
                output[i] += ridge * input[i];
            }
        };

        Vector correction(predicted.size(), 0);
        const SolverResult solver = preconditionedConjugateGradient(
            correction,
            rhs,
            diagonal,
            applyOperator,
            options.maxIterationsPerLevel,
            options.relativeTolerance);
        report.totalIterations += solver.iterations;
        report.converged = report.converged && solver.converged;
        values = std::move(predicted);
        for (std::size_t node = 0; node < values.size(); ++node) {
            values[node] += correction[node];
        }
        previousNx = levelNx;
        previousNy = levelNy;
    }

    if (options.enforcePointConstraints) {
        std::vector<PointStencil> defects = finalStencils;
        for (PointStencil& defect : defects) {
            defect.value -= evaluateStencil(defect, values);
        }
        Vector correction(values.size(), 0);
        const Vector zero(values.size(), 0);
        const ConstrainedSolverResult constrained = solveConstrainedMinimumCurvature(
            finalGrid,
            zero,
            defects,
            options,
            correction);
        for (std::size_t node = 0; node < values.size(); ++node) {
            values[node] += correction[node];
        }
        report.constraintIterations = constrained.iterations;
        report.totalIterations += constrained.linearIterations;
        report.converged = report.converged && constrained.linearSolvesConverged;

        const qreal projectionPenalty =
            constrained.finalPenalty
                    <= std::numeric_limits<qreal>::max()
                        / options.constraintPenaltyGrowth
            ? constrained.finalPenalty * options.constraintPenaltyGrowth
            : constrained.finalPenalty;
        const ExactProjectionResult projection = projectExactlyOntoConstraints(
            finalGrid,
            finalStencils,
            options,
            projectionPenalty,
            values);
        report.exactProjectionIterations = projection.iterations;
        report.totalIterations += projection.iterations;
        report.converged =
            report.converged && projection.linearSolvesConverged;
        report.pointConstraintsSatisfied = projection.constraintsSatisfied;
        if (!projection.constraintsSatisfied) {
            throw std::runtime_error(
                "convergent::interpolate: hard point constraints are inconsistent "
                "with the selected mesh interpolation or the global "
                "minimum-curvature constraint correction did not converge");
        }
    }

    for (qreal value : values) {
        if (!finite(value)) {
            throw std::runtime_error(
                "convergent::interpolate: numerical overflow produced a non-finite surface");
        }
    }

    surfaceValues = std::move(values);

    const ConstraintStatistics finalStatistics = constraintStatistics(
        finalStencils, surfaceValues);
    report.maxAbsolutePointResidual = finalStatistics.maxAbsoluteResidual;
    report.weightedRmsPointResidual = std::sqrt(
        finalStatistics.weightedSquaredResidual / finalStatistics.totalWeight);
    if (!options.enforcePointConstraints) {
        report.pointConstraintsSatisfied = finalStatistics.maxRelativeResidual
            <= options.pointConstraintRelativeTolerance;
    }
    return report;
}

qreal evaluateSurface(
    const std::vector<qreal>& surfaceValues,
    qreal minx,
    qreal maxx,
    qreal miny,
    qreal maxy,
    std::size_t nx,
    std::size_t ny,
    qreal x,
    qreal y,
    CellInterpolation interpolation)
{
    const GridGeometry grid{minx, maxx, miny, maxy, nx, ny};
    validateSurface(grid);
    if (surfaceValues.size() != nx * ny) {
        throw std::invalid_argument(
            "convergent::evaluateSurface: values size must be nx * ny");
    }

    const std::vector<Sample> checked = validateAndFilterPoints(
        grid, std::vector<Sample>{{x, y, 0, 1}}, false);
    const PointStencil stencil = makePointStencil(grid, checked.front(), interpolation);
    return evaluateStencil(stencil, surfaceValues);
}

} // namespace geo::convergent
