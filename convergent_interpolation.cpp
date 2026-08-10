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
    plane.cx = (surface.minx + surface.maxx) / 2;
    plane.cy = (surface.miny + surface.maxy) / 2;
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
    Vector target(targetNx * targetNy);
    for (std::size_t targetY = 0; targetY < targetNy; ++targetY) {
        const qreal sourceY = static_cast<qreal>(targetY)
            * static_cast<qreal>(sourceNy - 1) / static_cast<qreal>(targetNy - 1);
        const std::size_t y0 = std::min<std::size_t>(
            static_cast<std::size_t>(std::floor(sourceY)), sourceNy - 2);
        const std::size_t y1 = y0 + 1;
        const qreal fy = sourceY - static_cast<qreal>(y0);

        for (std::size_t targetX = 0; targetX < targetNx; ++targetX) {
            const qreal sourceX = static_cast<qreal>(targetX)
                * static_cast<qreal>(sourceNx - 1) / static_cast<qreal>(targetNx - 1);
            const std::size_t x0 = std::min<std::size_t>(
                static_cast<std::size_t>(std::floor(sourceX)), sourceNx - 2);
            const std::size_t x1 = x0 + 1;
            const qreal fx = sourceX - static_cast<qreal>(x0);

            const qreal z00 = source[surfaceIndex(x0, y0, sourceNx)];
            const qreal z10 = source[surfaceIndex(x1, y0, sourceNx)];
            const qreal z01 = source[surfaceIndex(x0, y1, sourceNx)];
            const qreal z11 = source[surfaceIndex(x1, y1, sourceNx)];
            target[surfaceIndex(targetX, targetY, targetNx)] =
                (1 - fy) * ((1 - fx) * z00 + fx * z10)
                + fy * ((1 - fx) * z01 + fx * z11);
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

void validateLocalConstraintCompatibility(
    const std::vector<PointStencil>& stencils,
    qreal relativeTolerance)
{
    using Support = std::array<std::size_t, 4>;
    const std::size_t unused = std::numeric_limits<std::size_t>::max();
    const qreal coefficientTolerance =
        64 * std::numeric_limits<qreal>::epsilon();

    std::map<Support, std::vector<const PointStencil*>> groups;
    for (const PointStencil& stencil : stencils) {
        Support support{unused, unused, unused, unused};
        std::size_t count = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            if (stencil.coefficient[i] > coefficientTolerance) {
                support[count++] = stencil.index[i];
            }
        }
        std::sort(support.begin(), support.begin() + count);
        groups[support].push_back(&stencil);
    }

    for (const auto& [support, group] : groups) {
        std::size_t supportSize = 0;
        while (supportSize < support.size() && support[supportSize] != unused) {
            ++supportSize;
        }
        std::vector<std::vector<qreal>> coefficients(
            group.size(), std::vector<qreal>(supportSize));
        std::vector<std::vector<qreal>> augmented(
            group.size(), std::vector<qreal>(supportSize + 1));
        for (std::size_t row = 0; row < group.size(); ++row) {
            const PointStencil& stencil = *group[row];
            for (std::size_t local = 0; local < supportSize; ++local) {
                for (std::size_t i = 0; i < 4; ++i) {
                    if (stencil.index[i] == support[local]) {
                        coefficients[row][local] = stencil.coefficient[i];
                        augmented[row][local] = stencil.coefficient[i];
                    }
                }
            }
            augmented[row][supportSize] = stencil.value;
        }

        const std::size_t coefficientRank = matrixRank(
            coefficients, relativeTolerance);
        const std::size_t augmentedRank = matrixRank(
            augmented, relativeTolerance);
        if (augmentedRank > coefficientRank) {
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
    for (std::size_t i = 0; i < left.size(); ++i) {
        result += left[i] * right[i];
    }
    return result;
}

[[nodiscard]] qreal norm(const Vector& vector)
{
    return std::sqrt(std::max<qreal>(0, dot(vector, vector)));
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
    bool linearSolvesConverged = true;
    bool constraintsSatisfied = false;
};

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
        validateLocalConstraintCompatibility(
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

    for (const auto [nx, ny] : levels) {
        const GridGeometry grid{
            surface.minx, surface.maxx, surface.miny, surface.maxy, nx, ny};
        const Vector trend = evaluatePlane(grid, plane);
        if (values.empty()) {
            values = trend;
        } else {
            values = resample(values, previousNx, previousNy, nx, ny);
        }

        const std::vector<PointStencil> stencils = makePointStencils(
            grid, validPoints, options.cellInterpolation);
        const qreal curvatureScale = meanPositive(curvatureDiagonal(grid, 1));
        Vector diagonal = curvatureDiagonal(grid, options.smoothness);
        const qreal dataScale = options.dataWeight * curvatureScale;
        const qreal ridge = options.regularization * curvatureScale;

        addPointDiagonal(stencils, dataScale, diagonal);
        for (qreal& value : diagonal) {
            value += ridge;
        }
        const Vector rhs = buildRightHandSide(
            values.size(), stencils, dataScale, trend, ridge);

        const auto applyOperator = [&](const Vector& input, Vector& output) {
            std::fill(output.begin(), output.end(), 0);
            addCurvatureOperator(grid, options.smoothness, input, output);
            addPointOperator(stencils, dataScale, input, output);
            for (std::size_t i = 0; i < input.size(); ++i) {
                output[i] += ridge * input[i];
            }
        };

        const SolverResult solver = preconditionedConjugateGradient(
            values,
            rhs,
            diagonal,
            applyOperator,
            options.maxIterationsPerLevel,
            options.relativeTolerance);
        report.totalIterations += solver.iterations;
        report.converged = report.converged && solver.converged;
        previousNx = nx;
        previousNy = ny;
    }

    if (options.enforcePointConstraints) {
        const Vector finalTrend = evaluatePlane(finalGrid, plane);
        const ConstrainedSolverResult constrained = solveConstrainedMinimumCurvature(
            finalGrid,
            finalTrend,
            finalStencils,
            options,
            values);
        report.constraintIterations = constrained.iterations;
        report.totalIterations += constrained.linearIterations;
        report.converged = report.converged && constrained.linearSolvesConverged;
        report.pointConstraintsSatisfied = constrained.constraintsSatisfied;
        if (!constrained.constraintsSatisfied) {
            throw std::runtime_error(
                "convergent::interpolate: hard point constraints are inconsistent "
                "with the selected mesh interpolation or the constrained solver "
                "did not converge");
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
