#include "gaussian_stochastic_interpolation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace geo::gaussian {
namespace {

using Real = double;
using Vector = std::vector<Real>;
using Complex = std::complex<Real>;
using ComplexVector = std::vector<Complex>;

constexpr Real pi = 3.141592653589793238462643383279502884;

struct GridGeometry {
    Real minx = 0;
    Real maxx = 0;
    Real miny = 0;
    Real maxy = 0;
    std::size_t nx = 0;
    std::size_t ny = 0;

    [[nodiscard]] Real dx() const
    {
        return (maxx - minx) / static_cast<Real>(nx - 1);
    }

    [[nodiscard]] Real dy() const
    {
        return (maxy - miny) / static_cast<Real>(ny - 1);
    }
};

struct PointStencil {
    std::array<std::size_t, 4> index{};
    std::array<Real, 4> coefficient{};
    Real value = 0;
    Real weight = 1;
};

struct Plane {
    Real a = 0;
    Real bx = 0;
    Real by = 0;
    Real cx = 0;
    Real cy = 0;
    Real sx = 1;
    Real sy = 1;

    [[nodiscard]] Real evaluate(Real x, Real y) const
    {
        return a + bx * ((x - cx) / sx) + by * ((y - cy) / sy);
    }
};

struct EffectiveParameters {
    Real majorRange = 0;
    Real minorRange = 0;
    Real structuredVariance = 0;
    Real nodeNuggetVariance = 0;
    Real cosAzimuth = 1;
    Real sinAzimuth = 0;
};

[[nodiscard]] bool finite(Real value)
{
    return std::isfinite(value);
}

void validateGrid(const GridGeometry& grid)
{
    if (grid.nx < 2 || grid.ny < 2) {
        throw std::invalid_argument(
            "gaussian::interpolate: nx and ny must be at least 2 (node counts)");
    }
    if (!finite(grid.minx) || !finite(grid.maxx)
        || !finite(grid.miny) || !finite(grid.maxy)
        || !(grid.maxx > grid.minx) || !(grid.maxy > grid.miny)) {
        throw std::invalid_argument(
            "gaussian::interpolate: bounds must be finite and non-degenerate");
    }
    if (grid.nx > std::numeric_limits<std::size_t>::max() / grid.ny) {
        throw std::overflow_error("gaussian::interpolate: surface is too large");
    }
}

void validateOptions(const Options& options)
{
    const auto nonNegativeFinite = [](qreal value) {
        return finite(static_cast<Real>(value)) && value >= 0;
    };
    if (!nonNegativeFinite(options.majorRange)
        || !nonNegativeFinite(options.minorRange)
        || !nonNegativeFinite(options.structuredVariance)
        || !nonNegativeFinite(options.nodeNuggetVariance)
        || !finite(static_cast<Real>(options.azimuthDegrees))) {
        throw std::invalid_argument(
            "gaussian::interpolate: covariance parameters must be finite and non-negative");
    }
    if (options.paddingFactor < 2 || options.maxEmbeddingExpansions == 0) {
        throw std::invalid_argument(
            "gaussian::interpolate: paddingFactor >= 2 and maxEmbeddingExpansions > 0 required");
    }
    if (options.enforcePointConstraints
        && (options.maxConditioningIterations == 0
            || options.maxConditioningRefinements == 0)) {
        throw std::invalid_argument(
            "gaussian::interpolate: conditioning iteration limits must be positive");
    }
    if (!finite(static_cast<Real>(options.conditioningRelativeTolerance))
        || options.conditioningRelativeTolerance <= 0) {
        throw std::invalid_argument(
            "gaussian::interpolate: conditioning tolerance must be finite and positive");
    }
    switch (options.cellInterpolation) {
    case CellInterpolation::Bilinear:
    case CellInterpolation::TriangleBottomLeftToTopRight:
    case CellInterpolation::TriangleBottomRightToTopLeft:
        break;
    default:
        throw std::invalid_argument(
            "gaussian::interpolate: unknown cell interpolation");
    }
    switch (options.correlationModel) {
    case CorrelationModel::Gaussian:
    case CorrelationModel::Exponential:
    case CorrelationModel::Spherical:
        break;
    default:
        throw std::invalid_argument(
            "gaussian::interpolate: unknown correlation model");
    }
    switch (options.trendModel) {
    case TrendModel::PlaneFromPoints:
    case TrendModel::ConstantFromPoints:
    case TrendModel::InputSurface:
        break;
    default:
        throw std::invalid_argument("gaussian::interpolate: unknown trend model");
    }
}

[[nodiscard]] Real coordinateTolerance(Real lo, Real hi)
{
    const Real scale = std::max<Real>(
        1, std::max(std::abs(lo), std::max(std::abs(hi), std::abs(hi - lo))));
    return 64 * std::numeric_limits<qreal>::epsilon() * scale;
}

[[nodiscard]] std::vector<Sample> validateAndFilterPoints(
    const GridGeometry& grid,
    const std::vector<Sample>& points,
    bool ignoreOutside)
{
    std::vector<Sample> result;
    result.reserve(points.size());
    const Real xtol = coordinateTolerance(grid.minx, grid.maxx);
    const Real ytol = coordinateTolerance(grid.miny, grid.maxy);

    for (const Sample& source : points) {
        if (!finite(source.x) || !finite(source.y)
            || !finite(source.value) || !finite(source.weight)) {
            throw std::invalid_argument(
                "gaussian::interpolate: point fields must be finite");
        }
        if (source.weight < 0) {
            throw std::invalid_argument(
                "gaussian::interpolate: point weight must not be negative");
        }
        if (source.weight == 0) {
            continue;
        }
        const bool outside = source.x < grid.minx - xtol
            || source.x > grid.maxx + xtol
            || source.y < grid.miny - ytol
            || source.y > grid.maxy + ytol;
        if (outside) {
            if (ignoreOutside) {
                continue;
            }
            throw std::out_of_range(
                "gaussian::interpolate: a conditioning point lies outside the surface");
        }
        Sample point = source;
        point.x = std::clamp(point.x, static_cast<qreal>(grid.minx),
            static_cast<qreal>(grid.maxx));
        point.y = std::clamp(point.y, static_cast<qreal>(grid.miny),
            static_cast<qreal>(grid.maxy));
        result.push_back(point);
    }
    return result;
}

[[nodiscard]] std::array<Real, 4> interpolationCoefficients(
    Real fx,
    Real fy,
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
            return {1 - fx, fx - fy, 0, fy};
        }
        return {1 - fy, 0, fy - fx, fx};
    case CellInterpolation::TriangleBottomRightToTopLeft:
        if (fx + fy <= 1) {
            return {1 - fx - fy, fx, fy, 0};
        }
        return {0, 1 - fy, 1 - fx, fx + fy - 1};
    }
    throw std::invalid_argument("gaussian::interpolate: unknown cell interpolation");
}

[[nodiscard]] PointStencil makePointStencil(
    const GridGeometry& grid,
    const Sample& point,
    CellInterpolation interpolation)
{
    const Real gx = (point.x - grid.minx) / grid.dx();
    const Real gy = (point.y - grid.miny) / grid.dy();
    const std::size_t ix = std::min<std::size_t>(
        static_cast<std::size_t>(std::floor(gx)), grid.nx - 2);
    const std::size_t iy = std::min<std::size_t>(
        static_cast<std::size_t>(std::floor(gy)), grid.ny - 2);
    const Real fx = std::clamp(gx - static_cast<Real>(ix), Real(0), Real(1));
    const Real fy = std::clamp(gy - static_cast<Real>(iy), Real(0), Real(1));

    PointStencil result;
    result.index = {
        surfaceIndex(ix, iy, grid.nx),
        surfaceIndex(ix + 1, iy, grid.nx),
        surfaceIndex(ix, iy + 1, grid.nx),
        surfaceIndex(ix + 1, iy + 1, grid.nx),
    };
    result.coefficient = interpolationCoefficients(fx, fy, interpolation);
    result.value = point.value;
    result.weight = point.weight;
    return result;
}

[[nodiscard]] std::vector<PointStencil> makePointStencils(
    const GridGeometry& grid,
    const std::vector<Sample>& points,
    CellInterpolation interpolation)
{
    std::vector<PointStencil> result;
    result.reserve(points.size());
    for (const Sample& point : points) {
        result.push_back(makePointStencil(grid, point, interpolation));
    }
    return result;
}

void validateActiveSupport(
    const std::vector<PointStencil>& stencils,
    const std::vector<std::uint8_t>* mask,
    std::size_t nodeCount)
{
    if (mask == nullptr) {
        return;
    }
    if (mask->size() != nodeCount) {
        throw std::invalid_argument(
            "gaussian::interpolate: activeNodeMask size must be nx * ny");
    }
    const Real tolerance = 64 * std::numeric_limits<Real>::epsilon();
    for (const PointStencil& stencil : stencils) {
        for (std::size_t local = 0; local < 4; ++local) {
            if (stencil.coefficient[local] > tolerance
                && (*mask)[stencil.index[local]] == 0) {
                throw std::out_of_range(
                    "gaussian::interpolate: a point lies on an inactive or clipped mesh cell");
            }
        }
    }
}

[[nodiscard]] std::size_t matrixRank(
    std::vector<std::vector<Real>> matrix,
    Real relativeTolerance)
{
    if (matrix.empty() || matrix.front().empty()) {
        return 0;
    }
    const std::size_t rows = matrix.size();
    const std::size_t columns = matrix.front().size();
    for (std::size_t column = 0; column < columns; ++column) {
        Real scale = 0;
        for (std::size_t row = 0; row < rows; ++row) {
            scale = std::max(scale, std::abs(matrix[row][column]));
        }
        if (scale > 0) {
            for (std::size_t row = 0; row < rows; ++row) {
                matrix[row][column] /= scale;
            }
        }
    }

    const Real tolerance = std::max(
        256 * std::numeric_limits<Real>::epsilon(), relativeTolerance);
    std::size_t rank = 0;
    for (std::size_t column = 0; column < columns && rank < rows; ++column) {
        std::size_t pivot = rank;
        for (std::size_t row = rank + 1; row < rows; ++row) {
            if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) {
                pivot = row;
            }
        }
        if (std::abs(matrix[pivot][column]) <= tolerance) {
            continue;
        }
        std::swap(matrix[pivot], matrix[rank]);
        const Real pivotValue = matrix[rank][column];
        for (std::size_t j = column; j < columns; ++j) {
            matrix[rank][j] /= pivotValue;
        }
        for (std::size_t row = 0; row < rows; ++row) {
            if (row == rank) {
                continue;
            }
            const Real factor = matrix[row][column];
            for (std::size_t j = column; j < columns; ++j) {
                matrix[row][j] -= factor * matrix[rank][j];
            }
        }
        ++rank;
    }
    return rank;
}

void validateLocalConstraintCompatibility(
    const std::vector<PointStencil>& stencils,
    Real relativeTolerance)
{
    using Support = std::array<std::size_t, 4>;
    const std::size_t unused = std::numeric_limits<std::size_t>::max();
    const Real coefficientTolerance = 64 * std::numeric_limits<Real>::epsilon();
    std::map<Support, std::vector<const PointStencil*>> groups;

    for (const PointStencil& stencil : stencils) {
        Support support{unused, unused, unused, unused};
        std::size_t count = 0;
        for (std::size_t local = 0; local < 4; ++local) {
            if (stencil.coefficient[local] > coefficientTolerance) {
                support[count++] = stencil.index[local];
            }
        }
        std::sort(support.begin(), support.begin() + count);
        groups[support].push_back(&stencil);
    }

    for (const auto& item : groups) {
        const Support& support = item.first;
        const auto& group = item.second;
        std::size_t supportSize = 0;
        while (supportSize < support.size() && support[supportSize] != unused) {
            ++supportSize;
        }
        std::vector<std::vector<Real>> coefficients(
            group.size(), std::vector<Real>(supportSize));
        std::vector<std::vector<Real>> augmented(
            group.size(), std::vector<Real>(supportSize + 1));
        for (std::size_t row = 0; row < group.size(); ++row) {
            for (std::size_t localSupport = 0;
                 localSupport < supportSize;
                 ++localSupport) {
                for (std::size_t local = 0; local < 4; ++local) {
                    if (group[row]->index[local] == support[localSupport]) {
                        coefficients[row][localSupport] =
                            group[row]->coefficient[local];
                        augmented[row][localSupport] =
                            group[row]->coefficient[local];
                    }
                }
            }
            augmented[row][supportSize] = group[row]->value;
        }
        if (matrixRank(augmented, relativeTolerance)
            > matrixRank(coefficients, relativeTolerance)) {
            throw std::invalid_argument(
                "gaussian::interpolate: hard points cannot be represented by "
                "the selected grid resolution and cell topology");
        }
    }
}

[[nodiscard]] Real evaluateStencil(
    const PointStencil& stencil,
    const Vector& values)
{
    Real result = 0;
    for (std::size_t local = 0; local < 4; ++local) {
        result += stencil.coefficient[local] * values[stencil.index[local]];
    }
    return result;
}

void scatterTranspose(
    const std::vector<PointStencil>& stencils,
    const Vector& pointValues,
    Vector& nodes)
{
    std::fill(nodes.begin(), nodes.end(), 0);
    for (std::size_t pointIndex = 0;
         pointIndex < stencils.size();
         ++pointIndex) {
        for (std::size_t local = 0; local < 4; ++local) {
            nodes[stencils[pointIndex].index[local]] +=
                stencils[pointIndex].coefficient[local] * pointValues[pointIndex];
        }
    }
}

void gather(
    const std::vector<PointStencil>& stencils,
    const Vector& nodes,
    Vector& pointValues)
{
    pointValues.resize(stencils.size());
    for (std::size_t pointIndex = 0;
         pointIndex < stencils.size();
         ++pointIndex) {
        pointValues[pointIndex] = evaluateStencil(stencils[pointIndex], nodes);
    }
}

[[nodiscard]] bool solve3x3(
    std::array<std::array<Real, 3>, 3> matrix,
    std::array<Real, 3> rhs,
    std::array<Real, 3>& solution)
{
    for (std::size_t column = 0; column < 3; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < 3; ++row) {
            if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) {
                pivot = row;
            }
        }
        const Real tolerance = 512 * std::numeric_limits<Real>::epsilon();
        if (std::abs(matrix[pivot][column]) <= tolerance) {
            return false;
        }
        std::swap(matrix[pivot], matrix[column]);
        std::swap(rhs[pivot], rhs[column]);
        for (std::size_t row = column + 1; row < 3; ++row) {
            const Real factor = matrix[row][column] / matrix[column][column];
            for (std::size_t j = column; j < 3; ++j) {
                matrix[row][j] -= factor * matrix[column][j];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    for (std::size_t reverse = 0; reverse < 3; ++reverse) {
        const std::size_t row = 2 - reverse;
        Real value = rhs[row];
        for (std::size_t column = row + 1; column < 3; ++column) {
            value -= matrix[row][column] * solution[column];
        }
        solution[row] = value / matrix[row][row];
    }
    return true;
}

[[nodiscard]] Plane fitPlane(
    const GridGeometry& grid,
    const std::vector<Sample>& points,
    bool allowSlopes)
{
    Plane result;
    result.cx = (grid.minx + grid.maxx) / 2;
    result.cy = (grid.miny + grid.maxy) / 2;
    result.sx = grid.maxx - grid.minx;
    result.sy = grid.maxy - grid.miny;
    if (points.empty()) {
        return result;
    }

    Real totalWeight = 0;
    Real weightedValue = 0;
    for (const Sample& point : points) {
        totalWeight += point.weight;
        weightedValue += point.weight * point.value;
        if (!finite(totalWeight) || !finite(weightedValue)) {
            throw std::overflow_error(
                "gaussian::interpolate: point weights/values overflow trend estimation");
        }
    }
    result.a = weightedValue / totalWeight;
    if (!allowSlopes) {
        return result;
    }

    std::array<std::array<Real, 3>, 3> normal{};
    std::array<Real, 3> rhs{};
    for (const Sample& point : points) {
        const std::array<Real, 3> row{
            1,
            (point.x - result.cx) / result.sx,
            (point.y - result.cy) / result.sy,
        };
        for (std::size_t i = 0; i < 3; ++i) {
            rhs[i] += point.weight * row[i] * point.value;
            for (std::size_t j = 0; j < 3; ++j) {
                normal[i][j] += point.weight * row[i] * row[j];
            }
            if (!finite(rhs[i])) {
                throw std::overflow_error(
                    "gaussian::interpolate: point values overflow plane estimation");
            }
        }
    }
    const Real slopeRidge = std::max<Real>(1, totalWeight) * 1e-12;
    normal[1][1] += slopeRidge;
    normal[2][2] += slopeRidge;
    std::array<Real, 3> parameters{};
    if (solve3x3(normal, rhs, parameters)) {
        result.a = parameters[0];
        result.bx = parameters[1];
        result.by = parameters[2];
    }
    return result;
}

[[nodiscard]] Vector makeTrend(
    const GridGeometry& grid,
    const std::vector<Sample>& points,
    const std::vector<qreal>& input,
    const Options& options)
{
    if (options.trendModel == TrendModel::InputSurface) {
        if (input.size() != grid.nx * grid.ny) {
            throw std::invalid_argument(
                "gaussian::interpolate: InputSurface trend requires nx * ny input values");
        }
        Vector result(input.begin(), input.end());
        for (Real value : result) {
            if (!finite(value)) {
                throw std::invalid_argument(
                    "gaussian::interpolate: input trend values must be finite");
            }
        }
        return result;
    }

    const Plane plane = fitPlane(
        grid, points, options.trendModel == TrendModel::PlaneFromPoints);
    Vector result(grid.nx * grid.ny);
    for (std::size_t iy = 0; iy < grid.ny; ++iy) {
        const Real y = grid.miny + static_cast<Real>(iy) * grid.dy();
        for (std::size_t ix = 0; ix < grid.nx; ++ix) {
            const Real x = grid.minx + static_cast<Real>(ix) * grid.dx();
            result[surfaceIndex(ix, iy, grid.nx)] = plane.evaluate(x, y);
            if (!finite(result[surfaceIndex(ix, iy, grid.nx)])) {
                throw std::overflow_error(
                    "gaussian::interpolate: fitted trend is not finite");
            }
        }
    }
    return result;
}

[[nodiscard]] Real estimateStructuredVariance(
    const std::vector<PointStencil>& stencils,
    const Vector& trend)
{
    if (stencils.empty()) {
        return 1;
    }
    Real totalWeight = 0;
    Real weightedMean = 0;
    for (const PointStencil& stencil : stencils) {
        const Real residual = stencil.value - evaluateStencil(stencil, trend);
        totalWeight += stencil.weight;
        weightedMean += stencil.weight * residual;
        if (!finite(totalWeight) || !finite(weightedMean)) {
            throw std::overflow_error(
                "gaussian::interpolate: residual statistics overflow");
        }
    }
    weightedMean /= totalWeight;
    Real variance = 0;
    for (const PointStencil& stencil : stencils) {
        const Real residual = stencil.value
            - evaluateStencil(stencil, trend) - weightedMean;
        variance += stencil.weight * residual * residual;
        if (!finite(variance)) {
            throw std::overflow_error(
                "gaussian::interpolate: residual variance overflow");
        }
    }
    variance /= totalWeight;
    const Real valueScale = std::accumulate(
        stencils.begin(), stencils.end(), Real(0),
        [](Real current, const PointStencil& stencil) {
            return std::max(current, std::abs(stencil.value));
        });
    const Real floor = std::max<Real>(1, valueScale * valueScale) * 1e-12;
    return std::max(variance, floor);
}

[[nodiscard]] EffectiveParameters effectiveParameters(
    const GridGeometry& grid,
    const std::vector<PointStencil>& stencils,
    const Vector& trend,
    const Options& options)
{
    EffectiveParameters result;
    result.majorRange = options.majorRange > 0
        ? options.majorRange
        : 0.2 * (grid.maxx - grid.minx);
    result.minorRange = options.minorRange > 0
        ? options.minorRange
        : 0.2 * (grid.maxy - grid.miny);
    result.structuredVariance = options.structuredVariance > 0
        ? options.structuredVariance
        : estimateStructuredVariance(stencils, trend);
    result.nodeNuggetVariance = options.nodeNuggetVariance;
    const Real radians = options.azimuthDegrees * pi / 180;
    result.cosAzimuth = std::cos(radians);
    result.sinAzimuth = std::sin(radians);
    if (!(result.majorRange > 0) || !(result.minorRange > 0)
        || !(result.structuredVariance > 0)
        || !finite(result.majorRange) || !finite(result.minorRange)
        || !finite(result.structuredVariance)
        || !finite(result.nodeNuggetVariance)) {
        throw std::invalid_argument(
            "gaussian::interpolate: effective covariance parameters are not finite and positive");
    }
    return result;
}

[[nodiscard]] Real correlation(
    Real dx,
    Real dy,
    const EffectiveParameters& parameters,
    CorrelationModel model)
{
    const Real major = parameters.cosAzimuth * dx
        + parameters.sinAzimuth * dy;
    const Real minor = -parameters.sinAzimuth * dx
        + parameters.cosAzimuth * dy;
    // hypot избегает underflow range^2 и промежуточного overflow при очень
    // сильной анизотропии. При бесконечном normalized корреляция корректно 0.
    const Real normalized = std::hypot(
        major / parameters.majorRange,
        minor / parameters.minorRange);
    switch (model) {
    case CorrelationModel::Gaussian:
        return std::exp(-3 * normalized * normalized);
    case CorrelationModel::Exponential:
        return std::exp(-3 * normalized);
    case CorrelationModel::Spherical:
        if (normalized >= 1) {
            return 0;
        }
        return 1 - 1.5 * normalized
            + 0.5 * normalized * normalized * normalized;
    }
    throw std::invalid_argument("gaussian::interpolate: unknown correlation model");
}

[[nodiscard]] Real covariance(
    Real dx,
    Real dy,
    bool sameNode,
    const EffectiveParameters& parameters,
    CorrelationModel model)
{
    return parameters.structuredVariance
            * correlation(dx, dy, parameters, model)
        + (sameNode ? parameters.nodeNuggetVariance : 0);
}

[[nodiscard]] std::size_t nextPowerOfTwo(std::size_t value)
{
    if (value == 0) {
        return 1;
    }
    std::size_t result = 1;
    while (result < value) {
        if (result > std::numeric_limits<std::size_t>::max() / 2) {
            throw std::overflow_error(
                "gaussian::interpolate: FFT embedding is too large");
        }
        result *= 2;
    }
    return result;
}

void fft1d(Complex* values, std::size_t size, bool inverse)
{
    for (std::size_t i = 1, j = 0; i < size; ++i) {
        std::size_t bit = size >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(values[i], values[j]);
        }
    }
    for (std::size_t length = 2; length <= size; length <<= 1) {
        const Real angle = (inverse ? 2 : -2) * pi / static_cast<Real>(length);
        const Complex root(std::cos(angle), std::sin(angle));
        for (std::size_t start = 0; start < size; start += length) {
            Complex factor(1, 0);
            for (std::size_t offset = 0; offset < length / 2; ++offset) {
                const Complex even = values[start + offset];
                const Complex odd = values[start + offset + length / 2] * factor;
                values[start + offset] = even + odd;
                values[start + offset + length / 2] = even - odd;
                factor *= root;
            }
        }
        if (length == size) {
            break;
        }
    }
    if (inverse) {
        for (std::size_t i = 0; i < size; ++i) {
            values[i] /= static_cast<Real>(size);
        }
    }
}

void fft2d(
    ComplexVector& values,
    std::size_t nx,
    std::size_t ny,
    bool inverse)
{
    for (std::size_t y = 0; y < ny; ++y) {
        fft1d(values.data() + y * nx, nx, inverse);
    }
    ComplexVector column(ny);
    for (std::size_t x = 0; x < nx; ++x) {
        for (std::size_t y = 0; y < ny; ++y) {
            column[y] = values[y * nx + x];
        }
        fft1d(column.data(), ny, inverse);
        for (std::size_t y = 0; y < ny; ++y) {
            values[y * nx + x] = column[y];
        }
    }
}

struct Embedding {
    std::size_t nx = 0;
    std::size_t ny = 0;
    Vector eigenvalues;
    std::size_t expansions = 0;
};

[[nodiscard]] std::ptrdiff_t signedLag(std::size_t index, std::size_t size)
{
    return index <= size / 2
        ? static_cast<std::ptrdiff_t>(index)
        : static_cast<std::ptrdiff_t>(index)
            - static_cast<std::ptrdiff_t>(size);
}

[[nodiscard]] Embedding buildEmbedding(
    const GridGeometry& grid,
    const EffectiveParameters& parameters,
    const Options& options)
{
    if (grid.nx > std::numeric_limits<std::size_t>::max() / options.paddingFactor
        || grid.ny > std::numeric_limits<std::size_t>::max() / options.paddingFactor) {
        throw std::overflow_error("gaussian::interpolate: FFT dimensions overflow");
    }
    std::size_t embeddingNx = nextPowerOfTwo(grid.nx * options.paddingFactor);
    std::size_t embeddingNy = nextPowerOfTwo(grid.ny * options.paddingFactor);

    for (std::size_t expansion = 0;
         expansion < options.maxEmbeddingExpansions;
         ++expansion) {
        if (embeddingNx > std::numeric_limits<std::size_t>::max() / embeddingNy) {
            throw std::overflow_error("gaussian::interpolate: FFT embedding is too large");
        }
        const std::size_t count = embeddingNx * embeddingNy;
        ComplexVector kernel(count);
        for (std::size_t y = 0; y < embeddingNy; ++y) {
            const Real dy = static_cast<Real>(signedLag(y, embeddingNy)) * grid.dy();
            for (std::size_t x = 0; x < embeddingNx; ++x) {
                const Real dx = static_cast<Real>(signedLag(x, embeddingNx)) * grid.dx();
                kernel[y * embeddingNx + x] = covariance(
                    dx, dy, x == 0 && y == 0, parameters, options.correlationModel);
            }
        }

        // У повёрнутой анизотропии неоднозначны только Nyquist-строка и
        // Nyquist-столбец: +P/2 и -P/2 представляют один периодический лаг.
        // Реальные лаги целевой сетки при paddingFactor >= 2 сюда не попадают.
        const auto symmetrizePair = [&](std::size_t x, std::size_t y) {
            const std::size_t oppositeX = (embeddingNx - x) % embeddingNx;
            const std::size_t oppositeY = (embeddingNy - y) % embeddingNy;
            const std::size_t first = y * embeddingNx + x;
            const std::size_t second = oppositeY * embeddingNx + oppositeX;
            const Real average = 0.5
                * (kernel[first].real() + kernel[second].real());
            kernel[first] = average;
            kernel[second] = average;
        };
        for (std::size_t x = 0; x < embeddingNx; ++x) {
            symmetrizePair(x, embeddingNy / 2);
        }
        for (std::size_t y = 0; y < embeddingNy; ++y) {
            symmetrizePair(embeddingNx / 2, y);
        }

        fft2d(kernel, embeddingNx, embeddingNy, false);
        Vector eigenvalues(count);
        Real largest = 0;
        Real minimum = std::numeric_limits<Real>::infinity();
        for (std::size_t i = 0; i < count; ++i) {
            eigenvalues[i] = kernel[i].real();
            largest = std::max(largest, std::abs(eigenvalues[i]));
            minimum = std::min(minimum, eigenvalues[i]);
        }
        const Real tolerance = std::max<Real>(1, largest)
            * 4096 * std::numeric_limits<Real>::epsilon();
        if (minimum >= -tolerance) {
            for (Real& eigenvalue : eigenvalues) {
                eigenvalue = std::max<Real>(0, eigenvalue);
            }
            return {embeddingNx, embeddingNy, std::move(eigenvalues), expansion};
        }

        if (embeddingNx > std::numeric_limits<std::size_t>::max() / 2
            || embeddingNy > std::numeric_limits<std::size_t>::max() / 2) {
            break;
        }
        embeddingNx *= 2;
        embeddingNy *= 2;
    }
    throw std::runtime_error(
        "gaussian::interpolate: covariance has a negative FFT embedding spectrum; "
        "increase padding/expansions or reduce correlation ranges");
}

class CovarianceOperator {
public:
    CovarianceOperator(
        GridGeometry grid,
        Embedding embedding)
        : grid_(grid)
        , embedding_(std::move(embedding))
        , frequency_(embedding_.nx * embedding_.ny)
    {
    }

    [[nodiscard]] const Embedding& embedding() const
    {
        return embedding_;
    }

    void apply(const Vector& input, Vector& output)
    {
        std::fill(frequency_.begin(), frequency_.end(), Complex(0, 0));
        for (std::size_t y = 0; y < grid_.ny; ++y) {
            for (std::size_t x = 0; x < grid_.nx; ++x) {
                frequency_[y * embedding_.nx + x] =
                    input[surfaceIndex(x, y, grid_.nx)];
            }
        }
        fft2d(frequency_, embedding_.nx, embedding_.ny, false);
        for (std::size_t i = 0; i < frequency_.size(); ++i) {
            frequency_[i] *= embedding_.eigenvalues[i];
        }
        fft2d(frequency_, embedding_.nx, embedding_.ny, true);
        output.resize(grid_.nx * grid_.ny);
        for (std::size_t y = 0; y < grid_.ny; ++y) {
            for (std::size_t x = 0; x < grid_.nx; ++x) {
                output[surfaceIndex(x, y, grid_.nx)] =
                    frequency_[y * embedding_.nx + x].real();
            }
        }
    }

    template <typename NormalGenerator>
    [[nodiscard]] Vector sample(NormalGenerator& normal)
    {
        for (Complex& value : frequency_) {
            value = Complex(normal(), 0);
        }
        fft2d(frequency_, embedding_.nx, embedding_.ny, false);
        // F не нормирован, F^-1 содержит 1/M. При Cov(white)=I выражение
        // F^-1 sqrt(lambda) F white уже имеет covariance C; дополнительного
        // sqrt(M) здесь быть не должно (для identity kernel sample == white).
        for (std::size_t i = 0; i < frequency_.size(); ++i) {
            frequency_[i] *= std::sqrt(embedding_.eigenvalues[i]);
        }
        fft2d(frequency_, embedding_.nx, embedding_.ny, true);
        Vector result(grid_.nx * grid_.ny);
        for (std::size_t y = 0; y < grid_.ny; ++y) {
            for (std::size_t x = 0; x < grid_.nx; ++x) {
                result[surfaceIndex(x, y, grid_.nx)] =
                    frequency_[y * embedding_.nx + x].real();
            }
        }
        return result;
    }

private:
    GridGeometry grid_;
    Embedding embedding_;
    ComplexVector frequency_;
};

class NormalGenerator {
public:
    explicit NormalGenerator(std::uint64_t seed)
        : engine_(seed)
    {
    }

    Real operator()()
    {
        if (hasSpare_) {
            hasSpare_ = false;
            return spare_;
        }
        const Real u1 = uniformOpen();
        const Real u2 = uniformOpen();
        const Real radius = std::sqrt(-2 * std::log(u1));
        const Real angle = 2 * pi * u2;
        spare_ = radius * std::sin(angle);
        hasSpare_ = true;
        return radius * std::cos(angle);
    }

private:
    Real uniformOpen()
    {
        // Явное преобразование верхних 53 бит не зависит от реализации
        // std::uniform_real_distribution.
        constexpr Real scale = 1.0 / 9007199254740992.0;
        return (static_cast<Real>(engine_() >> 11) + 0.5) * scale;
    }

    std::mt19937_64 engine_;
    bool hasSpare_ = false;
    Real spare_ = 0;
};

[[nodiscard]] Real dot(const Vector& left, const Vector& right)
{
    return std::inner_product(left.begin(), left.end(), right.begin(), Real(0));
}

struct ConstraintStatistics {
    Real maxAbsolute = 0;
    Real maxRelative = 0;
    Real weightedSquared = 0;
    Real totalWeight = 0;
};

[[nodiscard]] ConstraintStatistics constraintStatistics(
    const std::vector<PointStencil>& stencils,
    const Vector& values)
{
    ConstraintStatistics result;
    for (const PointStencil& stencil : stencils) {
        const Real residual = evaluateStencil(stencil, values) - stencil.value;
        result.maxAbsolute = std::max(result.maxAbsolute, std::abs(residual));
        result.maxRelative = std::max(result.maxRelative,
            std::abs(residual) / std::max<Real>(1, std::abs(stencil.value)));
        result.weightedSquared += stencil.weight * residual * residual;
        result.totalWeight += stencil.weight;
    }
    return result;
}

[[nodiscard]] Vector conditioningDiagonal(
    const GridGeometry& grid,
    const std::vector<PointStencil>& stencils,
    const EffectiveParameters& parameters,
    CorrelationModel model)
{
    Vector diagonal(stencils.size(), 0);
    for (std::size_t pointIndex = 0;
         pointIndex < stencils.size();
         ++pointIndex) {
        const PointStencil& stencil = stencils[pointIndex];
        for (std::size_t a = 0; a < 4; ++a) {
            if (stencil.coefficient[a] == 0) {
                continue;
            }
            const std::size_t ax = stencil.index[a] % grid.nx;
            const std::size_t ay = stencil.index[a] / grid.nx;
            for (std::size_t b = 0; b < 4; ++b) {
                if (stencil.coefficient[b] == 0) {
                    continue;
                }
                const std::size_t bx = stencil.index[b] % grid.nx;
                const std::size_t by = stencil.index[b] / grid.nx;
                diagonal[pointIndex] += stencil.coefficient[a]
                    * stencil.coefficient[b]
                    * covariance(
                        (static_cast<Real>(ax) - static_cast<Real>(bx)) * grid.dx(),
                        (static_cast<Real>(ay) - static_cast<Real>(by)) * grid.dy(),
                        stencil.index[a] == stencil.index[b],
                        parameters,
                        model);
            }
        }
        if (!(diagonal[pointIndex] > 0) || !finite(diagonal[pointIndex])) {
            throw std::runtime_error(
                "gaussian::interpolate: zero covariance at a conditioning point");
        }
    }
    return diagonal;
}

struct ConditioningResult {
    std::size_t iterations = 0;
    std::size_t refinements = 0;
    bool satisfied = false;
};

[[nodiscard]] Real effectiveConstraintTolerance(
    const std::vector<PointStencil>& stencils,
    const Options& options);

void projectOntoHardConstraints(
    const std::vector<PointStencil>& stencils,
    const Options& options,
    Vector& values)
{
    if (stencils.empty()) {
        return;
    }
    const Real tolerance = effectiveConstraintTolerance(stencils, options);
    Vector rowScale(stencils.size(), 1);
    for (std::size_t row = 0; row < stencils.size(); ++row) {
        Real squaredNorm = 0;
        for (Real coefficient : stencils[row].coefficient) {
            squaredNorm += coefficient * coefficient;
        }
        rowScale[row] = 1 / std::sqrt(squaredNorm);
    }

    for (std::size_t refinement = 0; refinement < 3; ++refinement) {
        Vector rhs(stencils.size(), 0);
        Real maximum = 0;
        for (std::size_t row = 0; row < stencils.size(); ++row) {
            const Real residual = stencils[row].value
                - evaluateStencil(stencils[row], values);
            rhs[row] = rowScale[row] * residual;
            maximum = std::max(maximum, std::abs(residual));
        }
        if (maximum <= tolerance) {
            return;
        }

        Vector multiplier(stencils.size(), 0);
        Vector residual = rhs;
        Vector direction = residual;
        Vector gram(stencils.size(), 0);
        Vector nodes(values.size(), 0);
        Real residualSquared = dot(residual, residual);
        for (std::size_t iteration = 0;
             iteration < options.maxConditioningIterations;
             ++iteration) {
            std::fill(nodes.begin(), nodes.end(), 0);
            for (std::size_t row = 0; row < stencils.size(); ++row) {
                const Real scaled = rowScale[row] * direction[row];
                for (std::size_t local = 0; local < 4; ++local) {
                    nodes[stencils[row].index[local]] +=
                        stencils[row].coefficient[local] * scaled;
                }
            }
            for (std::size_t row = 0; row < stencils.size(); ++row) {
                gram[row] = rowScale[row]
                    * evaluateStencil(stencils[row], nodes);
            }
            const Real denominator = dot(direction, gram);
            if (!(denominator > 0) || !finite(denominator)) {
                break;
            }
            const Real alpha = residualSquared / denominator;
            for (std::size_t row = 0; row < stencils.size(); ++row) {
                multiplier[row] += alpha * direction[row];
                residual[row] -= alpha * gram[row];
            }
            const Real next = dot(residual, residual);
            if (std::sqrt(std::max<Real>(0, next))
                <= tolerance * 0.1) {
                break;
            }
            if (!(residualSquared > 0) || !finite(next)) {
                break;
            }
            const Real beta = next / residualSquared;
            for (std::size_t row = 0; row < stencils.size(); ++row) {
                direction[row] = residual[row] + beta * direction[row];
            }
            residualSquared = next;
        }

        for (std::size_t row = 0; row < stencils.size(); ++row) {
            const Real scaled = rowScale[row] * multiplier[row];
            for (std::size_t local = 0; local < 4; ++local) {
                values[stencils[row].index[local]] +=
                    stencils[row].coefficient[local] * scaled;
            }
        }
    }
}

[[nodiscard]] Real effectiveConstraintTolerance(
    const std::vector<PointStencil>& stencils,
    const Options& options)
{
    Real largestValue = 1;
    for (const PointStencil& stencil : stencils) {
        largestValue = std::max(largestValue, std::abs(stencil.value));
    }
    const Real requested = options.conditioningRelativeTolerance * largestValue;
    const Real representable = 128
        * static_cast<Real>(std::numeric_limits<qreal>::epsilon()) * largestValue;
    return std::max(requested, representable);
}

[[nodiscard]] ConditioningResult conditionRealization(
    const GridGeometry& grid,
    const std::vector<PointStencil>& stencils,
    const EffectiveParameters& parameters,
    const Options& options,
    CovarianceOperator& covarianceOperator,
    Vector& values)
{
    ConditioningResult result;
    if (stencils.empty()) {
        result.satisfied = true;
        return result;
    }

    const Vector diagonal = conditioningDiagonal(
        grid, stencils, parameters, options.correlationModel);
    const Real absoluteTolerance = effectiveConstraintTolerance(stencils, options);
    Vector nodeInput(grid.nx * grid.ny, 0);
    Vector nodeOutput(grid.nx * grid.ny, 0);
    Vector gramOutput(stencils.size(), 0);

    const auto applyGram = [&](const Vector& input, Vector& output) {
        scatterTranspose(stencils, input, nodeInput);
        covarianceOperator.apply(nodeInput, nodeOutput);
        gather(stencils, nodeOutput, output);
    };

    const Real initialMaximumResidual =
        constraintStatistics(stencils, values).maxAbsolute;
    if (initialMaximumResidual <= absoluteTolerance) {
        result.satisfied = true;
        return result;
    }

    for (std::size_t refinement = 1;
         refinement <= options.maxConditioningRefinements;
         ++refinement) {
        result.refinements = refinement;
        Vector rhs(stencils.size());
        Real maximum = 0;
        for (std::size_t i = 0; i < stencils.size(); ++i) {
            rhs[i] = stencils[i].value - evaluateStencil(stencils[i], values);
            maximum = std::max(maximum, std::abs(rhs[i]));
        }
        if (maximum <= absoluteTolerance) {
            result.satisfied = true;
            return result;
        }

        Vector multiplier(stencils.size(), 0);
        Vector residual = rhs;
        Vector preconditioned(stencils.size());
        Vector direction(stencils.size());
        for (std::size_t i = 0; i < stencils.size(); ++i) {
            preconditioned[i] = residual[i] / diagonal[i];
            direction[i] = preconditioned[i];
        }
        Real residualPreconditioned = dot(residual, preconditioned);

        for (std::size_t iteration = 0;
             iteration < options.maxConditioningIterations;
             ++iteration) {
            applyGram(direction, gramOutput);
            const Real denominator = dot(direction, gramOutput);
            if (!(denominator > 0) || !finite(denominator)
                || !finite(residualPreconditioned)) {
                break;
            }
            const Real alpha = residualPreconditioned / denominator;
            for (std::size_t i = 0; i < stencils.size(); ++i) {
                multiplier[i] += alpha * direction[i];
                residual[i] -= alpha * gramOutput[i];
            }
            ++result.iterations;

            maximum = 0;
            for (Real value : residual) {
                maximum = std::max(maximum, std::abs(value));
            }
            if (maximum <= absoluteTolerance * 0.25) {
                break;
            }

            for (std::size_t i = 0; i < stencils.size(); ++i) {
                preconditioned[i] = residual[i] / diagonal[i];
            }
            const Real next = dot(residual, preconditioned);
            if (!finite(next) || !(residualPreconditioned > 0)) {
                break;
            }
            const Real beta = next / residualPreconditioned;
            for (std::size_t i = 0; i < stencils.size(); ++i) {
                direction[i] = preconditioned[i] + beta * direction[i];
            }
            residualPreconditioned = next;
        }

        scatterTranspose(stencils, multiplier, nodeInput);
        covarianceOperator.apply(nodeInput, nodeOutput);
        for (std::size_t node = 0; node < values.size(); ++node) {
            values[node] += nodeOutput[node];
        }

        if (constraintStatistics(stencils, values).maxAbsolute
            <= absoluteTolerance) {
            result.satisfied = true;
            return result;
        }
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
    const GridGeometry grid{
        static_cast<Real>(minx), static_cast<Real>(maxx),
        static_cast<Real>(miny), static_cast<Real>(maxy), nx, ny};
    validateGrid(grid);
    validateOptions(options);
    const std::vector<Sample> validPoints = validateAndFilterPoints(
        grid, points, options.ignoreOutsidePoints);
    std::vector<Sample> canonicalPoints = validPoints;
    std::sort(canonicalPoints.begin(), canonicalPoints.end(),
        [](const Sample& left, const Sample& right) {
            return std::tie(left.x, left.y, left.value, left.weight)
                < std::tie(right.x, right.y, right.value, right.weight);
        });
    const std::vector<PointStencil> stencils = makePointStencils(
        grid, canonicalPoints, options.cellInterpolation);
    validateActiveSupport(stencils, options.activeNodeMask, nx * ny);
    if (options.enforcePointConstraints) {
        validateLocalConstraintCompatibility(
            stencils, options.conditioningRelativeTolerance);
    }

    Vector trend = makeTrend(grid, canonicalPoints, surfaceValues, options);
    const EffectiveParameters parameters = effectiveParameters(
        grid, stencils, trend, options);
    const Embedding embedding = buildEmbedding(grid, parameters, options);
    CovarianceOperator covarianceOperator(grid, embedding);
    NormalGenerator normal(options.seed);
    Vector values = covarianceOperator.sample(normal);
    for (std::size_t node = 0; node < values.size(); ++node) {
        values[node] += trend[node];
    }

    ConditioningResult conditioning;
    if (options.enforcePointConstraints) {
        conditioning = conditionRealization(
            grid, stencils, parameters, options, covarianceOperator, values);
        const Real covarianceResidual =
            constraintStatistics(stencils, values).maxAbsolute;
        const Real requestedTolerance =
            effectiveConstraintTolerance(stencils, options);
        if (!conditioning.satisfied
            && covarianceResidual > requestedTolerance * 1000) {
            throw std::runtime_error(
                "gaussian::interpolate: covariance conditioning solve did not "
                "converge; exact geometric fallback was not applied because "
                "it would distort the conditional Gaussian realization");
        }
        // Последний solve остаётся ковариационным и задаёт распределение.
        // Эта коррекция затрагивает только остаток порядка solver tolerance,
        // чтобы округление qreal не оставило визуально заметной щели.
        projectOntoHardConstraints(stencils, options, values);
        conditioning.satisfied =
            constraintStatistics(stencils, values).maxAbsolute
            <= effectiveConstraintTolerance(stencils, options);
        if (!conditioning.satisfied) {
            throw std::runtime_error(
                "gaussian::interpolate: hard conditioning constraints are "
                "incompatible or the covariance solve did not converge");
        }
    }

    surfaceValues.assign(values.begin(), values.end());
    Vector stored(surfaceValues.begin(), surfaceValues.end());
    const ConstraintStatistics statistics = constraintStatistics(stencils, stored);

    Report report;
    report.seedUsed = options.seed;
    report.effectiveMajorRange = parameters.majorRange;
    report.effectiveMinorRange = parameters.minorRange;
    report.effectiveStructuredVariance = parameters.structuredVariance;
    report.effectiveNodeNuggetVariance = parameters.nodeNuggetVariance;
    report.embeddingNx = embedding.nx;
    report.embeddingNy = embedding.ny;
    report.embeddingExpansions = embedding.expansions;
    report.conditioningIterations = conditioning.iterations;
    report.conditioningRefinements = conditioning.refinements;
    report.maxAbsolutePointResidual = statistics.maxAbsolute;
    report.weightedRmsPointResidual = statistics.totalWeight > 0
        ? std::sqrt(statistics.weightedSquared / statistics.totalWeight)
        : 0;
    report.pointConstraintsSatisfied = !options.enforcePointConstraints
        ? stencils.empty()
        : (stencils.empty()
            || statistics.maxAbsolute
                <= effectiveConstraintTolerance(stencils, options));
    report.converged = true;
    if (options.enforcePointConstraints && !report.pointConstraintsSatisfied) {
        throw std::runtime_error(
            "gaussian::interpolate: output qreal precision is insufficient "
            "to retain the requested hard point constraints");
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
    validateGrid(grid);
    if (surfaceValues.size() != nx * ny) {
        throw std::invalid_argument(
            "gaussian::evaluateSurface: values size must be nx * ny");
    }
    const std::vector<Sample> point = validateAndFilterPoints(
        grid, std::vector<Sample>{{x, y, 0, 1}}, false);
    const PointStencil stencil = makePointStencil(
        grid, point.front(), interpolation);
    Vector values(surfaceValues.begin(), surfaceValues.end());
    return static_cast<qreal>(evaluateStencil(stencil, values));
}

} // namespace geo::gaussian
