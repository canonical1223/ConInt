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

using Scalar = double;

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
    if (!nonNegativeFinite(options.smoothness)
        || !positiveFinite(options.priorWeight)
        || !nonNegativeFinite(options.snapStrength)
        || !nonNegativeFinite(options.finalPointStrength)
        || !positiveFinite(options.gaussianSigma)
        || !positiveFinite(options.relativeTolerance)
        || !nonNegativeFinite(options.absoluteTolerance)) {
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
    if (surface.gorid.size() != expected) {
        throw std::invalid_argument("surface.gorid.size() must equal surface.nx * surface.ny");
    }
    for (qreal value : surface.gorid) {
        if (!finite(static_cast<Scalar>(value))) {
            throw std::invalid_argument("surface.gorid must contain only finite values");
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

    // A canonical order makes accumulation deterministic for any permutation of
    // the same points.
    std::sort(result.begin(), result.end(), [](const CanonicalPoint& a, const CanonicalPoint& b) {
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        if (a.value != b.value) return a.value < b.value;
        return a.weight < b.weight;
    });

    // Exact co-located controls are one physical constraint.  Combining them
    // before normalisation makes {(z=10,w=1),(z=30,w=3)} exactly equivalent
    // to {(z=25,w=4)} and avoids two redundant rank-one constraints.
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
    std::transform(surface.gorid.begin(), surface.gorid.end(), result.values.begin(),
        [](qreal value) { return static_cast<Scalar>(value); });
    return result;
}

std::size_t ceilDivide(std::size_t value, std::size_t divisor)
{
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

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

    // Very thin grids near a boundary can need more candidates than the local
    // square supplied.  Falling back to all nodes is rare and remains exact.
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
    long double result = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result += static_cast<long double>(a[i]) * static_cast<long double>(b[i]);
    }
    return static_cast<Scalar>(result);
}

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

        // Periodically remove recursive-residual drift.
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
        return report; // The documented no-controls identity operation.
    }

    const Grid input = surfaceAsGrid(surface);
    const auto hierarchy = buildHierarchy(surface.nx, surface.ny, options);
    Grid coarsest = resample(input, hierarchy.front().first, hierarchy.front().second);
    Grid solved = coarsest;

    for (std::size_t levelIndex = 0; levelIndex < hierarchy.size(); ++levelIndex) {
        const auto [nx, ny] = hierarchy[levelIndex];
        Grid prior = levelIndex == 0 ? coarsest : resample(solved, nx, ny);
        const bool finalLevel = levelIndex + 1 == hierarchy.size();
        const std::size_t snapNodes = snapNodeCount(prior, coarsest, finalLevel, options);

        std::vector<Scalar> snapDiagonal(prior.values.size(), Scalar(0));
        std::vector<Scalar> snapRhs(prior.values.size(), Scalar(0));
        addSnapConstraints(prior, controls, snapNodes, options, snapDiagonal, snapRhs);
        const std::vector<PointConstraint> finalConstraints = finalLevel
            ? makePointConstraints(prior, controls)
            : std::vector<PointConstraint>{};

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

    // Report the residual of the values that are actually returned, including
    // any loss of precision when a Qt build uses a narrower qreal.
    Grid returned = solved;
    std::transform(output.begin(), output.end(), returned.values.begin(),
        [](qreal value) { return static_cast<Scalar>(value); });
    report.maxControlError = static_cast<qreal>(maxControlError(returned, controls));

    surface.gorid = std::move(output);
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
