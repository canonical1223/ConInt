#include "convergent_interpolation.h"
#include "fault_geometry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>

namespace ci {
namespace {
using detail::Vec2;
using detail::FaultBarrier;
constexpr double nan = std::numeric_limits<double>::quiet_NaN();
constexpr std::size_t absent = std::numeric_limits<std::size_t>::max();

struct Datum {
    Vec2 p;
    double value, weight;
    std::size_t original;
};

struct Lattice {
    std::vector<std::size_t> ix, iy;
    std::vector<double> x, y, z;
    std::vector<unsigned char> valid;
    std::size_t nx() const { return x.size(); }
    std::size_t ny() const { return y.size(); }
    std::size_t at(std::size_t i, std::size_t j) const { return j * nx() + i; }
    Vec2 position(std::size_t id) const { return {x[id % nx()], y[id / nx()]}; }
};

struct Term { std::size_t id; double a; };
struct Row {
    std::array<Term, 8> terms{};
    std::size_t n = 0;
    double weight = 0, target = 0;
    double evaluate(const std::vector<double>& z) const {
        double r = 0;
        for (std::size_t k = 0; k < n; ++k) r += terms[k].a * z[terms[k].id];
        return r;
    }
};

double distance2(Vec2 a, Vec2 b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Pivoted 3x3 solve; a singular affine fit is explicitly rejected.
bool solve3(std::array<std::array<double, 3>, 3> a,
            std::array<double, 3> b, std::array<double, 3>& result) {
    double magnitude = 0;
    for (const auto& row : a) for (double v : row) magnitude = std::max(magnitude, std::abs(v));
    if (!(magnitude > 0)) return false;
    for (std::size_t j = 0; j < 3; ++j) {
        std::size_t pivot = j;
        for (std::size_t k = j + 1; k < 3; ++k)
            if (std::abs(a[k][j]) > std::abs(a[pivot][j])) pivot = k;
        if (std::abs(a[pivot][j]) < 1e-12 * magnitude) return false;
        std::swap(a[j], a[pivot]); std::swap(b[j], b[pivot]);
        const double d = a[j][j];
        for (std::size_t k = j; k < 3; ++k) a[j][k] /= d;
        b[j] /= d;
        for (std::size_t i = 0; i < 3; ++i) if (i != j) {
            const double f = a[i][j];
            for (std::size_t k = j; k < 3; ++k) a[i][k] -= f * a[j][k];
            b[i] -= f * b[j];
        }
    }
    result = b;
    return std::all_of(b.begin(), b.end(), [](double v) { return std::isfinite(v); });
}

std::vector<std::size_t> sampleIndices(std::size_t n, std::size_t stride) {
    std::vector<std::size_t> result{0};
    for (std::size_t i = stride; i < n - 1;) {
        result.push_back(i);
        if (stride >= n - 1 - i) break;
        i += stride;
    }
    if (result.back() != n - 1) result.push_back(n - 1);
    return result;
}

Lattice makeLattice(const Surface& surface, std::size_t stride,
                    double width, double height, const FaultBarrier& faults) {
    Lattice g;
    g.ix = sampleIndices(surface.nx, stride);
    g.iy = sampleIndices(surface.ny, stride);
    for (auto i : g.ix) g.x.push_back(width * static_cast<double>(i) / static_cast<double>(surface.nx - 1));
    for (auto j : g.iy) g.y.push_back(height * static_cast<double>(j) / static_cast<double>(surface.ny - 1));
    g.z.assign(g.nx() * g.ny(), nan);
    g.valid.resize(g.z.size());
    for (std::size_t i = 0; i < g.z.size(); ++i) g.valid[i] = !faults.onFault(g.position(i));
    return g;
}

// Safe off-grid value/first-order Taylor reconstruction. The complete filled
// support hull (including the query) must be free of faults. This condition is
// stronger than visibility of the query-to-node edges alone.
Row supportAt(Vec2 q, const Lattice& g, const FaultBarrier& faults) {
    Row result;
    auto xi = std::lower_bound(g.x.begin(), g.x.end(), q.x);
    auto yi = std::lower_bound(g.y.begin(), g.y.end(), q.y);
    const auto cx = static_cast<std::size_t>(xi - g.x.begin());
    const auto cy = static_cast<std::size_t>(yi - g.y.begin());
    std::vector<std::pair<double, std::size_t>> candidates;
    // Local extrapolation near a cut cell; do not silently search another block.
    const std::size_t radius = 3;
    const auto x0 = cx > radius ? cx - radius : 0;
    const auto y0 = cy > radius ? cy - radius : 0;
    const auto x1 = std::min(g.nx() - 1, cx + radius);
    const auto y1 = std::min(g.ny() - 1, cy + radius);
    for (std::size_t j = y0; j <= y1; ++j) for (std::size_t i = x0; i <= x1; ++i) {
        const auto id = g.at(i, j);
        if (!g.valid[id] || !std::isfinite(g.z[id])) continue;
        const Vec2 p = g.position(id);
        if (faults.blocked(q, p)) continue;
        const double d = distance2(q, p);
        if (d <= 1e-28) {
            result.n = 1; result.terms[0] = {id, 1}; return result;
        }
        candidates.emplace_back(d, id);
    }
    std::sort(candidates.begin(), candidates.end());
    std::vector<Vec2> hull{q};
    std::vector<std::pair<double, std::size_t>> selected;
    for (auto c : candidates) {
        hull.push_back(g.position(c.second));
        if (faults.hullBlocked(hull)) { hull.pop_back(); continue; }
        selected.push_back(c);
        if (selected.size() == result.terms.size()) break;
    }
    if (selected.empty()) return result;
    // Moving least squares shape functions: sum(a)=1, sum(a*dx)=sum(a*dy)=0.
    const double scale = std::sqrt(selected.back().first);
    const double d0 = selected.front().first;
    std::array<std::array<double, 3>, 3> moment{};
    std::array<std::array<double, 3>, 8> basis{};
    std::array<double, 8> weights{};
    for (std::size_t k = 0; k < selected.size(); ++k) {
        const Vec2 p = g.position(selected[k].second);
        basis[k] = {1, (p.x - q.x) / scale, (p.y - q.y) / scale};
        weights[k] = d0 / selected[k].first;
        for (std::size_t i = 0; i < 3; ++i) for (std::size_t j = 0; j < 3; ++j)
            moment[i][j] += weights[k] * basis[k][i] * basis[k][j];
    }
    std::array<double, 3> beta{};
    double absoluteSum = 0;
    if (selected.size() >= 3 && solve3(moment, {1, 0, 0}, beta)) {
        result.n = selected.size();
        double sum = 0;
        for (std::size_t k = 0; k < result.n; ++k) {
            const double a = weights[k] * std::inner_product(beta.begin(), beta.end(), basis[k].begin(), 0.0);
            result.terms[k] = {selected[k].second, a};
            sum += a; absoluteSum += std::abs(a);
        }
        if (absoluteSum <= 16 && std::abs(sum) > 0.5) {
            for (std::size_t k = 0; k < result.n; ++k) result.terms[k].a /= sum;
            return result;
        }
    }
    // Insufficient geometry (e.g. a one-node-wide block): constant reconstruction.
    result.n = 1; result.terms[0] = {selected.front().second, 1};
    return result;
}

// Initial first-order Taylor trend fitted only to directly visible controls.
// It is not the final interpolator: all controls are subsequently re-applied in
// the weighted least-squares system on every lattice.
double initialTrend(Vec2 q, const std::vector<Datum>& data, const FaultBarrier& faults) {
    std::vector<std::pair<double, std::size_t>> order;
    order.reserve(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) order.emplace_back(distance2(q, data[i].p), i);
    std::sort(order.begin(), order.end());
    std::vector<std::pair<double, std::size_t>> selected;
    std::vector<Vec2> hull{q};
    for (auto c : order) {
        if (faults.blocked(q, data[c.second].p)) continue;
        hull.push_back(data[c.second].p);
        if (faults.hullBlocked(hull)) { hull.pop_back(); continue; }
        selected.push_back(c);
        if (selected.size() == 16) break;
    }
    if (selected.empty()) return nan;
    const double scale = std::max(std::sqrt(selected.back().first), 1e-12);
    // Center values as well as coordinates to avoid loss of precision at depth.
    const double reference = data[selected.front().second].value;
    double maxWeight = 0;
    for (auto c : selected) maxWeight = std::max(maxWeight, data[c.second].weight);
    std::array<std::array<double, 3>, 3> a{};
    std::array<double, 3> b{};
    double sum = 0, sumValue = 0;
    for (auto c : selected) {
        const Datum& d = data[c.second];
        const std::array<double, 3> v{1, (d.p.x - q.x) / scale, (d.p.y - q.y) / scale};
        const double w = (d.weight / maxWeight) / (1 + c.first / (scale * scale));
        sum += w; sumValue += w * (d.value - reference);
        for (std::size_t i = 0; i < 3; ++i) {
            b[i] += w * v[i] * (d.value - reference);
            for (std::size_t j = 0; j < 3; ++j) a[i][j] += w * v[i] * v[j];
        }
    }
    // A weak slope regularizer selects zero slope in unconstrained directions.
    a[1][1] += sum * 1e-12; a[2][2] += sum * 1e-12;
    std::array<double, 3> fit{};
    if (solve3(a, b, fit)) return reference + fit[0];
    return reference + sumValue / sum;
}

// Fill only through uncut lattice edges. A component without a single seed
// remains NaN. Neither zeros nor another block's mean are invented for it.
void extendSeeds(Lattice& g, const FaultBarrier& faults) {
    std::queue<std::size_t> queue;
    for (std::size_t id = 0; id < g.z.size(); ++id)
        if (g.valid[id] && std::isfinite(g.z[id])) queue.push(id);
    while (!queue.empty()) {
        const auto id = queue.front(); queue.pop();
        const auto i = id % g.nx(), j = id / g.nx();
        const std::array<std::size_t, 4> near{
            i ? id - 1 : absent, i + 1 < g.nx() ? id + 1 : absent,
            j ? id - g.nx() : absent, j + 1 < g.ny() ? id + g.nx() : absent};
        for (auto other : near) {
            if (other == absent || !g.valid[other] || std::isfinite(g.z[other])) continue;
            if (faults.blocked(g.position(id), g.position(other))) continue;
            g.z[other] = g.z[id]; queue.push(other);
        }
    }
}

double cellWidth(const std::vector<double>& axis, std::size_t i) {
    if (i == 0) return (axis[1] - axis[0]) * 0.5;
    if (i + 1 == axis.size()) return (axis[i] - axis[i - 1]) * 0.5;
    return (axis[i + 1] - axis[i - 1]) * 0.5;
}

void appendCurvature(std::vector<Row>& rows, const Lattice& g,
                     const FaultBarrier& faults, const std::vector<std::size_t>& index,
                     double referenceArea, double lambda) {
    if (lambda == 0) return;
    // referenceArea is FINAL cell area; this fixes the energy scaling across
    // levels. Nonuniform last cells and unequal x/y spacing are both supported.
    for (std::size_t j = 0; j < g.ny(); ++j) for (std::size_t i = 0; i < g.nx(); ++i) {
        const auto id = g.at(i, j);
        if (index[id] == absent) continue;
        const double area = cellWidth(g.x, i) * cellWidth(g.y, j);
        if (i && i + 1 < g.nx() && index[id - 1] != absent && index[id + 1] != absent &&
            !faults.blocked(g.position(id - 1), g.position(id + 1))) {
            const double left = g.x[i] - g.x[i - 1], right = g.x[i + 1] - g.x[i];
            const double factor = 2 * referenceArea / (left + right);
            Row r; r.n = 3; r.weight = lambda * area / referenceArea;
            r.terms[0] = {index[id - 1], factor / left};
            r.terms[1] = {index[id], -factor * (1 / left + 1 / right)};
            r.terms[2] = {index[id + 1], factor / right}; rows.push_back(r);
        }
        if (j && j + 1 < g.ny() && index[id - g.nx()] != absent && index[id + g.nx()] != absent &&
            !faults.blocked(g.position(id - g.nx()), g.position(id + g.nx()))) {
            const double down = g.y[j] - g.y[j - 1], up = g.y[j + 1] - g.y[j];
            const double factor = 2 * referenceArea / (down + up);
            Row r; r.n = 3; r.weight = lambda * area / referenceArea;
            r.terms[0] = {index[id - g.nx()], factor / down};
            r.terms[1] = {index[id], -factor * (1 / down + 1 / up)};
            r.terms[2] = {index[id + g.nx()], factor / up}; rows.push_back(r);
        }
        if (i + 1 < g.nx() && j + 1 < g.ny()) {
            const auto right = id + 1, above = id + g.nx(), diagonal = above + 1;
            if (index[right] == absent || index[above] == absent || index[diagonal] == absent) continue;
            if (faults.intersectsBox(g.position(id), g.position(diagonal))) continue;
            const double rectangleArea = (g.x[i + 1] - g.x[i]) * (g.y[j + 1] - g.y[j]);
            const double factor = referenceArea / rectangleArea;
            Row r; r.n = 4; r.weight = 2 * lambda * rectangleArea / referenceArea;
            r.terms[0] = {index[id], factor}; r.terms[1] = {index[right], -factor};
            r.terms[2] = {index[above], -factor}; r.terms[3] = {index[diagonal], factor};
            rows.push_back(r);
        }
    }
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

struct SolverResult { bool converged; std::size_t iterations; double residual; };

// Matrix-free Jacobi-preconditioned CG for (sum w a a^T + epsilon I) z = b.
// The positive prior removes affine / disconnected-small-block nullspaces.
SolverResult solve(const std::vector<Row>& rows, std::vector<double>& z,
                   const Options& options) {
    if (z.empty()) return {true, 0, 0};
    const std::size_t n = z.size();
    std::vector<double> b(n), diagonal(n, options.priorWeight);
    for (std::size_t i = 0; i < n; ++i) b[i] = options.priorWeight * z[i];
    for (const Row& row : rows) for (std::size_t k = 0; k < row.n; ++k) {
        const auto t = row.terms[k];
        b[t.id] += row.weight * t.a * row.target;
        diagonal[t.id] += row.weight * t.a * t.a;
    }
    auto apply = [&](const std::vector<double>& x, std::vector<double>& y) {
        for (std::size_t i = 0; i < n; ++i) y[i] = options.priorWeight * x[i];
        for (const Row& row : rows) {
            const double v = row.weight * row.evaluate(x);
            for (std::size_t k = 0; k < row.n; ++k) y[row.terms[k].id] += row.terms[k].a * v;
        }
    };
    std::vector<double> r(n), p(n), preconditioned(n), ap(n);
    apply(z, ap);
    for (std::size_t i = 0; i < n; ++i) {
        r[i] = b[i] - ap[i]; preconditioned[i] = r[i] / diagonal[i]; p[i] = preconditioned[i];
    }
    const double denominator = std::max(1.0, std::sqrt(dot(b, b)));
    auto relativeResidual = [&]() { return std::sqrt(dot(r, r)) / denominator; };
    double residual = relativeResidual();
    if (!std::isfinite(residual)) throw std::runtime_error("Non-finite initial linear-system residual");
    if (residual <= options.relativeTolerance) return {true, 0, residual};
    double rz = dot(r, preconditioned);
    std::size_t iteration = 0;
    for (; iteration < options.maxIterations; ++iteration) {
        apply(p, ap);
        const double pap = dot(p, ap);
        if (!(pap > 0) || !std::isfinite(pap) || !std::isfinite(rz)) break;
        const double alpha = rz / pap;
        for (std::size_t i = 0; i < n; ++i) { z[i] += alpha * p[i]; r[i] -= alpha * ap[i]; }
        residual = relativeResidual();
        // Check the actual residual periodically and before declaring success.
        if ((iteration + 1) % 50 == 0 || residual <= options.relativeTolerance) {
            apply(z, ap);
            for (std::size_t i = 0; i < n; ++i) r[i] = b[i] - ap[i];
            residual = relativeResidual();
        }
        if (!std::isfinite(residual)) throw std::runtime_error("Non-finite linear-system residual");
        if (residual <= options.relativeTolerance) return {true, iteration + 1, residual};
        for (std::size_t i = 0; i < n; ++i) preconditioned[i] = r[i] / diagonal[i];
        const double nextRz = dot(r, preconditioned);
        const double beta = nextRz / rz;
        for (std::size_t i = 0; i < n; ++i) p[i] = preconditioned[i] + beta * p[i];
        rz = nextRz;
    }
    apply(z, ap);
    for (std::size_t i = 0; i < n; ++i) r[i] = b[i] - ap[i];
    return {false, iteration, relativeResidual()};
}

void validate(const Surface& s, const std::vector<Point>& points,
              const std::vector<Fault>& faults, const Options& o) {
    auto finite = [](qreal v) { return std::isfinite(static_cast<double>(v)); };
    if (s.nx < 2 || s.ny < 2 || s.nx > std::numeric_limits<std::size_t>::max() / s.ny)
        throw std::invalid_argument("nx and ny must be >= 2, and nx*ny must fit size_t");
    if (!finite(s.minx) || !finite(s.maxx) || !finite(s.miny) || !finite(s.maxy) ||
        !(s.maxx > s.minx) || !(s.maxy > s.miny)) throw std::invalid_argument("Invalid surface bounds");
    if (!s.grid.empty() && s.grid.size() != s.nx * s.ny) throw std::invalid_argument("grid size must equal nx*ny");
    for (qreal value : s.grid) if (std::isinf(static_cast<double>(value)))
        throw std::invalid_argument("grid may contain NaN, but not infinity");
    for (const Point& p : points) {
        if (!finite(p.x) || !finite(p.y) || !finite(p.value) || !finite(p.weight) || p.weight < 0)
            throw std::invalid_argument("Control coordinates/value/weight must be finite; weight >= 0");
        if (p.weight > 0 && (p.x < s.minx || p.x > s.maxx || p.y < s.miny || p.y > s.maxy))
            throw std::invalid_argument("Positive-weight control point is outside the surface");
    }
    for (const Fault& f : faults) for (const Point& p : f)
        if (!finite(p.x) || !finite(p.y)) throw std::invalid_argument("Non-finite fault vertex");
    if (o.initialCoarseningFactor < 1 || o.maxIterations < 1 ||
        !(o.relativeTolerance > 0) || !std::isfinite(o.relativeTolerance) ||
        o.smoothness < 0 || !std::isfinite(o.smoothness) ||
        !(o.priorWeight > 0) || !std::isfinite(o.priorWeight) ||
        o.geometryTolerance < 0 || !std::isfinite(o.geometryTolerance))
        throw std::invalid_argument("Invalid interpolation options");
}
} // namespace

Report interpolate(Surface& surface, const std::vector<Point>& points,
                   const std::vector<Fault>& polylines, const Options& options) {
    validate(surface, points, polylines, options);
    const double extentX = static_cast<double>(surface.maxx) - static_cast<double>(surface.minx);
    const double extentY = static_cast<double>(surface.maxy) - static_cast<double>(surface.miny);
    const double scale = std::max(extentX, extentY);
    if (!std::isfinite(scale) || !(scale > 0)) throw std::invalid_argument("Unrepresentable extent");
    const double width = extentX / scale, height = extentY / scale;
    const double dx = width / static_cast<double>(surface.nx - 1);
    const double dy = height / static_cast<double>(surface.ny - 1);
    const double tolerance = options.geometryTolerance > 0 ? options.geometryTolerance / scale : 1e-10;
    if (!(dx > 0) || !(dy > 0) || !(dx * dy > 0) || tolerance >= 0.25 * std::min(dx, dy))
        throw std::invalid_argument("Grid spacing too small for the geometry tolerance");
    auto normalize = [&](qreal x, qreal y) -> Vec2 {
        return {(static_cast<double>(x) - static_cast<double>(surface.minx)) / scale,
                (static_cast<double>(y) - static_cast<double>(surface.miny)) / scale};
    };
    std::vector<detail::Segment> segments;
    for (const Fault& f : polylines) for (std::size_t i = 1; i < f.size(); ++i)
        segments.push_back({normalize(f[i - 1].x, f[i - 1].y), normalize(f[i].x, f[i].y)});
    const FaultBarrier barriers(std::move(segments), tolerance);
    std::vector<Datum> data;
    double largestWeight = 0;
    for (const Point& p : points) largestWeight = std::max(largestWeight, static_cast<double>(p.weight));
    double relativeWeightSum = 0;
    for (const Point& p : points) if (p.weight > 0) relativeWeightSum += static_cast<double>(p.weight) / largestWeight;
    const auto positiveCount = static_cast<std::size_t>(std::count_if(points.begin(), points.end(),
        [](const Point& p) { return p.weight > 0; }));
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Point& p = points[i];
        if (p.weight == 0) continue;
        const Vec2 xy = normalize(p.x, p.y);
        if (barriers.onFault(xy)) throw std::invalid_argument("Positive-weight control point lies on a fault; a side ID is required");
        const double w = (static_cast<double>(p.weight) / largestWeight) *
            (static_cast<double>(positiveCount) / relativeWeightSum);
        if (!(w > 0)) throw std::invalid_argument("Control weight dynamic range is too large");
        data.push_back({xy, static_cast<double>(p.value), w, i});
    }

    Report report;
    Lattice parent;
    std::size_t stride = std::min(options.initialCoarseningFactor, std::max(surface.nx - 1, surface.ny - 1));
    for (;;) {
        Lattice g = makeLattice(surface, stride, width, height, barriers);
        for (std::size_t id = 0; id < g.z.size(); ++id) {
            if (!g.valid[id]) continue;
            const Vec2 p = g.position(id);
            const auto original = g.iy[id / g.nx()] * surface.nx + g.ix[id % g.nx()];
            const double initial = options.useInitialGrid && !surface.grid.empty()
                ? static_cast<double>(surface.grid[original]) : nan;
            g.z[id] = initial;
            if (!parent.z.empty()) {
                const Row support = supportAt(p, parent, barriers);
                if (support.n && !std::isfinite(initial)) g.z[id] = support.evaluate(parent.z);
                else if (support.n) {
                    // Keep fine detail of the supplied initial grid. Prolong only
                    // the correction, not the coarse approximation of that grid.
                    double correction = 0;
                    bool hasBaseline = true;
                    for (std::size_t k = 0; k < support.n; ++k) {
                        const auto parentId = support.terms[k].id;
                        const auto baselineId = parent.iy[parentId / parent.nx()] * surface.nx
                            + parent.ix[parentId % parent.nx()];
                        const double baseline = static_cast<double>(surface.grid[baselineId]);
                        if (!std::isfinite(baseline)) { hasBaseline = false; break; }
                        correction += support.terms[k].a * (parent.z[parentId] - baseline);
                    }
                    if (hasBaseline) g.z[id] += correction;
                }
            }
            if (!std::isfinite(g.z[id])) g.z[id] = initialTrend(p, data, barriers);
        }
        extendSeeds(g, barriers);
        std::vector<std::size_t> index(g.z.size(), absent);
        std::vector<double> unknown;
        double reference = 0;
        for (double v : g.z) if (std::isfinite(v)) { reference = v; break; }
        for (std::size_t id = 0; id < g.z.size(); ++id) if (std::isfinite(g.z[id])) {
            index[id] = unknown.size(); unknown.push_back(g.z[id] - reference);
        }
        std::vector<Row> rows;
        rows.reserve(data.size() + 3 * g.z.size());
        std::vector<Row> observations;
        std::vector<const Datum*> represented;
        for (const Datum& d : data) {
            Row row = supportAt(d.p, g, barriers);
            if (!row.n) {
                if (stride == 1) report.unrepresentedControlPoints.push_back(d.original);
                continue;
            }
            row.weight = d.weight; row.target = d.value;
            observations.push_back(row); represented.push_back(&d);
            double coefficientSum = 0;
            for (std::size_t k = 0; k < row.n; ++k) {
                coefficientSum += row.terms[k].a;
                row.terms[k].id = index[row.terms[k].id];
            }
            row.target -= reference * coefficientSum;
            rows.push_back(row);
        }
        appendCurvature(rows, g, barriers, index, dx * dy, options.smoothness);
        const SolverResult result = solve(rows, unknown, options);
        report.converged = report.converged && result.converged;
        report.iterations += result.iterations; ++report.levels;
        for (std::size_t id = 0; id < g.z.size(); ++id) if (index[id] != absent) {
            const double value = unknown[index[id]] + reference;
            if (!std::isfinite(value) || std::abs(value) > static_cast<double>(std::numeric_limits<qreal>::max()))
                throw std::runtime_error("Interpolation produced an unrepresentable value");
            g.z[id] = value;
        }
        if (stride == 1) {
            report.usedControlPoints = observations.size();
            report.finalRelativeResidual = result.residual;
            report.converged = report.converged && report.unrepresentedControlPoints.empty();
            double sumWeight = 0, squaredError = 0, maxError = 0;
            for (std::size_t i = 0; i < observations.size(); ++i) {
                const double error = observations[i].evaluate(g.z) - represented[i]->value;
                sumWeight += represented[i]->weight;
                squaredError += represented[i]->weight * error * error;
                maxError = std::max(maxError, std::abs(error));
            }
            if (sumWeight > 0) {
                report.weightedRms = std::sqrt(squaredError / sumWeight);
                report.maxAbsoluteResidual = maxError;
            }
            std::vector<qreal> output; output.reserve(g.z.size());
            for (double v : g.z) {
                if (!std::isfinite(v)) ++report.undefinedNodes;
                output.push_back(static_cast<qreal>(v));
            }
            surface.grid = std::move(output);
            return report;
        }
        parent = std::move(g);
        stride = std::max<std::size_t>(1, stride / 2);
    }
}
} // namespace ci
