#include "fault_geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ci::detail {
namespace {

constexpr std::size_t noNode = std::numeric_limits<std::size_t>::max();
constexpr std::size_t leafSize = 8;

bool finite(Vec2 p) {
    return std::isfinite(p.x) && std::isfinite(p.y);
}

void validate(Vec2 p) {
    if (!finite(p)) {
        throw std::invalid_argument("Fault geometry coordinates must be finite");
    }
}

long double orient(Vec2 a, Vec2 b, Vec2 p) {
    const long double dx = static_cast<long double>(b.x) - a.x;
    const long double dy = static_cast<long double>(b.y) - a.y;
    const long double px = static_cast<long double>(p.x) - a.x;
    const long double py = static_cast<long double>(p.y) - a.y;
    return dx * py - dy * px;
}

long double distanceSquared(Vec2 a, Vec2 b) {
    const long double dx = static_cast<long double>(a.x) - b.x;
    const long double dy = static_cast<long double>(a.y) - b.y;
    return dx * dx + dy * dy;
}

long double pointSegmentDistanceSquared(Vec2 p, Segment segment) {
    const long double dx = static_cast<long double>(segment.b.x) - segment.a.x;
    const long double dy = static_cast<long double>(segment.b.y) - segment.a.y;
    const long double px = static_cast<long double>(p.x) - segment.a.x;
    const long double py = static_cast<long double>(p.y) - segment.a.y;
    const long double lengthSquared = dx * dx + dy * dy;
    if (lengthSquared == 0.0L) {
        return px * px + py * py;
    }
    const long double t = std::clamp((px * dx + py * dy) / lengthSquared,
                                   0.0L, 1.0L);
    const long double ex = px - t * dx;
    const long double ey = py - t * dy;
    return ex * ex + ey * ey;
}

bool opposite(long double a, long double b) {
    return (a < 0.0L && b > 0.0L) || (a > 0.0L && b < 0.0L);
}

bool pointOnClosedSegment(Vec2 p, Segment segment) {
    return orient(segment.a, segment.b, p) == 0.0L &&
           p.x >= std::min(segment.a.x, segment.b.x) &&
           p.x <= std::max(segment.a.x, segment.b.x) &&
           p.y >= std::min(segment.a.y, segment.b.y) &&
           p.y <= std::max(segment.a.y, segment.b.y);
}

bool segmentNear(Segment a, Segment b, long double toleranceSquared) {
    // Strict crossings are the only intersections not captured by endpoint
    // distances. The distance tests include touches and collinear overlaps.
    if (opposite(orient(a.a, a.b, b.a), orient(a.a, a.b, b.b)) &&
        opposite(orient(b.a, b.b, a.a), orient(b.a, b.b, a.b))) {
        return true;
    }
    // Explicit closed containment also handles zero tolerance without asking
    // a projection with a rounded parameter to produce an exact zero distance.
    if (pointOnClosedSegment(a.a, b) || pointOnClosedSegment(a.b, b) ||
        pointOnClosedSegment(b.a, a) || pointOnClosedSegment(b.b, a)) {
        return true;
    }
    return pointSegmentDistanceSquared(a.a, b) <= toleranceSquared ||
           pointSegmentDistanceSquared(a.b, b) <= toleranceSquared ||
           pointSegmentDistanceSquared(b.a, a) <= toleranceSquared ||
           pointSegmentDistanceSquared(b.b, a) <= toleranceSquared;
}

std::vector<Vec2> convexHull(std::vector<Vec2> points) {
    std::sort(points.begin(), points.end(), [](Vec2 a, Vec2 b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    points.erase(std::unique(points.begin(), points.end(), [](Vec2 a, Vec2 b) {
        return a.x == b.x && a.y == b.y;
    }), points.end());
    if (points.size() <= 2) {
        return points;
    }

    std::vector<Vec2> hull;
    hull.reserve(points.size() * 2);
    for (Vec2 p : points) {
        while (hull.size() >= 2 &&
               orient(hull[hull.size() - 2], hull.back(), p) <= 0.0L) {
            hull.pop_back();
        }
        hull.push_back(p);
    }
    const std::size_t lowerSize = hull.size();
    for (std::size_t i = points.size() - 1; i > 0; --i) {
        const Vec2 p = points[i - 1];
        while (hull.size() > lowerSize &&
               orient(hull[hull.size() - 2], hull.back(), p) <= 0.0L) {
            hull.pop_back();
        }
        hull.push_back(p);
    }
    hull.pop_back();
    return hull;
}

bool insideConvex(Vec2 p, const std::vector<Vec2>& hull) {
    // A proper convex hull is counterclockwise. Boundary is included.
    for (std::size_t i = 0; i < hull.size(); ++i) {
        if (orient(hull[i], hull[(i + 1) % hull.size()], p) < 0.0L) {
            return false;
        }
    }
    return true;
}

bool segmentNearHull(Segment segment, const std::vector<Vec2>& hull,
                     long double toleranceSquared) {
    if (insideConvex(segment.a, hull) || insideConvex(segment.b, hull)) {
        return true;
    }
    for (std::size_t i = 0; i < hull.size(); ++i) {
        if (segmentNear(segment, {hull[i], hull[(i + 1) % hull.size()]},
                        toleranceSquared)) {
            return true;
        }
    }
    return false;
}

} // namespace

FaultBarrier::FaultBarrier(std::vector<Segment> segments, double tolerance)
    : tolerance_(tolerance) {
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        throw std::invalid_argument("Fault tolerance must be finite and nonnegative");
    }
    const long double toleranceSquared =
        static_cast<long double>(tolerance) * tolerance;
    for (Segment segment : segments) {
        validate(segment.a);
        validate(segment.b);
        if (distanceSquared(segment.a, segment.b) > toleranceSquared) {
            segments_.push_back(segment);
        }
    }
    if (!segments_.empty()) {
        build(0, segments_.size());
    }
}

FaultBarrier::Bounds FaultBarrier::boundsOf(Segment segment) {
    return {std::min(segment.a.x, segment.b.x),
            std::min(segment.a.y, segment.b.y),
            std::max(segment.a.x, segment.b.x),
            std::max(segment.a.y, segment.b.y)};
}

FaultBarrier::Bounds FaultBarrier::merged(Bounds a, Bounds b) {
    return {std::min(a.minx, b.minx), std::min(a.miny, b.miny),
            std::max(a.maxx, b.maxx), std::max(a.maxy, b.maxy)};
}

bool FaultBarrier::overlaps(Bounds a, Bounds b) {
    return a.minx <= b.maxx && b.minx <= a.maxx &&
           a.miny <= b.maxy && b.miny <= a.maxy;
}

FaultBarrier::Bounds FaultBarrier::expanded(Bounds bounds) const {
    // nextafter prevents pruning an exact tolerance contact due to a rounded
    // inward floating point bound. Leaf tests still use Euclidean distance.
    const double infinity = std::numeric_limits<double>::infinity();
    return {std::nextafter(bounds.minx - tolerance_, -infinity),
            std::nextafter(bounds.miny - tolerance_, -infinity),
            std::nextafter(bounds.maxx + tolerance_, infinity),
            std::nextafter(bounds.maxy + tolerance_, infinity)};
}

std::size_t FaultBarrier::build(std::size_t begin, std::size_t end) {
    Bounds bounds = boundsOf(segments_[begin]);
    for (std::size_t i = begin + 1; i < end; ++i) {
        bounds = merged(bounds, boundsOf(segments_[i]));
    }
    const std::size_t index = nodes_.size();
    nodes_.push_back({bounds, begin, end - begin, noNode, noNode});
    if (end - begin <= leafSize) {
        return index;
    }
    const bool useX = bounds.maxx - bounds.minx >= bounds.maxy - bounds.miny;
    const std::size_t middle = begin + (end - begin) / 2;
    std::nth_element(segments_.begin() + static_cast<std::ptrdiff_t>(begin),
                     segments_.begin() + static_cast<std::ptrdiff_t>(middle),
                     segments_.begin() + static_cast<std::ptrdiff_t>(end),
                     [useX](Segment a, Segment b) {
        if (useX) {
            return a.a.x * 0.5 + a.b.x * 0.5 < b.a.x * 0.5 + b.b.x * 0.5;
        }
        return a.a.y * 0.5 + a.b.y * 0.5 < b.a.y * 0.5 + b.b.y * 0.5;
    });
    const std::size_t left = build(begin, middle);
    const std::size_t right = build(middle, end);
    // Recursive build may reallocate nodes_, so retain indices, not references.
    nodes_[index].left = left;
    nodes_[index].right = right;
    return index;
}

template <class Predicate>
bool FaultBarrier::anyAt(std::size_t index, Bounds bounds,
                         const Predicate& predicate) const {
    const Node& node = nodes_[index];
    if (!overlaps(node.bounds, bounds)) {
        return false;
    }
    if (node.left != noNode) {
        return anyAt(node.left, bounds, predicate) ||
               anyAt(node.right, bounds, predicate);
    }
    for (std::size_t i = node.begin; i < node.begin + node.count; ++i) {
        if (predicate(segments_[i])) {
            return true;
        }
    }
    return false;
}

bool FaultBarrier::blocked(Vec2 a, Vec2 b) const {
    validate(a);
    validate(b);
    if (nodes_.empty()) {
        return false;
    }
    const long double toleranceSquared =
        static_cast<long double>(tolerance_) * tolerance_;
    return anyAt(0, expanded(boundsOf({a, b})), [&](Segment segment) {
        return segmentNear({a, b}, segment, toleranceSquared);
    });
}

bool FaultBarrier::onFault(Vec2 point) const {
    validate(point);
    if (nodes_.empty()) {
        return false;
    }
    const long double toleranceSquared =
        static_cast<long double>(tolerance_) * tolerance_;
    return anyAt(0, expanded(boundsOf({point, point})), [&](Segment segment) {
        return pointOnClosedSegment(point, segment) ||
               pointSegmentDistanceSquared(point, segment) <= toleranceSquared;
    });
}

bool FaultBarrier::intersectsBox(Vec2 lo, Vec2 hi) const {
    validate(lo);
    validate(hi);
    if (lo.x > hi.x || lo.y > hi.y) {
        throw std::invalid_argument("Fault query rectangle bounds are inverted");
    }
    if (lo.x == hi.x || lo.y == hi.y) {
        return blocked(lo, hi);
    }
    if (nodes_.empty()) {
        return false;
    }
    const long double toleranceSquared =
        static_cast<long double>(tolerance_) * tolerance_;
    const std::vector<Vec2> hull = {lo, {hi.x, lo.y}, hi, {lo.x, hi.y}};
    return anyAt(0, expanded({lo.x, lo.y, hi.x, hi.y}), [&](Segment segment) {
        return segmentNearHull(segment, hull, toleranceSquared);
    });
}

bool FaultBarrier::hullBlocked(const std::vector<Vec2>& points) const {
    for (Vec2 p : points) {
        validate(p);
    }
    if (points.empty() || nodes_.empty()) {
        return false;
    }
    const std::vector<Vec2> hull = convexHull(points);
    if (hull.size() == 1) {
        return onFault(hull.front());
    }
    if (hull.size() == 2) {
        return blocked(hull.front(), hull.back());
    }
    Bounds bounds = boundsOf({hull.front(), hull.front()});
    for (Vec2 p : hull) {
        bounds = merged(bounds, boundsOf({p, p}));
    }
    const long double toleranceSquared =
        static_cast<long double>(tolerance_) * tolerance_;
    return anyAt(0, expanded(bounds), [&](Segment segment) {
        return segmentNearHull(segment, hull, toleranceSquared);
    });
}

} // namespace ci::detail
