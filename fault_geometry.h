#pragma once

#include <cstddef>
#include <vector>

namespace ci::detail {

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

struct Segment {
    Vec2 a;
    Vec2 b;
};

// Closed fault segments are opaque barriers, including at their endpoints.
// Coordinates and tolerance must use the same Euclidean coordinate system.
// Segments no longer than tolerance are discarded as duplicate vertices.
// Construction is exclusive; concurrent const queries are safe.
class FaultBarrier {
public:
    explicit FaultBarrier(std::vector<Segment> segments,
                          double tolerance = 1e-10);

    // True when [a,b] intersects a fault or is within tolerance of a fault.
    bool blocked(Vec2 a, Vec2 b) const;
    bool onFault(Vec2 point) const;

    // Tests the filled rectangle, including faults entirely inside it.
    // Throws invalid_argument for an inverted rectangle.
    bool intersectsBox(Vec2 lo, Vec2 hi) const;

    // Tests the filled convex hull, including faults entirely inside it.
    // Input need not be ordered. Empty, point and line hulls are supported.
    bool hullBlocked(const std::vector<Vec2>& points) const;

private:
    struct Bounds {
        double minx;
        double miny;
        double maxx;
        double maxy;
    };
    struct Node {
        Bounds bounds;
        std::size_t begin;
        std::size_t count;
        std::size_t left;
        std::size_t right;
    };

    std::vector<Segment> segments_;
    std::vector<Node> nodes_;
    double tolerance_;

    static Bounds boundsOf(Segment segment);
    static Bounds merged(Bounds a, Bounds b);
    static bool overlaps(Bounds a, Bounds b);
    Bounds expanded(Bounds bounds) const;
    std::size_t build(std::size_t begin, std::size_t end);

    template <class Predicate>
    bool anyAt(std::size_t index, Bounds bounds,
               const Predicate& predicate) const;
};

} // namespace ci::detail
