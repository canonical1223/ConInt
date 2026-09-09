#pragma once

#include <QtCore/qglobal.h>
#include <cstddef>
#include <limits>
#include <vector>

namespace ci {

struct Point {
    qreal x = 0;
    qreal y = 0;
    qreal value = 0;
    qreal weight = 1;
};

struct Surface {
    // Nodes include both bounds. Index = iy * nx + ix; y increases upwards.
    std::vector<qreal> grid;
    qreal minx = 0, maxx = 0, miny = 0, maxy = 0;
    std::size_t nx = 0, ny = 0;
};

// Open polyline. Repeat the first vertex at the end to close it.
// Only x,y of a fault vertex are used.
using Fault = std::vector<Point>;

struct Options {
    std::size_t initialCoarseningFactor = 8;
    std::size_t maxIterations = 2000;
    double relativeTolerance = 1e-8;
    double smoothness = 0.1;
    // Small positive regularization toward the initial/refined trend.
    double priorWeight = 1e-5;
    // Zero selects an automatic tolerance (1e-10 of the larger extent).
    double geometryTolerance = 0;
    bool useInitialGrid = true;
};

struct Report {
    bool converged = true;
    std::size_t levels = 0;
    std::size_t iterations = 0;
    std::size_t usedControlPoints = 0;
    std::size_t undefinedNodes = 0;
    // Positive-weight points with no safe support in the FINAL grid.
    std::vector<std::size_t> unrepresentedControlPoints;
    double weightedRms = std::numeric_limits<double>::quiet_NaN();
    double maxAbsoluteResidual = std::numeric_limits<double>::quiet_NaN();
    double finalRelativeResidual = 0;
};

// Weighted, fault-aware multiresolution surface reconstruction, not a Petrel clone.
// grid may be empty or contain exactly nx*ny values (NaN = unknown).
// Positive-weight points on a fault / outside the bounds are rejected.
// Nodes on faults and unsupported components are returned as NaN.
// Weights are relative reliabilities: internally normalized to mean 1.
// Control points are soft constraints; see Report for actual fit accuracy.
// Invalid input throws std::invalid_argument. Exceptions leave surface unchanged.
// Nonconvergence returns the current approximation and Report::converged=false.
Report interpolate(Surface& surface, const std::vector<Point>& points,
                   const std::vector<Fault>& faults = {},
                   const Options& options = {});

} // namespace ci
