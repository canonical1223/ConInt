#pragma once

#include <cstddef>
#include <vector>

// qreal must be selected once for the whole target. CMake defines the
// standalone branch PUBLIC, so the library and every consumer share one ABI.
// A Qt/qmake build leaves the macro undefined and uses Qt's qreal.
#if defined(CONVERGENT_GRIDDING_STANDALONE_QREAL)
using qreal = double;
#elif defined(__has_include)
#  if __has_include(<QtCore/qglobal.h>)
#    include <QtCore/qglobal.h>
#  elif __has_include(<QtGlobal>)
#    include <QtGlobal>
#  else
#    error "QtGlobal was not found; define CONVERGENT_GRIDDING_STANDALONE_QREAL"
#  endif
#else
#  error "QtGlobal was not found; define CONVERGENT_GRIDDING_STANDALONE_QREAL"
#endif

namespace convergent {

// gorid is intentionally spelled exactly as in the requested input format.
// Layout: index = iy * nx + ix, ix grows left-to-right and iy bottom-to-top.
struct Surface {
    std::vector<qreal> gorid;
    qreal minx{};
    qreal maxx{};
    qreal miny{};
    qreal maxy{};
    std::size_t nx{};
    std::size_t ny{};
};

struct Point {
    qreal x{};
    qreal y{};
    qreal value{};
    qreal weight{qreal(1)};
};

struct ConvergentGriddingOptions {
    // Coarse-to-fine hierarchy.
    std::size_t initialSnapNodes{16};
    std::size_t coarsestIntervals{8};
    std::size_t maxLevels{8};

    // Energy weights.  The solved functional at a level is
    //   smoothness * ||curvature(u)||^2
    // + priorWeight * ||u - coarseTrend||^2
    // + Snap/Taylor constraints
    // + final bilinear point constraints.
    qreal smoothness{qreal(1)};
    qreal priorWeight{qreal(1e-3)};
    qreal snapStrength{qreal(100)};
    qreal finalPointStrength{qreal(1000)};

    // Distance kernel for Snap, measured in grid-cell units.
    qreal gaussianSigma{qreal(1)};

    // 0: value only, 1: value + slope, 2: value + slope + curvature.
    int taylorOrder{2};
    bool normalizePointWeights{true};

    // Matrix-free Jacobi-preconditioned conjugate gradients.
    std::size_t maxSolverIterations{1500};
    qreal relativeTolerance{qreal(1e-9)};
    qreal absoluteTolerance{qreal(0)};
    bool throwOnNonConvergence{true};
};

struct ConvergentGriddingLevelReport {
    std::size_t nx{};
    std::size_t ny{};
    std::size_t snapNodes{};
    std::size_t solverIterations{};
    qreal relativeResidual{};
    bool converged{};
};

struct ConvergentGriddingReport {
    std::vector<ConvergentGriddingLevelReport> levels;
    // Maximum absolute bilinear residual at canonical control locations.
    // Exact co-located controls are first replaced by their weighted mean.
    qreal maxControlError{};
};

// Strong exception guarantee: surface is changed only after all levels have
// completed successfully.  A point with weight == 0 is ignored.  Negative
// weights and points outside the closed surface bounds are rejected.
ConvergentGriddingReport convergentGridding(
    Surface& surface,
    const std::vector<Point>& points,
    const ConvergentGriddingOptions& options = {});

// Convenience non-mutating overload.
Surface convergentGriddedSurface(
    const Surface& surface,
    const std::vector<Point>& points,
    const ConvergentGriddingOptions& options = {},
    ConvergentGriddingReport* report = nullptr);

} // namespace convergent
