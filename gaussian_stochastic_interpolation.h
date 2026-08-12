#pragma once

#include "surface_grid_geometry.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace geo::gaussian {

using CellInterpolation = geo::surface_grid::CellInterpolation;

enum class CorrelationModel {
    Gaussian,
    Exponential,
    Spherical,
};

enum class TrendModel {
    // Взвешенная плоскость по контрольным точкам.
    PlaneFromPoints,
    // Взвешенное среднее контрольных точек.
    ConstantFromPoints,
    // Существующее surface.values используется как base-case/trend.
    InputSurface,
};

struct Sample {
    qreal x = 0;
    qreal y = 0;
    qreal value = 0;
    qreal weight = 1;
};

struct Options {
    CellInterpolation cellInterpolation =
        CellInterpolation::TriangleBottomRightToTopLeft;
    CorrelationModel correlationModel = CorrelationModel::Gaussian;
    TrendModel trendModel = TrendModel::PlaneFromPoints;

    // Геостатистический practical range: для Gaussian/Exponential корреляция
    // на этом расстоянии равна exp(-3) ~= 0.05; у Spherical — ровно нулю.
    // 0 означает 20% соответствующего размера поверхности.
    qreal majorRange = 0;
    qreal minorRange = 0;
    // Азимут большой оси в градусах против часовой стрелки от +X.
    qreal azimuthDegrees = 0;

    // Дисперсия пространственно коррелированной части. 0 — оценить по
    // остаткам точек относительно тренда, с безопасным fallback.
    qreal structuredVariance = 0;
    // Независимая вариация УЗЛОВ mesh, не ошибка измерения.
    qreal nodeNuggetVariance = 0;

    // Размер circulant embedding. Начальное значение 2 оставляет реальные
    // лаги внутри FFT-периода; при отрицательном спектре размер увеличивается.
    std::size_t paddingFactor = 2;
    std::size_t maxEmbeddingExpansions = 3;

    std::uint64_t seed = 1;

    // false создаёт безусловную реализацию и полностью игнорирует points как
    // ограничения (points по-прежнему могут использоваться для trend/variance).
    bool enforcePointConstraints = true;
    std::size_t maxConditioningIterations = 3000;
    std::size_t maxConditioningRefinements = 3;
    qreal conditioningRelativeTolerance = 1e-10;

    bool ignoreOutsidePoints = false;
    // Необязательная маска узлов rendered mesh размером nx * ny.
    const std::vector<std::uint8_t>* activeNodeMask = nullptr;
};

struct Report {
    std::uint64_t seedUsed = 0;
    qreal effectiveMajorRange = 0;
    qreal effectiveMinorRange = 0;
    qreal effectiveStructuredVariance = 0;
    qreal effectiveNodeNuggetVariance = 0;
    std::size_t embeddingNx = 0;
    std::size_t embeddingNy = 0;
    std::size_t embeddingExpansions = 0;
    std::size_t conditioningIterations = 0;
    std::size_t conditioningRefinements = 0;
    qreal maxAbsolutePointResidual = 0;
    qreal weightedRmsPointResidual = 0;
    bool pointConstraintsSatisfied = false;
    bool converged = false;
};

[[nodiscard]] constexpr std::size_t surfaceIndex(
    std::size_t ix,
    std::size_t iy,
    std::size_t nx) noexcept
{
    return geo::surface_grid::surfaceIndex(ix, iy, nx);
}

// Создаёт одну условную Gaussian random function realization.
// nx, ny — число УЗЛОВ. Результат: нижняя строка первой, внутри строки
// значения слева направо. При TrendModel::InputSurface старые значения сначала
// копируются как trend; в остальных режимах старое содержимое не используется.
Report interpolate(
    std::vector<qreal>& surfaceValues,
    qreal minx,
    qreal maxx,
    qreal miny,
    qreal maxy,
    std::size_t nx,
    std::size_t ny,
    const std::vector<Sample>& points,
    const Options& options = {});

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
    CellInterpolation interpolation);

namespace detail {

template <typename T, typename = void>
struct IsCompatiblePoint : std::false_type {};

template <typename T>
struct IsCompatiblePoint<T, std::void_t<
    decltype(std::declval<const T&>().x),
    decltype(std::declval<const T&>().y),
    decltype(std::declval<const T&>().value),
    decltype(std::declval<const T&>().weight)>> : std::true_type {};

template <typename T, typename = void>
struct IsCompatibleSurface : std::false_type {};

template <typename T>
struct IsCompatibleSurface<T, std::void_t<
    decltype(std::declval<T&>().values),
    decltype(std::declval<const T&>().minx),
    decltype(std::declval<const T&>().maxx),
    decltype(std::declval<const T&>().miny),
    decltype(std::declval<const T&>().maxy),
    decltype(std::declval<const T&>().nx),
    decltype(std::declval<const T&>().ny)>> : std::true_type {};

} // namespace detail

template <typename PointT,
          std::enable_if_t<detail::IsCompatiblePoint<PointT>::value, int> = 0>
Report interpolate(
    std::vector<qreal>& surfaceValues,
    qreal minx,
    qreal maxx,
    qreal miny,
    qreal maxy,
    std::size_t nx,
    std::size_t ny,
    const std::vector<PointT>& points,
    const Options& options = {})
{
    std::vector<Sample> samples;
    samples.reserve(points.size());
    for (const PointT& point : points) {
        samples.push_back(Sample{
            static_cast<qreal>(point.x),
            static_cast<qreal>(point.y),
            static_cast<qreal>(point.value),
            static_cast<qreal>(point.weight),
        });
    }
    return interpolate(
        surfaceValues, minx, maxx, miny, maxy, nx, ny, samples, options);
}

template <typename SurfaceT,
          typename PointT,
          std::enable_if_t<detail::IsCompatibleSurface<SurfaceT>::value
                  && detail::IsCompatiblePoint<PointT>::value,
              int> = 0>
Report interpolate(
    SurfaceT& surface,
    const std::vector<PointT>& points,
    const Options& options = {})
{
    return interpolate(
        surface.values,
        static_cast<qreal>(surface.minx),
        static_cast<qreal>(surface.maxx),
        static_cast<qreal>(surface.miny),
        static_cast<qreal>(surface.maxy),
        static_cast<std::size_t>(surface.nx),
        static_cast<std::size_t>(surface.ny),
        points,
        options);
}

template <typename SurfaceT,
          std::enable_if_t<detail::IsCompatibleSurface<SurfaceT>::value, int> = 0>
qreal evaluateSurface(
    const SurfaceT& surface,
    qreal x,
    qreal y,
    CellInterpolation interpolation)
{
    return evaluateSurface(
        surface.values,
        static_cast<qreal>(surface.minx),
        static_cast<qreal>(surface.maxx),
        static_cast<qreal>(surface.miny),
        static_cast<qreal>(surface.maxy),
        static_cast<std::size_t>(surface.nx),
        static_cast<std::size_t>(surface.ny),
        x,
        y,
        interpolation);
}

} // namespace geo::gaussian
