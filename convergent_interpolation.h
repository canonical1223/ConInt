#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#if __has_include(<QtCore/qglobal.h>)
#include <QtCore/qglobal.h>
#else
// Нужен только для автономной сборки тестов. В Qt-проекте используется
// qreal из QtCore/qglobal.h.
using qreal = double;
#endif

namespace geo::convergent {

// Способ, которым готовый проект восстанавливает поверхность внутри ячейки.
// Режим должен совпадать с построением индексного буфера отображаемого mesh.
enum class CellInterpolation {
    Bilinear,
    // Диагональ: нижний левый узел -> верхний правый узел.
    TriangleBottomLeftToTopRight,
    // Диагональ: нижний правый узел -> верхний левый узел.
    // Это распространённый порядок для регулярного треугольного mesh.
    TriangleBottomRightToTopLeft,
};

// Внутреннее представление входной точки. Пользовательскому проекту не нужно
// заменять им собственный Point: шаблонная перегрузка ниже выполнит конверсию.
struct Sample {
    qreal x = 0;
    qreal y = 0;
    qreal value = 0;
    qreal weight = 1;
};

struct Options {
    // 0 — автоматическое число уровней.
    std::size_t maxLevels = 0;
    std::size_t coarsestGridSize = 17;

    std::size_t maxIterationsPerLevel = 1500;
    qreal relativeTolerance = 1e-9;
    qreal smoothness = 1;

    CellInterpolation cellInterpolation =
        CellInterpolation::TriangleBottomRightToTopLeft;

    // Начальный штраф за невязку. В жёстком режиме это стартовый параметр
    // расширенного Лагранжиана, а не замена точного ограничения.
    qreal dataWeight = 1e3;
    qreal regularization = 1e-10;
    bool ignoreOutsidePoints = false;

    // Необязательная маска отображаемых узлов размером nx * ny:
    // 0 — узел отсутствует в rendered mesh, ненулевое значение — активен.
    // Объект маски должен существовать до завершения interpolate().
    const std::vector<std::uint8_t>* activeNodeMask = nullptr;

    // Решает задачу минимальной кривизны при ограничениях
    // B(point) * surface == point.value.
    bool enforcePointConstraints = true;
    std::size_t maxConstraintIterations = 30;
    qreal constraintPenaltyGrowth = 5;
    qreal pointConstraintRelativeTolerance = 1e-9;
};

struct Report {
    std::size_t levelsUsed = 0;
    std::size_t totalIterations = 0;
    qreal maxAbsolutePointResidual = 0;
    qreal weightedRmsPointResidual = 0;
    std::size_t constraintIterations = 0;
    bool pointConstraintsSatisfied = false;
    bool converged = false;
};

// Порядок хранения поверхности:
//   iy = 0      — нижняя строка (y = miny),
//   iy = ny - 1 — верхняя строка  (y = maxy),
// внутри строки ix растёт слева направо: minx -> maxx.
[[nodiscard]] constexpr std::size_t surfaceIndex(
    std::size_t ix,
    std::size_t iy,
    std::size_t nx) noexcept
{
    return iy * nx + ix;
}

// Основной ABI: не зависит от пользовательских типов Surface и Point.
//
// nx и ny — число узлов, включая границы. Результат записывается построчно:
// surfaceValues[surfaceIndex(ix, iy, nx)]. Первый элемент соответствует
// нижнему левому углу (minx, miny). Старое содержимое не используется.
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

// Вычисляет высоту именно по той модели ячейки, которая используется для
// ограничений. Полезно для независимой проверки против rendered mesh.
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

// Перегрузка для уже существующего Point. Требуются публичные поля
// x, y, value и weight, приводимые к qreal.
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

// Удобный адаптер для структуры Surface с полями:
// values, minx, maxx, miny, maxy, nx, ny.
// Если в вашем проекте вектор называется иначе, используйте перегрузку выше
// и явно передайте surface.data/surface.z/другое имя.
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

} // namespace geo::convergent
