#pragma once

#include <cstddef>

#if __has_include(<QtCore/qglobal.h>)
#include <QtCore/qglobal.h>
#else
// Нужен только для автономного использования вне Qt-проекта.
using qreal = double;
#endif

namespace geo::surface_grid {

// Способ восстановления поверхности между четырьмя узлами регулярной ячейки.
// Он обязан совпадать с индексным буфером отображаемого mesh.
enum class CellInterpolation {
    Bilinear,
    // Диагональ: нижний левый узел -> верхний правый узел.
    TriangleBottomLeftToTopRight,
    // Диагональ: нижний правый узел -> верхний левый узел.
    TriangleBottomRightToTopLeft,
};

// iy = 0 — нижняя строка (y = miny), ix растёт слева направо.
[[nodiscard]] constexpr std::size_t surfaceIndex(
    std::size_t ix,
    std::size_t iy,
    std::size_t nx) noexcept
{
    return iy * nx + ix;
}

} // namespace geo::surface_grid
