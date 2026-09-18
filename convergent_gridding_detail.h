#pragma once
#include "convergent_gridding.h"
#include "fault_geometry.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace convergent {
namespace detail {

// ВНУТРЕННИЙ контракт v1: не публичный API приложения. Отделяет данные сетки,
// интерполяционные строки и решатель; не зависит от версии convergent2.
// Внутренние вычисления всегда выполняются в double, даже если публичный qreal
// имеет меньшую точность. Это уменьшает накопление ошибок в итерационных
// методах; перед возвратом результат все равно проверяется после cast в qreal.
using Scalar = double;

// Канонический внутренний формат получается после сортировки, удаления точек
// с нулевым весом и объединения точек с одинаковыми x/y.
struct CanonicalPoint {
    Scalar x{};
    Scalar y{};
    Scalar value{};
    Scalar weight{};
};

// Внутреннее описание геометрии и значений одного уровня иерархии.
struct Grid {
    std::size_t nx{};
    std::size_t ny{};
    Scalar minx{};
    Scalar maxx{};
    Scalar miny{};
    Scalar maxy{};
    // Тот же порядок, что у Surface::grid: iy == 0 соответствует y == miny,
    // а index(ix, iy) идет слева направо и затем снизу вверх.
    std::vector<Scalar> values;
    // Геометрия живет на стеке управляющей функции дольше всех Grid. nullptr
    // сохраняет прежний численный путь без разломов (backward compatibility).
    const detail::FaultGeometry* faults{};

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

    Scalar x(std::size_t ix) const noexcept
    {
        return ix + 1 == nx ? maxx : minx + static_cast<Scalar>(ix) * dx();
    }

    Scalar y(std::size_t iy) const noexcept
    {
        return iy + 1 == ny ? maxy : miny + static_cast<Scalar>(iy) * dy();
    }
};

// Градиент и симметричный Hessian: gx=du/dx, gy=du/dy,
// gxx=d2u/dx2, gxy=d2u/(dx*dy), gyy=d2u/dy2.
struct Derivatives {
    Scalar gx{};
    Scalar gy{};
    Scalar gxx{};
    Scalar gxy{};
    Scalar gyy{};
};

// Один ненулевой элемент разреженной строки C_p.
struct StencilTerm {
    std::size_t index{};
    Scalar coefficient{};
};

// Разреженная строка билинейной выборки C_p. Регулярная 2D-точка зависит не
// более чем от четырех углов содержащей ячейки; на границе count может быть
// меньше после объединения одинаковых индексов.
struct PointConstraint {
    std::array<StencilTerm, 4> terms{};
    std::size_t count{};
    Scalar value{};
    Scalar weight{};
};

// Результат основного PCG для одного уровня.
struct SolverResult {
    std::vector<Scalar> values;
    // Ноль означает, что теплый старт уже удовлетворял критерию.
    std::size_t iterations{};
    // Норма хранимой невязки (residual) PCG; истинная невязка (true residual)
    // пересчитывается раз в 50 шагов.
    Scalar relativeResidual{};
    bool converged{};
};

// Результат отдельного Gram-PCG, исправляющего контрольную невязку.
struct ProjectionResult {
    // Ноль возможен, когда soft solution уже находится в exact-допуске.
    std::size_t iterations{};
    // Максимум |C_p*u-d_p| в текущей стадии проекции.
    Scalar maxError{};
    bool converged{};
};

inline bool finite(Scalar value) noexcept
{
    return std::isfinite(value);
}

// Безопасно вычисляет число узлов и не допускает переполнение size_t до
// выделения памяти. Проверка nx,ny>=2 выполняется отдельно в validateInput().
inline std::size_t checkedNodeCount(std::size_t nx, std::size_t ny)
{
    if (nx == 0 || ny > std::numeric_limits<std::size_t>::max() / nx) {
        throw std::invalid_argument("surface dimensions overflow size_t");
    }
    return nx * ny;
}


inline bool visibleNodes(const Grid& grid, std::size_t a, std::size_t b)
{
    return !grid.faults || grid.faults->visible(
        grid.x(a % grid.nx), grid.y(a / grid.nx),
        grid.x(b % grid.nx), grid.y(b / grid.nx));
}


// Разреженное скалярное произведение одной строки C_p с узловым вектором u.
inline Scalar constraintValue(const PointConstraint& constraint,
                       const std::vector<Scalar>& values)
{
    Scalar result = 0;
    for (std::size_t i = 0; i < constraint.count; ++i) {
        result += constraint.terms[i].coefficient * values[constraint.terms[i].index];
    }
    return result;
}


// Копировать только геометрию, не многомегабайтный values, который затем
// немедленно заменялся. В отличие от Grid copy это O(1) без выделения буфера.
inline Grid gridGeometry(const Grid& source)
{
    Grid grid;
    grid.nx = source.nx;
    grid.ny = source.ny;
    grid.minx = source.minx;
    grid.maxx = source.maxx;
    grid.miny = source.miny;
    grid.maxy = source.maxy;
    grid.faults = source.faults;
    return grid;
}

// Операторы доступны отдельным unit-тестам, но не входят в публичный ABI.
std::vector<std::uint8_t> faultCurvatureMask(const Grid&);
void applyCurvature(const Grid&, const std::vector<Scalar>&,
                    std::vector<Scalar>&, Scalar,
                    const std::vector<std::uint8_t>* mask = nullptr);
std::vector<Scalar> curvatureDiagonal(const Grid&, Scalar,
                                     const std::vector<std::uint8_t>* mask = nullptr);
SolverResult solveLevel(const Grid&, const std::vector<Scalar>& prior,
                        const std::vector<Scalar>& snapDiagonal,
                        const std::vector<Scalar>& snapRhs,
                        const std::vector<PointConstraint>&,
                        const ConvergentGriddingOptions&);
Scalar effectiveControlTolerance(const std::vector<PointConstraint>&,
                                 const ConvergentGriddingOptions&);
ProjectionResult projectOntoPointConstraints(Grid&, const std::vector<PointConstraint>&,
                                            const ConvergentGriddingOptions&);
} // namespace detail
} // namespace convergent
