#pragma once

#include "convergent_gridding_2.h"

#include <vector>

namespace convergent2 {
namespace detail {

// Две односторонние пробы (one-sided bank probes) задают численное условие
// около ориентированного отрезка разлома. Left/Right определяются в XY по
// направлению от предыдущей вершины к следующей, независимо от знака Z.
// В Throw target ВСЕГДА нормализован к u(left)-u(right); пользовательский
// ThrowSign учитывается при создании проб, а не повторно в solver.
// В AbsoluteElevation target — абсолютная отметка и используются отдельные
// равенства для каждой выбранной стороны. Point.value внутри left/right не
// участвует в этом контракте; истинная правая часть содержится в target.
struct FaultProbe {
    Point left, right;
    double target{};
    double weight{1};
    bool useLeft{true};
    bool useRight{true};
};

// В fixedSampling-режиме шаг внутри каждого исходного сегмента равен половине
// диагонали КОНЕЧНОЙ ячейки: s=0.5*hypot(hx,hy). Остаток сегмента короче s.
// Высота и вес линейно интерполируются; общая вершина не дублируется.
// Без fixedSampling возвращаются только исходные вершины — это явный выбор
// реализации v2, а не утверждение об автоматическом режиме Petrel.
// Пустые линии пропускаются, одиночная вершина становится контрольной точкой.
// Точки должны лежать в Surface; weight==0 не выдается, weight<0 запрещен.
std::vector<Point> sampleInputPolylines(
    const Surface& surface,
    const std::vector<Polyline>& polylines,
    bool fixedSampling);

// Midpoint-квадратура (midpoint quadrature) вдоль части разлома внутри Surface:
// сегмент делится на равные части длиной <= половины диагонали текущей ячейки,
// в середине каждой части строятся левый/правый выносы вдоль нормали.
// Их начальный размер = faultBankOffsetCells*min(hx,hy); если нужно, размер
// уменьшается, чтобы не выйти из Surface и не пересечь другой сегмент.
// Непредставимое ограничение вызывает исключение; ненулевой throw не теряется
// молча. Открытые концы не продолжаются, заданный throw не гасится автоматически.
std::vector<FaultProbe> makeFaultProbes(
    const Surface& levelGeometry,
    const std::vector<Fault>& faults,
    const ConvergentGriddingOptions& options);

// Объединение внутренностей ЯВНО замкнутых полилиний (union of interiors).
// Замыкание = строго одинаковые XY первой и последней вершины. Для каждой
// линии действует правило четности (even-odd rule), граница считается внутри.
// Вложенные полигоны дают объединение, а не отверстия; направление обхода
// не влияет на маску. Открытые линии внутреннюю область не определяют.
bool insideClosedFault(double x, double y, const std::vector<Fault>& faults);

} // namespace detail
} // namespace convergent2
