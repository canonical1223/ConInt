#include "convergent_gridding_2.h"
#include "fault_geometry_2.h"
#include "trend_model_2.h"
#include "input_processing_2.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace convergent2 {
namespace {

// ============================================================================
// CONVERGENT GRIDDING v2: ПОЛНАЯ СХЕМА И ГРАНИЦЫ СООТВЕТСТВИЯ PETREL
// ============================================================================
// Этот файл сначала скопирован из convergent_gridding.cpp, затем изменён.
// Исходные .h/.cpp v1 сохранены; namespace convergent2 позволяет линковать обе.
//
// 1. VALIDATION / PREPROCESSING.
//    Проверка геометрии, valid-маски, весов, настроек и дополнительного входа.
//    Полилинии при fixedSampling уплотняются с шагом 0.5*hypot(hx,hy).
//    Их Z интерполируются по ФАКТИЧЕСКИ представимым XY (важно на XY~1e12).
//    Одинаковые XY controls объединяются средневзвешенно, веса нормируются.
//    Геометрия Fault — последовательные сегменты, не бесконечные прямые.
//
// 2. INITIAL TREND / COARSENING.
//    По данным строится локальный полином Horizontal/Linear/Quadratic.
//    QR-fit, нормировка координат и rank fallback реализованы в TrendModel.
//    Опционально пользовательская grid служит prior. При одних относительных
//    throw/dip полная валидная grid обязательна как абсолютный height reference.
//    initialCoarseningFactor задаёт отношение шагов: c -> c/2 -> ... -> 1.
//    Автоматическая оценка по плотности — наша эвристика, не формула Petrel.
//    Размеры округляются с сохранением исходных min/max. Непредставимые
//    разломные coarse-уровни пропускаются с отражением размеров в отчёте.
//
// 3. COARSE-TO-FINE LOOP.
//    a) Fault influence включает барьеры на доле ПОСЛЕДНИХ проходов.
//       Последний всегда учитывает разломы; процент не масштабирует throw.
//    b) Prolongation переносит значения, а производные отдельно выбираются
//       из предыдущего решения в координатах controls ДО ресемплирования.
//       После включения барьера связи односторонние во всех операциях.
//    c) Snap проецирует значение p на соседние узлы рядом Тейлора:
//         T_p(n)=z_p+grad(prior)_p*delta+0.5*delta^T*Hessian_p*delta.
//       Пользовательский порядок и число Snap-узлов действуют ТОЛЬКО на
//       первом проходе. Позднее используются наша автоматическая стратегия
//       1..16 узлов и линейный/квадратичный прогноз по доступным производным.
//    d) Normal вводит мягкие anchors выполаживающего продолжения ВНЕ convex
//       hull видимых данных; Trend сохраняет продолжение начального тренда.
//       Точная формула Normal в Petrel неизвестна; в v2 это ограниченное
//       расстояние rho=L*tanh(r/L), подробно в trend_model_2.*.
//    e) Smooth решает SPD-систему (matrix-free Jacobi-PCG):
//         Phi(u) = 1/2 [smoothness*||B*u||² + priorWeight*||u-prior||²
//                     + Snap penalties + jump/dip/absolute-Z penalties].
//       На последнем уровне добавляются soft control equations C*u=d.
//       B — вторые разности thin-plate bending energy. Пересекающие разлом
//       строки B исключаются целиком, чтобы сохранить симметрию и PSD.
//       Это оператор бигармонического типа, НЕ доказанно точный stencil Briggs.
//
// 4. FINAL CONSTRAINTS.
//    Равенства точек C*u=d либо интервалы |C*u-d|<=residualTolerance,
//    скачки C_left*u-C_right*u=throw и высотные min/max выполняются совместно.
//    Для равенств используется Gram-PCG, для интервалов/bounds и вырожденных
//    наборов — Dykstra projection. Несовместимость вызывает исключение.
//    Это L2-проекция узловых поправок, не повторная минимизация энергии изгиба.
//
// 5. OPTIONAL FINAL SMOOTHING.
//    smoothingIterations — отдельное сглаживание НА КОНЕЧНОЙ сетке после
//    convergent-проходов, а не лимит PCG. После каждого устойчивого шага B^T B
//    восстанавливаются ограничения. Fixed принуждает все проходы, иначе
//    возможна ранняя остановка по изменению Z.
//
// 6. DOMAIN / NODATA / LINEAR FILL.
//    Узлы на разломе и внутренности при FillInside=false маскируются.
//    EnsureAllNodes заполняет оставшиеся доступные пропуски линейным
//    гармоническим продолжением по видимым рёбрам. Компоненты без источников
//    остаются NoData. При grid-only входе сначала заполняются пропуски,
//    затем сглаживаются только валидные stencil: фиктивные нули не участвуют.
//
// 7. VERIFY / COMMIT.
//    Значения переводятся в qreal; повторно проверяются point/jump/bounds.
//    На выходе valid[n]=0 и NaN (или буферный0 по опции) обозначают NoData.
//    Единственные записи в Surface — два noexcept swap после всех проверок.
//    Исходный объект сохраняется при любой ошибке.
//
// ВАЖНО: подсказки Petrel определяют семантику пользовательских переключателей,
// но не закрытые численные детали. Gaussian Snap, QR-neighborhood, Normal tanh,
// график числа узлов, PCG, Dykstra, harmonic fill и signed-throw bank probes —
// явно выбранная реконструкция. Throw по запросу пользователя не тождественен
// Petrel Use Z Values (абсолютные отметки fault polygons). Источники: README_2.

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
    // До 8 узлов: C_left-C_right для перепада, до 4 для обычной точки.
    std::array<StencilTerm, 8> terms{};
    std::size_t count{};
    Scalar value{};
    Scalar weight{};
    Scalar strength{-1}; // <0: finalPointStrength; иначе собственный penalty
    Scalar halfWidth{}; // 0=равенство; >0=допустимый интервал по данным
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

bool finite(Scalar value) noexcept
{
    return std::isfinite(value);
}

// Безопасно вычисляет число узлов и не допускает переполнение size_t до
// выделения памяти. Проверка nx,ny>=2 выполняется отдельно в validateInput().
std::size_t checkedNodeCount(std::size_t nx, std::size_t ny)
{
    if (nx == 0 || ny > std::numeric_limits<std::size_t>::max() / nx) {
        throw std::invalid_argument("surface dimensions overflow size_t");
    }
    return nx * ny;
}

// Проверка параметров одновременно фиксирует математические предпосылки:
// smoothness/Snap/final penalties можно отключить нулем, но priorWeight обязан
// быть строго положительным, иначе A может остаться лишь SPSD и обычный PCG не
// имеет требуемой гарантии. relativeTolerance и gaussianSigma также строго
// положительны; absolute/control tolerances могут быть нулевыми.
void validateOptions(const ConvergentGriddingOptions& options)
{
    const auto positiveFinite = [](qreal value) {
        return finite(static_cast<Scalar>(value)) && value > qreal(0);
    };
    const auto nonNegativeFinite = [](qreal value) {
        return finite(static_cast<Scalar>(value)) && value >= qreal(0);
    };

    if (options.initialSnapNodes == 0 || options.initialSnapNodes > 16
        || options.maxLevels == 0 || options.maxSolverIterations == 0) {
        throw std::invalid_argument("gridding counts must be greater than zero");
    }
    if (options.maxControlProjectionIterations == 0) {
        throw std::invalid_argument(
            "maxControlProjectionIterations must be positive in exact-control mode");
    }
    if (!nonNegativeFinite(options.smoothness)
        || !positiveFinite(options.priorWeight)
        || !nonNegativeFinite(options.snapStrength)
        || !nonNegativeFinite(options.finalPointStrength)
        || !positiveFinite(options.gaussianSigma)
        || !positiveFinite(options.relativeTolerance)
        || !nonNegativeFinite(options.absoluteTolerance)
        || !nonNegativeFinite(options.controlTolerance)) {
        throw std::invalid_argument("invalid convergent gridding numeric option");
    }
    const Scalar sigma = static_cast<Scalar>(options.gaussianSigma);
    if (!finite(sigma * sigma) || !(sigma * sigma > 0)) {
        throw std::invalid_argument("gaussianSigma squared must be finite and positive");
    }
    if (!nonNegativeFinite(options.initialCoarseningFactor)
        || (options.initialCoarseningFactor > 0 && options.initialCoarseningFactor < 1)
        || !positiveFinite(options.normalDecayCells)
        || !nonNegativeFinite(options.residualTolerance)
        || !nonNegativeFinite(options.smoothingChangeTolerance)
        || !nonNegativeFinite(options.faultInfluencePercent) || options.faultInfluencePercent > 100
        || !positiveFinite(options.faultBankOffsetCells) || options.faultBankOffsetCells > 0.5
        || !nonNegativeFinite(options.faultConstraintStrength)
        || !nonNegativeFinite(options.dipConstraintStrength)
        || !nonNegativeFinite(options.dipInfluenceRadius)
        || (options.minimumValue && !finite(*options.minimumValue))
        || (options.maximumValue && !finite(*options.maximumValue)))
        throw std::invalid_argument("invalid v2 option");
    if (static_cast<int>(options.initialProjection) < 0 || static_cast<int>(options.initialProjection) > 2
        || (options.extrapolationMethod != ExtrapolationMethod::Normal && options.extrapolationMethod != ExtrapolationMethod::Trend)
        || (options.faultZMode != FaultZMode::Ignore && options.faultZMode != FaultZMode::Throw && options.faultZMode != FaultZMode::AbsoluteElevation)
        || (options.throwSign != ThrowSign::LeftMinusRight && options.throwSign != ThrowSign::RightMinusLeft)
        || (options.absoluteElevationBank != FaultBank::Both && options.absoluteElevationBank != FaultBank::Left && options.absoluteElevationBank != FaultBank::Right)
        || (options.dipMode != DipMode::None && options.dipMode != DipMode::AzimuthOnly && options.dipMode != DipMode::DipAndAzimuth))
        throw std::invalid_argument("invalid v2 enum option");
}

std::vector<CanonicalPoint> validateInput(
    const Surface& surface,
    const std::vector<Point>& points,
    const ConvergentGriddingOptions& options)
{
    // Сначала проверяются параметры, чтобы даже ветвь «нет активных точек»
    // имела единый контракт ошибок с обычным расчетом.
    validateOptions(options);

    if (surface.nx < 2 || surface.ny < 2) {
        throw std::invalid_argument("surface.nx and surface.ny must both be at least 2");
    }

    const Scalar minx = static_cast<Scalar>(surface.minx);
    const Scalar maxx = static_cast<Scalar>(surface.maxx);
    const Scalar miny = static_cast<Scalar>(surface.miny);
    const Scalar maxy = static_cast<Scalar>(surface.maxy);
    if (!finite(minx) || !finite(maxx) || !finite(miny) || !finite(maxy)
        || !(minx < maxx) || !(miny < maxy)) {
        throw std::invalid_argument("surface bounds must be finite and strictly increasing");
    }

    const Scalar spanX = maxx - minx;
    const Scalar spanY = maxy - miny;
    const Scalar dx = spanX / static_cast<Scalar>(surface.nx - 1);
    const Scalar dy = spanY / static_cast<Scalar>(surface.ny - 1);
    if (!finite(spanX) || !finite(spanY) || !finite(dx) || !finite(dy)
        || !(spanX > 0) || !(spanY > 0) || !(dx > 0) || !(dy > 0)
        || !finite(dx / dy) || !finite(dy / dx)) {
        throw std::invalid_argument("surface bounds produce an unsafe grid spacing");
    }

    const std::size_t expected = checkedNodeCount(surface.nx, surface.ny);
    if (surface.grid.size() != expected) {
        throw std::invalid_argument("surface.grid.size() must equal surface.nx * surface.ny");
    }
    if (!surface.valid.empty() && surface.valid.size() != expected)
        throw std::invalid_argument("surface.valid must be empty or nx*ny");
    for (std::size_t i = 0; i < expected; ++i)
        if ((surface.valid.empty() || surface.valid[i]) && !finite(surface.grid[i]))
            throw std::invalid_argument("valid surface nodes must be finite");

    std::vector<CanonicalPoint> result;
    result.reserve(points.size());
    for (const Point& point : points) {
        const CanonicalPoint p{
            static_cast<Scalar>(point.x),
            static_cast<Scalar>(point.y),
            static_cast<Scalar>(point.value),
            static_cast<Scalar>(point.weight)};

        if (!finite(p.x) || !finite(p.y) || !finite(p.value) || !finite(p.weight)) {
            throw std::invalid_argument("control point fields must be finite");
        }
        if (p.weight < 0) {
            throw std::invalid_argument("control point weight must not be negative");
        }
        if (p.x < minx || p.x > maxx || p.y < miny || p.y > maxy) {
            throw std::invalid_argument("control point lies outside the surface bounds");
        }
        if (p.weight == 0) {
            continue;
        }
        result.push_back(p);
    }

    // Канонический порядок делает накопление воспроизводимым и не зависящим от
    // перестановки одного и того же набора входных точек.
    std::sort(result.begin(), result.end(), [](const CanonicalPoint& a, const CanonicalPoint& b) {
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        if (a.value != b.value) return a.value < b.value;
        return a.weight < b.weight;
    });

    // Совпадающие координаты задают одно физическое ограничение. Совпадение
    // проверяется строгим x==x и y==y уже после преобразования в double;
    // epsilon-кластеризации близких координат нет. Объединение до нормализации
    // делает {(z=10,w=1),(z=30,w=3)} эквивалентным {(z=25,w=4)} и не создает
    // заведомо повторяющиеся строки ограничений.
    std::vector<CanonicalPoint> grouped;
    grouped.reserve(result.size());
    for (std::size_t begin = 0; begin < result.size();) {
        std::size_t end = begin + 1;
        while (end < result.size()
               && result[end].x == result[begin].x
               && result[end].y == result[begin].y) {
            ++end;
        }
        long double weightSum = 0;
        long double weightedValue = 0;
        for (std::size_t i = begin; i < end; ++i) {
            weightSum += static_cast<long double>(result[i].weight);
            weightedValue += static_cast<long double>(result[i].weight)
                * static_cast<long double>(result[i].value);
        }
        const long double combinedValue = weightedValue / weightSum;
        if (!std::isfinite(weightSum) || !std::isfinite(combinedValue)
            || weightSum > static_cast<long double>(std::numeric_limits<Scalar>::max())
            || std::abs(combinedValue)
                > static_cast<long double>(std::numeric_limits<Scalar>::max())) {
            throw std::invalid_argument("co-located control point weights overflow");
        }
        grouped.push_back({result[begin].x, result[begin].y,
                           static_cast<Scalar>(combinedValue),
                           static_cast<Scalar>(weightSum)});
        begin = end;
    }
    result = std::move(grouped);

    Scalar maximumWeight = 0;
    for (const CanonicalPoint& point : result) {
        maximumWeight = std::max(maximumWeight, point.weight);
    }
    if (options.normalizePointWeights && maximumWeight > 0) {
        // Деление на общий максимум сохраняет относительные веса и делает
        // мягкие этапы инвариантными к умножению всех исходных весов на одну
        // константу. В exact-проекции вес не масштабирует отдельное равенство,
        // но уже повлиял на value объединенных совпадающих точек.
        for (CanonicalPoint& point : result) {
            point.weight /= maximumWeight;
        }
    }
    return result;
}

// Создает пустой уровень с границами публичной Surface, но с заданным числом
// узлов. Все уровни поэтому покрывают одну физическую область.
Grid makeGridGeometry(const Surface& surface, std::size_t nx, std::size_t ny)
{
    Grid result;
    result.nx = nx;
    result.ny = ny;
    result.minx = static_cast<Scalar>(surface.minx);
    result.maxx = static_cast<Scalar>(surface.maxx);
    result.miny = static_cast<Scalar>(surface.miny);
    result.maxy = static_cast<Scalar>(surface.maxy);
    result.values.resize(checkedNodeCount(nx, ny));
    return result;
}

void addTerm(PointConstraint& constraint, std::size_t index, Scalar coefficient);
Scalar constraintValue(const PointConstraint& constraint,
                       const std::vector<Scalar>& values);

bool visibleNodes(const Grid& grid, std::size_t a, std::size_t b)
{
    return !grid.faults || grid.faults->visible(
        grid.x(a % grid.nx), grid.y(a / grid.nx),
        grid.x(b % grid.nx), grid.y(b / grid.nx));
}

// Общий оператор выборки с барьерами (fault-aware sampling operator).
// Билинейные коэффициенты невидимых углов удаляются; оставшиеся образуют
// разбиение единицы после перенормировки. Так константа на каждой стороне
// воспроизводится точно. Линейная точность непосредственно у разлома не
// гарантируется: это дискретная аппроксимация разрыва на регулярных узлах.
// Дополнительно требуем взаимную видимость всех выбранных углов: штраф c*c^T
// не должен создавать поперечную связь, обходящую маску энергии B^T*B.
// При нескольких совместимых наборах выбирается набор максимального суммарного
// исходного веса (из максимум 15 подмножеств четырех углов).
bool samplingConstraint(const Grid& grid, Scalar x, Scalar y,
                        PointConstraint& result)
{
    result = {};
    if (grid.faults && grid.faults->onFault(x, y)) return false;
    const Scalar fx = std::clamp((x - grid.minx) / grid.dx(), Scalar(0),
                                 static_cast<Scalar>(grid.nx - 1));
    const Scalar fy = std::clamp((y - grid.miny) / grid.dy(), Scalar(0),
                                 static_cast<Scalar>(grid.ny - 1));
    const std::size_t ix = static_cast<std::size_t>(std::floor(fx));
    const std::size_t iy = static_cast<std::size_t>(std::floor(fy));
    const std::size_t ix1 = std::min(ix + 1, grid.nx - 1);
    const std::size_t iy1 = std::min(iy + 1, grid.ny - 1);
    const Scalar tx = fx - static_cast<Scalar>(ix);
    const Scalar ty = fy - static_cast<Scalar>(iy);
    PointConstraint corners;
    addTerm(corners, grid.index(ix, iy), (1 - tx) * (1 - ty));
    addTerm(corners, grid.index(ix1, iy), tx * (1 - ty));
    addTerm(corners, grid.index(ix, iy1), (1 - tx) * ty);
    addTerm(corners, grid.index(ix1, iy1), tx * ty);
    if (!grid.faults) { result = corners; return true; }

    unsigned visible = 0;
    for (std::size_t k = 0; k < corners.count; ++k) {
        const auto index = corners.terms[k].index;
        if (grid.faults->visible(x, y, grid.x(index % grid.nx), grid.y(index / grid.nx)))
            visible |= 1u << k;
    }
    unsigned best = 0;
    Scalar bestWeight = 0;
    for (unsigned mask = 1; mask < (1u << corners.count); ++mask) {
        if ((mask & visible) != mask) continue;
        bool compatible = true;
        Scalar weight = 0;
        for (std::size_t a = 0; a < corners.count && compatible; ++a) {
            if (!(mask & (1u << a))) continue;
            weight += corners.terms[a].coefficient;
            for (std::size_t b = a + 1; b < corners.count; ++b) {
                if ((mask & (1u << b)) && !visibleNodes(grid, corners.terms[a].index,
                                                        corners.terms[b].index)) {
                    compatible = false;
                    break;
                }
            }
        }
        if (compatible && weight > bestWeight) { best = mask; bestWeight = weight; }
    }
    if (bestWeight > 0) {
        for (std::size_t k = 0; k < corners.count; ++k) {
            if (best & (1u << k)) addTerm(result, corners.terms[k].index,
                                         corners.terms[k].coefficient / bestWeight);
        }
        return true;
    }

    // Если все углы лежат на разломе/за ним, ищем локальную одностороннюю опору.
    // Дальний глобальный поиск опасен: узкий блок без узлов нельзя незаметно
    // заменить значением из другой части области. false требует более частой
    // сетки либо пропуска неподходящего грубого уровня.
    Scalar bestDistance = std::numeric_limits<Scalar>::infinity();
    std::size_t bestIndex = 0;
    for (std::size_t j = iy > 2 ? iy - 2 : 0; j <= std::min(iy + 2, grid.ny - 1); ++j) {
        for (std::size_t i = ix > 2 ? ix - 2 : 0; i <= std::min(ix + 2, grid.nx - 1); ++i) {
            const Scalar dx = static_cast<Scalar>(i) - fx;
            const Scalar dy = static_cast<Scalar>(j) - fy;
            const Scalar distance = dx * dx + dy * dy;
            if (distance < bestDistance && grid.faults->visible(x, y, grid.x(i), grid.y(j))) {
                bestDistance = distance;
                bestIndex = grid.index(i, j);
            }
        }
    }
    if (!finite(bestDistance)) return false;
    addTerm(result, bestIndex, Scalar(1));
    return true;
}

// Билинейная интерполяция (bilinear interpolation) в прямоугольной ячейке.
// Физические координаты переводятся в дробные индексы fx,fy; четыре веса
// равны (1-tx)(1-ty), tx(1-ty), (1-tx)ty, tx*ty и образуют разбиение единицы
// (partition of unity).
// Поэтому оператор точно возвращает узловые значения и воспроизводит любую
// аффинную плоскость. clamp защищает только от округления на замкнутой границе:
// точки вне Surface отсеиваются раньше в validateInput().
Scalar rawBilinearSample(const Grid& grid, Scalar x, Scalar y)
{
    Scalar fx = (x - grid.minx) / grid.dx();
    Scalar fy = (y - grid.miny) / grid.dy();
    fx = std::clamp(fx, Scalar(0), static_cast<Scalar>(grid.nx - 1));
    fy = std::clamp(fy, Scalar(0), static_cast<Scalar>(grid.ny - 1));

    const std::size_t ix0 = std::min(
        static_cast<std::size_t>(std::floor(fx)), grid.nx - 1);
    const std::size_t iy0 = std::min(
        static_cast<std::size_t>(std::floor(fy)), grid.ny - 1);
    const std::size_t ix1 = std::min(ix0 + 1, grid.nx - 1);
    const std::size_t iy1 = std::min(iy0 + 1, grid.ny - 1);
    const Scalar tx = fx - static_cast<Scalar>(ix0);
    const Scalar ty = fy - static_cast<Scalar>(iy0);

    const Scalar v00 = grid.values[grid.index(ix0, iy0)];
    const Scalar v10 = grid.values[grid.index(ix1, iy0)];
    const Scalar v01 = grid.values[grid.index(ix0, iy1)];
    const Scalar v11 = grid.values[grid.index(ix1, iy1)];
    return (Scalar(1) - tx) * (Scalar(1) - ty) * v00
         + tx * (Scalar(1) - ty) * v10
         + (Scalar(1) - tx) * ty * v01
         + tx * ty * v11;
}

Scalar bilinearSample(const Grid& grid, Scalar x, Scalar y)
{
    if (!grid.faults) return rawBilinearSample(grid, x, y);
    PointConstraint row;
    if (!samplingConstraint(grid, x, y, row)) {
        throw std::invalid_argument(
            "cannot sample on a fault or in a fault block without local grid nodes; refine nx/ny");
    }
    return constraintValue(row, grid.values);
}

// Поточечный билинейный ресемплинг (pointwise bilinear resampling) между
// сетками с одинаковыми min/max. Он используется как для огрубления
// (coarsening) входной поверхности, так и для продолжения (prolongation)
// решения на более частый уровень. Это НЕ оператор ограничения (restriction)
// с полным взвешиванием (full weighting) и не фильтр низких частот (low-pass
// filter), поэтому при огрублении нет подавления наложения спектров
// (anti-aliasing).
//
// Последний узел каждой оси явно ставится в maxx/maxy, чтобы не накопить ошибку
// округления координаты. Переносятся только значения u; производные отдельно
// не хранятся В Grid; v2 переносит их отдельно в координатах controls до вызова.
Grid resample(const Grid& source, std::size_t nx, std::size_t ny)
{
    Grid result = source;
    result.nx = nx;
    result.ny = ny;
    result.values.assign(checkedNodeCount(nx, ny), Scalar(0));

    for (std::size_t iy = 0; iy < ny; ++iy) {
        const Scalar ty = static_cast<Scalar>(iy) / static_cast<Scalar>(ny - 1);
        // С разломами один и тот же способ вычисления координат обязателен
        // для resample, геометрических масок и Snap: разные порядки округления
        // на больших XY могут переместить пограничный узел на другую сторону.
        const Scalar y = source.faults ? result.y(iy) : (iy + 1 == ny
            ? result.maxy
            : result.miny + ty * (result.maxy - result.miny));
        for (std::size_t ix = 0; ix < nx; ++ix) {
            const Scalar tx = static_cast<Scalar>(ix) / static_cast<Scalar>(nx - 1);
            const Scalar x = source.faults ? result.x(ix) : (ix + 1 == nx
                ? result.maxx
                : result.minx + tx * (result.maxx - result.minx));
            // Узел ровно на разломе не представляет ни одну из двух сторон.
            // Его буферное значение не участвует в связях/выборках; сохраняем
            // обычный prior, чтобы grid оставался конечным и без NaN-масок.
            result.values[result.index(ix, iy)] = source.faults && source.faults->onFault(x, y)
                ? rawBilinearSample(source, x, y) : bilinearSample(source, x, y);
        }
    }
    return result;
}

// Копирует публичный qreal-вектор во внутренний double, не изменяя row-major
// порядок и геометрию.
Grid surfaceAsGrid(const Surface& surface)
{
    Grid result = makeGridGeometry(surface, surface.nx, surface.ny);
    std::transform(surface.grid.begin(), surface.grid.end(), result.values.begin(),
        [](qreal value) { return static_cast<Scalar>(value); });
    return result;
}

// Целочисленное деление вверх (ceiling division) без сложения value+divisor-1,
// которое могло бы переполнить size_t.
std::size_t ceilDivide(std::size_t value, std::size_t divisor)
{
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

// Иерархия строится по числу ИНТЕРВАЛОВ (nx-1, ny-1). Для коэффициента f размер
// уровня равен ceil((finalNx-1)/f)+1 на X и аналогично на Y. f начинается с
// initialCoarseningFactor и делится на2 до1: порядок сразу coarse-to-fine.
//
// ceilDivide(...)+1 сохраняет общие физические границы для размеров, которые
// не имеют вида 2^k+1; поэтому узлы соседних уровней не обязаны точно совпадать,
// а последний переход не обязательно является точным удвоением. maxLevels —
// maxLevels — жёсткий предел; при нехватке уровней ошибка не скрывается
// усечённой иерархией. Последний уровень совпадает с исходными nx,ny.
std::vector<std::pair<std::size_t, std::size_t>> buildHierarchy(
    std::size_t finalNx, std::size_t finalNy,
    const ConvergentGriddingOptions& options)
{
    // В отличие от v1, factor — отношение шагов. Не степени двойки тоже
    // разрешены: 6 -> 3 -> 1.5 -> 1. Границы сохраняются, интервалы округляются.
    Scalar factor = std::max(Scalar(1), Scalar(options.initialCoarseningFactor));
    std::vector<std::pair<std::size_t, std::size_t>> result;
    for (;;) {
        auto dimension = [factor](std::size_t n) {
            return static_cast<std::size_t>(std::ceil(Scalar(n - 1) / factor)) + 1;
        };
        const auto dims = std::make_pair(dimension(finalNx), dimension(finalNy));
        if (result.empty() || result.back() != dims) result.push_back(dims);
        if (factor <= 1) break;
        if (result.size() >= options.maxLevels)
            throw std::invalid_argument("maxLevels is too small for initialCoarseningFactor");
        factor = std::max(Scalar(1), factor / 2);
    }
    return result;
}

// Проверка представимости (representability): каждый активный узел целевого
// уровня должен иметь одностороннюю опору на исходном. Проверяются именно
// соседние переходы, поскольку для 512 узлов уровни не строго вложены.
bool supportsTransfer(const Grid& source, const Grid& target)
{
    PointConstraint row;
    for (std::size_t j = 0; j < target.ny; ++j) {
        for (std::size_t i = 0; i < target.nx; ++i) {
            const Scalar x = target.x(i), y = target.y(j);
            if (source.faults->onFault(x, y)) continue;
            if (!samplingConstraint(source, x, y, row)) return false;
        }
    }
    return true;
}

Scalar valueAt(const Grid& grid, std::size_t ix, std::size_t iy)
{
    return grid.values[grid.index(ix, iy)];
}

// Оценка производных конечными разностями (finite differences). Производные
// нужны только для Taylor-переноса контрольного значения на Snap-узлы:
//
//   gx  ~= (u[i+1,j]-u[i-1,j])/(2*hx),
//   gy  ~= (u[i,j+1]-u[i,j-1])/(2*hy),
//   gxx ~= (u[i-1,j]-2*u[i,j]+u[i+1,j])/(hx^2),
//   gyy ~= (u[i,j-1]-2*u[i,j]+u[i,j+1])/(hy^2).
//
// gxy получается перекрестной разностью по доступному прямоугольнику. Внутри
// сетки шаблоны центральные (centered differences), у края — односторонние или
// асимметричные и обычно имеют меньший порядок точности. Деление на hx/hy
// означает, что gradient и Hessian выражены в физических единицах x/y.
// Эти формулы НЕ задают граничное условие основного сглаживающего оператора.
Derivatives derivativesAtNode(const Grid& grid, std::size_t ix, std::size_t iy)
{
    const Scalar hx = grid.dx();
    const Scalar hy = grid.dy();
    Derivatives d;

    if (grid.faults) {
        const auto center = grid.index(ix, iy);
        if (grid.faults->onFault(grid.x(ix), grid.y(iy))) return d;
        // Односторонние конечные разности (one-sided finite differences).
        // Центральная разность допустима только при видимости обоих соседей.
        // Если доступно два узла с одной стороны, восстанавливаем и кривизну;
        // иначе производная второго порядка не наблюдается и принимается нулевой.
        const auto along = [&](bool alongX, Scalar& first, Scalar& second) {
            const auto position = alongX ? ix : iy;
            const auto size = alongX ? grid.nx : grid.ny;
            const Scalar step = alongX ? hx : hy;
            const auto indexAt = [&](std::size_t n) {
                return alongX ? grid.index(n, iy) : grid.index(ix, n);
            };
            const auto safe = [&](std::size_t n) { return visibleNodes(grid, center, indexAt(n)); };
            const auto v = [&](std::size_t n) { return grid.values[indexAt(n)]; };
            const Scalar u = grid.values[center];
            const bool left = position > 0 && safe(position - 1);
            const bool right = position + 1 < size && safe(position + 1);
            if (left && right) {
                first = (v(position + 1) - v(position - 1)) / (2 * step);
                second = (v(position - 1) - 2 * u + v(position + 1)) / (step * step);
            } else if (right) {
                first = (v(position + 1) - u) / step;
                if (position + 2 < size && safe(position + 2))
                    second = (u - 2 * v(position + 1) + v(position + 2)) / (step * step);
            } else if (left) {
                first = (u - v(position - 1)) / step;
                if (position >= 2 && safe(position - 2))
                    second = (u - 2 * v(position - 1) + v(position - 2)) / (step * step);
            }
        };
        along(true, d.gx, d.gxx);
        along(false, d.gy, d.gyy);
        const auto x0 = ix > 0 && visibleNodes(grid, center, grid.index(ix - 1, iy)) ? ix - 1 : ix;
        const auto x1 = ix + 1 < grid.nx && visibleNodes(grid, center, grid.index(ix + 1, iy)) ? ix + 1 : ix;
        const auto y0 = iy > 0 && visibleNodes(grid, center, grid.index(ix, iy - 1)) ? iy - 1 : iy;
        const auto y1 = iy + 1 < grid.ny && visibleNodes(grid, center, grid.index(ix, iy + 1)) ? iy + 1 : iy;
        if (x1 > x0 && y1 > y0) {
            const std::array<std::size_t, 4> corners{
                grid.index(x0, y0), grid.index(x1, y0), grid.index(x0, y1), grid.index(x1, y1)};
            bool safe = true;
            for (std::size_t a = 0; a < 4 && safe; ++a) {
                if (!visibleNodes(grid, center, corners[a])) safe = false;
                for (std::size_t b = a + 1; b < 4 && safe; ++b)
                    if (!visibleNodes(grid, corners[a], corners[b])) safe = false;
            }
            if (safe) d.gxy = (grid.values[corners[3]] - grid.values[corners[1]]
                               - grid.values[corners[2]] + grid.values[corners[0]])
                / (static_cast<Scalar>(x1 - x0) * hx * static_cast<Scalar>(y1 - y0) * hy);
        }
        return d;
    }

    if (ix == 0) {
        d.gx = (valueAt(grid, 1, iy) - valueAt(grid, 0, iy)) / hx;
    } else if (ix + 1 == grid.nx) {
        d.gx = (valueAt(grid, ix, iy) - valueAt(grid, ix - 1, iy)) / hx;
    } else {
        d.gx = (valueAt(grid, ix + 1, iy) - valueAt(grid, ix - 1, iy)) / (Scalar(2) * hx);
    }

    if (iy == 0) {
        d.gy = (valueAt(grid, ix, 1) - valueAt(grid, ix, 0)) / hy;
    } else if (iy + 1 == grid.ny) {
        d.gy = (valueAt(grid, ix, iy) - valueAt(grid, ix, iy - 1)) / hy;
    } else {
        d.gy = (valueAt(grid, ix, iy + 1) - valueAt(grid, ix, iy - 1)) / (Scalar(2) * hy);
    }

    if (grid.nx >= 3) {
        if (ix == 0) {
            d.gxx = (valueAt(grid, 0, iy) - Scalar(2) * valueAt(grid, 1, iy)
                     + valueAt(grid, 2, iy)) / (hx * hx);
        } else if (ix + 1 == grid.nx) {
            d.gxx = (valueAt(grid, ix, iy) - Scalar(2) * valueAt(grid, ix - 1, iy)
                     + valueAt(grid, ix - 2, iy)) / (hx * hx);
        } else {
            d.gxx = (valueAt(grid, ix - 1, iy) - Scalar(2) * valueAt(grid, ix, iy)
                     + valueAt(grid, ix + 1, iy)) / (hx * hx);
        }
    }

    if (grid.ny >= 3) {
        if (iy == 0) {
            d.gyy = (valueAt(grid, ix, 0) - Scalar(2) * valueAt(grid, ix, 1)
                     + valueAt(grid, ix, 2)) / (hy * hy);
        } else if (iy + 1 == grid.ny) {
            d.gyy = (valueAt(grid, ix, iy) - Scalar(2) * valueAt(grid, ix, iy - 1)
                     + valueAt(grid, ix, iy - 2)) / (hy * hy);
        } else {
            d.gyy = (valueAt(grid, ix, iy - 1) - Scalar(2) * valueAt(grid, ix, iy)
                     + valueAt(grid, ix, iy + 1)) / (hy * hy);
        }
    }

    const std::size_t ix0 = ix == 0 ? 0 : ix - 1;
    const std::size_t ix1 = ix + 1 < grid.nx ? ix + 1 : ix;
    const std::size_t iy0 = iy == 0 ? 0 : iy - 1;
    const std::size_t iy1 = iy + 1 < grid.ny ? iy + 1 : iy;
    const Scalar spanX = static_cast<Scalar>(ix1 - ix0) * hx;
    const Scalar spanY = static_cast<Scalar>(iy1 - iy0) * hy;
    if (spanX > 0 && spanY > 0) {
        d.gxy = (valueAt(grid, ix1, iy1) - valueAt(grid, ix1, iy0)
                 - valueAt(grid, ix0, iy1) + valueAt(grid, ix0, iy0))
            / (spanX * spanY);
    }
    return d;
}

// Производные в произвольной координате получают в два этапа: вычисляют пять
// компонент в четырех углах содержащей ячейки, затем билинейно интерполируют
// каждое поле. Это выбор данной реализации: отдельные derivative grids между
// уровнями не переносятся (not carried); Taylor-модель описывает prior текущего
// уровня, а не восстанавливает производные непосредственно из облака точек.
Derivatives derivativesAt(const Grid& grid, Scalar x, Scalar y)
{
    if (grid.faults) {
        PointConstraint row;
        if (!samplingConstraint(grid, x, y, row))
            throw std::invalid_argument("Taylor derivatives have no visible grid support; refine nx/ny");
        Derivatives result;
        for (std::size_t k = 0; k < row.count; ++k) {
            const auto& term = row.terms[k];
            const auto d = derivativesAtNode(grid, term.index % grid.nx, term.index / grid.nx);
            result.gx += term.coefficient * d.gx;
            result.gy += term.coefficient * d.gy;
            result.gxx += term.coefficient * d.gxx;
            result.gxy += term.coefficient * d.gxy;
            result.gyy += term.coefficient * d.gyy;
        }
        return result;
    }
    Scalar fx = std::clamp((x - grid.minx) / grid.dx(), Scalar(0),
                           static_cast<Scalar>(grid.nx - 1));
    Scalar fy = std::clamp((y - grid.miny) / grid.dy(), Scalar(0),
                           static_cast<Scalar>(grid.ny - 1));
    const std::size_t ix0 = std::min(static_cast<std::size_t>(std::floor(fx)), grid.nx - 1);
    const std::size_t iy0 = std::min(static_cast<std::size_t>(std::floor(fy)), grid.ny - 1);
    const std::size_t ix1 = std::min(ix0 + 1, grid.nx - 1);
    const std::size_t iy1 = std::min(iy0 + 1, grid.ny - 1);
    const Scalar tx = fx - static_cast<Scalar>(ix0);
    const Scalar ty = fy - static_cast<Scalar>(iy0);

    const Derivatives d00 = derivativesAtNode(grid, ix0, iy0);
    const Derivatives d10 = derivativesAtNode(grid, ix1, iy0);
    const Derivatives d01 = derivativesAtNode(grid, ix0, iy1);
    const Derivatives d11 = derivativesAtNode(grid, ix1, iy1);
    const auto blend = [&](Scalar Derivatives::*member) {
        return (Scalar(1) - tx) * (Scalar(1) - ty) * d00.*member
             + tx * (Scalar(1) - ty) * d10.*member
             + (Scalar(1) - tx) * ty * d01.*member
             + tx * ty * d11.*member;
    };
    return {blend(&Derivatives::gx), blend(&Derivatives::gy),
            blend(&Derivatives::gxx), blend(&Derivatives::gxy),
            blend(&Derivatives::gyy)};
}

struct CandidateNode {
    std::size_t index{};
    std::size_t ix{};
    std::size_t iy{};
    Scalar distanceSquared{};
};

// Поиск ближайших соседей (nearest-neighbor search) ведется по евклидову
// расстоянию в КООРДИНАТАХ ЯЧЕЕК:
//   distanceCells^2 = (ix-fx)^2 + (iy-fy)^2.
// При dx!=dy это не физическая евклидова метрика: одна ячейка по X и Y считается
// одинаковым расстоянием, поэтому область влияния в физических координатах
// растянута. Равные расстояния разрешаются плоским индексом для детерминизма.
std::vector<CandidateNode> nearestNodes(
    const Grid& grid, Scalar x, Scalar y, std::size_t requested)
{
    requested = std::min(requested, grid.values.size());
    const Scalar fx = (x - grid.minx) / grid.dx();
    const Scalar fy = (y - grid.miny) / grid.dy();
    const std::size_t centerX = std::min(
        static_cast<std::size_t>(std::floor(std::clamp(
            fx, Scalar(0), static_cast<Scalar>(grid.nx - 1)))), grid.nx - 1);
    const std::size_t centerY = std::min(
        static_cast<std::size_t>(std::floor(std::clamp(
            fy, Scalar(0), static_cast<Scalar>(grid.ny - 1)))), grid.ny - 1);
    const std::size_t radius = static_cast<std::size_t>(
        std::ceil(std::sqrt(static_cast<Scalar>(requested)))) + 3;
    const std::size_t minX = centerX > radius ? centerX - radius : 0;
    const std::size_t minY = centerY > radius ? centerY - radius : 0;
    const std::size_t maxX = std::min(centerX + radius, grid.nx - 1);
    const std::size_t maxY = std::min(centerY + radius, grid.ny - 1);

    std::vector<CandidateNode> candidates;
    candidates.reserve((maxX - minX + 1) * (maxY - minY + 1));
    for (std::size_t iy = minY; iy <= maxY; ++iy) {
        for (std::size_t ix = minX; ix <= maxX; ++ix) {
            if (grid.faults && !grid.faults->visible(x, y, grid.x(ix), grid.y(iy))) continue;
            const Scalar dxCells = static_cast<Scalar>(ix) - fx;
            const Scalar dyCells = static_cast<Scalar>(iy) - fy;
            candidates.push_back({grid.index(ix, iy), ix, iy,
                                  dxCells * dxCells + dyCells * dyCells});
        }
    }

    // Для очень тонкой сетки или точки у края локального квадрата может быть
    // недостаточно. Редкий полный просмотр сохраняет точный результат именно
    // поиска ближайших узлов (это не относится к exact-интерполяции значений).
    if (candidates.size() < requested && !grid.faults) {
        candidates.clear();
        candidates.reserve(grid.values.size());
        for (std::size_t iy = 0; iy < grid.ny; ++iy) {
            for (std::size_t ix = 0; ix < grid.nx; ++ix) {
                const Scalar dxCells = static_cast<Scalar>(ix) - fx;
                const Scalar dyCells = static_cast<Scalar>(iy) - fy;
                candidates.push_back({grid.index(ix, iy), ix, iy,
                                      dxCells * dxCells + dyCells * dyCells});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const CandidateNode& a,
                                                       const CandidateNode& b) {
        if (a.distanceSquared != b.distanceSquared) {
            return a.distanceSquared < b.distanceSquared;
        }
        return a.index < b.index;
    });
    if (candidates.empty()) {
        throw std::invalid_argument("control point has no visible Snap nodes; refine nx/ny");
    }
    // Барьер может оставить меньше узлов, чем номинальное расписание Snap.
    // Используем доступные, не расширяя связь через разлом ради нужного числа.
    candidates.resize(std::min(requested, candidates.size()));
    return candidates;
}

// Расписание Snap (Snap schedule) локализует влияние точки при уточнении.
// Точная формула для нефинального уровня:
//   base = min(initialSnapNodes, numberOfNodesOnCurrentLevel),
// Формирует мягкую Snap-часть квадратичного функционала (soft Snap penalty).
// Snap означает перенос влияния контрольной точки на несколько ближайших узлов
// текущего уровня. Для p и узла n вводится физическое смещение
//
//   delta_pn = [x_n-x_p, y_n-y_p]^T
//
// и прогноз по Тейлору (Taylor projection)
//
//   T_p(n) = d_p
//          + gradient(prior)(p)^T * delta_pn
//          + 1/2 * delta_pn^T * Hessian(prior)(p) * delta_pn.
//
// Переданный текущим проходом taylorOrder оставляет соответственно только d_p, линейную часть или
// полный квадратичный член. Обратите внимание: разложение привязано к
// ИЗМЕРЕННОМУ d_p, а градиент/матрица Гессиана (gradient/Hessian) берутся из
// prior; delta имеет физические единицы x/y. При этом расстояние для гауссова
// ядра (Gaussian kernel)
//
//   k_pn = exp(-distanceCells^2/(2*sigma^2))
//
// измеряется в ячейках. При постоянной sigma физические радиусы sigma*dx и
// sigma*dy уменьшаются на более частых уровнях, поэтому воздействие точки
// локализуется. k_pn нормируется ОТДЕЛЬНО для каждой точки, после чего
// w_pn=w_p*k_pn/sum_n(k_pn), то есть sum_n(w_pn)=w_p.
//
// Если несколько точек воздействуют на узел n, то
//   W_n    = sum_p w_pn,
//   Tbar_n = sum_p(w_pn*T_p(n))/W_n.
// С точностью до не зависящей от u константы их вклад эквивалентен одному
// штрафу W_n*(u_n-Tbar_n)^2. diagonal и rhs накапливают градиент функционала
//
//   1/2 * snapStrength * sum_pn w_pn*(u_n-T_p(n))^2.
//
// Следовательно, конечный snapStrength задает МЯГКУЮ привязку: u_n не обязан
// точно равняться T_p(n). Форма Gaussian-ядра — выбор данной реализации, а не
// утверждение о закрытой функции смешивания (blending function) Petrel.
void addSnapConstraints(
    const Grid& prior,
    const std::vector<CanonicalPoint>& points,
    std::size_t count,
    const ConvergentGriddingOptions& options,
    const std::vector<Derivatives>& pointDerivatives,
    int taylorOrder,
    std::vector<Scalar>& diagonal,
    std::vector<Scalar>& rhs)
{
    std::vector<Scalar> accumulatedWeight(prior.values.size(), Scalar(0));
    std::vector<Scalar> accumulatedValue(prior.values.size(), Scalar(0));
    const Scalar sigma2 = static_cast<Scalar>(options.gaussianSigma)
        * static_cast<Scalar>(options.gaussianSigma);

    for (std::size_t pi = 0; pi < points.size(); ++pi) {
        const CanonicalPoint& point = points[pi];
        const Derivatives d = pointDerivatives[pi];
        const std::vector<CandidateNode> nodes = nearestNodes(prior, point.x, point.y, count);
        std::vector<Scalar> kernels(nodes.size());
        Scalar kernelSum = 0;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            // Вычитание ближайшего r^2 сокращается при нормировке и защищает
            // fault-вариант от underflow, когда ближайший видимый узел дальше.
            const Scalar shifted = nodes[i].distanceSquared
                - (prior.faults ? nodes.front().distanceSquared : Scalar(0));
            kernels[i] = std::exp(-shifted / (Scalar(2) * sigma2));
            kernelSum += kernels[i];
        }
        if (!(kernelSum > 0) || !finite(kernelSum)) {
            throw std::runtime_error("failed to normalize the Snap distance kernel");
        }

        for (std::size_t i = 0; i < nodes.size(); ++i) {
            const CandidateNode& node = nodes[i];
            const Scalar nodeX = prior.faults ? prior.x(node.ix)
                : prior.minx + static_cast<Scalar>(node.ix) * prior.dx();
            const Scalar nodeY = prior.faults ? prior.y(node.iy)
                : prior.miny + static_cast<Scalar>(node.iy) * prior.dy();
            const Scalar deltaX = nodeX - point.x;
            const Scalar deltaY = nodeY - point.y;
            Scalar projected = point.value;
            if (taylorOrder >= 1) {
                projected += d.gx * deltaX + d.gy * deltaY;
            }
            if (taylorOrder >= 2) {
                projected += Scalar(0.5) * (d.gxx * deltaX * deltaX
                    + Scalar(2) * d.gxy * deltaX * deltaY
                    + d.gyy * deltaY * deltaY);
            }
            if (!finite(projected)) {
                throw std::runtime_error("Taylor projection produced a non-finite value");
            }

            const Scalar weight = point.weight * kernels[i] / kernelSum;
            const Scalar nextWeight = accumulatedWeight[node.index] + weight;
            const Scalar nextValue = accumulatedValue[node.index] + weight * projected;
            if (!finite(weight) || !finite(nextWeight) || !finite(nextValue)) {
                throw std::runtime_error("Snap constraint accumulation overflowed");
            }
            accumulatedWeight[node.index] = nextWeight;
            accumulatedValue[node.index] = nextValue;
        }
    }

    const Scalar strength = static_cast<Scalar>(options.snapStrength);
    for (std::size_t i = 0; i < diagonal.size(); ++i) {
        diagonal[i] += strength * accumulatedWeight[i];
        rhs[i] += strength * accumulatedValue[i];
        if (!finite(diagonal[i]) || !finite(rhs[i])) {
            throw std::runtime_error("scaled Snap constraint overflowed");
        }
    }
}

// Добавляет коэффициент в разреженный разностный шаблон (sparse stencil)
// ограничения. Совпавшие индексы объединяются; это происходит, например, на
// maxx/maxy, где два формальных угла билинейной ячейки обозначают один узел.
void addTerm(PointConstraint& constraint, std::size_t index, Scalar coefficient)
{
    if (coefficient == 0) return;
    for (std::size_t i = 0; i < constraint.count; ++i) {
        if (constraint.terms[i].index == index) {
            constraint.terms[i].coefficient += coefficient;
            return;
        }
    }
    if (constraint.count == constraint.terms.size())
        throw std::logic_error("constraint exceeds eight-node support");
    constraint.terms[constraint.count++] = {index, coefficient};
}

// Строит строку C_p оператора билинейной выборки (bilinear sampling operator)
// для каждой точки:
//   C_p * u = sum_{k=1..4} c_k * u_k.
// В строке остается от одного до четырех уникальных узлов, коэффициенты
// неотрицательны и в точной арифметике суммируются в единицу. Поэтому C_p*u
// является значением кусочно-билинейной поверхности именно в (x_p,y_p), а не
// значением одного ближайшего узла. Разные точки одной ячейки создают разные
// строки C, которые, однако, могут оказаться линейно зависимыми.
std::vector<PointConstraint> makePointConstraints(
    const Grid& grid, const std::vector<CanonicalPoint>& points)
{
    std::vector<PointConstraint> result;
    result.reserve(points.size());
    for (const CanonicalPoint& point : points) {
        if (grid.faults) {
            PointConstraint constraint;
            if (!samplingConstraint(grid, point.x, point.y, constraint))
                throw std::invalid_argument("control point has no fault-aware interpolation support; refine nx/ny");
            constraint.value = point.value;
            constraint.weight = point.weight;
            result.push_back(constraint);
            continue;
        }
        Scalar fx = std::clamp((point.x - grid.minx) / grid.dx(), Scalar(0),
                               static_cast<Scalar>(grid.nx - 1));
        Scalar fy = std::clamp((point.y - grid.miny) / grid.dy(), Scalar(0),
                               static_cast<Scalar>(grid.ny - 1));
        const std::size_t ix0 = std::min(static_cast<std::size_t>(std::floor(fx)), grid.nx - 1);
        const std::size_t iy0 = std::min(static_cast<std::size_t>(std::floor(fy)), grid.ny - 1);
        const std::size_t ix1 = std::min(ix0 + 1, grid.nx - 1);
        const std::size_t iy1 = std::min(iy0 + 1, grid.ny - 1);
        const Scalar tx = fx - static_cast<Scalar>(ix0);
        const Scalar ty = fy - static_cast<Scalar>(iy0);

        PointConstraint constraint;
        constraint.value = point.value;
        constraint.weight = point.weight;
        addTerm(constraint, grid.index(ix0, iy0), (Scalar(1) - tx) * (Scalar(1) - ty));
        addTerm(constraint, grid.index(ix1, iy0), tx * (Scalar(1) - ty));
        addTerm(constraint, grid.index(ix0, iy1), (Scalar(1) - tx) * ty);
        addTerm(constraint, grid.index(ix1, iy1), tx * ty);
        result.push_back(constraint);
    }
    return result;
}

// Применяет нормальный оператор (normal operator) дискретной полунормы
// Гессиана (Hessian seminorm) и НАКАПЛИВАЕТ K*input в output.
//
// Сначала определим СЫРЫЕ, то есть еще не поделенные на шаги, разности:
//
//   delta_xx u[i,j] = u[i-1,j] - 2*u[i,j] + u[i+1,j],
//   delta_yy u[i,j] = u[i,j-1] - 2*u[i,j] + u[i,j+1],
//   delta_xy4 u[i,j] = u[i+1,j+1] - u[i+1,j-1]
//                     - u[i-1,j+1] + u[i-1,j-1].
//
// Строки B имеют вид
//
//   Bx  = (hy/hx) * delta_xx,
//   By  = (hx/hy) * delta_yy,
//   Bxy = (sqrt(2)/4) * delta_xy4.
//
// Поэтому точная дискретная энергия кода равна
//
//   J_code(u) = ||B*u||_2^2
//             = sum ((hy/hx)*delta_xx u)^2
//             + sum ((hx/hy)*delta_yy u)^2
//             + 2*sum ((1/4)*delta_xy4 u)^2.
//
// Коэффициент 2 появляется потому, что в норме Фробениуса симметричной матрицы
// Гессиана (Hessian) смешанная производная учитывается дважды:
// u_xy^2+u_yx^2=2*u_xy^2.
// Непрерывный аналог — энергия изгиба тонкой пластины (thin-plate bending
// energy):
//
//   J_TP(u) = integral [u_xx^2 + 2*u_xy^2 + u_yy^2] dx dy
//           = integral ||H(u)||_F^2 dx dy.
//
// Для прямоугольной квадратурной дискретизации (quadrature discretization) на
// тех же внутренних шаблонах J_code=hx*hy*J_TP,h. Множитель hx*hy общий внутри
// одного уровня и не меняет минимизатор (minimizer) чистой задачи кривизны, но
// меняется между уровнями. Поэтому относительный баланс smoothness против
// prior/Snap/final penalties зависит от разрешения.
//
// Градиент 1/2*smoothness*||B*u||^2 равен
// K*u=smoothness*B^T*B*u. В непрерывном интерьере соответствующий оператор
// Эйлера—Лагранжа имеет четвертый порядок:
//   u_xxxx + 2*u_xxyy + u_yyyy = Delta^2*u,
// то есть является бигармоническим оператором (biharmonic operator). Однако
// выбранный B^T*B не равен буквально квадрату одного стандартного пятиузлового
// дискретного Лапласиана: отличаются высокочастотный разностный шаблон
// (high-frequency stencil) и граничные строки.
// Поэтому точнее говорить «дискретный оператор бигармонического типа»
// (discrete biharmonic-type operator).
//
// Явные условия Дирихле/Неймана (Dirichlet/Neumann), периодические условия
// (periodic conditions) и фиктивные узлы (ghost nodes) отсутствуют. Граничные
// узлы остаются неизвестными, а их уравнения индуцируются только теми строками
// B, которые целиком помещаются в grid. Это дискретная вариационная обработка
// свободного края (free/natural boundary treatment), но не заявление о точном
// воспроизведении конкретных непрерывных условий свободной пластины (free-plate
// boundary conditions). При обычных размерах любое аффинное поле лежит в ядре
// (null space) B; на очень тонких сетках ядро может быть шире.
//
// Функция не очищает output: вызывающий код обязан сделать это сам или намеренно
// суммировать K*input с другими операторами.
// Геометрическая маска строк B (cut-stencil mask), рассчитанная один раз до PCG.
// Если опорные узлы строки видят друг друга через разлом, строка удаляется целиком.
// Простое зануление отдельных элементов готовой B^T*B могло бы разрушить
// симметрию/положительную полуопределенность; удаление строк B сохраняет обе.
// Проверяются также диагонали четырехузлового mixed-шаблона: разлом может
// пересечь диагональную связь, не пересекая один из выбранных осевых отрезков.
std::vector<std::uint8_t> faultCurvatureMask(const Grid& grid)
{
    if (!grid.faults) return {};
    std::vector<std::uint8_t> mask(grid.values.size(), 0);
    const auto allowed = [&](std::initializer_list<std::size_t> nodes) {
        for (auto a = nodes.begin(); a != nodes.end(); ++a)
            for (auto b = a + 1; b != nodes.end(); ++b)
                if (!visibleNodes(grid, *a, *b)) return false;
        return true;
    };
    for (std::size_t j = 0; j < grid.ny; ++j) {
        for (std::size_t i = 0; i < grid.nx; ++i) {
            const auto center = grid.index(i, j);
            if (i > 0 && i + 1 < grid.nx && allowed({grid.index(i - 1, j), center, grid.index(i + 1, j)}))
                mask[center] |= 1;
            if (j > 0 && j + 1 < grid.ny && allowed({grid.index(i, j - 1), center, grid.index(i, j + 1)}))
                mask[center] |= 2;
            if (i > 0 && i + 1 < grid.nx && j > 0 && j + 1 < grid.ny
                && allowed({grid.index(i + 1, j + 1), grid.index(i + 1, j - 1),
                            grid.index(i - 1, j + 1), grid.index(i - 1, j - 1)}))
                mask[center] |= 4;
        }
    }
    return mask;
}

void applyCurvature(
    const Grid& grid,
    const std::vector<Scalar>& input,
    std::vector<Scalar>& output,
    Scalar smoothness,
    const std::vector<std::uint8_t>* mask = nullptr)
{
    if (smoothness == 0) return;
    // ax масштабирует вторую разность по X, ay — по Y. Названия относятся к
    // направлению производной, а не к числителю отношения шагов.
    const Scalar ax = grid.dy() / grid.dx();
    const Scalar ay = grid.dx() / grid.dy();
    const auto accumulate3 = [&](std::size_t a, std::size_t b, std::size_t c,
                                 Scalar scale) {
        const Scalar d = scale * (input[a] - Scalar(2) * input[b] + input[c]);
        const Scalar weighted = smoothness * scale * d;
        output[a] += weighted;
        output[b] -= Scalar(2) * weighted;
        output[c] += weighted;
    };

    if (grid.nx >= 3) {
        for (std::size_t iy = 0; iy < grid.ny; ++iy) {
            for (std::size_t ix = 1; ix + 1 < grid.nx; ++ix) {
                if (mask && !((*mask)[grid.index(ix, iy)] & 1)) continue;
                accumulate3(grid.index(ix - 1, iy), grid.index(ix, iy),
                            grid.index(ix + 1, iy), ax);
            }
        }
    }
    if (grid.ny >= 3) {
        for (std::size_t iy = 1; iy + 1 < grid.ny; ++iy) {
            for (std::size_t ix = 0; ix < grid.nx; ++ix) {
                if (mask && !((*mask)[grid.index(ix, iy)] & 2)) continue;
                accumulate3(grid.index(ix, iy - 1), grid.index(ix, iy),
                            grid.index(ix, iy + 1), ay);
            }
        }
    }
    if (grid.nx >= 3 && grid.ny >= 3) {
        // delta_xy4/(4*hx*hy) — стандартная центральная оценка u_xy. Физические
        // шаги уже сократились в выбранном cell-scaled B, поэтому здесь c=1/4.
        constexpr Scalar c = Scalar(0.25);
        for (std::size_t iy = 1; iy + 1 < grid.ny; ++iy) {
            for (std::size_t ix = 1; ix + 1 < grid.nx; ++ix) {
                if (mask && !((*mask)[grid.index(ix, iy)] & 4)) continue;
                const std::size_t pp = grid.index(ix + 1, iy + 1);
                const std::size_t pm = grid.index(ix + 1, iy - 1);
                const std::size_t mp = grid.index(ix - 1, iy + 1);
                const std::size_t mm = grid.index(ix - 1, iy - 1);
                const Scalar d = c * (input[pp] - input[pm] - input[mp] + input[mm]);
                const Scalar weighted = Scalar(2) * smoothness * c * d;
                output[pp] += weighted;
                output[pm] -= weighted;
                output[mp] -= weighted;
                output[mm] += weighted;
            }
        }
    }
}

// Вычисляет точную диагональ того же K=smoothness*B^T*B для диагонального
// предобуславливателя Якоби (Jacobi preconditioner). Для строки B с
// коэффициентами b_k вклад B^T*B в diagonal[k] равен b_k^2; отсюда коэффициенты
// 1,4,1 для второй разности и одинаковые квадраты четырех mixed-коэффициентов.
//
// При изменении stencil или его масштаба эту функцию необходимо менять
// синхронно с applyCurvature(). Несовпадение не меняет сам оператор applyA, но
// делает предобуславливатель неточным и способно резко ухудшить сходимость PCG.
std::vector<Scalar> curvatureDiagonal(const Grid& grid, Scalar smoothness,
                                     const std::vector<std::uint8_t>* mask = nullptr)
{
    std::vector<Scalar> diagonal(grid.values.size(), Scalar(0));
    if (smoothness == 0) return diagonal;
    const Scalar ax = grid.dy() / grid.dx();
    const Scalar ay = grid.dx() / grid.dy();
    const auto accumulate3 = [&](std::size_t a, std::size_t b, std::size_t c,
                                 Scalar scale) {
        const Scalar factor = smoothness * scale * scale;
        diagonal[a] += factor;
        diagonal[b] += Scalar(4) * factor;
        diagonal[c] += factor;
    };
    if (grid.nx >= 3) {
        for (std::size_t iy = 0; iy < grid.ny; ++iy) {
            for (std::size_t ix = 1; ix + 1 < grid.nx; ++ix) {
                if (mask && !((*mask)[grid.index(ix, iy)] & 1)) continue;
                accumulate3(grid.index(ix - 1, iy), grid.index(ix, iy),
                            grid.index(ix + 1, iy), ax);
            }
        }
    }
    if (grid.ny >= 3) {
        for (std::size_t iy = 1; iy + 1 < grid.ny; ++iy) {
            for (std::size_t ix = 0; ix < grid.nx; ++ix) {
                if (mask && !((*mask)[grid.index(ix, iy)] & 2)) continue;
                accumulate3(grid.index(ix, iy - 1), grid.index(ix, iy),
                            grid.index(ix, iy + 1), ay);
            }
        }
    }
    if (grid.nx >= 3 && grid.ny >= 3) {
        constexpr Scalar coefficientSquared = Scalar(1) / Scalar(16);
        const Scalar factor = Scalar(2) * smoothness * coefficientSquared;
        for (std::size_t iy = 1; iy + 1 < grid.ny; ++iy) {
            for (std::size_t ix = 1; ix + 1 < grid.nx; ++ix) {
                if (mask && !((*mask)[grid.index(ix, iy)] & 4)) continue;
                diagonal[grid.index(ix + 1, iy + 1)] += factor;
                diagonal[grid.index(ix + 1, iy - 1)] += factor;
                diagonal[grid.index(ix - 1, iy + 1)] += factor;
                diagonal[grid.index(ix - 1, iy - 1)] += factor;
            }
        }
    }
    return diagonal;
}

// Разреженное скалярное произведение одной строки C_p с узловым вектором u.
Scalar constraintValue(const PointConstraint& constraint,
                       const std::vector<Scalar>& values)
{
    Scalar result = 0;
    for (std::size_t i = 0; i < constraint.count; ++i) {
        result += constraint.terms[i].coefficient * values[constraint.terms[i].index];
    }
    return result;
}

Scalar dot(const std::vector<Scalar>& a, const std::vector<Scalar>& b)
{
    // Накопление выполняется в long double, чтобы использовать дополнительную
    // точность там, где ее предоставляет ABI. На MSVC long double обычно имеет
    // ту же точность, что double, поэтому улучшение не гарантируется платформой.
    // Наружу результат в любом случае возвращается как внутренний Scalar.
    long double result = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result += static_cast<long double>(a[i]) * static_cast<long double>(b[i]);
    }
    return static_cast<Scalar>(result);
}

// Решает на одном уровне неявную симметричную положительно определенную систему
// (Symmetric Positive Definite system, SPD) A*u=b. Она является условием
// стационарности регуляризованной least-squares задачи из общей схемы:
//
//   A = K + priorWeight*I + diag(snapDiagonal)
//         + sum_p finalStrength*w_p*c_p*c_p^T,
//   b = priorWeight*prior + snapRhs
//         + sum_p finalStrength*w_p*c_p*value_p.
//
// Здесь K=smoothness*B^T*B; c_p — строка билинейного оператора точки. Последняя
// сумма присутствует только на финальном уровне и является суммой обновлений
// ранга один (rank-one updates). K, Snap diagonal и c_p*c_p^T симметричны и
// положительно полуопределены (Symmetric Positive Semidefinite, SPSD).
//
// priorWeight=mu>0 — регуляризация Тихонова (Tikhonov regularization). Для
// любого ненулевого v
//
//   v^T*A*v = smoothness*||B*v||^2 + mu*||v||^2 + otherNonnegativeTerms
//           >= mu*||v||^2 > 0.
//
// Следовательно, priorWeight сдвигает ВСЕ ядро K и гарантирует SPD независимо
// от расположения точек — это математическое основание применять CG/PCG.
// Большой разброс весов все же может сделать A плохо обусловленной
// (ill-conditioned), хотя положительная определенность сохранится.
//
// «Матрично-свободный» (matrix-free) означает, что полная sparse-матрица A не
// собирается: applyA применяет K, диагональные части и точечные rank-one terms.
// Явно хранится только diag(A), необходимая для Jacobi. prior используется как
// теплый старт (warm start) x0.
SolverResult solveLevelCentered(
    const Grid& geometry,
    const std::vector<Scalar>& prior,
    const std::vector<Scalar>& extraDiagonal,
    const std::vector<Scalar>& extraRhs,
    const std::vector<PointConstraint>& finalConstraints,
    const ConvergentGriddingOptions& options)
{
    const std::size_t size = prior.size();
    const Scalar priorWeight = static_cast<Scalar>(options.priorWeight);
    const Scalar smoothness = static_cast<Scalar>(options.smoothness);
    const Scalar finalStrength = static_cast<Scalar>(options.finalPointStrength);

    // prior term дает mu*I и mu*prior; мягкий Snap уже агрегирован в
    // extraDiagonal/extraRhs как сумма узловых least-squares штрафов.
    std::vector<Scalar> rhs(size);
    const auto maskStorage = faultCurvatureMask(geometry);
    const auto* mask = geometry.faults ? &maskStorage : nullptr;
    std::vector<Scalar> diagonal = curvatureDiagonal(geometry, smoothness, mask);
    for (std::size_t i = 0; i < size; ++i) {
        diagonal[i] += priorWeight + extraDiagonal[i];
        rhs[i] = priorWeight * prior[i] + extraRhs[i];
        if (!finite(diagonal[i]) || !(diagonal[i] > 0) || !finite(rhs[i])) {
            throw std::runtime_error("level-system coefficients are not finite");
        }
    }
    // Для 1/2*rho*w*(c^T*u-d)^2 Hessian равен rho*w*c*c^T, а правая
    // часть — rho*w*c*d. В Jacobi diagonal попадают только rho*w*c_k^2.
    for (const PointConstraint& constraint : finalConstraints) {
        const Scalar strength = (constraint.strength < 0 ? finalStrength : constraint.strength) * constraint.weight;
        if (!finite(strength)) {
            throw std::runtime_error("point-constraint strength overflowed");
        }
        for (std::size_t a = 0; a < constraint.count; ++a) {
            const StencilTerm& term = constraint.terms[a];
            rhs[term.index] += strength * term.coefficient * constraint.value;
            diagonal[term.index] += strength * term.coefficient * term.coefficient;
            if (!finite(rhs[term.index]) || !finite(diagonal[term.index])) {
                throw std::runtime_error("point-constraint coefficients overflowed");
            }
        }
    }

    // Полное матрично-свободное умножение (matrix-free product) A*input.
    // В отличие от applyCurvature эта лямбда-функция (lambda function) начинает
    // с нулевого output, затем складывает все четыре части A.
    const auto applyA = [&](const std::vector<Scalar>& input,
                            std::vector<Scalar>& output) {
        std::fill(output.begin(), output.end(), Scalar(0));
        applyCurvature(geometry, input, output, smoothness, mask);
        for (std::size_t i = 0; i < size; ++i) {
            output[i] += (priorWeight + extraDiagonal[i]) * input[i];
        }
        for (const PointConstraint& constraint : finalConstraints) {
            const Scalar strength = (constraint.strength < 0 ? finalStrength : constraint.strength) * constraint.weight;
            const Scalar projected = constraintValue(constraint, input);
            for (std::size_t i = 0; i < constraint.count; ++i) {
                const StencilTerm& term = constraint.terms[i];
                output[term.index] += strength * term.coefficient * projected;
            }
        }
    };

    // Инициализация PCG:
    //   M=diag(A), r0=b-A*x0, z0=M^{-1}*r0, p0=z0.
    // Обозначение p для направления в коде представлено переменной direction,
    // чтобы не смешивать его с контрольной точкой Point.
    std::vector<Scalar> x = prior;
    std::vector<Scalar> ax(size), residual(size), z(size), direction(size), ad(size);
    applyA(x, ax);
    for (std::size_t i = 0; i < size; ++i) {
        residual[i] = rhs[i] - ax[i];
        z[i] = residual[i] / diagonal[i];
    }
    direction = z;
    Scalar rz = dot(residual, z);
    const Scalar rhsNorm = std::sqrt(std::max(Scalar(0), dot(rhs, rhs)));
    const Scalar initialAxNorm = std::sqrt(std::max(Scalar(0), dot(ax, ax)));
    const Scalar normalizer = std::max(rhsNorm, initialAxNorm);
    const Scalar tolerance = static_cast<Scalar>(options.relativeTolerance);
    const Scalar absoluteTolerance = static_cast<Scalar>(options.absoluteTolerance);
    // Нормировка фиксируется по начальному состоянию и не меняется во время
    // итераций: ||r||_2 <= absTol+relTol*max(||b||_2,||A*x0||_2).
    // Это не относительная невязка ||r_k||/||r_0||. Если normalizer==0,
    // relativeResidual по соглашению равен нулю.
    const Scalar threshold = absoluteTolerance + tolerance * normalizer;
    if (!finite(normalizer) || !finite(threshold)) {
        throw std::runtime_error("PCG residual scale is not finite");
    }
    Scalar residualNorm = std::sqrt(std::max(Scalar(0), dot(residual, residual)));
    Scalar relativeResidual = normalizer > 0 ? residualNorm / normalizer : Scalar(0);

    SolverResult result;
    result.values = x;
    result.relativeResidual = relativeResidual;
    result.converged = residualNorm <= threshold;
    if (result.converged) return result;

    // Одна стандартная PCG-итерация:
    //   alpha=(r^T*z)/(p^T*A*p), x<-x+alpha*p, r<-r-alpha*A*p,
    //   z<-M^{-1}r, beta=(r_new^T*z_new)/(r_old^T*z_old),
    //   p<-z+beta*p.
    for (std::size_t iteration = 0; iteration < options.maxSolverIterations; ++iteration) {
        applyA(direction, ad);
        const Scalar denominator = dot(direction, ad);
        if (!(denominator > 0) || !finite(denominator) || !finite(rz)) {
            throw std::runtime_error("PCG breakdown while solving the biharmonic system");
        }
        const Scalar alpha = rz / denominator;
        if (!finite(alpha)) {
            throw std::runtime_error("PCG produced a non-finite step");
        }
        for (std::size_t i = 0; i < size; ++i) {
            x[i] += alpha * direction[i];
            residual[i] -= alpha * ad[i];
        }

        // Обычно r обновляется дешевой рекуррентной формулой выше. Из-за
        // округления она постепенно расходится с истинной b-A*x, поэтому каждые
        // 512 шагов выполняются true-residual replacement и restart направления.
        // Частый restart разрушал низкочастотные моды при одних jump-данных
        // и малом priorWeight. Даём PCG накопить пространство Крылова.
        bool restartDirection = (iteration + 1) % 512 == 0;
        if (restartDirection) {
            applyA(x, ax);
            for (std::size_t i = 0; i < size; ++i) residual[i] = rhs[i] - ax[i];
        }
        residualNorm = std::sqrt(std::max(Scalar(0), dot(residual, residual)));
        relativeResidual = normalizer > 0 ? residualNorm / normalizer : Scalar(0);
        result.iterations = iteration + 1;
        if (residualNorm <= threshold && !restartDirection) {
            // Прежде чем объявить сходимость, обязательно проверить b-A*x.
            applyA(x, ax);
            for (std::size_t i=0;i<size;++i) residual[i]=rhs[i]-ax[i];
            residualNorm=std::sqrt(std::max(Scalar(0),dot(residual,residual)));
            relativeResidual=normalizer>0?residualNorm/normalizer:Scalar(0);
            restartDirection=true;
        }
        if (residualNorm <= threshold) {
            result.values = std::move(x);
            result.relativeResidual = relativeResidual;
            result.converged = true;
            return result;
        }

        for (std::size_t i = 0; i < size; ++i) z[i] = residual[i] / diagonal[i];
        const Scalar nextRz = dot(residual, z);
        if (!finite(nextRz)) {
            throw std::runtime_error("PCG produced a non-finite residual");
        }
        if (restartDirection) {
            direction = z;
        } else {
            const Scalar beta = nextRz / rz;
            for (std::size_t i = 0; i < size; ++i) {
                direction[i] = z[i] + beta * direction[i];
            }
        }
        rz = nextRz;
    }

    // При исчерпании лимита возвращается последнее приближение (last iterate).
    // Вызывающий код либо выбросит исключение, либо сохранит его в зависимости
    // от throwOnNonConvergence.
    result.values = std::move(x);
    result.relativeResidual = relativeResidual;
    result.converged = false;
    return result;
}

// Сдвиг начала Z (vertical datum centering). Из всех величин вычитается
// единая средняя отметка prior; строки jumps/dip имеют sum(c)=0 и не меняют
// правую часть. Это устраняет зависимость допуска PCG от произвольного датума
// +100/+10000 и потерю точности при вычитании больших близких b и A*prior.
SolverResult solveLevel(
    const Grid& geometry, const std::vector<Scalar>& prior,
    const std::vector<Scalar>& extraDiagonal, const std::vector<Scalar>& extraRhs,
    const std::vector<PointConstraint>& constraints,
    const ConvergentGriddingOptions& options)
{
    long double sum=0;
    for(Scalar v:prior)sum+=static_cast<long double>(v)/prior.size();
    const Scalar datum=Scalar(sum);
    std::vector<Scalar> shiftedPrior=prior,shiftedRhs=extraRhs;
    for(std::size_t n=0;n<prior.size();++n){
        shiftedPrior[n]-=datum;shiftedRhs[n]-=extraDiagonal[n]*datum;
    }
    auto shiftedConstraints=constraints;
    for(auto& c:shiftedConstraints){
        Scalar coefficientSum=0;
        for(std::size_t k=0;k<c.count;++k)coefficientSum+=c.terms[k].coefficient;
        c.value-=datum*coefficientSum;
    }
    auto result=solveLevelCentered(geometry,shiftedPrior,extraDiagonal,shiftedRhs,
                                   shiftedConstraints,options);
    for(auto& v:result.values)v+=datum;
    return result;
}

Scalar maximumAbsolute(const std::vector<Scalar>& values)
{
    Scalar result = 0;
    for (Scalar value : values) {
        if (!finite(value)) {
            throw std::runtime_error("point-projection residual is not finite");
        }
        result = std::max(result, std::abs(value));
    }
    return result;
}

// «Точное» прохождение здесь означает не алгебраическое равенство, а выполнение
// в эффективном абсолютном допуске (effective absolute tolerance). Масштаб
// начинается с 1, поэтому почти нулевые d_p не делают rounding floor нулевым.
Scalar effectiveControlTolerance(
    const std::vector<PointConstraint>& constraints,
    const ConvergentGriddingOptions& options)
{
    Scalar valueScale = 1;
    for (const PointConstraint& constraint : constraints) {
        valueScale = std::max(valueScale, std::abs(constraint.value));
    }
    // controlTolerance — абсолютный пользовательский допуск. Нижняя граница
    // учитывает точность публичного qreal, потому что именно в него в конце
    // преобразуется grid; внутренняя точность double сама по себе недостаточна.
    const Scalar roundingFloor = Scalar(64)
        * static_cast<Scalar>(std::numeric_limits<qreal>::epsilon()) * valueScale;
    return std::max(static_cast<Scalar>(options.controlTolerance), roundingFloor);
}

// Матрично-свободное умножение (C*C^T)*v в пространстве контрольных точек:
// сначала рассеивание (scatter) nodeWork=C^T*v, затем сборка (gather)
// output=C*nodeWork. Размерный вектор узлов переиспользуется как workspace.
void applyConstraintGram(
    const std::vector<PointConstraint>& constraints,
    const std::vector<Scalar>& input,
    std::vector<Scalar>& output,
    std::vector<Scalar>& nodeWork)
{
    std::fill(nodeWork.begin(), nodeWork.end(), Scalar(0));
    for (std::size_t i = 0; i < constraints.size(); ++i) {
        for (std::size_t termIndex = 0;
             termIndex < constraints[i].count; ++termIndex) {
            const StencilTerm& term = constraints[i].terms[termIndex];
            nodeWork[term.index] += term.coefficient * input[i];
        }
    }
    for (std::size_t i = 0; i < constraints.size(); ++i) {
        output[i] = constraintValue(constraints[i], nodeWork);
        if (!finite(output[i])) {
            throw std::runtime_error("point-constraint Gram product overflowed");
        }
    }
}

// Выполняет ортогональную евклидову проекцию (Euclidean orthogonal projection)
// гладкого решения на множество билинейных равенств. Пусть C имеет m строк
// (канонические точки) и N столбцов (узлы), r=d-C*u. Решается задача
//
//   minimize_delta  1/2*||delta||_2^2
//   subject to      C*delta = r.
//
// Из условий Каруша—Куна—Таккера (KKT conditions) следует
//
//   delta = C^T*lambda,
//   G*lambda = r,  где G=C*C^T.
//
// G называется матрицей Грама ограничений (constraint Gram matrix). Она всегда
// SPSD и становится SPD при линейно независимых строках C. Полная G не хранится:
// applyConstraintGram умножает на нее через рассеивание/сборку (scatter/gather).
// Диагональ G_pp=||c_p||_2^2 используется как Jacobi preconditioner, а Gram-PCG работает
// в m-мерном пространстве точек, а не в N-мерном пространстве grid.
//
// Критерий остановки здесь задан максимальной нормой (infinity norm):
//   ||r_k||_inf <= max(controlTolerance,
//                      64*epsilon(qreal)*max(1,max_p|d_p|)).
// При этом внутренние скалярные произведения CG остаются евклидовыми. В жесткой
// фазе (hard phase) Point::weight не масштабирует отдельное равенство: после
// объединения совпадающих точек все оставшиеся C_p*u=d_p обязательны с одним
// допуском.
//
// Важное ограничение: это НЕ A-ортогональная проекция минимальной энергии
// (minimum-energy / A-orthogonal projection). Для нее поправка имела бы вид
//
//   delta_A = A^{-1}*C^T*(C*A^{-1}*C^T)^{-1}*r.
//
// Здесь же delta принадлежит образу (range) C^T, поэтому поправка изменяет лишь
// объединение 1..4-узловых билинейных шаблонов (stencils) всех точек.
// Повторное сглаживание после нее не выполняется, и итог уже не обязан
// минимизировать Phi(u). Реализация не вычисляет ранг (rank) и псевдообратную
// матрицу (pseudoinverse): рангово-дефицитные строки, в том числе от разных
// точек одной ячейки, могут вызвать аварийную остановку (breakdown) PCG, а
// несовместимые равенства не могут пройти финальную проверку невязки.
ProjectionResult projectOntoPointConstraints(
    Grid& grid,
    const std::vector<PointConstraint>& constraints,
    const ConvergentGriddingOptions& options)
{
    ProjectionResult result;
    result.converged = true;
    if (constraints.empty()) return result;

    const std::size_t count = constraints.size();
    const Scalar tolerance = effectiveControlTolerance(constraints, options);
    // Правая часть Gram-системы — исходная контрольная невязка r=d-C*u.
    // inverseDiagonal содержит M^{-1}, M_pp=G_pp=||c_p||_2^2.
    std::vector<Scalar> rhs(count), residual(count), inverseDiagonal(count);
    for (std::size_t i = 0; i < count; ++i) {
        rhs[i] = constraints[i].value
            - constraintValue(constraints[i], grid.values);
        Scalar diagonal = 0;
        for (std::size_t termIndex = 0;
             termIndex < constraints[i].count; ++termIndex) {
            const Scalar coefficient = constraints[i].terms[termIndex].coefficient;
            diagonal += coefficient * coefficient;
        }
        if (!finite(rhs[i]) || !finite(diagonal) || !(diagonal > 0)) {
            throw std::runtime_error("invalid exact point constraint");
        }
        inverseDiagonal[i] = Scalar(1) / diagonal;
    }

    // Если soft-решение уже удовлетворяет допуску, correction равна нулю и
    // controlProjectionIterations закономерно остается равным 0.
    result.maxError = maximumAbsolute(rhs);
    if (result.maxError <= tolerance) return result;

    std::vector<Scalar> multiplier(count, Scalar(0));
    residual = rhs;
    std::vector<Scalar> preconditioned(count), direction(count), gramDirection(count);
    std::vector<Scalar> gramMultiplier(count), nodeWork(grid.values.size(), Scalar(0));
    for (std::size_t i = 0; i < count; ++i) {
        preconditioned[i] = residual[i] * inverseDiagonal[i];
    }
    direction = preconditioned;
    Scalar residualPreconditioned = dot(residual, preconditioned);

    // Далее используется та же PCG-рекуррентность, но для G*lambda=r. В отличие
    // от основного solver критерий проверяет max-absolute residual.
    result.converged = false;
    for (std::size_t iteration = 0;
         iteration < options.maxControlProjectionIterations; ++iteration) {
        applyConstraintGram(constraints, direction, gramDirection, nodeWork);
        const Scalar denominator = dot(direction, gramDirection);
        if (!(denominator > 0) || !finite(denominator)
            || !(residualPreconditioned > 0)
            || !finite(residualPreconditioned)) {
            throw std::runtime_error(
                "exact control equations are dependent or incompatible at this grid resolution");
        }
        const Scalar alpha = residualPreconditioned / denominator;
        if (!finite(alpha)) {
            throw std::runtime_error("exact-control projection produced a non-finite step");
        }
        for (std::size_t i = 0; i < count; ++i) {
            multiplier[i] += alpha * direction[i];
            residual[i] -= alpha * gramDirection[i];
        }

        // Каждые 25 шагов и перед возможным успешным выходом восстанавливается
        // истинная невязка rhs-G*lambda, чтобы exact-проверка не основывалась
        // только на накопленной рекуррентной невязке.
        const bool restartDirection = (iteration + 1) % 25 == 0;
        if (restartDirection || maximumAbsolute(residual) <= tolerance) {
            applyConstraintGram(constraints, multiplier, gramMultiplier, nodeWork);
            for (std::size_t i = 0; i < count; ++i) {
                residual[i] = rhs[i] - gramMultiplier[i];
            }
        }

        result.iterations = iteration + 1;
        result.maxError = maximumAbsolute(residual);
        if (result.maxError <= tolerance) {
            result.converged = true;
            break;
        }

        for (std::size_t i = 0; i < count; ++i) {
            preconditioned[i] = residual[i] * inverseDiagonal[i];
        }
        const Scalar nextResidualPreconditioned = dot(residual, preconditioned);
        if (!(nextResidualPreconditioned > 0)
            || !finite(nextResidualPreconditioned)) {
            throw std::runtime_error(
                "exact control equations are dependent or incompatible at this grid resolution");
        }
        if (restartDirection) {
            direction = preconditioned;
        } else {
            const Scalar beta = nextResidualPreconditioned / residualPreconditioned;
            for (std::size_t i = 0; i < count; ++i) {
                direction[i] = preconditioned[i] + beta * direction[i];
            }
        }
        residualPreconditioned = nextResidualPreconditioned;
    }

    if (!result.converged) {
        throw std::runtime_error(
            "exact control projection did not converge; controls may be incompatible "
            "at this grid resolution");
    }

    // После решения Gram-системы scatter C^T*lambda дает минимальную узловую
    // correction delta; она прибавляется к еще не опубликованной копии grid.
    std::fill(nodeWork.begin(), nodeWork.end(), Scalar(0));
    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t termIndex = 0;
             termIndex < constraints[i].count; ++termIndex) {
            const StencilTerm& term = constraints[i].terms[termIndex];
            nodeWork[term.index] += term.coefficient * multiplier[i];
        }
    }
    for (std::size_t i = 0; i < grid.values.size(); ++i) {
        grid.values[i] += nodeWork[i];
        if (!finite(grid.values[i])) {
            throw std::runtime_error("exact-control correction produced a non-finite grid value");
        }
    }

    // Финальная независимая проверка вычисляет C*(u+delta)-d непосредственно по
    // обновленным узлам, а не доверяет последней внутренней residual PCG.
    result.maxError = 0;
    for (const PointConstraint& constraint : constraints) {
        result.maxError = std::max(result.maxError,
            std::abs(constraintValue(constraint, grid.values) - constraint.value));
    }
    if (result.maxError > tolerance) {
        throw std::runtime_error(
            "exact-control correction lost accuracy while updating the grid");
    }
    return result;
}

// Диагностическая infinity-норма |C*u-d| по каноническим точкам. При наличии
// разломов C совпадает с fault-aware sampleSurface. Она считается
// одинаково в soft- и exact-режиме и не является residual основного solver.
Scalar maxControlError(const Grid& grid, const std::vector<CanonicalPoint>& points)
{
    Scalar result = 0;
    for (const CanonicalPoint& point : points) {
        result = std::max(result, std::abs(bilinearSample(grid, point.x, point.y) - point.value));
    }
    return result;
}

#include "gridding_pipeline_2.inc"

} // namespace

// Оркестратор полной схемы, приведенной в начале файла. Все промежуточные
// объекты локальны, поэтому исключение до последнего присваивания не меняет
// переданную пользователем Surface.
ConvergentGriddingReport convergentGridding(
    Surface& surface,
    const std::vector<Point>& points,
    const ConvergentGriddingOptions& options)
{
    return convergentGridding(surface, points, std::vector<Fault>{}, options);
}

ConvergentGriddingReport convergentGridding(
    Surface& surface, const std::vector<Point>& points,
    const std::vector<Fault>& faults, const ConvergentGriddingOptions& options,
    const AdditionalInputs& additional)
{
    // До любого чтения геометрии проверяем даже пустой вход. Оригинальный
    // Surface остаётся неизменным до последней пары swap.
    validateInput(surface, {}, options);
    validateDip(surface, additional);
    std::vector<Point> data = points;
    const auto lineData = detail::sampleInputPolylines(surface, additional.polylines,
                                                      options.ensureFixedSamplingInterval);
    data.insert(data.end(),lineData.begin(),lineData.end());
    const auto controls = validateInput(surface,data,options);
    const detail::FaultGeometry barriers(surface,faults);
    const auto* geometry = barriers.empty()?nullptr:&barriers;
    for(const auto& p:controls) {
        if(barriers.onFault(p.x,p.y))
            throw std::invalid_argument("control lies on a fault; specify one bank");
        if(!options.fillInsideFaultPolygons && detail::insideClosedFault(p.x,p.y,faults))
            throw std::invalid_argument("control lies in an excluded closed-fault region");
    }
    Grid input=surfaceAsGrid(surface);
    input.faults=geometry;
    std::vector<std::uint8_t> inputValid(input.values.size(),1);
    for(std::size_t n=0;n<input.values.size();++n) {
        inputValid[n]=(surface.valid.empty()||surface.valid[n])&&finite(input.values[n]);
        if(!inputValid[n])input.values[n]=0; // только численный буфер, не данные
    }

    const auto finalFaultRows=faultRows(input,faults,options);
    // Абсолютные Z разломов могут инициализировать тренд; перепады не могут:
    // jump задаёт разность, но не произвольную аддитивную константу поверхности.
    std::vector<Point> trendData;
    for(const auto& p:controls)trendData.push_back({qreal(p.x),qreal(p.y),qreal(p.value),qreal(p.weight)});
    if(options.faultZMode==FaultZMode::AbsoluteElevation)
        for(const auto& p:detail::makeFaultProbes(surface,faults,options)) {
            if(p.useLeft)trendData.push_back({p.left.x,p.left.y,qreal(p.target),qreal(p.weight)});
            if(p.useRight)trendData.push_back({p.right.x,p.right.y,qreal(p.target),qreal(p.weight)});
        }
    ConvergentGriddingReport report;
    std::vector<CanonicalPoint> boundData;
    for(const auto& p:trendData)boundData.push_back({p.x,p.y,p.value,p.weight});
    if(boundData.empty()) {
        Scalar lo=std::numeric_limits<Scalar>::infinity(),hi=-lo;
        for(std::size_t n=0;n<input.values.size();++n)if(inputValid[n]) {
            lo=std::min(lo,input.values[n]);hi=std::max(hi,input.values[n]);
        }
        if(finite(lo)){boundData.push_back({0,0,lo,1});boundData.push_back({0,0,hi,1});}
    }
    const auto bounds=heightBounds(boundData,options);
    const bool activeDip=options.dipMode!=DipMode::None && options.dipConstraintStrength>0
        && std::any_of(additional.dipAzimuth.begin(),additional.dipAzimuth.end(),
                       [](const DipAzimuthPoint& p){return p.weight>0;});
    if(trendData.empty()&&(!finalFaultRows.empty()||activeDip)
        &&std::any_of(inputValid.begin(),inputValid.end(),[](std::uint8_t v){return !v;}))
        throw std::invalid_argument("relative throw/dip data alone require a fully defined input height reference");
    if(controls.empty()&&finalFaultRows.empty()&&!activeDip) {
        // Нет новых ограничений: входная поверхность является единственным
        // источником. Не превращаем все-NoData в фиктивную нулевую карту.
        Grid result=input;
        std::vector<std::uint8_t> allowed(inputValid.size(),1),valid=inputValid;
        for(std::size_t j=0;j<input.ny;++j)for(std::size_t i=0;i<input.nx;++i) {
            const auto n=input.index(i,j);
            allowed[n]=(!geometry||!geometry->onFault(input.x(i),input.y(j)))
                &&(options.fillInsideFaultPolygons||!detail::insideClosedFault(input.x(i),input.y(j),faults));
            if(!allowed[n])valid[n]=0;
        }
        enforceConstraints(result,{},bounds,options);
        fillHoles(result,valid,allowed,options,report);
        postSmooth(result,{},bounds,options,report,&valid);
        std::vector<qreal> output(result.values.size());
        for(std::size_t n=0;n<output.size();++n) {
            output[n]=valid[n]?qreal(result.values[n]):(options.writeNoDataAsNaN
                ?std::numeric_limits<qreal>::quiet_NaN():qreal(0));
            if(!valid[n])++report.undefinedNodeCount;
            else if(!finite(output[n]))throw std::runtime_error("result does not fit qreal");
        }
        surface.grid.swap(output);surface.valid.swap(valid);return report;
    }

    auto working=options;
    if(working.initialCoarseningFactor==0) {
        // Эвристика авто-режима: примерно один coarse-узел на измерение,
        // коэффициент — степень двойки не больше характерного расстояния.
        const Scalar density=std::sqrt(Scalar(surface.nx-1)*Scalar(surface.ny-1)
                                      /Scalar(std::max<std::size_t>(1,trendData.size())));
        Scalar factor=1;
        while(factor*2<=density && factor*2<=Scalar(std::max(surface.nx-1,surface.ny-1)))
            factor*=2;
        working.initialCoarseningFactor=qreal(factor);
    }
    report.requestedCoarseningFactor=working.initialCoarseningFactor;
    auto hierarchy=buildHierarchy(surface.nx,surface.ny,working);
    if(geometry) {
        // Удаление непредставимого грубого префикса проверяет только те уровни,
        // на которых барьеры действительно включены. Расписание пересчитывается.
        for(;;) {
            std::size_t bad=hierarchy.size();
            for(std::size_t l=0;l<hierarchy.size();++l) {
                if(!faultsEnabled(l,hierarchy.size(),options.faultInfluencePercent))continue;
                Grid g=makeGridGeometry(surface,hierarchy[l].first,hierarchy[l].second);
                g.faults=geometry;
                PointConstraint row;
                bool ok=true;
                for(const auto& p:controls)if(!samplingConstraint(g,p.x,p.y,row)){ok=false;break;}
                if(ok && l>0) {
                    Grid prev=makeGridGeometry(surface,hierarchy[l-1].first,hierarchy[l-1].second);
                    prev.faults=geometry;
                    if(!supportsTransfer(prev,g)){bad=l-1;break;}
                }
                if(ok)try{(void)faultRows(g,faults,options);}
                    catch(const std::invalid_argument&){ok=false;}
                if(!ok){bad=l;break;}
            }
            if(bad==hierarchy.size())break;
            if(bad+1==hierarchy.size())
                throw std::invalid_argument("fault data cannot be represented on final grid; refine nx/ny");
            hierarchy.erase(hierarchy.begin(),hierarchy.begin()+static_cast<std::ptrdiff_t>(bad+1));
        }
    }

    const Scalar finalStep=std::max(input.dx(),input.dy());
    const Scalar decay=options.normalDecayCells*finalStep;
    const detail::TrendModel freeTrend(surface,trendData,nullptr,options.initialProjection,
                                       options.extrapolationMethod,decay);
    const detail::TrendModel faultTrend(surface,trendData,geometry,options.initialProjection,
                                        options.extrapolationMethod,decay);
    Grid solved;
    std::vector<std::uint8_t> valid;
    std::vector<Derivatives> carried;
    Scalar initialDx=0,initialDy=0;
    for(std::size_t l=0;l<hierarchy.size();++l) {
        const bool active=geometry&&faultsEnabled(l,hierarchy.size(),options.faultInfluencePercent);
        const auto& model=active?faultTrend:freeTrend;
        Grid prior=makeGridGeometry(surface,hierarchy[l].first,hierarchy[l].second);
        prior.faults=active?geometry:nullptr;
        std::vector<std::uint8_t> currentValid(prior.values.size(),0);
        if(l==0) {
            initialDx=prior.dx();initialDy=prior.dy();
            Grid source=input;source.faults=prior.faults;
            Grid validity=source;validity.values.assign(inputValid.begin(),inputValid.end());
            for(std::size_t j=0;j<prior.ny;++j)for(std::size_t i=0;i<prior.nx;++i) {
                const auto n=prior.index(i,j);
                const Scalar x=prior.x(i),y=prior.y(j);
                if(active&&geometry->onFault(x,y)) {
                    // Буфер не используется для бортов, но сохраняем датум
                    // input, чтобы фиктивный0 не влиял на численное центрирование.
                    prior.values[n]=rawBilinearSample(input,x,y);continue;
                }
                const auto t=model.evaluate(x,y);
                bool fromInput=false;
                if(options.useInputSurfaceAsPrior || trendData.empty()) {
                    PointConstraint row;
                    if(samplingConstraint(source,x,y,row)
                        &&constraintValue(row,validity.values)>1-1e-12) {
                        prior.values[n]=constraintValue(row,source.values);fromInput=true;
                    }
                }
                if(!fromInput)prior.values[n]=t.supported?t.value:0;
                currentValid[n]=fromInput||(t.supported
                    &&t.distanceToHull<=2*std::hypot(initialDx,initialDy));
            }
        } else {
            // При включении разломов оставляем ранее полученный региональный
            // тренд, но ВСЕ новые связи уже односторонние. Производные переносим
            // в координатах controls до prolongation (не из bilinear Hessian).
            Grid source=solved;source.faults=prior.faults;
            carried.clear();
            for(const auto& p:controls)carried.push_back(derivativesAt(source,p.x,p.y));
            Grid validity=source;validity.values.assign(valid.begin(),valid.end());
            Grid transferredMask=resample(validity,prior.nx,prior.ny);
            prior=resample(source,prior.nx,prior.ny);
            for(std::size_t n=0;n<currentValid.size();++n)currentValid[n]=transferredMask.values[n]>0.5;
        }
        const int levelOrder=l==0?static_cast<int>(options.initialProjection)
            :((prior.nx>=3&&prior.ny>=3)?2:1);
        if(l==0) {
            carried.clear();
            for(const auto& p:controls) {
                if(options.useInputSurfaceAsPrior)carried.push_back(derivativesAt(prior,p.x,p.y));
                else {
                    const auto t=model.evaluate(p.x,p.y);
                    carried.push_back({t.gx,t.gy,t.gxx,t.gxy,t.gyy});
                }
            }
        }
        const bool finalLevel=l+1==hierarchy.size();
        // initialSnapNodes действует ТОЛЬКО в первом проходе, включая случай
        // единственного уровня. Далее — наша автоматическая стратегия 1..16.
        const std::size_t snap=l==0?std::min(options.initialSnapNodes,prior.values.size())
            :(finalLevel?1:std::max<std::size_t>(1,std::min<std::size_t>(16,
                std::size_t(std::lround(16*std::sqrt(prior.dx()/initialDx*prior.dy()/initialDy))))));
        std::vector<Scalar> diagonal(prior.values.size(),0),rhs(prior.values.size(),0);
        addSnapConstraints(prior,controls,snap,working,carried,levelOrder,diagonal,rhs);
        if(options.extrapolationMethod==ExtrapolationMethod::Normal) {
            // Вдали от данных мягко удерживаем ограниченное продолжение тренда.
            // Внутри hull penalty равен нулю: это режим экстраполяции, не clamp Z.
            for(std::size_t j=0;j<prior.ny;++j)for(std::size_t i=0;i<prior.nx;++i) {
                const auto t=model.evaluate(prior.x(i),prior.y(j));
                if(!t.supported||t.distanceToHull<=0)continue;
                const Scalar fraction=t.distanceToHull/(t.distanceToHull+decay);
                const Scalar w=options.smoothness*fraction*fraction;
                const auto n=prior.index(i,j);diagonal[n]+=w;rhs[n]+=w*t.value;
            }
        }
        auto rows=active?faultRows(prior,faults,options):std::vector<PointConstraint>{};
        const auto dip=dipRows(prior,additional,finalStep,options);
        rows.insert(rows.end(),dip.begin(),dip.end());
        if(finalLevel) {
            const auto pointRows=makePointConstraints(prior,controls);
            rows.insert(rows.end(),pointRows.begin(),pointRows.end());
        }
        auto level=solveLevel(prior,prior.values,diagonal,rhs,rows,working);
        report.levels.push_back({prior.nx,prior.ny,snap,level.iterations,
            qreal(level.relativeResidual),level.converged,active,levelOrder});
        if(!level.converged&&options.throwOnNonConvergence)
            throw std::runtime_error("biharmonic PCG did not converge at grid level "
                                     +std::to_string(prior.nx)+"x"+std::to_string(prior.ny));
        solved=std::move(prior);solved.values=std::move(level.values);
        valid=std::move(currentValid);
        // Каждая явно заданная строка является источником, даже в блоке без
        // обычных controls (например, AbsoluteElevation на fault polygon).
        for(const auto& row:rows)for(std::size_t k=0;k<row.count;++k)
            if(row.terms[k].coefficient!=0)valid[row.terms[k].index]=1;
    }

    const auto pointRows=makePointConstraints(solved,controls);
    auto hardRows=finalFaultRows;
    if(options.enforceExactControls||options.allowResidual) {
        for(auto row:pointRows) {
            row.halfWidth=options.allowResidual?Scalar(options.residualTolerance):0;
            hardRows.push_back(row);
        }
    }
    report.faultConstraintCount=finalFaultRows.size();
    report.controlProjectionIterations=enforceConstraints(solved,hardRows,bounds,options);
    postSmooth(solved,hardRows,bounds,options,report);
    std::vector<std::uint8_t> allowed(valid.size(),1);
    for(std::size_t j=0;j<solved.ny;++j)for(std::size_t i=0;i<solved.nx;++i) {
        const auto n=solved.index(i,j);
        allowed[n]=(!geometry||!geometry->onFault(solved.x(i),solved.y(j)))
            &&(options.fillInsideFaultPolygons||!detail::insideClosedFault(solved.x(i),solved.y(j),faults));
        if(!allowed[n])valid[n]=0;
    }
    fillHoles(solved,valid,allowed,options,report);
    // Контроль после cast, но ДО NoData-маски: внутренние пробы бортов могут
    // лежать в намеренно скрытой FillInside=false области.
    for(auto& value:solved.values) {
        value=Scalar(qreal(value));
        if(!finite(value))throw std::runtime_error("convergent result does not fit qreal");
    }
    report.maxControlError=qreal(maxControlError(solved,controls));
    const Scalar tol=effectiveControlTolerance(hardRows,options);
    report.controlsSatisfied=report.maxControlError<=tol+(options.allowResidual?options.residualTolerance:0);
    if((options.enforceExactControls||options.allowResidual)&&!report.controlsSatisfied)
        throw std::runtime_error("qreal precision cannot retain point constraints");
    for(const auto& r:finalFaultRows)
        report.maxFaultConstraintError=qreal(std::max(Scalar(report.maxFaultConstraintError),
                                         std::abs(constraintValue(r,solved.values)-r.value)));
    if(report.maxFaultConstraintError>tol)
        throw std::runtime_error("qreal precision cannot retain fault constraints");
    if(bounds.enabled())for(Scalar v:solved.values)
        if(v<bounds.low-tol||v>bounds.high+tol)
            throw std::runtime_error("final output violates height bounds");
    std::vector<qreal> output(solved.values.size());
    for(std::size_t n=0;n<output.size();++n) {
        output[n]=valid[n]?qreal(solved.values[n]):(options.writeNoDataAsNaN
            ?std::numeric_limits<qreal>::quiet_NaN():qreal(0));
        if(!valid[n])++report.undefinedNodeCount;
    }
    surface.grid.swap(output);surface.valid.swap(valid);
    return report;
}

Surface convergentGriddedSurface(
    const Surface& surface,
    const std::vector<Point>& points,
    const ConvergentGriddingOptions& options,
    ConvergentGriddingReport* report)
{
    // Немутирующая перегрузка реализована через локальную копию и основной
    // оркестратор, поэтому численный путь обеих публичных функций идентичен.
    Surface result = surface;
    ConvergentGriddingReport localReport = convergentGridding(result, points, options);
    if (report != nullptr) *report = std::move(localReport);
    return result;
}

Surface convergentGriddedSurface(
    const Surface& surface,
    const std::vector<Point>& points,
    const std::vector<Fault>& faults,
    const ConvergentGriddingOptions& options,
    ConvergentGriddingReport* report,
    const AdditionalInputs& additional)
{
    Surface result = surface;
    auto localReport = convergentGridding(result, points, faults, options, additional);
    if (report) *report = std::move(localReport);
    return result;
}

qreal sampleSurface(const Surface& surface, qreal x, qreal y,
                    const std::vector<Fault>& faults)
{
    // Для одной выборки не копируем и не сканируем весь grid: после проверки
    // геометрии читаются только значения фактического support (до 4 узлов).
    if (surface.nx < 2 || surface.ny < 2
        || surface.grid.size() != checkedNodeCount(surface.nx, surface.ny)
        || (!surface.valid.empty() && surface.valid.size() != surface.grid.size()))
        throw std::invalid_argument("invalid surface dimensions for sampling");
    Grid grid;
    grid.nx = surface.nx; grid.ny = surface.ny;
    grid.minx = surface.minx; grid.maxx = surface.maxx;
    grid.miny = surface.miny; grid.maxy = surface.maxy;
    if (!finite(grid.minx) || !finite(grid.maxx) || !finite(grid.miny) || !finite(grid.maxy)
        || !(grid.maxx > grid.minx) || !(grid.maxy > grid.miny)
        || !finite(grid.dx()) || !finite(grid.dy()) || !(grid.dx() > 0) || !(grid.dy() > 0)
        || !finite(static_cast<Scalar>(x)) || !finite(static_cast<Scalar>(y))
        || x < grid.minx || x > grid.maxx || y < grid.miny || y > grid.maxy)
        throw std::invalid_argument("sampling coordinates or surface bounds are invalid");
    const detail::FaultGeometry barriers(surface, faults);
    grid.faults = barriers.empty() ? nullptr : &barriers;
    PointConstraint row;
    if (!samplingConstraint(grid, x, y, row))
        throw std::invalid_argument("cannot sample on a fault or in an unresolved fault block; refine nx/ny");
    Scalar value = 0;
    for (std::size_t k = 0; k < row.count; ++k) {
        if ((!surface.valid.empty() && !surface.valid[row.terms[k].index])
            || std::isnan(surface.grid[row.terms[k].index]))
            return std::numeric_limits<qreal>::quiet_NaN();
        const Scalar sample = static_cast<Scalar>(surface.grid[row.terms[k].index]);
        if (!finite(sample)) throw std::invalid_argument("sampling support contains a non-finite value");
        value += row.terms[k].coefficient * sample;
    }
    const qreal result = static_cast<qreal>(value);
    if (!finite(static_cast<Scalar>(result))) throw std::runtime_error("sample does not fit in qreal");
    return result;
}

} // namespace convergent2
