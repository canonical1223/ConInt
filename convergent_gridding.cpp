#include "convergent_gridding.h"
#include "convergent_gridding_detail.h"

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

namespace convergent {
namespace {

// Внутренние типы/решатель общие для двух единиц компиляции; остальные
// помощники конвейера сохраняют внутреннее связывание (internal linkage).
using namespace detail;

// ============================================================================
// ПОЛНАЯ СХЕМА РАБОТЫ (FULL ALGORITHM PIPELINE)
// ============================================================================
//
// Обозначения:
//   u       — вектор значений искомой поверхности во всех узлах уровня;
//   prior   — поверхность до коррекции на текущем уровне;
//   p       — контрольная точка (x_p, y_p, d_p, w_p);
//   C       — разреженный оператор выборки значений в точках: билинейный без
//             разломов, с перенормировкой видимых углов при наличии разломов;
//   B       — оператор дискретных вторых разностей;
//   K=smoothness*B^T*B — нормальный оператор (normal operator) энергии
//                        кривизны;
//
// Поток данных:
//
//   Surface.grid + Points + Options + необязательный vector<Fault>
//                 |
//                 v
//   1. Проверка и канонизация (validation and canonicalization)
//      - проверяются размеры, границы, конечность чисел и веса;
//      - точки с weight==0 удаляются;
//      - результат сортируется для воспроизводимого накопления;
//      - одинаковые x/y заменяются одной точкой со средневзвешенным value;
//      - при normalizePointWeights веса делятся на максимальный вес.
//      - Fault — последовательная XY-ломаная; точки на ней отклоняются.
//        Сегменты превращаются в геометрические барьеры видимости.
//                 |
//                 +---- нет активных точек ----> surface не изменяется
//                 |
//                 v
//   2. Иерархия coarse-to-fine (multiresolution hierarchy)
//      - от исходных nx-1,ny-1 интервалов строятся более грубые уровни;
//      - разрешение приблизительно удваивается при переходе к следующему;
//      - все уровни имеют одинаковые minx/maxx/miny/maxy;
//      - последний уровень всегда совпадает с исходными nx,ny.
//      - с разломами удаляется префикс слишком грубых уровней, не способных
//        представить блоки контролей или перенести prior на следующий уровень.
//                 |
//                 v
//   3. Начальный prior
//      - входная Surface.grid поточечно билинейно ресемплируется на самый
//        грубый уровень; отдельный тренд по контрольным точкам не строится.
//      - при наличии разломов перенос использует только видимые опорные узлы.
//                 |
//                 v
//   4. Цикл по уровням
//      4.1 Уточнение / продолжение (refinement / prolongation)
//          Решение предыдущего уровня билинейно переносится на текущую сетку.
//          На первом уровне prior уже равен огрубленной входной поверхности.
//
//      4.2 Производные prior
//          Конечными разностями оцениваются градиент (gradient) и матрица
//          Гессиана (Hessian), затем их значения билинейно выбираются в
//          координате каждой точки.
//          Возле разломов используются доступные односторонние разности.
//
//      4.3 Привязка / проекция Тейлора (Snap / Taylor projection)
//          Для точки выбираются ближайшие узлы. Ее значение переносится в узел
//          разложением Тейлора (Taylor expansion) порядка 0, 1 или 2. Влияние
//          распределяется гауссовым ядром (Gaussian kernel) в метрике ячеек.
//          Получаются мягкие квадратичные штрафы (soft quadratic penalties),
//          а не жестко зафиксированные значения узлов.
//          Связь point->node допустима только без пересечения разлома.
//
//      4.4 Мягкая привязка исходных точек на последнем уровне
//          Для каждой точки строится строка C_p с 1..4 весами выборки и
//          добавляется штраф (C_p*u-d_p)^2. На промежуточных уровнях его нет.
//          При наличии разломов используется тот же оператор sampleSurface,
//          включая взаимную видимость опорных узлов: C^T*C не соединяет берега.
//
//      4.5 Сглаживание / минимум кривизны (smooth / minimum curvature)
//          Минимизируется сумма энергии изгиба, отклонения от prior, Snap и
//          финальных точечных штрафов. Условие стационарности дает симметричную
//          положительно определенную систему (Symmetric Positive Definite,
//          SPD) A*u=b, решаемую матрично-свободным PCG (matrix-free PCG) с
//          предобуславливателем Якоби (Jacobi preconditioner).
//          Пересекающие разлом строки B удаляются целиком; одинаковая маска
//          применяется к B^T*B и его Jacobi-диагонали. Геометрия считается до PCG.
//
//      4.6 Полученное u становится prior следующего, более частого уровня.
//                 |
//                 v
//   5. Необязательная точная коррекция (exact-control projection)
//      Если enforceExactControls=true, после гладкого решения численно
//      выполняется C*u=d. Решается система размера m*m, где m — число
//      канонических точек, с матрицей Грама ограничений (constraint Gram
//      matrix) G=C*C^T; после этого к узлам добавляется C^T*lambda.
//                 |
//                 v
//   6. Выход и транзакционная фиксация результата
//      - значения преобразуются из внутреннего double в публичный qreal;
//      - повторно проверяются конечность и фактическая контрольная невязка;
//      - только после всех проверок surface.grid заменяется одним присваиванием.
//
// На каждом уровне фактически минимизируется
//
//   Phi(u) = 1/2 * [
//       smoothness * ||B*u||_2^2
//     + priorWeight * ||u-prior||_2^2
//     + snapStrength * sum_pn w_pn*(u_n-T_p(n))^2
//     + finalPointStrength * sum_p w_p*(C_p*u-d_p)^2 ],
//
// где последнее слагаемое присутствует только на финальном уровне.
//
// ВАЖНЫЕ ГРАНИЦЫ СООТВЕТСТВИЯ PETREL
// -----------------------------------
// Публичные источники описывают общую идею Convergent Gridding, но не все
// численные детали Petrel. Гауссово Snap-ядро, билинейный перенос уровней,
// повторное вычисление производных, дискретные свободные границы и евклидова
// exact-проекция — явно выбранные свойства ЭТОЙ реализации. Они не выдаются
// за побитовое воспроизведение закрытой реализации Petrel.

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

    if (options.initialSnapNodes == 0 || options.coarsestIntervals == 0
        || options.maxLevels == 0 || options.maxSolverIterations == 0) {
        throw std::invalid_argument("gridding counts must be greater than zero");
    }
    if (options.enforceExactControls
        && options.maxControlProjectionIterations == 0) {
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
    if (options.taylorOrder < 0 || options.taylorOrder > 2) {
        throw std::invalid_argument("taylorOrder must be 0, 1, or 2");
    }
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
    for (qreal value : surface.grid) {
        if (!finite(static_cast<Scalar>(value))) {
            throw std::invalid_argument("surface.grid must contain only finite values");
        }
    }

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
    // Каждая пара углов встречается в нескольких из 15 подмножеств.
    // Проверяем геометрию пары только один раз, сохраняя прежний порядок
    // перебора подмножеств и сложения весов (результат выборки не меняется).
    std::array<unsigned, 4> compatibleWith{};
    for (std::size_t a = 0; a < corners.count; ++a) {
        if (!(visible & (1u << a))) continue;
        for (std::size_t b = a + 1; b < corners.count; ++b) {
            if ((visible & (1u << b)) && visibleNodes(grid, corners.terms[a].index,
                                                        corners.terms[b].index))
                compatibleWith[a] |= 1u << b;
        }
    }
    for (unsigned mask = 1; mask < (1u << corners.count); ++mask) {
        if ((mask & visible) != mask) continue;
        bool compatible = true;
        Scalar weight = 0;
        for (std::size_t a = 0; a < corners.count && compatible; ++a) {
            if (!(mask & (1u << a))) continue;
            weight += corners.terms[a].coefficient;
            for (std::size_t b = a + 1; b < corners.count; ++b) {
                if ((mask & (1u << b)) && !(compatibleWith[a] & (1u << b))) {
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
// не хранятся и на новом уровне снова вычисляются из ресемплированного prior.
Grid resample(const Grid& source, std::size_t nx, std::size_t ny)
{
    Grid result = gridGeometry(source);
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
// уровня равен ceil((finalNx-1)/f)+1 на X и аналогично на Y. f последовательно
// принимает 1,2,4,...; затем список разворачивается в порядок coarse-to-fine.
//
// ceilDivide(...)+1 сохраняет общие физические границы для размеров, которые
// не имеют вида 2^k+1; поэтому узлы соседних уровней не обязаны точно совпадать,
// а последний переход не обязательно является точным удвоением. maxLevels —
// жесткий предел: coarsestIntervals может не быть достигнут. Исходный размер
// всегда добавляется последним. Например, для 512x512 и стандартных параметров
// получается 9,17,33,65,129,257,512 узлов по каждой оси.
std::vector<std::pair<std::size_t, std::size_t>> buildHierarchy(
    std::size_t finalNx,
    std::size_t finalNy,
    const ConvergentGriddingOptions& options)
{
    std::vector<std::size_t> factors{1};
    std::size_t factor = 1;
    while (factors.size() < options.maxLevels) {
        const std::size_t intervalsX = ceilDivide(finalNx - 1, factor);
        const std::size_t intervalsY = ceilDivide(finalNy - 1, factor);
        if (intervalsX <= options.coarsestIntervals
            && intervalsY <= options.coarsestIntervals) {
            break;
        }
        if (factor > std::numeric_limits<std::size_t>::max() / 2) {
            break;
        }
        factor *= 2;
        factors.push_back(factor);
    }

    std::vector<std::pair<std::size_t, std::size_t>> hierarchy;
    hierarchy.reserve(factors.size());
    for (auto it = factors.rbegin(); it != factors.rend(); ++it) {
        const std::size_t nx = ceilDivide(finalNx - 1, *it) + 1;
        const std::size_t ny = ceilDivide(finalNy - 1, *it) + 1;
        if (hierarchy.empty() || hierarchy.back() != std::make_pair(nx, ny)) {
            hierarchy.emplace_back(nx, ny);
        }
    }
    if (hierarchy.empty() || hierarchy.back() != std::make_pair(finalNx, finalNy)) {
        hierarchy.emplace_back(finalNx, finalNy);
    }
    return hierarchy;
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

// Тонкий блок между двумя разломами может отсутствовать на coarse grid.
// Обычная prolongation тогда молча переносила бы соседний блок через разлом.
// Мы удаляем неподходящий начальный префикс иерархии; геометрия разломов при
// этом никогда не огрубляется. В худшем случае остаётся только final grid.
std::vector<std::pair<std::size_t, std::size_t>> faultHierarchy(
    const Surface& surface, const Grid& input,
    const std::vector<CanonicalPoint>& controls,
    const ConvergentGriddingOptions& options)
{
    auto hierarchy = buildHierarchy(surface.nx, surface.ny, options);
    if (!input.faults) return hierarchy;
    std::size_t first = 0;
    for (std::size_t level = 0; level < hierarchy.size(); ++level) {
        Grid current = makeGridGeometry(surface, hierarchy[level].first, hierarchy[level].second);
        current.faults = input.faults;
        PointConstraint row;
        bool supported = true;
        for (const auto& point : controls) {
            if (!samplingConstraint(current, point.x, point.y, row)) { supported = false; break; }
        }
        if (supported && level + 1 < hierarchy.size()) {
            Grid next = makeGridGeometry(surface, hierarchy[level + 1].first,
                                         hierarchy[level + 1].second);
            next.faults = input.faults;
            supported = supportsTransfer(current, next);
        }
        if (supported) supported = supportsTransfer(input, current);
        if (!supported) first = level + 1;
    }
    if (first == hierarchy.size()) {
        throw std::invalid_argument(
            "a control point is in a fault block without local grid nodes; refine nx/ny");
    }
    hierarchy.erase(hierarchy.begin(), hierarchy.begin() + static_cast<std::ptrdiff_t>(first));
    return hierarchy;
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

    // Нужны только первые requested кандидатов. Частичная сортировка
    // (partial sort) сохраняет тот же упорядоченный префикс, но не сортирует
    // отброшенный хвост. Пара (distanceSquared, index) задает полный порядок.
    const std::size_t selected = std::min(requested, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + selected,
                     candidates.end(), [](const CandidateNode& a,
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
    candidates.resize(selected);
    return candidates;
}

// Расписание Snap (Snap schedule) локализует влияние точки при уточнении.
// Точная формула для нефинального уровня:
//   base = min(initialSnapNodes, numberOfNodesOnCurrentLevel),
//   N = max(1, round(base * sqrt((dx/dx0)*(dy/dy0)))),
// где dx0,dy0 — шаги самой грубой сетки. При обычном двукратном изотропном
// уточнении и initialSnapNodes=16 получается 16->8->4->2->1. Из-за повторного
// ограничения base на каждом уровне строгая монотонность не гарантирована для
// экстремально большого initialSnapNodes. На финальном уровне N всегда равно 1.
std::size_t snapNodeCount(
    const Grid& grid,
    const Grid& coarsest,
    bool finalLevel,
    const ConvergentGriddingOptions& options)
{
    if (finalLevel) return 1;
    const std::size_t cappedInitial = std::min(
        options.initialSnapNodes, grid.values.size());
    const long double scaleSquared =
        static_cast<long double>(grid.dx() / coarsest.dx())
        * static_cast<long double>(grid.dy() / coarsest.dy());
    const long double scaled = static_cast<long double>(cappedInitial)
        * std::sqrt(scaleSquared);
    if (!std::isfinite(scaled) || scaled < 0
        || scaled > static_cast<long double>(grid.values.size())) {
        throw std::invalid_argument("initialSnapNodes cannot be scaled safely");
    }
    return std::max<std::size_t>(1,
        static_cast<std::size_t>(std::floor(scaled + 0.5L)));
}

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
// options.taylorOrder оставляет соответственно только d_p, линейную часть или
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
    std::vector<Scalar>& diagonal,
    std::vector<Scalar>& rhs)
{
    std::vector<Scalar> accumulatedWeight(prior.values.size(), Scalar(0));
    std::vector<Scalar> accumulatedValue(prior.values.size(), Scalar(0));
    const Scalar sigma2 = static_cast<Scalar>(options.gaussianSigma)
        * static_cast<Scalar>(options.gaussianSigma);

    for (const CanonicalPoint& point : points) {
        const Derivatives d = derivativesAt(prior, point.x, point.y);
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
            if (options.taylorOrder >= 1) {
                projected += d.gx * deltaX + d.gy * deltaY;
            }
            if (options.taylorOrder >= 2) {
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

} // namespace

using namespace detail;

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
    Surface& surface,
    const std::vector<Point>& points,
    const std::vector<Fault>& faults,
    const ConvergentGriddingOptions& options)
{
    const std::vector<CanonicalPoint> controls = validateInput(surface, points, options);
    const detail::FaultGeometry barriers(surface, faults);
    for (const auto& point : controls) {
        if (barriers.onFault(point.x, point.y))
            throw std::invalid_argument("a control point lies on a fault; specify a point on one side");
    }
    ConvergentGriddingReport report;
    if (controls.empty()) {
        // Валидная поверхность без ненулевых контрольных точек не изменяется.
        return report;
    }

    // Самый грубый prior получается ресемплированием входной surface.grid.
    // Отдельный тренд по точкам здесь не строится. Иерархия содержит общие
    // min/max и заканчивается строго исходными surface.nx/surface.ny.
    Grid input = surfaceAsGrid(surface);
    input.faults = barriers.empty() ? nullptr : &barriers;
    const auto hierarchy = faultHierarchy(surface, input, controls, options);
    Grid coarsest = input.faults && hierarchy.front() == std::make_pair(surface.nx, surface.ny)
        ? input : resample(input, hierarchy.front().first, hierarchy.front().second);
    Grid solved = std::move(coarsest);
    // Для правила Snap нужны только шаги самого грубого уровня.
    coarsest = gridGeometry(solved);

    for (std::size_t levelIndex = 0; levelIndex < hierarchy.size(); ++levelIndex) {
        const auto [nx, ny] = hierarchy[levelIndex];
        // Уточнение (Refine/prolongation): первый prior уже имеет нужный грубый
        // размер; далее решение предыдущего уровня билинейно переносится на
        // более частую сетку.
        Grid prior = levelIndex == 0 ? std::move(solved) : resample(solved, nx, ny);
        const bool finalLevel = levelIndex + 1 == hierarchy.size();
        const std::size_t snapNodes = snapNodeCount(prior, coarsest, finalLevel, options);

        // Привязка Snap: контрольные значения проецируются по Тейлору (Taylor
        // projection) в ближайшие узлы и превращаются в диагональные мягкие
        // штрафы текущего уровня.
        std::vector<Scalar> snapDiagonal(prior.values.size(), Scalar(0));
        std::vector<Scalar> snapRhs(prior.values.size(), Scalar(0));
        addSnapConstraints(prior, controls, snapNodes, options, snapDiagonal, snapRhs);

        // На финальном уровне добавляется мягкая билинейная привязка C_p*u=d_p.
        // Она уменьшает величину последующей exact-поправки, но не заменяет ее.
        const std::vector<PointConstraint> finalConstraints = finalLevel
            ? makePointConstraints(prior, controls)
            : std::vector<PointConstraint>{};

        // Сглаживание (Smooth): PCG балансирует Hessian energy, сохранение prior,
        // Snap и, только на последнем уровне, билинейные ограничения. Это
        // регуляризованная система бигармонического типа, а не решение одного
        // чистого уравнения Delta^2*u=0. При разрешенной несходимости последняя
        // итерация (last iterate) все равно передается на следующий этап.
        SolverResult level = solveLevel(prior, prior.values, snapDiagonal, snapRhs,
                                        finalConstraints, options);
        report.levels.push_back({nx, ny, snapNodes, level.iterations,
                                 static_cast<qreal>(level.relativeResidual), level.converged});
        if (!level.converged && options.throwOnNonConvergence) {
            // Термин biharmonic указывает на доминирующий оператор K, хотя A
            // также содержит prior и точечные penalties. Размер в сообщении —
            // текущий промежуточный level, например 33x33, а не итоговый grid.
            throw std::runtime_error("biharmonic PCG did not converge at grid level "
                + std::to_string(nx) + "x" + std::to_string(ny));
        }
        solved = std::move(prior);
        solved.values = std::move(level.values);
    }

    // Мягкое решение (soft solution) обычно лишь приближенно выполняет C*u=d.
    // Необязательная exact-control projection доводит билинейные значения до
    // effective tolerance отдельным решением в пространстве точек.
    const std::vector<PointConstraint> outputConstraints =
        makePointConstraints(solved, controls);
    if (options.enforceExactControls) {
        const ProjectionResult projection = projectOntoPointConstraints(
            solved, outputConstraints, options);
        report.controlProjectionIterations = projection.iterations;
    }

    if (input.faults) {
        // Изолированные узлы на геометрическом разломе не определяют ни один
        // берег. Для стабильного хранения оставляем именно исходное значение;
        // fault-aware sampler никогда не использует его как опорное.
        for (std::size_t j = 0; j < solved.ny; ++j)
            for (std::size_t i = 0; i < solved.nx; ++i)
                if (input.faults->onFault(solved.x(i), solved.y(j)))
                    solved.values[solved.index(i, j)] = input.values[input.index(i, j)];
    }

    // Публичный qreal может иметь меньшую разрядность, чем внутренний double,
    // поэтому преобразование выполняется во временный буфер и проверяется до
    // изменения surface.grid.
    std::vector<qreal> output(solved.values.size());
    std::transform(solved.values.begin(), solved.values.end(), output.begin(),
        [](Scalar value) {
            if (!finite(value)) {
                throw std::runtime_error("convergent gridding produced a non-finite result");
            }
            const qreal converted = static_cast<qreal>(value);
            if (!finite(static_cast<Scalar>(converted))) {
                throw std::runtime_error("convergent gridding result does not fit in qreal");
            }
            return converted;
        });

    // Отчет измеряет невязку именно возвращаемых значений, включая возможную
    // потерю точности, когда qreal в Qt-сборке имеет меньшую разрядность, чем
    // внутренний Scalar.
    std::transform(output.begin(), output.end(), solved.values.begin(),
        [](qreal value) { return static_cast<Scalar>(value); });
    report.maxControlError = static_cast<qreal>(maxControlError(solved, controls));
    const Scalar acceptedControlError = effectiveControlTolerance(
        outputConstraints, options);
    report.controlsSatisfied = static_cast<Scalar>(report.maxControlError)
        <= acceptedControlError;
    // В soft-режиме false — допустимый диагностический результат. В exact-
    // режиме это означает, что cast в qreal разрушил уже выполненную коррекцию.
    if (options.enforceExactControls && !report.controlsSatisfied) {
        throw std::runtime_error(
            "qreal precision is insufficient to retain the exact control-point correction");
    }

    // Единственная запись в объект пользователя: все вычисления, exact-проекция,
    // преобразование и проверки завершены. Это строгая гарантия исключений
    // (strong exception guarantee), но не atomic/thread-safe операция.
    surface.grid = std::move(output);
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
    ConvergentGriddingReport* report)
{
    Surface result = surface;
    auto localReport = convergentGridding(result, points, faults, options);
    if (report) *report = std::move(localReport);
    return result;
}

qreal sampleSurface(const Surface& surface, qreal x, qreal y,
                    const std::vector<Fault>& faults)
{
    // Для одной выборки не копируем и не сканируем весь grid: после проверки
    // геометрии читаются только значения фактического support (до 4 узлов).
    if (surface.nx < 2 || surface.ny < 2
        || surface.grid.size() != checkedNodeCount(surface.nx, surface.ny))
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
        const Scalar sample = static_cast<Scalar>(surface.grid[row.terms[k].index]);
        if (!finite(sample)) throw std::invalid_argument("sampling support contains a non-finite value");
        value += row.terms[k].coefficient * sample;
    }
    const qreal result = static_cast<qreal>(value);
    if (!finite(static_cast<Scalar>(result))) throw std::runtime_error("sample does not fit in qreal");
    return result;
}

} // namespace convergent
