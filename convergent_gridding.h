#pragma once

#include <cstddef>
#include <vector>

// Тип qreal должен выбираться одинаково для библиотеки и всех ее потребителей.
// CMake задает standalone-макрос как PUBLIC, а Qt/qmake-сборка использует qreal
// из Qt. Это предотвращает несовместимость ABI между библиотекой и приложением.
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

// Регулярная поверхность. min/max — координаты крайних УЗЛОВ, поэтому число
// интервалов равно nx - 1 и ny - 1.
//
// Порядок grid: index = iy * nx + ix. Сначала хранится нижняя строка от
// minx к maxx, затем следующая строка выше, то есть слева направо и снизу вверх.
struct Surface {
    std::vector<qreal> grid;
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
    // Нулевой вес исключает точку. Положительный вес влияет на Snap, мягкую
    // финальную привязку и среднее совпадающих точек. В exact-режиме каждая
    // оставшаяся уникальная координата является обязательным ограничением.
    qreal weight{qreal(1)};
};

struct ConvergentGriddingOptions {
    // Иерархия coarse-to-fine. На первом уровне число Snap-узлов не превышает
    // initialSnapNodes. coarsestIntervals — целевой верхний предел числа
    // интервалов по каждой оси на самом грубом уровне, а maxLevels — жесткое
    // ограничение количества уровней; при малом maxLevels предел может быть
    // не достигнут. Последний уровень всегда имеет исходные nx и ny.
    std::size_t initialSnapNodes{16};
    std::size_t coarsestIntervals{8};
    std::size_t maxLevels{8};

    // Относительные веса функционала, решаемого на каждом уровне:
    //   smoothness * ||curvature(u)||^2
    // + priorWeight * ||u - prior||^2
    // + snapStrength * sum w_pn * (u_n - Taylor_p(n))^2
    // + finalPointStrength * sum w_p * (C_p*u - value_p)^2.
    // Здесь prior на первом уровне — ресемплированная входная surface.grid,
    // далее — интерполированное решение предыдущего уровня; C_p — билинейная
    // выборка поверхности в контрольной точке. Последнее слагаемое добавляется
    // только на финальном уровне. Веса относятся к дискретизации данной
    // реализации и не являются параметрами Petrel 1:1.
    qreal smoothness{qreal(1)};
    qreal priorWeight{qreal(1e-3)};
    qreal snapStrength{qreal(100)};
    // Мягкая финальная привязка делает последующую exact-коррекцию меньше, но
    // сама по себе не гарантирует прохождение поверхности через точки.
    qreal finalPointStrength{qreal(1000)};

    // После гладкого решения потребовать точное билинейное значение во всех
    // канонических контрольных координатах. Коррекция минимизирует евклидову
    // норму изменения узлов, а не повторно минимизирует кривизну поверхности.
    // Точность ограничена controlTolerance и машинной точностью выходного qreal.
    // Зависимые или несовместимые уравнения могут привести к исключению.
    bool enforceExactControls{true};
    std::size_t maxControlProjectionIterations{2000};
    qreal controlTolerance{qreal(1e-10)};

    // Сигма Gaussian-ядра Snap в координатах ячеек. Это расстояние по индексам
    // сетки, а не физическое расстояние в единицах x/y.
    qreal gaussianSigma{qreal(1)};

    // Порядок Taylor-проекции точки в соседний узел:
    // 0 — значение; 1 — значение и наклон; 2 — значение, наклон и кривизна.
    int taylorOrder{2};
    // После объединения совпадающих точек разделить все веса на максимальный.
    // Соотношения весов сохраняются, общий масштаб перестает влиять на решение.
    bool normalizePointWeights{true};

    // Параметры matrix-free PCG с диагональным (Jacobi) предобуславливателем.
    // Критерий: ||r|| <= absoluteTolerance
    //                    + relativeTolerance * max(||b||, ||A*x0||).
    // Лимит применяется отдельно к каждому уровню. Если
    // throwOnNonConvergence == false, последний iterate принимается, а
    // несходимость фиксируется в отчете.
    std::size_t maxSolverIterations{1500};
    qreal relativeTolerance{qreal(1e-9)};
    qreal absoluteTolerance{qreal(0)};
    bool throwOnNonConvergence{true};
};

struct ConvergentGriddingLevelReport {
    std::size_t nx{};
    std::size_t ny{};
    // Число ближайших Snap-узлов на одну точку на этом уровне.
    std::size_t snapNodes{};
    std::size_t solverIterations{};
    // ||r|| / max(||b||, ||A*x0||) для возвращенного iterate.
    qreal relativeResidual{};
    bool converged{};
};

struct ConvergentGriddingReport {
    std::vector<ConvergentGriddingLevelReport> levels;
    // Максимальная абсолютная билинейная невязка в канонических точках.
    // Совпадающие точки предварительно заменяются их взвешенным средним.
    qreal maxControlError{};
    // Число итераций PCG в пространстве контрольных точек при exact-коррекции.
    std::size_t controlProjectionIterations{};
    // В soft-режиме это только диагностика относительно exact-допуска, а не
    // признак сходимости основного PCG.
    bool controlsSatisfied{true};
};

// Полный алгоритм:
// 1. проверить и канонизировать точки;
// 2. построить иерархию сеток от грубой к исходной;
// 3. на каждом уровне перенести предыдущую поверхность, сформировать
//    Snap/Taylor-ограничения и решить регуляризованную minimum-curvature задачу;
// 4. при enforceExactControls выполнить точную билинейную коррекцию;
// 5. проверить результат после преобразования в qreal и только затем заменить
//    surface.grid.
//
// Обеспечивается strong exception guarantee: при ошибке surface не изменяется.
// Точки с weight == 0 игнорируются; отрицательные веса и точки вне замкнутых
// границ поверхности отклоняются.
ConvergentGriddingReport convergentGridding(
    Surface& surface,
    const std::vector<Point>& points,
    const ConvergentGriddingOptions& options = {});

// Удобная немутирующая перегрузка: возвращает измененную копию поверхности.
Surface convergentGriddedSurface(
    const Surface& surface,
    const std::vector<Point>& points,
    const ConvergentGriddingOptions& options = {},
    ConvergentGriddingReport* report = nullptr);

} // namespace convergent
