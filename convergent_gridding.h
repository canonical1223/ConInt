#pragma once

#include <cstddef>
#include <vector>

// Тип qreal должен выбираться одинаково для библиотеки и всех ее потребителей.
// CMake задает standalone-макрос как PUBLIC, а Qt/qmake-сборка использует qreal
// из Qt. Это предотвращает несовместимость двоичного интерфейса приложения
// (Application Binary Interface, ABI) между библиотекой и вызывающим кодом.
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

// Регулярная прямоугольная поверхность (regular Cartesian grid). min/max —
// координаты крайних УЗЛОВ, поэтому число интервалов равно nx-1 и ny-1, а
// dx=(maxx-minx)/(nx-1), dy=(maxy-miny)/(ny-1). Требуется nx,ny>=2 и
// grid.size()==nx*ny.
//
// Порядок хранения по строкам (row-major layout):
//   index = iy*nx + ix,
//   ix=0..nx-1 растет от minx к maxx,
//   iy=0..ny-1 растет от miny к maxy.
// Первые nx значений — нижняя строка y=miny слева направо; затем идут строки
// выше. Алгоритм не транспонирует сетку и не переворачивает ось Y.
struct Surface {
    std::vector<qreal> grid;
    qreal minx{};
    qreal maxx{};
    qreal miny{};
    qreal maxy{};
    std::size_t nx{};
    std::size_t ny{};
};

// Контрольная точка (control point), задающая измеренное значение поверхности.
// Точки должны находиться внутри замкнутых границ Surface.
struct Point {
    qreal x{};
    qreal y{};
    qreal value{};
    // Нулевой вес исключает точку, отрицательный запрещен. Положительный вес
    // задает относительную силу мягких штрафов (soft penalties) и участвует во
    // взвешенном среднем совпадающих x/y. После такого объединения независимые
    // равенства exact-режима не масштабируются весом: каждое обязательно с
    // одинаковым численным допуском.
    qreal weight{qreal(1)};
};

// Разлом — последовательная ломаная (fault polyline): [p0,p1], [p1,p2], ... .
// Используются только x/y; value и weight вершин НЕ задают ни высоту, ни сброс.
// Замыкание задается явно повторением первой вершины. Допустимы концы за
// пределами Surface, пересечения разломов и повторные соседние вершины.
// После удаления повторов необходимо хотя бы две различные вершины.
using Fault = std::vector<Point>;

struct ConvergentGriddingOptions {
    // Многоуровневая иерархия от грубой сетки к частой (coarse-to-fine
    // multiresolution hierarchy). Размеры строятся обратным проходом от
    // исходного числа интервалов с коэффициентами 1,2,4,... .
    //
    // initialSnapNodes — номинальное число ближайших Snap-узлов на грубом
    // уровне. На каждом уровне оно ограничивается числом доступных узлов и
    // масштабируется относительно шага самой грубой сетки; обычно получается
    // 16->8->4->2->1, но для экстремальных размеров строгая монотонность не
    // гарантируется. На финальном уровне принудительно используется один узел.
    std::size_t initialSnapNodes{16};
    // Желаемый верхний предел интервалов по КАЖДОЙ оси самого грубого уровня.
    // Он может быть не достигнут, если раньше сработает maxLevels.
    std::size_t coarsestIntervals{8};
    // Жесткий предел числа уровней, включая исходное разрешение.
    std::size_t maxLevels{8};

    // На каждом уровне минимизируется квадратичный функционал
    //
    //   Phi(u) = 1/2 * [
    //       smoothness * ||B*u||_2^2
    //     + priorWeight * ||u-prior||_2^2
    //     + snapStrength * sum_pn w_pn*(u_n-T_p(n))^2
    //     + finalPointStrength * sum_p w_p*(C_p*u-value_p)^2 ].
    //
    // B — дискретные вторые разности, T_p(n) — Taylor-прогноз точки p в узле
    // n, C_p — строка билинейной выборки. Последняя сумма используется только
    // на финальном уровне.
    // Здесь prior на первом уровне — ресемплированная входная surface.grid,
    // далее — интерполированное решение предыдущего уровня; C_p — билинейная
    // выборка поверхности в контрольной точке. Все коэффициенты относительны,
    // зависят от дискретизации и не являются параметрами Petrel 1:1.
    // Вес энергии изгиба тонкой пластины (thin-plate bending energy).
    qreal smoothness{qreal(1)};
    // Регуляризация Тихонова нулевого порядка (zero-order Tikhonov
    // regularization). priorWeight>0 также делает систему строго SPD.
    qreal priorWeight{qreal(1e-3)};
    // Сила мягких Snap/Taylor-штрафов. Конечное значение не фиксирует узел
    // точно на Taylor target.
    qreal snapStrength{qreal(100)};
    // Сила мягкой билинейной привязки только на финальном уровне. Она уменьшает
    // последующую exact-коррекцию, но сама не задает окончательную точность.
    qreal finalPointStrength{qreal(1000)};

    // После гладкого решения численно выполнить C*u=d во всех канонических
    // контрольных координатах. Используется евклидова ортогональная проекция
    // (Euclidean orthogonal projection) минимальной узловой L2-поправки, а не
    // повторная минимизация энергии кривизны.
    bool enforceExactControls{true};
    // Лимит отдельного PCG для матрицы Грама (constraint Gram matrix) C*C^T.
    std::size_t maxControlProjectionIterations{2000};
    // Абсолютный допуск. Фактический порог равен максимуму этого значения и
    // 64*epsilon(qreal)*max(1,max_p|value_p|), чтобы учесть выходной тип qreal.
    // Зависимые ограничения могут вызвать breakdown; несовместимые не могут
    // пройти финальную проверку невязки.
    qreal controlTolerance{qreal(1e-10)};

    // Сигма гауссова ядра (Gaussian kernel) Snap в координатах ЯЧЕЕК текущего
    // уровня. Это расстояние по индексам, а не в физических единицах x/y.
    qreal gaussianSigma{qreal(1)};

    // Порядок разложения Тейлора (Taylor expansion), которым значение точки
    // переносится в соседний узел: 0 — value; 1 — value+gradient;
    // 2 — value+gradient+Hessian (наклон и кривизна prior).
    int taylorOrder{2};
    // После объединения совпадающих точек разделить все веса на максимальный.
    // Отношения весов сохраняются, а общий множитель перестает менять баланс
    // мягких штрафов относительно smoothness и priorWeight.
    bool normalizePointWeights{true};

    // Предобусловленный метод сопряженных градиентов (Preconditioned Conjugate
    // Gradient, PCG). Полная разреженная A не хранится (matrix-free), но явно
    // вычисляется ее диагональ M=diag(A) для предобуславливателя Якоби
    // (Jacobi preconditioner): z=M^{-1}r.
    //
    // Критерий в евклидовой 2-норме:
    //   ||r||_2 <= absoluteTolerance
    //             + relativeTolerance*max(||b||_2,||A*x0||_2).
    // Лимит применяется отдельно к каждому уровню. Если
    // throwOnNonConvergence==false, последняя итерация (last iterate)
    // принимается, а несходимость фиксируется в отчете.
    std::size_t maxSolverIterations{1500};
    qreal relativeTolerance{qreal(1e-9)};
    qreal absoluteTolerance{qreal(0)};
    bool throwOnNonConvergence{true};
};

struct ConvergentGriddingLevelReport {
    // Размер именно узлов текущего промежуточного уровня. Поэтому сообщение о
    // несходимости на 33x33 не означает, что итоговая Surface имеет размер 33x33.
    std::size_t nx{};
    std::size_t ny{};
    // Номинальное число Snap-узлов на одну точку. Разломы могут уменьшить
    // фактическое число доступных видимых узлов для отдельных точек.
    std::size_t snapNodes{};
    // Выполненные итерации основного level PCG; может быть 0, если x0 уже
    // удовлетворяет допуску.
    std::size_t solverIterations{};
    // ||r||_2/max(||b||_2,||A*x0||_2) для возвращенной итерации. Между
    // периодическими пересчетами b-A*x используется рекуррентная невязка PCG,
    // поэтому это не обязательно повторно вычисленная true residual.
    qreal relativeResidual{};
    // Относится только к основной системе уровня, не к exact-проекции.
    bool converged{};
};

struct ConvergentGriddingReport {
    std::vector<ConvergentGriddingLevelReport> levels;
    // Максимальная абсолютная невязка оператора sampleSurface (infinity norm
    // residual): билинейного без разломов, одностороннего с разломами.
    // в канонических точках уже после преобразования результата в qreal.
    // Совпадающие точки предварительно заменяются их взвешенным средним.
    qreal maxControlError{};
    // Число итераций Gram-PCG в пространстве контрольных точек. Равно 0, если
    // exact-режим выключен либо начальная невязка уже находится в допуске.
    std::size_t controlProjectionIterations{};
    // В soft-режиме это только диагностика относительно exact-допуска, а не
    // признак сходимости основного PCG.
    bool controlsSatisfied{true};
};

// Краткая схема (полная схема и теория приведены в convergent_gridding.cpp):
// validation -> canonicalization -> coarse-to-fine hierarchy ->
// [bilinear prolongation -> Snap/Taylor -> minimum-curvature PCG] ->
// optional exact-control projection -> qreal verification -> commit.
//
// Строгая гарантия исключений (strong exception guarantee): при исключении
// surface не изменяется. Единственная запись выполняется после всех уровней,
// exact-проекции, преобразования в qreal и финальных проверок.
// Точки с weight == 0 игнорируются; отрицательные веса и точки вне замкнутых
// границ поверхности отклоняются.
ConvergentGriddingReport convergentGridding(
    Surface& surface,
    const std::vector<Point>& points,
    const ConvergentGriddingOptions& options = {});

// Разломы служат непроницаемыми барьерами (barrier faults) для локальных связей:
// разностные шаблоны, Snap, перенос prior и выборка точек не пересекают ломаные.
// Влияние может обойти свободный конец разлома. Сброс возникает из данных,
// задающих разные значения по сторонам; сама геометрия сброс не назначает.
// Слишком грубые уровни без узлов для представления отдельных блоков пропускаются.
// Если блок точки не представлен даже в конечной сетке, требуется увеличить nx/ny.
// Точки непосредственно на разломе отклоняются: сторона не определена.
// Узел на разломе сохраняет исходное значение и исключается из связей; grid не
// хранит два предельных значения. Для выборки/отрисовки используйте sampleSurface
// с теми же faults и не соединяйте треугольниками противоположные стороны.
// Четвертый аргумент обязателен, чтобы старый вызов (..., {}) был однозначен.
ConvergentGriddingReport convergentGridding(
    Surface& surface,
    const std::vector<Point>& points,
    const std::vector<Fault>& faults,
    const ConvergentGriddingOptions& options);

// Удобная немутирующая перегрузка: возвращает измененную копию поверхности.
Surface convergentGriddedSurface(
    const Surface& surface,
    const std::vector<Point>& points,
    const ConvergentGriddingOptions& options = {},
    ConvergentGriddingReport* report = nullptr);

Surface convergentGriddedSurface(
    const Surface& surface,
    const std::vector<Point>& points,
    const std::vector<Fault>& faults,
    const ConvergentGriddingOptions& options,
    ConvergentGriddingReport* report = nullptr);

// Значение в физической координате; без разломов — обычная билинейная выборка.
// Возле разлома веса видимых углов перенормируются (visibility-normalized
// bilinear interpolation); при отсутствии углов используется ближайший видимый
// узел в локальном радиусе двух ячеек. Этот же оператор C используется в soft-
// и exact-привязке и в maxControlError. Выборка на разломе вызывает исключение.
// Faults не сохраняются в Surface: их необходимо передавать вместе с grid.
qreal sampleSurface(
    const Surface& surface, qreal x, qreal y,
    const std::vector<Fault>& faults = {});

} // namespace convergent
