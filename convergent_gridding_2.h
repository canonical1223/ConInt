#pragma once

#include <cstddef>
#include <vector>
#include <cstdint>
#include <optional>

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

namespace convergent2 {

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
    // Пусто = все конечные значения валидны; 0 = NoData. Порядок как у grid.
    // На выходе маска всегда заполнена. NaN разрешён в невалидном входе.
    std::vector<std::uint8_t> valid;
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
// Замыкание явно повтором первой вершины. В v2 value — по умолчанию перепад
// бортов (Throw), либо абсолютная отметка (AbsoluteElevation), либо не
// используется (Ignore). weight>=0 — вес мягкого ограничения, 0 его исключает.
// Контракт знака задаёт ThrowSign. Пересекающиеся ограничения могут быть
// несовместимы; неоднозначность локальной пробы вызывает исключение.
using Fault = std::vector<Point>;

using Polyline = std::vector<Point>;
enum class ProjectionOrder { Horizontal = 0, Linear = 1, Quadratic = 2 };
enum class ExtrapolationMethod { Normal, Trend };
enum class FaultZMode { Ignore, Throw, AbsoluteElevation };
// Это явная конвенция API, не универсальный стандарт Petrel/геологии.
// Left/Right определяются направлением p[i]->p[i+1]. При развороте линии
// signed throw нужно инвертировать, чтобы сохранить физические стороны.
enum class ThrowSign { LeftMinusRight, RightMinusLeft };
enum class FaultBank { Both, Left, Right };
enum class DipMode { None, AzimuthOnly, DipAndAzimuth };
struct DipAzimuthPoint {
    qreal x{}, y{}, dipDegrees{}, azimuthDegrees{};
    qreal weight{1};
    // Азимут ПАДЕНИЯ: по часовой стрелке от +Y; Z растёт вверх.
    // gx=-tan(dip)*sin(azimuth), gy=-tan(dip)*cos(azimuth).
};
struct AdditionalInputs {
    std::vector<Polyline> polylines; // обычные линии с абсолютными отметками Z
    std::vector<DipAzimuthPoint> dipAzimuth;
};

struct ConvergentGriddingOptions {
    // Настройки v2. 0 = автоматический coarsening по плотности (эвристика v2),
    // >=1 = отношение первого шага к конечному. На следующих проходах /2.
    qreal initialCoarseningFactor{0};
    ProjectionOrder initialProjection{ProjectionOrder::Linear};
    ExtrapolationMethod extrapolationMethod{ExtrapolationMethod::Normal};
    qreal normalDecayCells{4};
    bool useInputSurfaceAsPrior{false};

    // Допуск по ДАННЫМ max|C*u-d| в единицах Z, не допуск PCG.
    bool allowResidual{false};
    qreal residualTolerance{1};
    // Дополнительное сглаживание только на конечном разрешении. Это не
    // основной Smooth внутри convergent-проходов и не число шагов PCG.
    std::size_t smoothingIterations{0};
    bool fixedSmoothing{false};
    qreal smoothingChangeTolerance{1e-6};

    qreal faultInfluencePercent{100}; // доля ПОСЛЕДНИХ gridding passes
    FaultZMode faultZMode{FaultZMode::Throw};
    ThrowSign throwSign{ThrowSign::LeftMinusRight};
    FaultBank absoluteElevationBank{FaultBank::Both};
    qreal faultConstraintStrength{1000};
    bool fillInsideFaultPolygons{true};
    // Две односторонние пробы на offset*min(hx,hy) от линии задают перепад.
    // Это разрешение разрыва сеткой, не геологическая ширина разлома.
    qreal faultBankOffsetCells{0.25};

    bool ensureAllNodesGetValue{true};
    bool ensureFixedSamplingInterval{false};
    bool writeNoDataAsNaN{true};
    DipMode dipMode{DipMode::None};
    bool useDipInfluenceRadius{false}; // false = бесконечный радиус
    qreal dipInfluenceRadius{0}; // 0 при включении = 2*max(final hx,hy)
    qreal dipConstraintStrength{100};

    std::optional<qreal> minimumValue, maximumValue;
    // Явная конвенция v2: lower=min(data)+p_min/100*range(data),
    // upper=max(data)+p_max/100*range(data). -10/+10 расширяют диапазон.
    // Формула процентов в предоставленных подсказках Petrel не раскрыта.
    bool minimumInPercentOfDataRange{false};
    bool maximumInPercentOfDataRange{false};
    // Первый проход: 1..16 ближайших узлов НА ТОЧКУ. Далее собственная
    // автоматическая стратегия 1..16; точное правило Petrel неизвестно.
    std::size_t initialSnapNodes{16};
    // Верхний предел числа уровней. Недостаточный лимит вызывает исключение.
    std::size_t maxLevels{32};

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
    // В v2 начальный prior строится по данным (локальный polynomial QR fit),
    // либо берётся из grid при useInputSurfaceAsPrior. Далее — prolongation.
    // Добавлены отдельные soft rows перепадов и dip/azimuth, Normal-anchors.
    // Все коэффициенты — инженерные численные настройки, не Petrel 1:1.
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

    // После smooth численно выполнить ограничения по данным. При allowResidual
    // используются интервалы даже если enforceExactControls=false.
    // Gram-PCG для равенств; Dykstra для совместных интервалов/bounds/jumps
    // либо зависимых строк. Это Euclidean projection, не energy minimization.
    bool enforceExactControls{true};
    // Лимит Gram-PCG и/или циклов Dykstra; несовместимость даёт исключение.
    std::size_t maxControlProjectionIterations{4000};
    // Абсолютный допуск. Фактический порог равен максимуму этого значения и
    // 64*epsilon(qreal)*max(1,max_p|value_p|), чтобы учесть выходной тип qreal.
    // Зависимые ограничения могут вызвать breakdown; несовместимые не могут
    // пройти финальную проверку невязки.
    qreal controlTolerance{qreal(1e-9)};

    // Сигма гауссова ядра (Gaussian kernel) Snap в координатах ЯЧЕЕК текущего
    // уровня. Это расстояние по индексам, а не в физических единицах x/y.
    qreal gaussianSigma{qreal(1)};

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
    // После центрирования вертикального датума (среднее prior вычитается
    // из неизвестных и абсолютных targets) применяется критерий:
    //   ||r||_2 <= absoluteTolerance
    //             + relativeTolerance*max(||b||_2,||A*x0||_2).
    // Лимит применяется отдельно к каждому уровню. Если
    // throwOnNonConvergence==false, последняя итерация (last iterate)
    // принимается, а несходимость фиксируется в отчете.
    std::size_t maxSolverIterations{5000};
    qreal relativeTolerance{qreal(1e-7)};
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
    // периодическими пересчетами b-A*x используется рекуррентная невязка PCG.
    // Перед объявлением сходимости проверяется true residual. Все нормы
    // относятся к системе с вычтенным вертикальным datum.
    qreal relativeResidual{};
    // Относится только к основной системе уровня, не к exact-проекции.
    bool converged{};
    bool faultsActive{};
    int projectionOrder{};
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
    qreal maxFaultConstraintError{};
    std::size_t faultConstraintCount{};
    std::size_t smoothingPasses{};
    std::size_t filledNodeCount{}, undefinedNodeCount{};
    qreal requestedCoarseningFactor{};
};

// Схема v2: validation -> polyline sampling -> trend -> hierarchy ->
// [fault schedule -> prolongation + derivative transfer -> Taylor Snap ->
// curvature PCG + jump/dip rows] -> constraints -> optional final smoothing ->
// domain mask -> linear hole fill -> qreal verification -> commit.
// Подробнее: README_2.md и gridding_pipeline_2.inc.
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

// Fault influence включает барьеры на последних проходах. Jump/absolute
// constraints учитываются отдельно от геометрии; открытые концы не продлеваются.
// Слишком грубые непредставимые уровни удаляются. На узлах разлома valid=0:
// обычная single-valued сетка не хранит два предельных значения.
// Для выборки и отрисовки нужны ТЕ ЖЕ faults, треугольники через разлом запрещены.
// Четвёртый аргумент обязателен для отсутствия неоднозначности (..., {}).
ConvergentGriddingReport convergentGridding(
    Surface& surface,
    const std::vector<Point>& points,
    const std::vector<Fault>& faults,
    const ConvergentGriddingOptions& options,
    const AdditionalInputs& additional = {});

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
    ConvergentGriddingReport* report = nullptr,
    const AdditionalInputs& additional = {});

// Значение в физической координате; без разломов — обычная билинейная выборка.
// Возле разлома веса видимых углов перенормируются (visibility-normalized
// bilinear interpolation); при отсутствии углов используется ближайший видимый
// узел в локальном радиусе двух ячеек. Этот же оператор C используется в soft-
// и exact-привязке и в maxControlError. Выборка на разломе вызывает исключение.
// Faults не сохраняются в Surface: их необходимо передавать вместе с grid.
// При NoData на опоре возвращается NaN (учитывается Surface.valid).
qreal sampleSurface(
    const Surface& surface, qreal x, qreal y,
    const std::vector<Fault>& faults = {});

} // namespace convergent2
