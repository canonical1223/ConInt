#include "convergent_gridding_detail.h"
#include <algorithm>
#include <initializer_list>
#include <utility>

// Численное ядро v1: оператор кривизны, система уровня, основной PCG и
// точная привязка. Формулы, допуски и частота рестартов исходной v1 сохранены.
namespace convergent {
namespace detail {

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
// Если хотя бы одна связь опорных узлов пересекает разлом, строка удаляется целиком.
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
    // Кандидат на parallel for по j: каждый поток записывает только свои
    // mask[center], а FaultGeometry читает неизменяемый пространственный индекс.
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
    const std::vector<std::uint8_t>* mask)
{
    if (smoothness == 0) return;
    // ВНИМАНИЕ: ниже scatter, соседние строки пишут в общие output[]. Простое
    // добавление parallel for создаст гонку (data race). Для параллелизма нужно
    // перейти к сборке вклада для каждого выходного узла (gather) либо двум
    // фазам Bu и B^T(Bu) с единственным владельцем каждого выходного узла.
    // При этом необходимо сохранить граничные строки и cut-stencil mask.
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
                                     const std::vector<std::uint8_t>* mask)
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

Scalar dot(const std::vector<Scalar>& a, const std::vector<Scalar>& b)
{
    // Возможна параллельная редукция (parallel reduction), но другой порядок
    // сложения меняет округления и иногда число PCG-итераций. Последовательный
    // порядок оставлен для совместимости; см. PERFORMANCE_V1.md.
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
SolverResult solveLevel(
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
        const Scalar strength = finalStrength * constraint.weight;
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
            const Scalar strength = finalStrength * constraint.weight;
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
    result.relativeResidual = relativeResidual;
    result.converged = residualNorm <= threshold;
    if (result.converged) {
        result.values = std::move(x);
        return result;
    }

    // Одна стандартная PCG-итерация:
    //   alpha=(r^T*z)/(p^T*A*p), x<-x+alpha*p, r<-r-alpha*A*p,
    //   z<-M^{-1}r, beta=(r_new^T*z_new)/(r_old^T*z_old),
    //   p<-z+beta*p.
    // Итерации зависят друг от друга и выполняются последовательно. Внутри
    // шага независимые обновления x[i], residual[i], z[i], direction[i]
    // допускают parallel for / SIMD с барьерами перед dot и applyA.
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
        // 50 шагов выполняются true-residual replacement и restart направления.
        const bool restartDirection = (iteration + 1) % 50 == 0;
        if (restartDirection) {
            applyA(x, ax);
            for (std::size_t i = 0; i < size; ++i) residual[i] = rhs[i] - ax[i];
        }
        residualNorm = std::sqrt(std::max(Scalar(0), dot(residual, residual)));
        relativeResidual = normalizer > 0 ? residualNorm / normalizer : Scalar(0);
        result.iterations = iteration + 1;
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

namespace {

// Сжатая нумерация носителя ограничений (compressed constraint support).
// У C не более четырех ненулевых элементов в строке, поэтому C^T*v затрагивает
// U <= min(N, 4*m) узлов. Остальные N-U столбцов C тождественно нулевые.
// Исключение этих столбцов НЕ меняет G=C*C^T: мы лишь перенумеровываем узлы,
// сохраняя коэффициенты, порядок строк и порядок арифметических операций.
//
// Подготовка требует O(m log m), одно умножение — O(U+m), workspace — O(U+m).
// В старой реализации каждый шаг очищал N значений даже при нескольких точках.
// Матрица G размера m*m по-прежнему не формируется. Рабочий массив принадлежит
// одному вызову проекции: для одновременных проекций нужен отдельный экземпляр.
class ConstraintGramOperator {
public:
    explicit ConstraintGramOperator(const std::vector<PointConstraint>& constraints)
        : rows_(constraints)
    {
        for (const PointConstraint& row : rows_) {
            for (std::size_t t = 0; t < row.count; ++t) {
                globalIndices_.push_back(row.terms[t].index);
            }
        }
        std::sort(globalIndices_.begin(), globalIndices_.end());
        globalIndices_.erase(std::unique(globalIndices_.begin(), globalIndices_.end()),
                             globalIndices_.end());
        for (PointConstraint& row : rows_) {
            for (std::size_t t = 0; t < row.count; ++t) {
                row.terms[t].index = static_cast<std::size_t>(std::lower_bound(
                    globalIndices_.begin(), globalIndices_.end(), row.terms[t].index)
                    - globalIndices_.begin());
            }
        }
        nodeWork_.resize(globalIndices_.size());
    }

    // Матрично-свободное умножение: scatter C^T*v, затем gather C*(C^T*v).
    // Простой parallel for для scatter НЕБЕЗОПАСЕН: строки имеют общие узлы.
    void apply(const std::vector<Scalar>& input, std::vector<Scalar>& output)
    {
        scatter(input);
        for (std::size_t i = 0; i < rows_.size(); ++i) {
            output[i] = constraintValue(rows_[i], nodeWork_);
            if (!finite(output[i])) {
                throw std::runtime_error("point-constraint Gram product overflowed");
            }
        }
    }

    void addCorrection(const std::vector<Scalar>& multiplier, std::vector<Scalar>& values)
    {
        scatter(multiplier);
        for (std::size_t i = 0; i < globalIndices_.size(); ++i) {
            values[globalIndices_[i]] += nodeWork_[i];
        }
        // Полная проверка оставлена для прежней диагностики нечисловых данных.
        // Это единственный O(N)-проход здесь, а не O(N) на каждой Gram-итерации.
        for (Scalar value : values) {
            if (!finite(value)) {
                throw std::runtime_error("exact-control correction produced a non-finite grid value");
            }
        }
    }

private:
    void scatter(const std::vector<Scalar>& input)
    {
        std::fill(nodeWork_.begin(), nodeWork_.end(), Scalar(0));
        for (std::size_t i = 0; i < rows_.size(); ++i) {
            for (std::size_t t = 0; t < rows_[i].count; ++t) {
                const StencilTerm& term = rows_[i].terms[t];
                nodeWork_[term.index] += term.coefficient * input[i];
            }
        }
    }

    std::vector<PointConstraint> rows_;
    std::vector<std::size_t> globalIndices_;
    std::vector<Scalar> nodeWork_;
};

} // namespace

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
// ConstraintGramOperator умножает на нее через рассеивание/сборку (scatter/gather).
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
    std::vector<Scalar> gramMultiplier(count);
    ConstraintGramOperator gram(constraints);
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
        gram.apply(direction, gramDirection);
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
            gram.apply(multiplier, gramMultiplier);
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
    gram.addCorrection(multiplier, grid.values);

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


} // namespace detail
} // namespace convergent
