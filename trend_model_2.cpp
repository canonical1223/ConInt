#include "trend_model_2.h"
#include "fault_geometry_2.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace convergent2 {
namespace detail {
namespace {

constexpr std::size_t maximumNeighbors = 32;
constexpr std::size_t maximumTerms = 6;

struct XY {
    double x{};
    double y{};
};

struct Datum {
    XY xy; // Координаты относительно центра Surface / общий масштаб XY.
    double physicalX{};
    double physicalY{};
    double value{};
    double weight{};
};

double cross(XY a, XY b, XY c)
{
    return (b.x - a.x) * (c.y - a.y)
        - (b.y - a.y) * (c.x - a.x);
}

// Монотонная цепь (monotone chain / Andrew convex hull). Вход уже отсортирован
// по x,y; фильтрация по видимости этот порядок сохраняет. Поэтому построение
// занимает O(n), а не требует повторной сортировки на каждом узле сетки.
// Коллинеарное облако превращается в отрезок; одна координата — в одну вершину.
std::vector<XY> convexHull(const std::vector<const Datum*>& points)
{
    std::vector<XY> unique;
    unique.reserve(points.size());
    for (const Datum* p : points) {
        if (unique.empty() || p->xy.x != unique.back().x || p->xy.y != unique.back().y)
            unique.push_back(p->xy);
    }
    if (unique.size() < 3) return unique;
    std::vector<XY> hull(2 * unique.size());
    std::size_t count = 0;
    for (XY p : unique) {
        while (count >= 2 && cross(hull[count - 2], hull[count - 1], p) <= 0) --count;
        hull[count++] = p;
    }
    const std::size_t lower = count + 1;
    for (std::size_t i = unique.size() - 1; i-- > 0;) {
        const XY p = unique[i];
        while (count >= lower && cross(hull[count - 2], hull[count - 1], p) <= 0) --count;
        hull[count++] = p;
    }
    hull.resize(count - 1);
    return hull;
}

struct Projection {
    XY anchor;
    // Якобиан ближайшей проекции q(x) на активный участок оболочки: I внутри,
    // t*t^T внутри ребра, 0 у вершины. На переходах между участками используем
    // производные выбранного участка; в точности на границе возвращаем I.
    double qxx{1};
    double qxy{};
    double qyy{1};
};

Projection projectToHull(XY x, const std::vector<XY>& hull)
{
    if (hull.size() == 1) return {hull.front(), 0, 0, 0};
    if (hull.size() >= 3) {
        bool inside = true;
        for (std::size_t i = 0; i < hull.size(); ++i) {
            if (cross(hull[i], hull[(i + 1) % hull.size()], x) < 0) {
                inside = false;
                break;
            }
        }
        if (inside) return {x, 1, 0, 1};
    }
    Projection result{x, 1, 0, 1};
    double best = std::numeric_limits<double>::infinity();
    const std::size_t edges = hull.size() == 2 ? 1 : hull.size();
    for (std::size_t i = 0; i < edges; ++i) {
        const XY a = hull[i];
        const XY b = hull[(i + 1) % hull.size()];
        const double dx = b.x - a.x, dy = b.y - a.y;
        const double squared = dx * dx + dy * dy;
        if (squared == 0) continue;
        const double t = std::clamp(((x.x - a.x) * dx + (x.y - a.y) * dy) / squared,
                                    0.0, 1.0);
        const XY q{a.x + t * dx, a.y + t * dy};
        const double r2 = (x.x - q.x) * (x.x - q.x) + (x.y - q.y) * (x.y - q.y);
        if (r2 < best) {
            best = r2;
            result = {q, 0, 0, 0};
            if (t > 0 && t < 1) {
                result.qxx = dx * dx / squared;
                result.qxy = dx * dy / squared;
                result.qyy = dy * dy / squared;
            }
        }
    }
    return result;
}

struct Polynomial {
    XY center;
    XY axis{1, 0};
    double radius{1};
    double valueScale{1};
    // P(s,t)=c0+c1*s+c2*t+c3*s*s+c4*s*t+c5*t*t.
    std::array<double, maximumTerms> coefficient{};
};

// Взвешенный least-squares через модифицированный Грам—Шмидт с повторной
// ортогонализацией (reorthogonalized modified Gram-Schmidt QR). Не строим A^T A:
// normal equations возводят число обусловленности в квадрат. Здесь максимум
// 32 строки и 6 столбцов, поэтому стоимость QR ограничена O(32*6^2), независимо
// от общего числа контролей. При малой диагонали R вызывающий код понизит степень.
bool fitDegree(const std::vector<const Datum*>& points, Polynomial& model,
               int degree, bool oneDimensional)
{
    const std::array<int, maximumTerms> allTerms{0, 1, 2, 3, 4, 5};
    const std::array<int, maximumTerms> lineTerms{0, 1, 3, 0, 0, 0};
    const auto& terms = oneDimensional ? lineTerms : allTerms;
    const std::size_t columns = degree == 0 ? 1 : oneDimensional
        ? static_cast<std::size_t>(degree + 1) : degree == 1 ? 3 : 6;
    if (points.size() < columns) return false;
    double maxWeight = 0;
    for (const Datum* p : points) maxWeight = std::max(maxWeight, p->weight);
    std::array<std::array<double, maximumNeighbors>, maximumTerms> q{};
    std::array<std::array<double, maximumTerms>, maximumTerms> r{};
    std::array<double, maximumNeighbors> rhs{};
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Datum& p = *points[i];
        const double dx = (p.xy.x - model.center.x) / model.radius;
        const double dy = (p.xy.y - model.center.y) / model.radius;
        const double s = model.axis.x * dx + model.axis.y * dy;
        const double t = -model.axis.y * dx + model.axis.x * dy;
        const double w = std::sqrt(p.weight / maxWeight);
        const std::array<double, maximumTerms> basis{1, s, t, s * s, s * t, t * t};
        for (std::size_t j = 0; j < columns; ++j) q[j][i] = w * basis[terms[j]];
        rhs[i] = w * (p.value / model.valueScale);
    }
    double referenceNorm = 0;
    for (std::size_t j = 0; j < columns; ++j) {
        double initialNorm2 = 0;
        for (std::size_t i = 0; i < points.size(); ++i) initialNorm2 += q[j][i] * q[j][i];
        referenceNorm = std::max(referenceNorm, std::sqrt(initialNorm2));
        for (int pass = 0; pass < 2; ++pass) {
            for (std::size_t k = 0; k < j; ++k) {
                double dot = 0;
                for (std::size_t i = 0; i < points.size(); ++i) dot += q[k][i] * q[j][i];
                r[k][j] += dot;
                for (std::size_t i = 0; i < points.size(); ++i) q[j][i] -= dot * q[k][i];
            }
        }
        double norm2 = 0;
        for (std::size_t i = 0; i < points.size(); ++i) norm2 += q[j][i] * q[j][i];
        const double norm = std::sqrt(norm2);
        if (!(norm > 1e-10 * referenceNorm)) return false;
        r[j][j] = norm;
        for (std::size_t i = 0; i < points.size(); ++i) q[j][i] /= norm;
    }
    std::array<double, maximumTerms> coefficients{};
    for (std::size_t j = columns; j-- > 0;) {
        double value = 0;
        for (std::size_t i = 0; i < points.size(); ++i) value += q[j][i] * rhs[i];
        for (std::size_t k = j + 1; k < columns; ++k) value -= r[j][k] * coefficients[k];
        coefficients[j] = value / r[j][j];
        if (!std::isfinite(coefficients[j])) return false;
    }
    model.coefficient.fill(0);
    for (std::size_t j = 0; j < columns; ++j) model.coefficient[terms[j]] = coefficients[j];
    return true;
}

Polynomial fitPolynomial(const std::vector<const Datum*>& neighbors, int degree)
{
    Polynomial model;
    double maxWeight = 0;
    for (const Datum* p : neighbors) {
        maxWeight = std::max(maxWeight, p->weight);
        model.valueScale = std::max(model.valueScale, std::abs(p->value));
    }
    double weightSum = 0;
    for (const Datum* p : neighbors) {
        const double w = p->weight / maxWeight;
        weightSum += w;
        model.center.x += w * p->xy.x;
        model.center.y += w * p->xy.y;
    }
    model.center.x /= weightSum;
    model.center.y /= weightSum;
    model.radius = 0;
    for (const Datum* p : neighbors)
        model.radius = std::max(model.radius, std::hypot(p->xy.x - model.center.x,
                                                      p->xy.y - model.center.y));
    if (!(model.radius > 0)) model.radius = 1;

    // Главные оси (principal component analysis, PCA) отделяют измеряемый
    // продольный тренд от неопределенного поперечного при коллинеарных данных.
    // Общий, а не раздельный масштаб X/Y сохраняет углы и физическую геометрию.
    double xx = 0, xy = 0, yy = 0;
    for (const Datum* p : neighbors) {
        const double w = p->weight / maxWeight;
        const double x = (p->xy.x - model.center.x) / model.radius;
        const double y = (p->xy.y - model.center.y) / model.radius;
        xx += w * x * x;
        xy += w * x * y;
        yy += w * y * y;
    }
    const double angle = 0.5 * std::atan2(2 * xy, xx - yy);
    model.axis = {std::cos(angle), std::sin(angle)};
    const double spread = std::hypot(xx - yy, 2 * xy);
    const double major = 0.5 * (xx + yy + spread);
    const double minor = std::max(0.0, 0.5 * (xx + yy - spread));
    const bool oneDimensional = minor <= 1e-12 * major;
    for (int attempt = degree; attempt >= 0; --attempt) {
        if (fitDegree(neighbors, model, attempt, oneDimensional)) return model;
    }
    // При наличии хотя бы одного положительного веса столбец константы имеет
    // ненулевую норму. Сюда можно попасть лишь при выходе чисел за диапазон.
    throw std::runtime_error("failed to fit a finite initial trend");
}

TrendSample evaluatePolynomial(const Polynomial& model, XY x)
{
    const double dx = (x.x - model.center.x) / model.radius;
    const double dy = (x.y - model.center.y) / model.radius;
    const double a = model.axis.x, b = model.axis.y;
    const double s = a * dx + b * dy, t = -b * dx + a * dy;
    const auto& c = model.coefficient;
    const double ps = c[1] + 2 * c[3] * s + c[4] * t;
    const double pt = c[2] + c[4] * s + 2 * c[5] * t;
    const double gscale = model.valueScale / model.radius;
    const double hscale = gscale / model.radius;
    TrendSample out;
    out.value = model.valueScale * (c[0] + c[1] * s + c[2] * t
                                   + c[3] * s * s + c[4] * s * t + c[5] * t * t);
    out.gx = gscale * (a * ps - b * pt);
    out.gy = gscale * (b * ps + a * pt);
    out.gxx = hscale * (2 * c[3] * a * a - 2 * c[4] * a * b + 2 * c[5] * b * b);
    out.gxy = hscale * (2 * c[3] * a * b + c[4] * (a * a - b * b) - 2 * c[5] * a * b);
    out.gyy = hscale * (2 * c[3] * b * b + 2 * c[4] * a * b + 2 * c[5] * a * a);
    out.supported = true;
    return out;
}

// Выполаживание Normal: q — ближайшая точка convex hull, d=x-q, r=|d|,
// n=d/r. Вместо P(x) вычисляем P(q+L*tanh(r/L)*n). Внутри оболочки P сохраняется.
// rho(0)=0, rho'(0)=1, rho''(0)=0, rho(infinity)=L: при удалении от данных
// нормальное продолжение ограничено, а наклон в направлении удаления гаснет.
// Тангенциальный тренд вдоль края сохраняется. У вершины предельное значение
// может зависеть от направления; «Normal» не означает одну константу на весь край.
// Правило полностью определено здесь и не выдается за точную формулу Petrel.
TrendSample evaluateNormal(const Polynomial& model, XY x, const Projection& projection,
                           double length)
{
    const double dx = x.x - projection.anchor.x, dy = x.y - projection.anchor.y;
    const double distance = std::hypot(dx, dy);
    if (!(distance > 0)) return evaluatePolynomial(model, x);
    const double nx = dx / distance, ny = dy / distance;
    const double z = distance / length;
    const double t = std::tanh(z);
    const double rho = length * t;
    const double first = 1 - t * t;
    const double second = -2 * t * first / length;
    const double scale = rho / distance;

    // a=rho/r; a' и (r*a''-a') нужны для второй производной отображения.
    // Ряд при r/L << 1 предотвращает потерю точности в разностях близких чисел.
    double da = (first - scale) / distance;
    double cubic = second - 3 * da;
    if (z < 1e-4) {
        const double z2 = z * z;
        da = z * (-2.0 / 3 + z2 * (8.0 / 15 - z2 * 34.0 / 105)) / length;
        cubic = z * z2 * (16.0 / 15 - z2 * 136.0 / 105) / length;
    }
    const double rxx = 1 - projection.qxx, rxy = -projection.qxy;
    const double ryy = 1 - projection.qyy;
    const double jxx = projection.qxx + scale * rxx + (first - scale) * nx * nx;
    const double jxy = projection.qxy + scale * rxy + (first - scale) * nx * ny;
    const double jyy = projection.qyy + scale * ryy + (first - scale) * ny * ny;
    TrendSample p = evaluatePolynomial(model, {projection.anchor.x + rho * nx,
                                               projection.anchor.y + rho * ny});
    TrendSample out = p;
    out.gx = jxx * p.gx + jxy * p.gy;
    out.gy = jxy * p.gx + jyy * p.gy;
    const double dot = p.gx * nx + p.gy * ny;
    const double rx = rxx * p.gx + rxy * p.gy;
    const double ry = rxy * p.gx + ryy * p.gy;
    out.gxx = p.gxx * jxx * jxx + 2 * p.gxy * jxx * jxy + p.gyy * jxy * jxy
        + da * (2 * nx * rx + dot * rxx) + cubic * dot * nx * nx;
    out.gxy = p.gxx * jxx * jxy + p.gxy * (jxx * jyy + jxy * jxy) + p.gyy * jxy * jyy
        + da * (ny * rx + nx * ry + dot * rxy) + cubic * dot * nx * ny;
    out.gyy = p.gxx * jxy * jxy + 2 * p.gxy * jxy * jyy + p.gyy * jyy * jyy
        + da * (2 * ny * ry + dot * ryy) + cubic * dot * ny * ny;
    return out;
}

} // namespace

struct TrendModel::Impl {
    double originX{};
    double originY{};
    double coordinateScale{1};
    double decayLength{1}; // В нормированных координатах Surface.
    int degree{};
    ExtrapolationMethod extrapolation{};
    const FaultGeometry* barriers{};
    std::vector<Datum> points;
    std::vector<XY> fullHull;
};

TrendModel::TrendModel(const Surface& surface, const std::vector<Point>& points,
                       const FaultGeometry* barriers, ProjectionOrder order,
                       ExtrapolationMethod extrapolation, double decayLength)
{
    auto impl = std::make_shared<Impl>();
    const double minx = surface.minx, maxx = surface.maxx;
    const double miny = surface.miny, maxy = surface.maxy;
    if (!std::isfinite(minx) || !std::isfinite(maxx) || !std::isfinite(miny)
        || !std::isfinite(maxy) || !(maxx > minx) || !(maxy > miny)
        || !std::isfinite(maxx - minx) || !std::isfinite(maxy - miny))
        throw std::invalid_argument("invalid bounds for initial trend");
    if (!std::isfinite(decayLength) || !(decayLength > 0))
        throw std::invalid_argument("trend decay length must be finite and positive");
    switch (order) {
    case ProjectionOrder::Horizontal: impl->degree = 0; break;
    case ProjectionOrder::Linear: impl->degree = 1; break;
    case ProjectionOrder::Quadratic: impl->degree = 2; break;
    default: throw std::invalid_argument("unknown initial projection order");
    }
    if (extrapolation != ExtrapolationMethod::Normal && extrapolation != ExtrapolationMethod::Trend)
        throw std::invalid_argument("unknown extrapolation method");
    impl->extrapolation = extrapolation;
    impl->coordinateScale = std::max(maxx - minx, maxy - miny);
    impl->originX = minx + (maxx - minx) / 2;
    impl->originY = miny + (maxy - miny) / 2;
    impl->decayLength = decayLength / impl->coordinateScale;
    if (!std::isfinite(impl->decayLength) || !(impl->decayLength > 0))
        throw std::invalid_argument("trend decay length cannot be normalized safely");
    impl->barriers = barriers && !barriers->empty() ? barriers : nullptr;
    impl->points.reserve(points.size());
    for (const Point& p : points) {
        const double x = p.x, y = p.y, value = p.value, weight = p.weight;
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(value)
            || !std::isfinite(weight) || weight < 0)
            throw std::invalid_argument("invalid control point for initial trend");
        if (weight == 0) continue;
        if (x < minx || x > maxx || y < miny || y > maxy)
            throw std::invalid_argument("initial trend control is outside surface bounds");
        if (impl->barriers && impl->barriers->onFault(x, y)) continue;
        impl->points.push_back({{(x - impl->originX) / impl->coordinateScale,
                                  (y - impl->originY) / impl->coordinateScale},
                                x, y, value, weight});
    }
    std::sort(impl->points.begin(), impl->points.end(), [](const Datum& a, const Datum& b) {
        if (a.xy.x != b.xy.x) return a.xy.x < b.xy.x;
        if (a.xy.y != b.xy.y) return a.xy.y < b.xy.y;
        if (a.value != b.value) return a.value < b.value;
        return a.weight < b.weight;
    });
    std::vector<const Datum*> all;
    all.reserve(impl->points.size());
    for (const Datum& p : impl->points) all.push_back(&p);
    impl->fullHull = convexHull(all);
    impl_ = std::move(impl);
}

TrendSample TrendModel::evaluate(double x, double y) const
{
    if (!std::isfinite(x) || !std::isfinite(y))
        throw std::invalid_argument("initial trend query must be finite");
    const Impl& model = *impl_;
    if (model.barriers && model.barriers->onFault(x, y)) return {};
    const XY query{(x - model.originX) / model.coordinateScale,
                   (y - model.originY) / model.coordinateScale};
    if (!std::isfinite(query.x) || !std::isfinite(query.y))
        throw std::invalid_argument("initial trend query cannot be normalized safely");
    std::vector<const Datum*> visible;
    visible.reserve(model.points.size());
    for (const Datum& p : model.points) {
        if (!model.barriers || model.barriers->visible(x, y, p.physicalX, p.physicalY))
            visible.push_back(&p);
    }
    if (visible.empty()) return {};
    // Normal использует hull ВСЕХ видимых данных: ограничение до 32 соседей
    // относится только к fit и не должно ошибочно выполаживать точку внутри
    // полного облака из-за случайно узкой оболочки выбранного соседства.
    const std::vector<XY> localHull = model.barriers ? convexHull(visible) : std::vector<XY>{};
    const std::vector<XY>& hull = model.barriers ? localHull : model.fullHull;
    const auto lessDistance = [query](const Datum* a, const Datum* b) {
        const double da = std::hypot(a->xy.x - query.x, a->xy.y - query.y);
        const double db = std::hypot(b->xy.x - query.x, b->xy.y - query.y);
        if (da != db) return da < db;
        // Указатели относятся к одному вектору; порядок обеспечивает одинаковый
        // выбор при равных расстояниях и не зависит от перестановки входных точек.
        return a < b;
    };
    if (visible.size() > maximumNeighbors) {
        std::nth_element(visible.begin(), visible.begin() + maximumNeighbors,
                         visible.end(), lessDistance);
        visible.resize(maximumNeighbors);
    }
    std::sort(visible.begin(), visible.end(), lessDistance);
    const Polynomial polynomial = fitPolynomial(visible, model.degree);
    const Projection projection = projectToHull(query, hull);
    TrendSample sample = model.extrapolation == ExtrapolationMethod::Normal
        ? evaluateNormal(polynomial, query, projection, model.decayLength)
        : evaluatePolynomial(polynomial, query);
    sample.distanceToHull = std::hypot(query.x - projection.anchor.x,
                                       query.y - projection.anchor.y) * model.coordinateScale;
    sample.gx /= model.coordinateScale;
    sample.gy /= model.coordinateScale;
    sample.gxx = sample.gxx / model.coordinateScale / model.coordinateScale;
    sample.gxy = sample.gxy / model.coordinateScale / model.coordinateScale;
    sample.gyy = sample.gyy / model.coordinateScale / model.coordinateScale;
    if (!std::isfinite(sample.value) || !std::isfinite(sample.gx) || !std::isfinite(sample.gy)
        || !std::isfinite(sample.gxx) || !std::isfinite(sample.gxy) || !std::isfinite(sample.gyy)
        || !std::isfinite(sample.distanceToHull))
        throw std::runtime_error("initial trend produced a non-finite value or derivative");
    return sample;
}

} // namespace detail
} // namespace convergent2
