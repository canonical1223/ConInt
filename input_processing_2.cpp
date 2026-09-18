#include "input_processing_2.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace convergent2 {
namespace detail {
namespace {

// Ограничение защищает от непреднамеренного выделения памяти при линии длиной
// 10^12 ячеек или экстремально мелкой сетке. Вместо частичного результата
// функция сообщает об ошибке: молча прореживать данные здесь нельзя.
constexpr std::size_t maximumSamples = 1000000;
constexpr std::size_t maximumIntersectionChecks = 100000000;
constexpr long double geometryTolerance =
    64.0L * static_cast<long double>(std::numeric_limits<double>::epsilon());

struct XY { long double x, y; };

struct Geometry {
    long double minx, maxx, miny, maxy, scale;
    long double hx, hy, width, height;

    explicit Geometry(const Surface& surface)
        : minx(surface.minx), maxx(surface.maxx),
          miny(surface.miny), maxy(surface.maxy)
    {
        if (surface.nx < 2 || surface.ny < 2 ||
            !std::isfinite(minx) || !std::isfinite(maxx) ||
            !std::isfinite(miny) || !std::isfinite(maxy) ||
            !(maxx > minx) || !(maxy > miny)) {
            throw std::invalid_argument("Input processing requires a finite valid grid geometry");
        }
        scale = std::max(maxx - minx, maxy - miny);
        if (!std::isfinite(scale)) {
            throw std::invalid_argument("Surface extents exceed the numerical range");
        }
        hx = (maxx - minx) / static_cast<long double>(surface.nx - 1);
        hy = (maxy - miny) / static_cast<long double>(surface.ny - 1);
        if (!(hx > 0) || !(hy > 0)) {
            throw std::invalid_argument("Grid intervals are below the numerical range");
        }
        width = (maxx - minx) / scale;
        height = (maxy - miny) / scale;
    }

    XY normalized(long double x, long double y) const
    {
        const XY result{(x - minx) / scale, (y - miny) / scale};
        const long double limit = std::sqrt(std::numeric_limits<long double>::max()) / 32;
        if (!std::isfinite(result.x) || !std::isfinite(result.y) ||
            std::abs(result.x) > limit || std::abs(result.y) > limit) {
            throw std::invalid_argument("Coordinates exceed the safe normalized range");
        }
        return result;
    }

    bool contains(const Point& p) const
    {
        return p.x >= minx && p.x <= maxx && p.y >= miny && p.y <= maxy;
    }
};

void validatePoint(const Point& p, const char* kind)
{
    if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
        !std::isfinite(p.value) || !std::isfinite(p.weight) || p.weight < 0) {
        throw std::invalid_argument(std::string(kind) +
            " requires finite XY/value/weight and nonnegative weight");
    }
}

bool sameXY(const Point& a, const Point& b)
{
    return a.x == b.x && a.y == b.y;
}

Point interpolate(const Point& a, const Point& b, long double t)
{
    const auto lerp = [](long double from, long double to, long double parameter) {
        const long double difference = to - from;
        // При больших абсолютных XY и малом размере области центрированная
        // форма не теряет точность на сложении двух почти равных больших
        // слагаемых. Для экстремальных Z разных знаков difference может
        // переполниться; тогда безопасна выпуклая комбинация.
        return std::isfinite(difference) ? from + parameter * difference
            : (1 - parameter) * from + parameter * to;
    };
    Point p{
        static_cast<qreal>(lerp(a.x, b.x, t)),
        static_cast<qreal>(lerp(a.y, b.y, t)), 0, 0
    };
    // Round-trip parameter: после записи XY в qreal фактическая точка может
    // сдвинуться на несколько ULP. При XY~1e12 это уже порядка 1e-4, заметно
    // больше допуска exact-контролей. Нельзя оставить Z от теоретического t:
    // несколько таких точек одной ячейки нарушили бы даже точную плоскость.
    // Пересчитываем t проекцией ВЫДАННОЙ координаты на исходный сегмент:
    //   t_real = dot(p-a,b-a)/|b-a|^2.
    // Масштабирование направления защищает квадрат нормы от under/overflow;
    // центрирование вычитает исходную вершину до скалярного произведения.
    // На диагонали поперечное округление не вносит отдельного Z-смещения:
    // все точки сохраняют одну аффинную модель с градиентом вдоль сегмента.
    const long double dx = static_cast<long double>(b.x) - a.x;
    const long double dy = static_cast<long double>(b.y) - a.y;
    const long double directionScale = std::max(std::abs(dx), std::abs(dy));
    if (directionScale > 0 && std::isfinite(directionScale)) {
        const long double sx = dx / directionScale, sy = dy / directionScale;
        const long double px = (static_cast<long double>(p.x) - a.x) / directionScale;
        const long double py = (static_cast<long double>(p.y) - a.y) / directionScale;
        t = std::clamp((px * sx + py * sy) / (sx * sx + sy * sy), 0.0L, 1.0L);
    }
    p.value = static_cast<qreal>(lerp(a.value, b.value, t));
    p.weight = static_cast<qreal>(lerp(a.weight, b.weight, t));
    validatePoint(p, "Interpolated point");
    return p;
}

void appendPoint(std::vector<Point>& result, const Point& p)
{
    if (p.weight == 0) return;
    if (result.size() >= maximumSamples) {
        throw std::length_error("Input polyline sampling exceeds one million points");
    }
    result.push_back(p);
}

int orientation(const XY& a, const XY& b, const XY& p)
{
    // Перенос и общий изотропный масштаб сохраняют знак векторного
    // произведения, но уменьшают потерю точности при больших координатах.
    // det>0 — слева от направления a->b, det<0 — справа.
    const long double dx = b.x - a.x, dy = b.y - a.y;
    const long double first = dx * (p.y - a.y);
    const long double second = dy * (p.x - a.x);
    const long double tolerance =
        geometryTolerance * std::max(std::abs(dx), std::abs(dy)) +
        32 * std::numeric_limits<long double>::epsilon() *
        (std::abs(first) + std::abs(second));
    const long double determinant = first - second;
    return determinant > tolerance ? 1 : (determinant < -tolerance ? -1 : 0);
}

bool onSegment(const XY& a, const XY& b, const XY& p)
{
    return p.x >= std::min(a.x, b.x) - geometryTolerance &&
           p.x <= std::max(a.x, b.x) + geometryTolerance &&
           p.y >= std::min(a.y, b.y) - geometryTolerance &&
           p.y <= std::max(a.y, b.y) + geometryTolerance &&
           orientation(a, b, p) == 0;
}

bool intersects(const XY& a, const XY& b, const XY& c, const XY& d)
{
    if (std::max(a.x, b.x) + geometryTolerance < std::min(c.x, d.x) ||
        std::max(c.x, d.x) + geometryTolerance < std::min(a.x, b.x) ||
        std::max(a.y, b.y) + geometryTolerance < std::min(c.y, d.y) ||
        std::max(c.y, d.y) + geometryTolerance < std::min(a.y, b.y)) return false;
    const int s1 = orientation(a, b, c), s2 = orientation(a, b, d);
    const int t1 = orientation(c, d, a), t2 = orientation(c, d, b);
    return (s1 == 0 && onSegment(a, b, c)) ||
           (s2 == 0 && onSegment(a, b, d)) ||
           (t1 == 0 && onSegment(c, d, a)) ||
           (t2 == 0 && onSegment(c, d, b)) || (s1 * s2 < 0 && t1 * t2 < 0);
}

struct Segment {
    Point a, b;
    XY na, nb;
    std::size_t faultIndex, vertexIndex;
};

// Liang–Barsky clipping: каждая координата ограничивает параметр t отрезка
// a+t*(b-a). Значения/веса затем вычисляются в исходном t, поэтому обрезка
// геометрии не переопределяет профиль Z/throw вдоль входного разлома.
bool clipSegment(const Segment& segment, const Geometry& geometry,
                 long double& t0, long double& t1)
{
    t0 = 0;
    t1 = 1;
    const auto clipAxis = [&](long double a, long double b, long double hi) {
        const long double d = b - a;
        if (d == 0) return a >= 0 && a <= hi;
        long double near = -a / d, far = (hi - a) / d;
        if (near > far) std::swap(near, far);
        t0 = std::max(t0, near);
        t1 = std::min(t1, far);
        return t1 > t0;
    };
    return clipAxis(segment.na.x, segment.nb.x, geometry.width) &&
           clipAxis(segment.na.y, segment.nb.y, geometry.height) && t1 > t0;
}

std::string probeContext(const Segment& segment)
{
    return "Fault " + std::to_string(segment.faultIndex) +
           ", segment " + std::to_string(segment.vertexIndex);
}

} // namespace

std::vector<Point> sampleInputPolylines(
    const Surface& surface, const std::vector<Polyline>& polylines,
    bool fixedSampling)
{
    const Geometry geometry(surface);
    const long double step = std::hypot(geometry.hx, geometry.hy) / 2;
    std::vector<Point> result;
    std::size_t generated = 0;
    for (const auto& line : polylines) {
        for (const auto& p : line) {
            validatePoint(p, "Input polyline");
            if (!geometry.contains(p)) {
                throw std::invalid_argument("Input polyline vertices must be inside Surface");
            }
        }
        if (line.empty()) continue;
        if (!fixedSampling) {
            for (const auto& p : line) appendPoint(result, p);
            continue;
        }
        appendPoint(result, line.front());
        for (std::size_t i = 1; i < line.size(); ++i) {
            const Point& a = line[i - 1];
            const Point& b = line[i];
            const long double length = std::hypot(
                static_cast<long double>(b.x) - a.x,
                static_cast<long double>(b.y) - a.y);
            const long double ratio = length / step;
            if (!std::isfinite(ratio) || ratio > maximumSamples ||
                static_cast<long double>(generated) + ratio > maximumSamples) {
                throw std::length_error("Fixed polyline subdivision exceeds one million steps");
            }
            // k*step, а не равное деление всего сегмента: подсказка Petrel
            // требует фиксированный шаг и отдельный короткий остаток.
            const auto count = static_cast<std::size_t>(std::ceil(ratio));
            generated += count;
            for (std::size_t k = 1; k < count; ++k) {
                const long double t = static_cast<long double>(k) * step / length;
                if (t < 1) appendPoint(result, interpolate(a, b, t));
            }
            // Общая вершина двух соседних сегментов уже является концом
            // предыдущего; начало следующего отдельно не добавляется.
            appendPoint(result, b);
        }
    }
    return result;
}

std::vector<FaultProbe> makeFaultProbes(
    const Surface& levelGeometry, const std::vector<Fault>& faults,
    const ConvergentGriddingOptions& options)
{
    const Geometry geometry(levelGeometry);
    if (options.faultZMode == FaultZMode::Ignore) return {};
    if (options.faultZMode != FaultZMode::Throw &&
        options.faultZMode != FaultZMode::AbsoluteElevation) {
        throw std::invalid_argument("Unknown fault Z mode");
    }
    if (!std::isfinite(options.faultBankOffsetCells) ||
        !(options.faultBankOffsetCells > 0)) {
        throw std::invalid_argument("faultBankOffsetCells must be finite and positive");
    }
    if (options.throwSign != ThrowSign::LeftMinusRight &&
        options.throwSign != ThrowSign::RightMinusLeft) {
        throw std::invalid_argument("Unknown throw sign convention");
    }
    if (options.absoluteElevationBank != FaultBank::Both &&
        options.absoluteElevationBank != FaultBank::Left &&
        options.absoluteElevationBank != FaultBank::Right) {
        throw std::invalid_argument("Unknown absolute elevation bank");
    }
    const long double baseOffset = options.faultBankOffsetCells *
        std::min(geometry.hx, geometry.hy);
    const long double step = std::hypot(geometry.hx, geometry.hy) / 2;
    if (!std::isfinite(baseOffset)) {
        throw std::invalid_argument("Fault bank offset exceeds the numerical range");
    }
    std::vector<Segment> segments;
    for (std::size_t f = 0; f < faults.size(); ++f) {
        const Fault& fault = faults[f];
        if (fault.size() < 2) throw std::invalid_argument("Fault requires at least two XY vertices");
        for (const auto& p : fault) validatePoint(p, "Fault vertex");
        const auto firstSegment = segments.size();
        for (std::size_t i = 1; i < fault.size(); ++i) {
            const Point& a = fault[i - 1];
            const Point& b = fault[i];
            if (sameXY(a, b)) {
                if (a.value != b.value && (a.weight > 0 || b.weight > 0)) {
                    throw std::invalid_argument("Repeated fault XY vertices have conflicting Z/throw");
                }
                continue;
            }
            if (segments.size() >= maximumSamples) {
                throw std::length_error("Fault geometry exceeds one million segments");
            }
            segments.push_back({a, b, geometry.normalized(a.x, a.y),
                geometry.normalized(b.x, b.y), f, i - 1});
        }
        if (segments.size() == firstSegment) {
            throw std::invalid_argument("Fault requires at least two distinct XY vertices");
        }
    }

    std::vector<FaultProbe> result;
    std::size_t generated = 0, intersectionChecks = 0;
    for (std::size_t index = 0; index < segments.size(); ++index) {
        const Segment& segment = segments[index];
        if (segment.a.weight == 0 && segment.b.weight == 0) continue;
        long double t0, t1;
        if (!clipSegment(segment, geometry, t0, t1)) continue;
        const long double dx = segment.nb.x - segment.na.x;
        const long double dy = segment.nb.y - segment.na.y;
        const long double length = std::hypot(dx, dy);
        const long double ratio = (t1 - t0) * length * (geometry.scale / step);
        if (!std::isfinite(ratio) || ratio > maximumSamples ||
            static_cast<long double>(generated) + std::max(1.0L, std::ceil(ratio)) > maximumSamples) {
            throw std::length_error("Fault probe subdivision exceeds one million samples");
        }
        const auto count = static_cast<std::size_t>(std::max(1.0L, std::ceil(ratio)));
        generated += count;
        // Левая единичная нормаль: nL=(-dy,dx)/|segment|. Правая — -nL.
        const long double nx = -dy / length, ny = dx / length;
        for (std::size_t k = 0; k < count; ++k) {
            const long double t = t0 + (t1 - t0) *
                ((static_cast<long double>(k) + 0.5L) / count);
            const Point center = interpolate(segment.a, segment.b, t);
            if (center.weight == 0) continue;
            const XY normalizedCenter = geometry.normalized(center.x, center.y);
            FaultProbe probe;
            probe.target = static_cast<double>(center.value);
            probe.weight = static_cast<double>(center.weight);
            if (options.faultZMode == FaultZMode::Throw) {
                if (options.throwSign == ThrowSign::RightMinusLeft) probe.target = -probe.target;
            } else {
                probe.useLeft = options.absoluteElevationBank != FaultBank::Right;
                probe.useRight = options.absoluteElevationBank != FaultBank::Left;
            }

            // Вынос не должен перескочить соседний разлом: такой скачок дал бы
            // условие для другой пары блоков. Уменьшаем расстояние до тех пор,
            // пока оба необходимых пути от центра до банка не станут видимы.
            // Точку на пересечении нельзя однозначно отнести к двум банкам;
            // никакое уменьшение offset это не исправляет, поэтому ошибка
            // обнаруживается явно, вместо потери заданной амплитуды сброса.
            bool represented = false;
            long double offset = baseOffset;
            for (unsigned attempt = 0; attempt < 40; ++attempt, offset /= 2) {
                probe.left = {static_cast<qreal>(center.x + nx * offset),
                              static_cast<qreal>(center.y + ny * offset), 0, center.weight};
                probe.right = {static_cast<qreal>(center.x - nx * offset),
                               static_cast<qreal>(center.y - ny * offset), 0, center.weight};
                if ((probe.useLeft && !geometry.contains(probe.left)) ||
                    (probe.useRight && !geometry.contains(probe.right))) continue;
                const XY left = geometry.normalized(probe.left.x, probe.left.y);
                const XY right = geometry.normalized(probe.right.x, probe.right.y);
                if ((probe.useLeft && orientation(segment.na, segment.nb, left) != 1) ||
                    (probe.useRight && orientation(segment.na, segment.nb, right) != -1)) {
                    // Дальнейшее уменьшение уже не вернет потерянное различие
                    // сторон в qreal или в геометрическом допуске.
                    break;
                }
                bool clear = true;
                for (std::size_t other = 0; other < segments.size(); ++other) {
                    if (other == index) continue;
                    if (++intersectionChecks > maximumIntersectionChecks) {
                        throw std::length_error("Fault probe visibility exceeds the safety work limit");
                    }
                    const Segment& obstacle = segments[other];
                    if (onSegment(obstacle.na, obstacle.nb, normalizedCenter)) {
                        throw std::runtime_error(probeContext(segment) +
                            ": a bank probe lies on a fault intersection; split or refine fault sampling");
                    }
                    if ((probe.useLeft && intersects(normalizedCenter, left, obstacle.na, obstacle.nb)) ||
                        (probe.useRight && intersects(normalizedCenter, right, obstacle.na, obstacle.nb))) {
                        clear = false;
                        break;
                    }
                }
                if (clear) { represented = true; break; }
            }
            if (!represented) {
                throw std::runtime_error(probeContext(segment) +
                    ": requested fault banks cannot be represented inside Surface; refine geometry or select an existing bank");
            }
            result.push_back(probe);
        }
    }
    return result;
}

bool insideClosedFault(double x, double y, const std::vector<Fault>& faults)
{
    if (!std::isfinite(x) || !std::isfinite(y)) {
        throw std::invalid_argument("Polygon query requires finite XY");
    }
    for (const auto& fault : faults) {
        if (fault.size() < 4 || !sameXY(fault.front(), fault.back())) continue;
        // Для маски нужны только XY: выключение Z-входа или нулевые веса
        // не отменяют геометрическую внутреннюю область замкнутого контура.
        long double minx = fault.front().x, maxx = minx;
        long double miny = fault.front().y, maxy = miny;
        for (const auto& vertex : fault) {
            if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y)) {
                throw std::invalid_argument("Fault polygon requires finite XY");
            }
            minx = std::min(minx, static_cast<long double>(vertex.x));
            maxx = std::max(maxx, static_cast<long double>(vertex.x));
            miny = std::min(miny, static_cast<long double>(vertex.y));
            maxy = std::max(maxy, static_cast<long double>(vertex.y));
        }
        if (x < minx || x > maxx || y < miny || y > maxy) continue;
        const long double scale = std::max(maxx - minx, maxy - miny);
        if (!(scale > 0) || !std::isfinite(scale)) continue;
        const XY point{(static_cast<long double>(x) - minx) / scale,
                       (static_cast<long double>(y) - miny) / scale};
        bool inside = false;
        for (std::size_t i = 1; i < fault.size(); ++i) {
            const XY a{(static_cast<long double>(fault[i - 1].x) - minx) / scale,
                       (static_cast<long double>(fault[i - 1].y) - miny) / scale};
            const XY b{(static_cast<long double>(fault[i].x) - minx) / scale,
                       (static_cast<long double>(fault[i].y) - miny) / scale};
            if (onSegment(a, b, point)) return true;
            // Полуоткрытый Y-интервал считает общую вершину ребер ровно один
            // раз. Горизонтальные ребра не меняют четность пересечений луча.
            if ((a.y > point.y) != (b.y > point.y)) {
                const long double intersection = a.x +
                    (point.y - a.y) * (b.x - a.x) / (b.y - a.y);
                if (point.x < intersection) inside = !inside;
            }
        }
        if (inside) return true; // union: вложенные контуры не вырезают дырки
    }
    return false;
}

} // namespace detail
} // namespace convergent2
