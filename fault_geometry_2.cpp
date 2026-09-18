#include "fault_geometry_2.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace convergent2 {
namespace detail {
namespace {

// Допуск выражен в долях максимальной длины стороны поверхности, а не в
// абсолютных мировых координатах. Например, для ширины области 20 км это
// около 3e-10 м. Он защищает классификацию общего узла от последних битов
// округления и не является пользовательской шириной/буфером разлома.
constexpr long double geometryTolerance =
    64.0L * static_cast<long double>(std::numeric_limits<double>::epsilon());

} // namespace

FaultGeometry::FaultGeometry(const Surface& surface,
                             const std::vector<Fault>& faults)
    : originX_(static_cast<long double>(surface.minx)),
      originY_(static_cast<long double>(surface.miny))
{
    const long double width =
        static_cast<long double>(surface.maxx) - originX_;
    const long double height =
        static_cast<long double>(surface.maxy) - originY_;
    if (!std::isfinite(originX_) || !std::isfinite(originY_) ||
        !std::isfinite(width) || !std::isfinite(height) ||
        !(width > 0) || !(height > 0)) {
        throw std::invalid_argument("Fault geometry requires finite positive surface extents");
    }
    scale_ = std::max(width, height);

    for (std::size_t faultIndex = 0; faultIndex < faults.size(); ++faultIndex) {
        const auto& fault = faults[faultIndex];
        if (fault.size() < 2) {
            throw std::invalid_argument("Fault " + std::to_string(faultIndex) +
                                        " requires at least two distinct XY vertices");
        }

        XY previous = normalize(static_cast<double>(fault.front().x),
                                static_cast<double>(fault.front().y));
        const std::size_t initialSegmentCount = segments_.size();
        for (std::size_t vertex = 1; vertex < fault.size(); ++vertex) {
            const XY next = normalize(static_cast<double>(fault[vertex].x),
                                      static_cast<double>(fault[vertex].y));
            // Строго повторенные последовательные вершины не добавляют
            // отрезок нулевой длины. Близкие вершины намеренно не сливаются:
            // такое слияние могло бы удалить небольшой реальный изгиб.
            if (previous.x == next.x && previous.y == next.y) {
                continue;
            }
            segments_.push_back({previous, next, bounds(previous, next)});
            previous = next;
        }
        if (segments_.size() == initialSegmentCount) {
            throw std::invalid_argument("Fault " + std::to_string(faultIndex) +
                                        " requires at least two distinct XY vertices");
        }
    }
}

FaultGeometry::XY FaultGeometry::normalize(double x, double y) const
{
    if (!std::isfinite(x) || !std::isfinite(y)) {
        throw std::invalid_argument("Fault geometry coordinates must be finite");
    }
    const XY result{
        (static_cast<long double>(x) - originX_) / scale_,
        (static_cast<long double>(y) - originY_) / scale_
    };
    // Вершины за границами поверхности разрешены: пересекающий область
    // отрезок часто задается именно внешними концами. Ограничение ниже
    // запрещает только экстремальное отношение координат к размеру области,
    // способное переполнить произведения в детерминанте ориентации. Запас 32
    // учитывает вычитание координат, два произведения и оценку погрешности.
    const long double safeMagnitude =
        std::sqrt(std::numeric_limits<long double>::max()) / 32.0L;
    if (!std::isfinite(result.x) || !std::isfinite(result.y) ||
        std::abs(result.x) > safeMagnitude ||
        std::abs(result.y) > safeMagnitude) {
        throw std::invalid_argument("Fault geometry coordinates exceed the safe normalized range");
    }
    return result;
}

FaultGeometry::Box FaultGeometry::bounds(const XY& a, const XY& b)
{
    return {std::min(a.x, b.x), std::max(a.x, b.x),
            std::min(a.y, b.y), std::max(a.y, b.y)};
}

bool FaultGeometry::overlaps(const Box& a, const Box& b)
{
    return !(a.maxx + geometryTolerance < b.minx ||
             b.maxx + geometryTolerance < a.minx ||
             a.maxy + geometryTolerance < b.miny ||
             b.maxy + geometryTolerance < a.miny);
}

bool FaultGeometry::contains(const Box& box, const XY& point)
{
    return point.x >= box.minx - geometryTolerance &&
           point.x <= box.maxx + geometryTolerance &&
           point.y >= box.miny - geometryTolerance &&
           point.y <= box.maxy + geometryTolerance;
}

int FaultGeometry::orientation(const XY& a, const XY& b, const XY& point)
{
    // Знак двумерного векторного произведения (orientation predicate):
    //   det = (bx-ax)*(py-ay) - (by-ay)*(px-ax).
    // Положительный/отрицательный знак означает левую/правую полуплоскость.
    // Нуль в пределах допуска означает коллинеарность. Сам по себе det
    // измеряется в квадрате длины, поэтому геометрический допуск умножается
    // на длину направления отрезка. Добавочная оценка roundoff масштабируется
    // величиной произведений: почти компенсирующиеся большие числа требуют
    // более осторожной классификации, чем хорошо различимый детерминант.
    const long double dx = b.x - a.x;
    const long double dy = b.y - a.y;
    const long double term1 = dx * (point.y - a.y);
    const long double term2 = dy * (point.x - a.x);
    const long double determinant = term1 - term2;
    const long double roundoff =
        32.0L * std::numeric_limits<long double>::epsilon() *
        (std::abs(term1) + std::abs(term2));
    const long double threshold =
        geometryTolerance * std::max(std::abs(dx), std::abs(dy)) + roundoff;
    if (determinant > threshold) {
        return 1;
    }
    if (determinant < -threshold) {
        return -1;
    }
    return 0;
}

bool FaultGeometry::intersects(const XY& a, const XY& b, const Box& box,
                               const Segment& segment)
{
    // Дешевый фильтр осевых прямоугольников (axis-aligned bounding boxes,
    // AABB) пропускает большинство далеких сегментов без детерминантов.
    if (!overlaps(box, segment.box)) {
        return false;
    }
    const int abA = orientation(a, b, segment.a);
    const int abB = orientation(a, b, segment.b);
    const int faultA = orientation(segment.a, segment.b, a);
    const int faultB = orientation(segment.a, segment.b, b);

    // Раздельные проверки касаний необходимы для общих концов, Т-образных
    // пересечений и коллинеарного наложения. Они также обрабатывают запрос
    // a==b: точка на разломе блокируется, точка вне разлома остается видимой.
    if ((abA == 0 && contains(box, segment.a)) ||
        (abB == 0 && contains(box, segment.b)) ||
        (faultA == 0 && contains(segment.box, a)) ||
        (faultB == 0 && contains(segment.box, b))) {
        return true;
    }
    return abA * abB < 0 && faultA * faultB < 0;
}

bool FaultGeometry::onFault(double x, double y) const
{
    const XY point = normalize(x, y);
    for (const auto& segment : segments_) {
        if (contains(segment.box, point) &&
            orientation(segment.a, segment.b, point) == 0) {
            return true;
        }
    }
    return false;
}

bool FaultGeometry::visible(double ax, double ay, double bx, double by) const
{
    const XY a = normalize(ax, ay);
    const XY b = normalize(bx, by);
    const Box queryBounds = bounds(a, b);
    for (const auto& segment : segments_) {
        if (intersects(a, b, queryBounds, segment)) {
            return false;
        }
    }
    return true;
}

} // namespace detail
} // namespace convergent2

