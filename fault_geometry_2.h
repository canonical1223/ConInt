#pragma once

#include "convergent_gridding_2.h"

#include <vector>

namespace convergent2 {
namespace detail {

// Геометрический барьер (geometric barrier) для всех операторов интерполяции.
// Разлом — объединение последовательных отрезков polyline; соединение
// последней вершины с первой появляется только при явно повторенной вершине.
// value и weight вершин не задают ни высоту, ни амплитуду смещения разлома.
//
// Видимость (line of sight) здесь означает отсутствие пересечения прямого
// отрезка между двумя точками с любым разломом. Касание вершины, движение
// вдоль разлома и начало/конец на разломе также считаются блокировкой. Такое
// консервативное правило не позволяет численным шаблонам незаметно соединять
// разные стороны разлома через общую вершину. Обход конечного разлома через
// другие узлы остается возможным: локальная видимость не является глобальной
// меткой связной компоненты (connected-component label).
class FaultGeometry {
public:
    FaultGeometry(const Surface& surface, const std::vector<Fault>& faults);

    bool empty() const noexcept { return segments_.empty(); }
    bool onFault(double x, double y) const;
    bool visible(double ax, double ay, double bx, double by) const;

private:
    struct XY {
        long double x;
        long double y;
    };

    struct Box {
        long double minx;
        long double maxx;
        long double miny;
        long double maxy;
    };

    struct Segment {
        XY a;
        XY b;
        Box box;
    };

    XY normalize(double x, double y) const;
    static Box bounds(const XY& a, const XY& b);
    static bool overlaps(const Box& a, const Box& b);
    static bool contains(const Box& box, const XY& point);
    static int orientation(const XY& a, const XY& b, const XY& point);
    static bool intersects(const XY& a, const XY& b, const Box& box,
                           const Segment& segment);

    // Общий изотропный масштаб сохраняет углы, коллинеарность и пересечения.
    // Перенос начала координат уменьшает потерю значащих разрядов при больших
    // географических координатах. long double дает дополнительную точность
    // на поддерживающих ее платформах; корректность не требует, чтобы этот
    // тип был шире double (на MSVC они обычно имеют одинаковую точность).
    long double originX_{};
    long double originY_{};
    long double scale_{1};
    std::vector<Segment> segments_;
};

} // namespace detail
} // namespace convergent2

