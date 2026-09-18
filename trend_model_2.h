#pragma once

#include "convergent_gridding_2.h"

#include <memory>
#include <vector>

namespace convergent2 {
namespace detail {

class FaultGeometry;

// Значение начального тренда и его локальные производные в физических
// координатах XY. supported=false означает отсутствие видимых контрольных
// точек (либо запрос непосредственно на геометрическом разломе). Это не маска
// convex hull: экстраполированное значение при наличии данных тоже supported.
struct TrendSample {
    double value{};
    double gx{};
    double gy{};
    double gxx{};
    double gxy{};
    double gyy{};
    // Физическое расстояние до convex hull всех видимых контролей; 0 внутри.
    // Для supported=false результат равен 0 и не обозначает достоверную опору.
    double distanceToHull{};
    bool supported{};
};

// Начальный тренд из данных (data-driven initial trend). Нулевая входная grid
// не делает начальные наклоны нулевыми: полином строится по контрольным точкам.
// Для запроса используются до 32 ближайших ВИДИМЫХ контролей; порядок 0/1/2
// означает горизонтальную, линейную либо квадратичную локальную модель.
// При вырожденности степень понижается; коллинеарные данные аппроксимируются
// вдоль их главной оси, без искусственного наклона поперек этой оси.
//
// Normal ограничивает расстояние продолжения от выпуклой оболочки видимых
// данных формулой rho=L*tanh(r/L); Trend сохраняет продолжение полинома.
// Эта формула — собственная реализация поведения «выполаживать вдали», а не
// утверждение о закрытой формуле Petrel. L=decayLength задан в единицах XY.
//
// Производные относятся к полиному с зафиксированными соседями и коэффициентами
// и к аналитическому преобразованию Normal. Переключение набора соседей не
// дифференцируется: это локальная оценка для Taylor projection, а не обещание
// глобальной C2-гладкости исходного тренда. Гладкость строит основной gridding.
//
// Объект сохраняет копию контролей. barriers — невладеющий указатель: геометрия
// должна существовать дольше TrendModel. nullptr либо empty() отключает барьеры.
class TrendModel {
public:
    TrendModel(const Surface& surface, const std::vector<Point>& points,
               const FaultGeometry* barriers, ProjectionOrder order,
               ExtrapolationMethod extrapolation, double decayLength);

    TrendSample evaluate(double x, double y) const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

} // namespace detail
} // namespace convergent2
