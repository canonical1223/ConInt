// Внутренние стадии v2; включается в anonymous namespace основного .cpp.
// Отдельный файл облегчает сопоставление скопированного численного ядра v1
// с новыми настройками. Здесь нет самостоятельной единицы компиляции.

bool faultsEnabled(std::size_t level, std::size_t total, Scalar percent)
{
    // Округление вверх — конвенция v2. Даже 0% оставляет последний проход:
    // подсказка Petrel говорит, что разломы всегда используются в конце.
    const auto active = std::max<std::size_t>(1,
        static_cast<std::size_t>(std::ceil(percent * Scalar(total) / 100)));
    return level >= total - std::min(total, active);
}

Surface gridSurface(const Grid& grid)
{
    Surface s;
    s.nx = grid.nx; s.ny = grid.ny;
    s.minx = qreal(grid.minx); s.maxx = qreal(grid.maxx);
    s.miny = qreal(grid.miny); s.maxy = qreal(grid.maxy);
    return s; // helpers геометрии не требуют массива значений
}

std::vector<PointConstraint> faultRows(
    const Grid& grid, const std::vector<Fault>& faults,
    const ConvergentGriddingOptions& options)
{
    std::vector<PointConstraint> rows;
    if (!grid.faults || options.faultZMode == FaultZMode::Ignore) return rows;
    for (const auto& probe : detail::makeFaultProbes(gridSurface(grid), faults, options)) {
        PointConstraint left, right;
        if ((probe.useLeft && !samplingConstraint(grid, probe.left.x, probe.left.y, left))
            || (probe.useRight && !samplingConstraint(grid, probe.right.x, probe.right.y, right)))
            throw std::invalid_argument("fault bank has no grid support; refine nx/ny");
        auto finish = [&](PointConstraint row) {
            row.value = probe.target; row.weight = probe.weight;
            row.strength = options.faultConstraintStrength;
            rows.push_back(row);
        };
        if (options.faultZMode == FaultZMode::Throw) {
            // Jump condition: C_left*u - C_right*u = throw. В отличие от
            // назначения двух абсолютных Z это не фиксирует среднюю отметку.
            for (std::size_t k = 0; k < right.count; ++k)
                addTerm(left, right.terms[k].index, -right.terms[k].coefficient);
            finish(left);
        } else {
            if (probe.useLeft) finish(left);
            if (probe.useRight) finish(right);
        }
    }
    return rows;
}

void validateDip(const Surface& surface, const AdditionalInputs& additional)
{
    for (const auto& d : additional.dipAzimuth) {
        if (!finite(d.x) || !finite(d.y) || !finite(d.weight) || d.weight < 0
            || !finite(d.dipDegrees) || std::abs(d.dipDegrees) >= 90
            || !finite(d.azimuthDegrees) || d.x < surface.minx || d.x > surface.maxx
            || d.y < surface.miny || d.y > surface.maxy)
            throw std::invalid_argument("invalid dip/azimuth input (dip must be in (-90,90))");
    }
}

std::vector<PointConstraint> dipRows(
    const Grid& g, const AdditionalInputs& input, Scalar finalStep,
    const ConvergentGriddingOptions& options)
{
    std::vector<PointConstraint> rows;
    if (options.dipMode == DipMode::None || options.dipConstraintStrength == 0) return rows;
    const Scalar pi = std::acos(Scalar(-1));
    const Scalar radius = options.dipInfluenceRadius > 0
        ? Scalar(options.dipInfluenceRadius) : 2 * finalStep;
    for (const auto& p : input.dipAzimuth) {
        if (p.weight == 0) continue;
        if (g.faults && g.faults->onFault(p.x, p.y))
            throw std::invalid_argument("dip/azimuth point must belong to one fault bank");
        const Scalar angle = p.azimuthDegrees * pi / 180;
        const Scalar gx = -std::tan(p.dipDegrees * pi / 180) * std::sin(angle);
        const Scalar gy = -std::tan(p.dipDegrees * pi / 180) * std::cos(angle);
        for (std::size_t j = 0; j < g.ny; ++j) for (std::size_t i = 0; i < g.nx; ++i) {
            const Scalar r = std::hypot(g.x(i)-p.x, g.y(j)-p.y);
            if (options.useDipInfluenceRadius && r >= radius) continue;
            if (g.faults && !g.faults->visible(p.x, p.y, g.x(i), g.y(j))) continue;
            // Односторонние производные у границ и барьеров. Строки задают
            // hx*du/dx и hy*du/dy: без опасного 1/h при очень мелкой сетке.
            auto derivative = [&](bool xAxis) {
                const std::size_t at = xAxis ? i : j, n = xAxis ? g.nx : g.ny;
                const auto idx = [&](std::size_t v) { return xAxis ? g.index(v,j) : g.index(i,v); };
                std::size_t lo = at, hi = at;
                if (at > 0 && visibleNodes(g, idx(at), idx(at-1))) lo = at-1;
                if (at+1 < n && visibleNodes(g, idx(at), idx(at+1))) hi = at+1;
                PointConstraint row;
                if (hi != lo) {
                    const Scalar f = 1 / Scalar(hi-lo);
                    addTerm(row, idx(lo), -f); addTerm(row, idx(hi), f);
                }
                return row;
            };
            PointConstraint rx = derivative(true), ry = derivative(false);
            Scalar w = p.weight;
            if (options.useDipInfluenceRadius) w *= std::pow(1-r/radius, 2);
            auto append = [&](PointConstraint row, Scalar value) {
                if (row.count == 0) return;
                row.value = value; row.weight = w;
                row.strength = options.dipConstraintStrength; rows.push_back(row);
            };
            if (options.dipMode == DipMode::DipAndAzimuth) {
                append(rx, gx*g.dx()); append(ry, gy*g.dy());
            } else if (rx.count && ry.count) {
                // Азимут без dip задаёт только направление: grad(f) dot t=0
                // вдоль простирания. Модуль градиента не навязывается.
                PointConstraint row;
                const Scalar h = std::min(g.dx(),g.dy());
                for (std::size_t k=0;k<rx.count;++k)
                    addTerm(row,rx.terms[k].index,rx.terms[k].coefficient*std::cos(angle)*h/g.dx());
                for (std::size_t k=0;k<ry.count;++k)
                    addTerm(row,ry.terms[k].index,-ry.terms[k].coefficient*std::sin(angle)*h/g.dy());
                append(row,0);
            }
        }
    }
    return rows;
}

struct HeightBounds {
    Scalar low{-std::numeric_limits<Scalar>::infinity()};
    Scalar high{std::numeric_limits<Scalar>::infinity()};
    bool enabled() const { return finite(low) || finite(high); }
};

HeightBounds heightBounds(const std::vector<CanonicalPoint>& data,
                         const ConvergentGriddingOptions& options)
{
    HeightBounds b;
    Scalar lo=0,hi=0;
    if (!data.empty()) {
        lo=hi=data.front().value;
        for (const auto& p:data) {lo=std::min(lo,p.value);hi=std::max(hi,p.value);}
    }
    if ((options.minimumValue && options.minimumInPercentOfDataRange)
        || (options.maximumValue && options.maximumInPercentOfDataRange))
        if (data.empty()) throw std::invalid_argument("percent bounds require absolute height data");
    if (options.minimumValue) b.low = options.minimumInPercentOfDataRange
        ? lo + Scalar(*options.minimumValue)/100*(hi-lo) : Scalar(*options.minimumValue);
    if (options.maximumValue) b.high = options.maximumInPercentOfDataRange
        ? hi + Scalar(*options.maximumValue)/100*(hi-lo) : Scalar(*options.maximumValue);
    if (std::isnan(b.low) || std::isnan(b.high) || b.low>b.high
        || (options.minimumValue && !finite(b.low)) || (options.maximumValue && !finite(b.high)))
        throw std::invalid_argument("height bounds are reversed or overflowed");
    return b;
}

Scalar constraintViolation(const PointConstraint& c, const std::vector<Scalar>& u)
{
    return std::max(Scalar(0),std::abs(constraintValue(c,u)-c.value)-c.halfWidth);
}

// Проекция на пересечение выпуклых множеств (Dykstra projection): точечные
// равенства, интервалы residual tolerance, jump conditions и box constraints.
// В отличие от простого clamp ПОСЛЕ exact-проекции, эти ограничения проверяются
// совместно. При несовместимости/исчерпании лимита результат не публикуется.
// Для равенств без bounds используем старую быструю Gram-PCG; при зависимых
// строках переходим к Dykstra, который не требует невырожденной C*C^T.
std::size_t enforceConstraints(Grid& grid, const std::vector<PointConstraint>& rows,
                               const HeightBounds& bounds,
                               const ConvergentGriddingOptions& options)
{
    bool equalities = !bounds.enabled();
    for (const auto& r:rows) if (r.halfWidth>0) equalities=false;
    if (equalities) {
        const auto backup = grid.values;
        try { return projectOntoPointConstraints(grid, rows, options).iterations; }
        catch (const std::runtime_error&) { grid.values=backup; }
    }
    if (rows.empty() && !bounds.enabled()) return 0;
    const Scalar tolerance = effectiveControlTolerance(rows, options);
    std::vector<Scalar> alpha(rows.size(),0), norm2(rows.size(),0);
    std::vector<Scalar> boxCorrection(bounds.enabled()?grid.values.size():0,0);
    for (std::size_t r=0;r<rows.size();++r) {
        for (std::size_t k=0;k<rows[r].count;++k)
            norm2[r]+=rows[r].terms[k].coefficient*rows[r].terms[k].coefficient;
        if (!(norm2[r]>0) || !finite(norm2[r]))
            throw std::invalid_argument("zero/invalid constraint support");
    }
    for (std::size_t it=0;it<options.maxControlProjectionIterations;++it) {
        for (std::size_t r=0;r<rows.size();++r) {
            const auto& c=rows[r];
            const Scalar v=constraintValue(c,grid.values)+alpha[r]*norm2[r];
            const Scalar target=std::clamp(v,c.value-c.halfWidth,c.value+c.halfWidth);
            const Scalar next=(v-target)/norm2[r];
            for (std::size_t k=0;k<c.count;++k)
                grid.values[c.terms[k].index]+=(alpha[r]-next)*c.terms[k].coefficient;
            alpha[r]=next;
        }
        if (bounds.enabled()) for (std::size_t n=0;n<grid.values.size();++n) {
            const Scalar v=grid.values[n]+boxCorrection[n];
            grid.values[n]=std::clamp(v,bounds.low,bounds.high);
            boxCorrection[n]=v-grid.values[n];
        }
        Scalar error=0;
        for (const auto& c:rows) error=std::max(error,constraintViolation(c,grid.values));
        for (Scalar v:grid.values) if (!finite(v))
            throw std::runtime_error("constraint projection overflow");
        if (error<=tolerance) return it+1;
    }
    throw std::runtime_error("point/throw/bound constraints are incompatible or projection did not converge");
}

void postSmooth(Grid& grid, const std::vector<PointConstraint>& rows,
                const HeightBounds& bounds, const ConvergentGriddingOptions& options,
                ConvergentGriddingReport& report,
                const std::vector<std::uint8_t>* valid = nullptr)
{
    if (options.smoothingIterations==0) return;
    auto masks=faultCurvatureMask(grid);
    if(valid) {
        if(masks.empty())masks.assign(grid.values.size(),7);
        // Исключаем целую строку B, если в её опоре есть NoData, а не
        // подставляем фиктивные нули в производные около пропуска.
        for(std::size_t j=0;j<grid.ny;++j)for(std::size_t i=0;i<grid.nx;++i) {
            const auto n=grid.index(i,j);
            if(i>0&&i+1<grid.nx&&(!(*valid)[n-1]||!(*valid)[n]||!(*valid)[n+1]))masks[n]&=6;
            if(j>0&&j+1<grid.ny&&(!(*valid)[n-grid.nx]||!(*valid)[n]||!(*valid)[n+grid.nx]))masks[n]&=5;
            if(i>0&&i+1<grid.nx&&j>0&&j+1<grid.ny
                &&(!(*valid)[n-grid.nx-1]||!(*valid)[n-grid.nx+1]
                   ||!(*valid)[n+grid.nx-1]||!(*valid)[n+grid.nx+1]))masks[n]&=3;
        }
    }
    const auto* mask=(grid.faults||valid)?&masks:nullptr;
    const auto diag=curvatureDiagonal(grid,1,mask);
    const Scalar maxDiag=*std::max_element(diag.begin(),diag.end());
    // Каждая строка B имеет <=4 элементов. Поэтому ||B^T B||_2 <=
    // 4*max(diag(B^T B)). Шаг 1/(4 maxDiag) устойчив и сохраняет плоскости
    // (их вторые разности нулевые). Не утверждаем, что это stencil Petrel.
    const Scalar step=maxDiag>0 ? 1/(4*maxDiag) : 0;
    for (std::size_t it=0;it<options.smoothingIterations;++it) {
        const auto previous=grid.values;
        std::vector<Scalar> curvature(grid.values.size(),0);
        applyCurvature(grid,previous,curvature,1,mask);
        for (std::size_t n=0;n<grid.values.size();++n) grid.values[n]-=step*curvature[n];
        report.controlProjectionIterations+=enforceConstraints(grid,rows,bounds,options);
        ++report.smoothingPasses;
        Scalar change=0;
        for (std::size_t n=0;n<grid.values.size();++n)
            change=std::max(change,std::abs(grid.values[n]-previous[n]));
        if (!options.fixedSmoothing && change<=options.smoothingChangeTolerance) break;
    }
}

// Линейное заполнение пропусков: дискретное гармоническое продолжение
// (harmonic extension), а не назначение нуля. Узлы с известными значениями
// закреплены; по видимым рёбрам графа решается L_uu*z=-L_uk*z_known.
// Компоненты без единого известного узла остаются NoData: информации нет.
void fillHoles(Grid& grid, std::vector<std::uint8_t>& valid,
               const std::vector<std::uint8_t>& allowed,
               const ConvergentGriddingOptions& options, ConvergentGriddingReport& report)
{
    if (!options.ensureAllNodesGetValue) return;
    const std::size_t n=grid.values.size();
    struct Edge {std::size_t a,b;Scalar w;};
    std::vector<Edge> edges;
    std::vector<std::vector<std::size_t>> neighbors(n);
    const Scalar h=std::min(grid.dx(),grid.dy());
    auto edge=[&](std::size_t a,std::size_t b,Scalar w) {
        if (allowed[a] && allowed[b] && visibleNodes(grid,a,b)) {
            edges.push_back({a,b,w}); neighbors[a].push_back(b);neighbors[b].push_back(a);
        }
    };
    for(std::size_t j=0;j<grid.ny;++j)for(std::size_t i=0;i<grid.nx;++i) {
        const auto a=grid.index(i,j);
        if(i+1<grid.nx)edge(a,a+1,h*h/(grid.dx()*grid.dx()));
        if(j+1<grid.ny)edge(a,a+grid.nx,h*h/(grid.dy()*grid.dy()));
    }
    std::vector<std::uint8_t> reachable(n,0);
    std::vector<std::size_t> queue;
    for(std::size_t a=0;a<n;++a)if(valid[a]&&allowed[a]){reachable[a]=1;queue.push_back(a);}
    for(std::size_t pos=0;pos<queue.size();++pos)for(auto b:neighbors[queue[pos]])
        if(!reachable[b]){reachable[b]=1;queue.push_back(b);}
    std::vector<std::uint8_t> unknown(n,0);
    std::size_t count=0;
    for(std::size_t a=0;a<n;++a)if(reachable[a]&&!valid[a]){unknown[a]=1;++count;}
    if(!count)return;
    std::vector<Scalar> diagonal(n,0),rhs(n,0),x(n,0),r(n,0),p(n,0),z(n,0),ap(n,0);
    for(const auto& e:edges){
        if(unknown[e.a]){diagonal[e.a]+=e.w;if(valid[e.b])rhs[e.a]+=e.w*grid.values[e.b];}
        if(unknown[e.b]){diagonal[e.b]+=e.w;if(valid[e.a])rhs[e.b]+=e.w*grid.values[e.a];}
    }
    auto apply=[&](const std::vector<Scalar>& v,std::vector<Scalar>& out){
        for(std::size_t a=0;a<n;++a)out[a]=diagonal[a]*v[a];
        for(const auto& e:edges)if(unknown[e.a]&&unknown[e.b]){
            out[e.a]-=e.w*v[e.b];out[e.b]-=e.w*v[e.a];}
    };
    for(std::size_t a=0;a<n;++a)if(unknown[a])x[a]=grid.values[a];
    apply(x,ap);
    for(std::size_t a=0;a<n;++a)if(unknown[a]){r[a]=rhs[a]-ap[a];z[a]=r[a]/diagonal[a];}
    p=z;Scalar rz=dot(r,z);
    const Scalar tol=std::max(Scalar(options.absoluteTolerance),
        Scalar(options.relativeTolerance)*std::max(Scalar(1),std::sqrt(dot(rhs,rhs))));
    std::size_t it=0;
    while(std::sqrt(dot(r,r))>tol&&it<options.maxSolverIterations){
        apply(p,ap);const Scalar denom=dot(p,ap);
        if(!(denom>0)||!finite(denom))throw std::runtime_error("hole fill PCG breakdown");
        const Scalar alpha=rz/denom;
        for(std::size_t a=0;a<n;++a)if(unknown[a]){x[a]+=alpha*p[a];r[a]-=alpha*ap[a];z[a]=r[a]/diagonal[a];}
        const Scalar next=dot(r,z),beta=next/rz;
        for(std::size_t a=0;a<n;++a)p[a]=z[a]+beta*p[a];
        rz=next;++it;
    }
    if(std::sqrt(dot(r,r))>tol)
        throw std::runtime_error("linear hole fill did not converge");
    for(std::size_t a=0;a<n;++a)if(unknown[a]){
        grid.values[a]=x[a];valid[a]=1;++report.filledNodeCount;}
}
