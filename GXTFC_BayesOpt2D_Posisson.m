clc; clear; close all;

%% ---------------- User controls ----------------
nu          = 0.01;                 % source width in S(x,y)
% w           = [0.5, 0.5, 0.15];     % initial local blob: [cx, cy, radius]
w           = [0.4, 0.6, 0.2];     % initial local blob: [cx, cy, radius]
alpha_grid  = 0.8;                  % Chebyshev-like blend for global grid

Nx_train    = 70; Ny_train = 70;    % collocation grid
Np          = 501;                  % plotting grid
Nx_res      = 200; Ny_res   = 200;  % residual check grid

% Bayesian optimization controls
useAbsResidual = true;   % minimize max|Residual| if true, else max(Residual)
MaxEvals       = 50;     % number of BO objective evals

rng(1);

%% ---------------- Collocation grid & source ----------------
[xf, yf] = ndgrid(linspace(0,1,Nx_train), linspace(0,1,Ny_train));
Xf = xf(:); Yf = yf(:);

Sfun = @(x,y) (1/(2*pi*nu^2)) * exp( -((x-0.5).^2 + (y-0.5).^2) / (2*nu^2) );
s    = Sfun(Xf, Yf);                      % Nf x 1

% Weighted LS (uniform weights here; keep structure for future)
Wsqrt = ones(numel(Xf),1)/sqrt(numel(Xf));

%% ---------------- Residual evaluation grid (warped toward center) ----------------
xgg = linspace(0,1,Nx_res); 
ygg = linspace(0,1,Ny_res);
pwarp = 2;   % power-warp exponent (>1 concentrates to 0.5)
ux = (sign(2*xgg-1).*abs(2*xgg-1).^pwarp + 1)/2;   
vy = (sign(2*ygg-1).*abs(2*ygg-1).^pwarp + 1)/2;
[xr, yr] = ndgrid(ux, vy);   % used for BO objective

tolB = 1e-8;                  % boundary tolerance for atom cull

%% ---------------- Bayesian Optimization of w = [cx, cy, radius] ----------------
cxVar = optimizableVariable('cx',[0.45, 0.55],'Type','real');
cyVar = optimizableVariable('cy',[0.45, 0.55],'Type','real');
rVar  = optimizableVariable('radius',[0.10, 0.20],'Type','real');
initTbl = table(w(1), w(2), w(3), 'VariableNames', {'cx','cy','radius'});

% Wrap every dependency explicitly into the objective
obj_wrapper = @(T) obj_residual_inf([T.cx, T.cy, T.radius], ...
    alpha_grid, Xf, Yf, Wsqrt, s, xr, yr, Sfun, tolB, useAbsResidual);

if exist('bayesopt','file') == 2
    BO = bayesopt(@(T)obj_wrapper(T), [cxVar, cyVar, rVar], ...
        'MaxObjectiveEvaluations', MaxEvals, ...
        'IsObjectiveDeterministic', true, ...
        'AcquisitionFunctionName','expected-improvement-plus', ...
        'UseParallel', false, ...
        'Verbose', 1, ...
        'InitialX', initTbl, ...
        'PlotFcn', {});  % headless
    best = BO.XAtMinObjective;
    w = [best.cx, best.cy, best.radius];
    fprintf('BayesOpt best w = [%.4f, %.4f, %.4f], max-residual = %.3e\n', ...
            w(1), w(2), w(3), BO.MinObjective);
% else
%     warning('bayesopt() not found. Falling back to random search.');
%     nTry = 40;
%     bestVal = inf; bestW = w;
%     for k = 1:nTry
%         wk = [ 0.45 + 0.10*rand, 0.45 + 0.10*rand, 0.10 + 0.10*rand ];
%         val = obj_residual_inf(wk, alpha_grid, Xf, Yf, Wsqrt, s, xr, yr, Sfun, tolB, useAbsResidual);
%         if val < bestVal, bestVal = val; bestW = wk; end
%     end
%     w = bestW;
%     fprintf('Random best w = [%.4f, %.4f, %.4f], max-residual = %.3e\n', ...
%             w(1), w(2), w(3), bestVal);
% end
else
    warning('bayesopt() not found. Falling back to random search.');
    nTry = 40;
    trialW   = zeros(nTry,3);
    trialObj = inf(nTry,1);

    bestVal = inf; bestW = w;
    for k = 1:nTry
        wk = [ 0.45 + 0.10*rand, 0.45 + 0.10*rand, 0.10 + 0.10*rand ];
        val = obj_residual_inf(wk, alpha_grid, Xf, Yf, Wsqrt, s, xr, yr, Sfun, tolB, useAbsResidual);
        trialW(k,:)   = wk;
        trialObj(k,1) = val;
        if val < bestVal, bestVal = val; bestW = wk; end
    end
    w = bestW;
    fprintf('Random best w = [%.4f, %.4f, %.4f], max-residual = %.3e\n', ...
            w(1), w(2), w(3), bestVal);
end

%% ---------------- Training history: Best-So-Far (residual + hyperparameters) ----------------
% Build the history (works for both bayesopt and random fallback)
if exist('BO','var') == 1
    Xtbl  = BO.XTrace;                   % table with cx, cy, radius
    fvals = BO.ObjectiveTrace(:);        % numeric vector
    good  = isfinite(fvals);
    Xtbl  = Xtbl(good,:); 
    fvals = fvals(good);
else
    Xtbl  = array2table(trialW, 'VariableNames', {'cx','cy','radius'});
    fvals = trialObj(:);
end

% --- Best-so-far without cummin (robust) ---
n = numel(fvals);
idxBest = zeros(n,1);
f_bsf   = zeros(n,1);
bestVal = inf; bestIdx = 1;
for k = 1:n
    if fvals(k) < bestVal
        bestVal = fvals(k);
        bestIdx = k;
    end
    idxBest(k) = bestIdx;
    f_bsf(k)   = bestVal;
end

% Extract best-so-far parameter sequences
cx_bsf = Xtbl.cx(idxBest);
cy_bsf = Xtbl.cy(idxBest);
r_bsf  = Xtbl.radius(idxBest);
iter   = (1:n)';
improv = [true; diff(idxBest)~=0];       % iterations where a new best occurred

% Label for the residual objective (matches your useAbsResidual choice)
if exist('useAbsResidual','var') && useAbsResidual
    objLabel = '$\max|r(x,y)|$';
else
    objLabel = '$\max r(x,y)$';
end

% Plot (4×1: residual + cx + cy + radius)
fHist = figure('Units','inches','Position',[0.5 0.5 8.5 9.5], 'Color','w');
set(fHist, 'defaultTextInterpreter','latex', ...
           'defaultAxesTickLabelInterpreter','latex', ...
           'defaultLegendInterpreter','latex');

tiledlayout(4,1,'Padding','compact','TileSpacing','compact');

% (1) Best-So-Far residual
nexttile;
plot(iter, f_bsf, 'LineWidth', 1.5); hold on;
plot(iter(improv), f_bsf(improv), 'o', 'MarkerSize', 5);
ylabel('$\mathrm{Best\!-\!So\!-\!Far}\ J_{val}$');


grid on; xlim([1 n]);
% auto log scale if positive
if all(f_bsf > 0), set(gca,'YScale','log'); end

% (2) Best-So-Far c_x
nexttile;
stairs(iter, cx_bsf, 'LineWidth', 1.5); hold on;
plot(iter(improv), cx_bsf(improv), 'o', 'MarkerSize', 5);
ylabel('$\mathrm{Best\!-\!So\!-\!Far}\ c_x$'); grid on; xlim([1 n]);

% (3) Best-So-Far c_y
nexttile;
stairs(iter, cy_bsf, 'LineWidth', 1.5); hold on;
plot(iter(improv), cy_bsf(improv), 'o', 'MarkerSize', 5);
ylabel('$\mathrm{Best\!-\!So\!-\!Far}\ c_y$'); grid on; xlim([1 n]);

% (4) Best-So-Far radius
nexttile;
stairs(iter, r_bsf, 'LineWidth', 1.5); hold on;
plot(iter(improv), r_bsf(improv), 'o', 'MarkerSize', 5);
xlabel('Iteration'); ylabel('$\mathrm{Best\!-\!So\!-\!Far}\ \mathrm{radius}$');
grid on; xlim([1 n]);

outdir = 'figs'; if ~exist(outdir,'dir'), mkdir(outdir); end
exportgraphics(fHist, fullfile(outdir,'bo_best_so_far_hist.png'), 'Resolution',300, 'BackgroundColor','white');
exportgraphics(fHist, fullfile(outdir,'bo_best_so_far_hist.pdf'), 'ContentType','vector', 'BackgroundColor','white');



%% ---------------- Recompute centers/solution with optimal w ----------------
[Centers, Sigma] = fallback_generate_centers_sigma(w, alpha_grid);

% Drop atoms right on the boundary (mask will squash them anyway)
mask = Centers(:,1)>tolB & Centers(:,1)<1-tolB & Centers(:,2)>tolB & Centers(:,2)<1-tolB;
Centers = Centers(mask,:);  Sigma = Sigma(mask);
Ns = size(Centers,1);
fprintf('Ns (atoms) = %d\n', Ns);

m = 1 ./ (sqrt(2)*Sigma(:));           % isotropic slopes

% Assemble PDE matrix and solve (QR via backslash)
R  = assemble_laplacian_Gphi(Xf, Yf, Centers, m);   % Nf x Ns
c  = (R .* Wsqrt) \ (s .* Wsqrt);

%% ---------------- Predict & diagnostics at optimum ----------------
% Plotting grid prediction
[xp, yp] = ndgrid(linspace(0,1,Np), linspace(0,1,Np));
Up   = xtfc_eval_predictor(xp, yp, Centers, m, c);

% FD reference "exact" (coarse solve + interpolation to plotting grid)
Uex = Poisson_Exact_FD_local(nu, 100);
[xe, ye] = ndgrid(linspace(0,1,size(Uex,1)), linspace(0,1,size(Uex,2)));
UexI = interp2(ye, xe, Uex, yp, xp, 'linear', 0);

% Residual on the warped grid used for optimization
LapUh = xtfc_eval_laplacian(xr, yr, Centers, m, c);
Res   = LapUh - Sfun(xr, yr);

% Proper L2 on warped grid (use ux, vy vectors)
I = trapz(ux, Res.^2, 1);   % integrate in x (dim 1)
I = trapz(vy, I, 2);        % integrate in y (dim 2)
L2res = sqrt(I);
fprintf('Validation residual L2 ≈ %.3e\n', L2res);

%% ---------------- Plots ----------------
figs = plot_xtfc_pub(Up, UexI, Res, Centers, Sigma, Np, Nx_res, Ny_res, ...
    'OutDir','figs','DPI',300,'TitleUH','X\!-\!TFC','TitleRef','Finite Difference', ...
    'L2res', L2res);

%% ===================== Objective (standalone local function) =====================
function f = obj_residual_inf(w_try, alpha_grid, Xf, Yf, Wsqrt, s, xr, yr, Sfun, tolB, useAbsResidual)
    % Make evaluation deterministic despite random center generation
    oldStream = RandStream.getGlobalStream;
    cleaner   = onCleanup(@() RandStream.setGlobalStream(oldStream));
    RandStream.setGlobalStream(RandStream('mt19937ar','Seed',1));

    % Generate atoms from candidate w, apply boundary mask
    [Ck, Sk] = fallback_generate_centers_sigma(w_try, alpha_grid);
    msk = Ck(:,1)>tolB & Ck(:,1)<1-tolB & Ck(:,2)>tolB & Ck(:,2)<1-tolB;
    Ck = Ck(msk,:);  Sk = Sk(msk);
    mk = 1 ./ (sqrt(2)*Sk(:));

    % Assemble, solve, evaluate residual on [xr,yr]
    Rk = assemble_laplacian_Gphi(Xf, Yf, Ck, mk);
    ck = (Rk .* Wsqrt) \ (s .* Wsqrt);
    LapUh_k = xtfc_eval_laplacian(xr, yr, Ck, mk, ck);
    Res_k   = LapUh_k - Sfun(xr, yr);

    f = max(abs(Res_k(:)));
    if ~useAbsResidual
        f = max(Res_k(:));
    end
end

%% ===================== Helper: PDE assembly / evaluation =====================
function R = assemble_laplacian_Gphi(X, Y, Centers, m)
% R_{j,i} = Lap( G * phi_i )(X_j,Y_j), G=x(1-x)y(1-y), isotropic m(i).
Nf_local = numel(X); Ns_local = size(Centers,1);
R = zeros(Nf_local, Ns_local);

G   = @(x,y) x.*(1-x).*y.*(1-y);
Gx  = @(x,y) (1-2*x).*y.*(1-y);
Gy  = @(x,y) (1-2*y).*x.*(1-x);
LapG= @(x,y) -2*( x.*(1-x) + y.*(1-y) );

Gf   = G(X,Y); Gxf = Gx(X,Y); Gyf = Gy(X,Y); LapGf = LapG(X,Y);

for i = 1:Ns_local
    mi = m(i); ax = Centers(i,1); ay = Centers(i,2);
    zx = mi*(X - ax);
    zy = mi*(Y - ay);
    phi   = exp(-(zx.^2 + zy.^2));
    phix  = -2*mi*zx .* phi;
    phiy  = -2*mi*zy .* phi;
    phixx = (4*mi^2*zx.^2 - 2*mi^2) .* phi;
    phiyy = (4*mi^2*zy.^2 - 2*mi^2) .* phi;
    Lphi  = phixx + phiyy;

    R(:,i) = Gf .* Lphi + 2*( Gxf .* phix + Gyf .* phiy ) + phi .* LapGf;
end
end

function U = xtfc_eval_predictor(X, Y, Centers, m, c)
% u_h = sum_i c_i * G * phi_i
G = @(x,y) x.*(1-x).*y.*(1-y);
U = zeros(size(X));
for i = 1:numel(c)
    mi = m(i); ax = Centers(i,1); ay = Centers(i,2);
    zx = mi*(X - ax);
    zy = mi*(Y - ay);
    phi = exp(-(zx.^2 + zy.^2));
    U = U + c(i) * ( G(X,Y) .* phi );
end
end

function L = xtfc_eval_laplacian(X, Y, Centers, m, c)
G   = @(x,y) x.*(1-x).*y.*(1-y);
Gx  = @(x,y) (1-2*x).*y.*(1-y);
Gy  = @(x,y) (1-2*y).*x.*(1-x);
LapG= @(x,y) -2*( x.*(1-x) + y.*(1-y) );
Gf   = G(X,Y); Gxf = Gx(X,Y); Gyf = Gy(X,Y); LapGf = LapG(X,Y);

L = zeros(size(X));
for i = 1:numel(c)
    mi = m(i); ax = Centers(i,1); ay = Centers(i,2);
    zx = mi*(X - ax);
    zy = mi*(Y - ay);
    phi   = exp(-(zx.^2 + zy.^2));
    phix  = -2*mi*zx .* phi;
    phiy  = -2*mi*zy .* phi;
    phixx = (4*mi^2*zx.^2 - 2*mi^2) .* phi;
    phiyy = (4*mi^2*zy.^2 - 2*mi^2) .* phi;
    Lphi  = phixx + phiyy;
    L = L + c(i) * ( Gf .* Lphi + 2*(Gxf.*phix + Gyf.*phiy) + phi.*LapGf );
end
end

%% ===================== Fallback generator =====================
function [Centers, Sigma] = fallback_generate_centers_sigma(wloc, alpha_locgrid)
% Lightweight stand-in:
% - Chebyshev-like global grid (nx=ny=31)
% - + N_inside points uniform in a circle centered at (w(1), w(2)) with radius w(3)
% - Isotropic sigma from local kNN (k=6) with caps based on global spacing

    % global grid
    nx=31; ny=31; c_sigma=5.0; gamma_end=0.8;
    x1 = cheb_like_1d(nx, alpha_locgrid, gamma_end);
    y1 = cheb_like_1d(ny, alpha_locgrid, gamma_end);
    [Xg, Yg] = meshgrid(x1, y1);
    dx = nodal_spacing(x1); dy = nodal_spacing(y1);
    [DX, DY] = meshgrid(dx, dy);
    Hg = sqrt(DX .* DY);
    sigma_g = c_sigma * Hg(:);
    Cg = [Xg(:), Yg(:)];

    % local points in a circle (uniform by area)
    N_inside = 200;  % tweakable
    cx=wloc(1); cy=wloc(2); rad=wloc(3);
    U = rand(N_inside,1); T = 2*pi*rand(N_inside,1);
    r = rad*sqrt(U);                          % area-uniform
    Px = cx + r.*cos(T);  Py = cy + r.*sin(T);
    Pin = [Px, Py];
    Pin = Pin(Px>=0 & Px<=1 & Py>=0 & Py<=1, :);

    % kNN widths for local
    k_local=6; alpha_loc=0.9;
    sigma_loc = local_sigma_knn(Pin, k_local, alpha_loc);

    % cap local by global sigma at those locations
    Fsg = scatteredInterpolant(Cg(:,1), Cg(:,2), sigma_g, 'natural', 'nearest');
    sigma_cap = Fsg(Pin(:,1), Pin(:,2));
    beta_cap = 1.0;
    sigma_loc = min(sigma_loc, beta_cap * sigma_cap);

    Centers = [Cg; Pin];
    Sigma   = [sigma_g; sigma_loc];
end

function t = cheb_like_1d(N, alpha, gamma)
    if N <= 1, t = (N==1)*0.0; return; end
    i = (0:N-1).';
    t_lin  = i/(N-1);
    t_cheb = 0.5*(1 - cos(pi*t_lin));            % Clenshaw–Curtis on [0,1]
    t = (1 - alpha)*t_lin + alpha*t_cheb;        % blend
    s = 2*t - 1; s = sign(s) .* (abs(s) .^ gamma); % end-warp
    t = 0.5*(s + 1);
    t(1) = 0; t(end) = 1;
end

function d = nodal_spacing(x)
    N = numel(x);
    d = zeros(N,1);
    if N==1, d(:)=1; return; end
    if N>=2
        d(1) = x(2) - x(1);
        d(N) = x(N) - x(N-1);
    end
    if N>=3
        d(2:N-1) = 0.5 * ((x(3:N) - x(2:N-1)) + (x(2:N-1) - x(1:N-2)));
    end
end

function sigma = local_sigma_knn(P, k, alpha_loc)
    [N,dim] = size(P); if dim~=2, error('P must be N-by-2'); end
    k = max(1, min(k, N-1));
    % simple O(N^2) distances (fine at these sizes)
    sq = sum(P.^2,2);
    D2 = max(0, sq + sq' - 2*(P*P.'));
    D = sqrt(D2); D(1:N+1:end) = inf;
    dk = zeros(N,1);
    for i=1:N
        ds = sort(D(i,:));
        dk(i) = ds(k);
    end
    href  = median(dk);
    sigma = alpha_loc * dk;
    sigma = max(sigma, 0.25*href);
    sigma = min(sigma, 4.0*href);
end

%% ===================== Minimal FD "exact" =====================
function u = Poisson_Exact_FD_local(nu_local, N)
% Minimal self-contained FD solve for ∆u = S, u|∂Ω=0 on [0,1]^2
xL=0; xR=1; yB=0; yT=1; x0=0.5; y0=0.5;
x = linspace(xL,xR,N); y = linspace(yB,yT,N);
[X,Y] = meshgrid(x,y); dx = x(2)-x(1);
f = (1/(2*pi*nu_local^2))*exp(-((X-x0).^2 + (Y-y0).^2)/(2*nu_local^2));
u = zeros(N,N);
e = ones(N-2,1);
D2 = spdiags([e -2*e e], -1:1, N-2, N-2)/dx^2;
I  = speye(N-2);
A  = kron(I,D2) + kron(D2,I);
f_inner = f(2:end-1, 2:end-1);
rhs = f_inner(:);
u_inner = A \ rhs;
u(2:end-1, 2:end-1) = reshape(u_inner, [N-2, N-2]);
end

%% ===================== Publication-quality plots =====================
function figs = plot_xtfc_pub(Up, UexI, Res, Centers, Sigma, Np_loc, Nx_res_loc, Ny_res_loc, varargin)
% (same plotting function we used earlier; unchanged core, using exportgraphics)
p = inputParser;
addParameter(p,'OutDir','figs'); addParameter(p,'DPI',300);
addParameter(p,'UseLaTeX',true); addParameter(p,'FontAxis',18);
addParameter(p,'FontTitle',20); addParameter(p,'FontCbar',16);
addParameter(p,'MarkerSize',18); addParameter(p,'CMapField','parula');
addParameter(p,'CMapMag','turbo'); addParameter(p,'TitleUH','X\!-\!TFC');
addParameter(p,'TitleRef','Finite Difference'); addParameter(p,'SuperTitle','');
addParameter(p,'L2res',[]); parse(p,varargin{:}); opt = p.Results;

if opt.UseLaTeX
    set(groot,'defaultTextInterpreter','latex','defaultAxesTickLabelInterpreter','latex',...
        'defaultLegendInterpreter','latex','defaultFigureColor','w','defaultAxesFontSize',16,...
        'defaultAxesLineWidth',1.0,'defaultLineLineWidth',1.5);
else
    set(groot,'defaultFigureColor','w','defaultAxesFontSize',16,...
        'defaultAxesLineWidth',1.0,'defaultLineLineWidth',1.5);
end

xNp = linspace(0,1,Np_loc); yNp = linspace(0,1,Np_loc);
xNr = linspace(0,1,Nx_res_loc); yNr = linspace(0,1,Ny_res_loc);
sol_clim = [min([Up(:); UexI(:)]), max([Up(:); UexI(:)])];
outdir = char(opt.OutDir); if ~exist(outdir,'dir'); mkdir(outdir); end

figs.f1 = figure('Units','inches','Position',[0.5 0.5 12 4.2]);
tiledlayout(figs.f1,1,3,'Padding','compact','TileSpacing','compact');
nexttile; imagesc(xNp,yNp,Up.'); axis image xy; colormap(gca,opt.CMapField); caxis(sol_clim);
title(['$u_h \ \mathrm{(' char(opt.TitleUH) ')}$'],'FontSize',opt.FontTitle);
xlabel('$x$','FontSize',opt.FontAxis); ylabel('$y$','FontSize',opt.FontAxis);
cb=colorbar('Location','eastoutside'); if opt.UseLaTeX, cb.TickLabelInterpreter='latex'; end
ylabel(cb,'$u_h$','Interpreter',ternary(opt.UseLaTeX,'latex','tex'),'FontSize',opt.FontCbar);
set(gca,'XMinorTick','on','YMinorTick','on');

nexttile; imagesc(xNp,yNp,UexI.'); axis image xy; colormap(gca,opt.CMapField); caxis(sol_clim);
title(['$u_{\mathrm{ref}} \ \mathrm{(' char(opt.TitleRef) ')}$'],'FontSize',opt.FontTitle);
xlabel('$x$','FontSize',opt.FontAxis); ylabel('$y$','FontSize',opt.FontAxis);
cb=colorbar('Location','eastoutside'); if opt.UseLaTeX, cb.TickLabelInterpreter='latex'; end
ylabel(cb,'$u_{\mathrm{ref}}$','Interpreter',ternary(opt.UseLaTeX,'latex','tex'),'FontSize',opt.FontCbar);
set(gca,'XMinorTick','on','YMinorTick','on');

nexttile; E=abs(Up-UexI); imagesc(xNp,yNp,E.'); axis image xy; colormap(gca,opt.CMapMag);
title('$|u_h - u_{\mathrm{ref}}|$','FontSize',opt.FontTitle);
xlabel('$x$','FontSize',opt.FontAxis); ylabel('$y$','FontSize',opt.FontAxis);
cb=colorbar('Location','eastoutside'); if opt.UseLaTeX, cb.TickLabelInterpreter='latex'; end
ylabel(cb,'Absolute error','Interpreter',ternary(opt.UseLaTeX,'latex','tex'),'FontSize',opt.FontCbar);
set(gca,'XMinorTick','on','YMinorTick','on');
save_figure(figs.f1, fullfile(outdir,'cmp_solution_ref_error'), opt.DPI);

figs.f2 = figure('Units','inches','Position',[0.5 0.5 9 3.8]);
imagesc(xNr,yNr,Res.'); axis image xy; colormap(opt.CMapMag);
% if isempty(opt.L2res), ttl='Pointwise residual $\Delta u_h - S$';
% else, ttl=sprintf('Pointwise residual $\\Delta u_h - S$, $\\|\\cdot\\|_{2} \\approx %.2e$', opt.L2res); end
% title(ttl,'FontSize',opt.FontTitle); 
xlabel('$x$','FontSize',opt.FontAxis); ylabel('$y$','FontSize',opt.FontAxis);
cb=colorbar('Location','eastoutside'); if opt.UseLaTeX, cb.TickLabelInterpreter='latex'; end
ylabel(cb,'Residual','Interpreter',ternary(opt.UseLaTeX,'latex','tex'),'FontSize',opt.FontCbar);
set(gca,'XMinorTick','on','YMinorTick','on');
save_figure(figs.f2, fullfile(outdir,'residual_map'), opt.DPI);

figs.f3 = figure('Units','inches','Position',[0.5 0.5 4.6 4.6]);
scatter(Centers(:,1), Centers(:,2), opt.MarkerSize, Sigma, 'filled', 'MarkerEdgeColor','k');
axis square; xlim([0 1]); ylim([0 1]); grid on; box on; colormap(opt.CMapMag);
% if opt.UseLaTeX, title('RBF Width Profile','FontSize',opt.FontTitle,'Interpreter','latex');
% else, title('RBF Width Profile','FontSize',opt.FontTitle); end
xlabel('$x$','FontSize',opt.FontAxis); ylabel('$y$','FontSize',opt.FontAxis);
cb=colorbar('Location','eastoutside'); if opt.UseLaTeX, cb.TickLabelInterpreter='latex'; end
ylabel(cb,'$\sigma$','Interpreter',ternary(opt.UseLaTeX,'latex','tex'),'FontSize',opt.FontCbar);
set(gca,'XMinorTick','on','YMinorTick','on');
save_figure(figs.f3, fullfile(outdir,'rbf_centers_sigma'), opt.DPI);
end

function save_figure(fh, basepath, dpi)
set(fh, 'InvertHardcopy','off');
pngfile = [basepath '.png']; pdffile = [basepath '.pdf'];
try
    exportgraphics(fh, pngfile, 'Resolution', dpi, 'BackgroundColor','white');
    exportgraphics(fh, pdffile, 'ContentType','vector', 'BackgroundColor','white');
catch
    set(fh,'PaperUnits','inches'); pos=get(fh,'Position');
    set(fh,'PaperPosition',[0 0 pos(3) pos(4)],'PaperSize',[pos(3) pos(4)]);
    print(fh, pngfile, '-dpng', sprintf('-r%d',dpi));
    print(fh, pdffile, '-dpdf', '-painters');
end
end

function out = ternary(cond, a, b); if cond, out=a; else, out=b; end; end
