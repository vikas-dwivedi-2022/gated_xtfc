% INVERSE_Variational_XTFC_SoftSplit_BO_safe.m
% Inverse X-TFC with soft domain split (xs, eps_scale) and Bayesian Optimization.
% Robust objective: returns a finite real scalar, with try/catch penalty.
% Requires Statistics and Machine Learning Toolbox for 'bayesopt'.

clc; clear; close all; %rng(0);
rng(4)

if exist('bayesopt','file')==0 || exist('optimizableVariable','file')==0
    error('This script requires the Statistics and Machine Learning Toolbox (bayesopt).');
end

%% -------------------- Problem + synthetic data --------------------
nu_true = 0.005;
xL = 0; xR = 1;

exact = @(x,nu) EXACT_SOLUTION(x, nu);  % stable exact

N_data    = 50;
p_data    = 3;                               
x_data    = right_clustered_linspace(xL, xR, N_data, p_data);
noise_std = 1e-2;                            
u_data    = exact(x_data, nu_true) + noise_std*randn(N_data,1);

% Boundary values and constrained part
BL = 0; BR = 1;
gfun = @(x) (1 - x).*BL + x.*BR;

%% -------------------- Layout (per block) + physics width ----------------
Nc_per_block    = 2*200;       % PDE collocation per block (total = 2*Nc_per_block)
Nstar_per_block = 2*150;       % RBF centers per block    (total = 2*Nstar_per_block)
k_width         = 1.5;       % geometric width multiplier
kappa_nu        = 5.0;       % eps_phy = kappa_nu * nu

%% -------------------- Precisions (whitening) ----------------------------
beta_data = 1 / (noise_std^2 + 1e-12);
beta_pde  = 1e2;             % tune if needed

%% -------------------- Bayesian Optimization space ----------------------
vars = [ ...
    optimizableVariable('log10nu',   [-3, 1],            'Type','real')   % nu in [1e-3, 10]
    optimizableVariable('xs',        [0.85, 0.995], 'Type','real')   % split point
    optimizableVariable('eps_scale', [10, 100],         'Type','real')   % blend scale
];

% Robust scalar objective (BO minimizes; we pass negative log-evidence)
objFcn = @(T) bo_neglog_evidence_scalar_safe(T, ...
    xL, xR, Nc_per_block, Nstar_per_block, ...
    k_width, kappa_nu, ...
    x_data, u_data, beta_data, beta_pde, BL, BR, gfun);

results = bayesopt(objFcn, vars, ...
    'IsObjectiveDeterministic', true, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 30, ...         % raise to 100–200 for higher quality
    'UseParallel', false, ...
    'Verbose', 1);


%--------------------------------------------------------------------------------
% %% --------- BO trace: best-so-far evidence and parameters ----------
% objTrace = results.ObjectiveTrace;   % bayesopt minimizes this = - log-evidence
% XTrace   = results.XTrace;           % table: log10nu, xs, eps_scale
% it       = (1:numel(objTrace)).';
% 
% % Convert to log-evidence observed at each eval
% logeObs  = -objTrace;
% 
% % Best-so-far (BSF) sequences
% bestLoge = zeros(size(logeObs));
% bestIdx  = zeros(size(logeObs));
% bestNu   = zeros(size(logeObs));
% bestXs   = zeros(size(logeObs));
% bestEps  = zeros(size(logeObs));
% 
% currBest = -Inf; currIdx = 1;
% for i = 1:numel(logeObs)
%     if logeObs(i) > currBest
%         currBest = logeObs(i);
%         currIdx  = i;
%     end
%     bestLoge(i) = currBest;
%     bestIdx(i)  = currIdx;
%     bestNu(i)   = 10.^XTrace.log10nu(currIdx);
%     bestXs(i)   = XTrace.xs(currIdx);
%     bestEps(i)  = XTrace.eps_scale(currIdx);
% end
% 
% % Report the final best-so-far point
% fprintf('[BO trace] best-so-far at iter %d: loge=%.3e, nu=%.6g, xs=%.4f, eps=%.3f\n', ...
%     bestIdx(end), bestLoge(end), bestNu(end), bestXs(end), bestEps(end));
% 
% % (Optional) save trace for later analysis
% BO_trace.iter      = it;
% BO_trace.loge_obs  = logeObs;
% BO_trace.loge_bsf  = bestLoge;
% BO_trace.nu_bsf    = bestNu;
% BO_trace.xs_bsf    = bestXs;
% BO_trace.eps_bsf   = bestEps;
% save('BO_trace.mat','BO_trace');
% 
% % --------- Plot: evidence & parameter evolution (best-so-far) ----------
% S = neurips_style();
% figT = figure('Color','w'); 
% tiledlayout(figT,2,2,'Padding','compact','TileSpacing','compact');
% 
% % (a) log-evidence
% nexttile; 
% plot(it, logeObs, ':', 'LineWidth', 1.0); hold on;
% plot(it, bestLoge, '-', 'LineWidth', 1.8);
% grid on; box on;
% xlabel('BO iteration','Interpreter','latex','FontSize',S.LabelFontSize);
% ylabel('log-evidence','Interpreter','latex','FontSize',S.LabelFontSize);
% legend({'Observed','Best so far'},'Interpreter','latex','Location','best','Box','off');
% title('Evidence trace','Interpreter','latex','FontSize',S.TitleFontSize);
% 
% % (b) nu (best-so-far)
% nexttile; 
% plot(it, log10(bestNu), '-', 'LineWidth', 1.8); grid on; box on;
% xlabel('BO iteration','Interpreter','latex','FontSize',S.LabelFontSize);
% ylabel('$log_{10}(\nu$) (best-so-far)','Interpreter','latex','FontSize',S.LabelFontSize);
% title('Best-so-far $log_{10}(\nu$)','Interpreter','latex','FontSize',S.TitleFontSize);
% 
% % (c) x_s (best-so-far)
% nexttile;
% plot(it, bestXs, '-', 'LineWidth', 1.8); grid on; box on;
% xlabel('BO iteration','Interpreter','latex','FontSize',S.LabelFontSize);
% ylabel('$x_s$ (best-so-far)','Interpreter','latex','FontSize',S.LabelFontSize);
% title('Best-so-far split','Interpreter','latex','FontSize',S.TitleFontSize);
% 
% % (d) eps_scale (best-so-far)
% nexttile;
% plot(it, bestEps, '-', 'LineWidth', 1.8); grid on; box on;
% xlabel('BO iteration','Interpreter','latex','FontSize',S.LabelFontSize);
% ylabel('$\varepsilon_{\rm scale}$ (best-so-far)','Interpreter','latex','FontSize',S.LabelFontSize);
% title('Best-so-far gate scale','Interpreter','latex','FontSize',S.TitleFontSize);
% 
% save_figure(figT, 'FIG_BO_TRACE.pdf', S.FigW*1.2, S.FigH*2.4, S.DPI);

%% --------- BO trace: best-so-far evidence and parameters ----------
objTrace = results.ObjectiveTrace;   % bayesopt minimizes this = - log-evidence
XTrace   = results.XTrace;           % table: log10nu, xs, eps_scale
it       = (1:numel(objTrace)).';

% Convert to log-evidence observed at each eval
logeObs  = -objTrace;

% Best-so-far (BSF) sequences
bestLoge = zeros(size(logeObs));
bestIdx  = zeros(size(logeObs));
bestNu   = zeros(size(logeObs));
bestXs   = zeros(size(logeObs));
bestEps  = zeros(size(logeObs));

currBest = -Inf; currIdx = 1;
for i = 1:numel(logeObs)
    if logeObs(i) > currBest
        currBest = logeObs(i);
        currIdx  = i;
    end
    bestLoge(i) = currBest;
    bestIdx(i)  = currIdx;
    bestNu(i)   = 10.^XTrace.log10nu(currIdx);
    bestXs(i)   = XTrace.xs(currIdx);
    bestEps(i)  = XTrace.eps_scale(currIdx);
end

% Final best-so-far point (for star marker)
idxStar   = bestIdx(end);
xStar     = it(idxStar);
yStarLoge = bestLoge(end);
yStarNu   = log10(bestNu(end));
yStarXs   = bestXs(end);
yStarEps  = bestEps(end);

% Report the final best-so-far point
fprintf('[BO trace] best-so-far at iter %d: loge=%.3e, log10(nu)=%.3f, xs=%.4f, eps=%.3f\n', ...
    idxStar, yStarLoge, yStarNu, yStarXs, yStarEps);

% (Optional) save trace for later analysis
BO_trace.iter      = it;
BO_trace.loge_obs  = logeObs;
BO_trace.loge_bsf  = bestLoge;
BO_trace.nu_bsf    = bestNu;
BO_trace.xs_bsf    = bestXs;
BO_trace.eps_bsf   = bestEps;
save('BO_trace.mat','BO_trace');

% --------- Plot: evidence & parameter evolution (best-so-far) ----------
S = neurips_style();
figT = figure('Color','w');
figT.Position = [1 1 S.FigW*1.8  S.FigH*1.05];  % wider, not so tall

tiledlayout(figT,2,2,'Padding','loose','TileSpacing','compact');  % more vertical room

% (a) log-evidence
nexttile; 
plot(it, logeObs, ':', 'LineWidth', 1.0); hold on;
plot(it, bestLoge, '-', 'LineWidth', 1.8);

plot(xStar, yStarLoge, 'rp', 'MarkerSize', 11, 'LineWidth', 1.2, ...
     'MarkerFaceColor','r', 'MarkerEdgeColor','k');   % pentagram
grid on; box on;
xlabel('BO iteration','Interpreter','latex','FontSize',S.LabelFontSize);
ylabel('log-evidence','Interpreter','latex','FontSize',S.LabelFontSize);
legend({'Observed','Best so far','Final best'},'Interpreter','latex','Location','best','Box','off');
title('Evidence trace','Interpreter','latex','FontSize',S.TitleFontSize);

% (b) log10(nu) (best-so-far)
nexttile; 
plot(it, log10(bestNu), '-', 'LineWidth', 1.8); hold on;
% plot(xStar, yStarNu, 'k*', 'MarkerSize', 10, 'LineWidth', 1.2);
plot(xStar, yStarNu, 'rp', 'MarkerSize', 11, 'LineWidth', 1.2, ...
     'MarkerFaceColor','r', 'MarkerEdgeColor','k');   % pentagram
grid on; box on;
xlabel('BO iteration','Interpreter','latex','FontSize',S.LabelFontSize);
ylabel('$\log_{10}\nu$ (best-so-far)','Interpreter','latex','FontSize',S.LabelFontSize);
title('Best-so-far $\log_{10}\nu$','Interpreter','latex','FontSize',S.TitleFontSize);

% (c) x_s (best-so-far)
nexttile;
plot(it, bestXs, '-', 'LineWidth', 1.8); hold on;
% plot(xStar, yStarXs, 'k*', 'MarkerSize', 10, 'LineWidth', 1.2);
plot(xStar, yStarXs, 'rp', 'MarkerSize', 11, 'LineWidth', 1.2, ...
     'MarkerFaceColor','r', 'MarkerEdgeColor','k');   % pentagram
grid on; box on;
xlabel('BO iteration','Interpreter','latex','FontSize',S.LabelFontSize);
ylabel('$x_s$ (best-so-far)','Interpreter','latex','FontSize',S.LabelFontSize);
title('Best-so-far split','Interpreter','latex','FontSize',S.TitleFontSize);

% (d) eps_scale (best-so-far)
nexttile;
plot(it, bestEps, '-', 'LineWidth', 1.8); hold on;
% plot(xStar, yStarEps, 'k*', 'MarkerSize', 10, 'LineWidth', 1.2);
plot(xStar, yStarEps, 'rp', 'MarkerSize', 11, 'LineWidth', 1.2, ...
     'MarkerFaceColor','r', 'MarkerEdgeColor','k');   % pentagram

grid on; box on;
xlabel('BO iteration','Interpreter','latex','FontSize',S.LabelFontSize);
ylabel('$\varepsilon_{\rm scale}$ (best-so-far)','Interpreter','latex','FontSize',S.LabelFontSize);
title('Best-so-far gate scale','Interpreter','latex','FontSize',S.TitleFontSize);

% Taller figure so bottom x-labels fit; higher DPI for clarity
save_figure(figT, 'FIG_BO_TRACE.pdf', S.FigW*1.8, S.FigH*1.05, 400);


%--------------------------------------------------------------------------------


best   = results.XAtMinObjective;
nu_hat = 10.^best.log10nu;
xs_hat = best.xs;
eps_hat= best.eps_scale;

% Rebuild at BO optimum; get eta_hat & log-evidence
[Phi_hat, y_hat, basis_opt] = build_whitened_xtfc_soft( ...
    nu_hat, xs_hat, eps_hat, ...
    xL, xR, Nc_per_block, Nstar_per_block, ...
    k_width, kappa_nu, ...
    x_data, u_data, beta_data, beta_pde, BL, BR, gfun);

eta_bounds = [1e-12, 1e-2]; eta0 = 1e-7;
[eta_hat, ~, ~, loge_hat] = auto_eta_for_Phi(Phi_hat, y_hat, eta0, eta_bounds);

fprintf('\n[BO] nu_hat=%.6g, xs_hat=%.4f, eps_scale_hat=%.3f, loge=%.3e, eta_hat=%.3e\n', ...
    nu_hat, xs_hat, eps_hat, loge_hat, eta_hat);

%% -------------------- Posterior + predictions ---------------------------
M  = size(Phi_hat,2);
A  = eta_hat*eye(M) + (Phi_hat.'*Phi_hat);
A  = 0.5*(A + A.');  % symmetrize
[R, ok, ~] = chol_spd_with_jitter(A);
if ~ok, error('Cholesky failed even after jitter.'); end
mN = R \ (R.' \ (Phi_hat.'*y_hat));

N_pred = 500; x_pred = linspace(xL, xR, N_pred).';
H_pred = build_rows_block(x_pred, basis_opt.psi_rows);
u_mean = gfun(x_pred) + H_pred*mN;

std_pred = zeros(N_pred,1);
for k = 1:N_pred
    h  = H_pred(k,:).';
    y1 = R.' \ h;
    z1 = R   \ y1;
    std_pred(k) = sqrt(max(0, h.'*z1));
end

u_exact_true = exact(x_pred, nu_true);
u_exact_hat  = exact(x_pred, nu_hat);

plot_solution_neurips_soft( ...
    x_pred, u_mean, std_pred, ...
    x_data, u_data, ...
    u_exact_true, u_exact_hat, ...
    nu_true, nu_hat, xs_hat, eps_hat, ...
    'FIG_INV_GATED_X-TFC');

%% ===================== BO objective (robust scalar) =====================
function f = bo_neglog_evidence_scalar_safe(T, ...
        xL, xR, Nc_pb, Ns_pb, k_width, kappa_nu, ...
        x_data, u_data, beta_data, beta_pde, BL, BR, gfun)
    % Always return a finite real scalar. On any failure, return a big penalty.
    PENALTY = 1e12;
    try
        log10nu   = T.log10nu;
        xs        = T.xs;
        eps_scale = T.eps_scale;
        nu        = 10.^log10nu;

        [Phi, y] = build_whitened_xtfc_soft( ...
            nu, xs, eps_scale, ...
            xL, xR, Nc_pb, Ns_pb, ...
            k_width, kappa_nu, ...
            x_data, u_data, beta_data, beta_pde, BL, BR, gfun);

        % Auto-tune eta and compute log-evidence
        eta_bounds = [1e-12, 1e-2]; eta0 = 1e-7;
        [~, ~, ~, loge] = auto_eta_for_Phi(Phi, y, eta0, eta_bounds);

        f = -loge;                                   % we minimize negative log-evidence
        if ~isfinite(f) || ~isreal(f) || ~isscalar(f)
            f = PENALTY;
        end
    catch
        f = PENALTY;
    end
end

%% ===================== Builders: soft-split XTFC blocks ==================
function [Phi, y, basis] = build_whitened_xtfc_soft( ...
        nu, xs, eps_scale, ...
        xL, xR, Nc_pb, Ns_pb, k_width, kappa_nu, ...
        x_data, u_data, beta_data, beta_pde, BL, BR, gfun)

    [X_pde, alpha_star, sigma_x] = PDE_and_Kernel_Centers( ...
        xL, xR, xs, Nc_pb, Ns_pb, k_width, eps_scale, nu, kappa_nu);

    % Basis params
    m  = 1./(sqrt(2)*sigma_x);
    b  = -m.*alpha_star;
    Ai = exp( - (b     ).^2 );
    Bi = exp( - (m + b ).^2 );

    % Row builders
    psi_rows = @(x) exp( - (m'.*x + b').^2 ) ...
                    - (1 - x)*Ai' - x*Bi';
    pde_rows_xtfc = @(x,nu_) ...
        (Ai' - Bi') + ...
        exp( - (m'.*x + b').^2 ) .* ( ...
           -2*(m'.*(m'.*x + b')) ...
           - nu_*(m'.^2).*(4*(m'.*x + b').^2 - 2) );

    % Data block
    H_data = build_rows_block(x_data, psi_rows);
    y_data = u_data - gfun(x_data);

    % PDE residual block
    LHS_PDE = build_rows_block(X_pde, @(x) pde_rows_xtfc(x,nu));
    f_g     = (BR - BL); % = 1
    y_pde   = - f_g * ones(size(LHS_PDE,1),1);

    % Whitening
    Phi = [ sqrt(beta_data)*H_data;
            sqrt(beta_pde )*LHS_PDE ];
    y   = [ sqrt(beta_data)*y_data;
            sqrt(beta_pde )*y_pde   ];

    basis.psi_rows      = psi_rows;
    basis.pde_rows_xtfc = @(x) pde_rows_xtfc(x,nu);
end

function [x_pde, alpha_star, sigma_x] = PDE_and_Kernel_Centers( ...
        xL, xR, xs, Nc, Nstar, k, eps_scale, nu, kappa_nu)
    assert(xL < xs && xs < xR, 'Require xL < xs < xR');
    assert(Nc > 0 && Nstar > 0, 'Nc and Nstar must be positive');

    % Collocation (PDE)
    xg = linspace(xL, xs, Nc+1)'; xg(end)=[];
    xl = linspace(xs, xR, Nc)'; 
    x_pde = [xg; xl];            % total = 2*Nc

    % RBF centers
    ag = linspace(xL, xs, Nstar+1)'; ag(end)=[];
    al = linspace(xs, xR, Nstar)'; 
    alpha_star = [ag; al];       % total = 2*Nstar

    % Block spacings
    dL = (xs - xL) / max(Nstar,1);
    dR = (xR - xs) / max(Nstar,1);
    sigma_L = k*dL; sigma_R = k*dR;

    % Transition scale: geometry vs physics
    eps_geo = eps_scale * min(dL, dR);
    eps_phy = kappa_nu * nu;
    epsb    = max(eps_geo, eps_phy);

    t = (alpha_star - xs) / epsb;
    s = 1 ./ (1 + exp(-t));

    sigma_x = sigma_L * (1 - s) + sigma_R * s;

    % Optional floor for conditioning:
    % sigma_min = 0.25*min(dL,dR);
    % sigma_x   = max(sigma_x, sigma_min);
end

%% -------------------- Core inference helpers ----------------------------
function [eta_hat, mN, R, loge] = auto_eta_for_Phi(Phi, y, eta0, bounds)
    eta_min = bounds(1); eta_max = bounds(2);
    eta = min(max(eta0, eta_min), eta_max);

    [~,S,~] = svd(Phi, 'econ'); s2 = diag(S).^2;
    M = size(Phi,2); N = size(Phi,1); 

    maxit = 50; tol = 1e-6; damp = 0.3; ok_last = true;

    for it = 1:maxit
        A = eta*eye(M) + (Phi.'*Phi); A = 0.5*(A + A.');
        [R, ok, ~] = chol_spd_with_jitter(A);
        if ~ok
            eta = min(10*max(eta, 1e-12), eta_max);
            ok_last = false; continue;
        end
        ok_last = true;

        rhs = Phi.'*y; mN = R \ (R.' \ rhs);
        gamma   = sum( s2 ./ (s2 + eta) );
        eta_new = gamma / max(mN.'*mN, 1e-30);
        eta_new = min(max(eta_new, eta_min), eta_max);
        eta_upd = (1-damp)*eta + damp*eta_new;

        if abs(eta_upd - eta) <= tol*(eta + 1e-12)
            eta = eta_upd; break;
        end
        eta = eta_upd;
    end

    if ~ok_last
        [eta, mN, R, ~] = refine_eta_grid(Phi, y, min(max(eta, 1e-9), 1e-3), 10, 9);
    end

    A = eta*eye(M) + (Phi.'*Phi); A = 0.5*(A + A.');
    [R, ok, ~] = chol_spd_with_jitter(A);
    if ~ok
        eta = min(max(eta*10, eta_min), eta_max);
        A = eta*eye(M) + (Phi.'*Phi); A = 0.5*(A + A.');
        [R, ok, ~] = chol_spd_with_jitter(A);
        if ~ok, error('Failed to stabilize A for evidence.'); end
    end
    rhs = Phi.'*y; mN = R \ (R.' \ rhs);
    r   = y - Phi*mN;
    E   = 0.5*( r.'*r + eta*(mN.'*mN) );
    logdetA = 2*sum(log(diag(R)));
    loge = (M/2)*log(eta) - E - 0.5*logdetA - (N/2)*log(2*pi);

    eta_hat = eta;
end

function [eta_best, mN_best, R_best, loge_best] = refine_eta_grid(Phi, y, eta_center, span, npts)
    M = size(Phi,2);
    grid = logspace(log10(eta_center/span), log10(eta_center*span), npts);
    loge_best = -inf; eta_best = grid(1);
    mN_best = []; R_best = [];
    for e = grid
        A = e*eye(M) + (Phi.'*Phi); A = 0.5*(A + A.');
        [R, ok, ~] = chol_spd_with_jitter(A);
        if ~ok, continue; end
        rhs = Phi.'*y; mN = R \ (R.' \ rhs);
        r = y - Phi*mN;
        E = 0.5*( r.'*r + e*(mN.'*mN) );
        logdetA = 2*sum(log(diag(R)));
        loge = (M/2)*log(e) - E - 0.5*logdetA - (N/2)*log(2*pi);
        if loge > loge_best
            loge_best = loge; eta_best = e; mN_best = mN; R_best = R;
        end
    end
end

function [R, ok, ridge] = chol_spd_with_jitter(A)
    [R,p] = chol(A);
    if p==0, ok=true; ridge=0; return; end
    ok=false;
    ridge = max(1e-12, eps(norm(A,1)));
    I = eye(size(A));
    for k = 1:8
        [R,p] = chol(A + ridge*I);
        if p==0, ok=true; return; end
        ridge = ridge*10;
    end
    R = [];
end

%% -------------------- Utilities (grids, rows, exact, plotting) ----------
function x = right_clustered_linspace(xL,xR,N,p)
    t = linspace(0,1,N).';
    x = xL + (xR - xL) * (1 - (1 - t).^p);
end

function H = build_rows_block(xvec, rowfun)
    n = numel(xvec);
    r = rowfun(xvec(1));
    H = zeros(n, numel(r));
    H(1,:) = r;
    for k = 2:n
        H(k,:) = rowfun(xvec(k));
    end
end

function u_exact = EXACT_SOLUTION(X, nu)
    overflow_threshold = 1 / log(realmax('double'));
    if nu > overflow_threshold
        u_exact = expm1(X ./ nu) ./ expm1(1 / nu);
    else
        exponent = (X - 1) ./ nu;
        threshold = -log(eps(class(X)));
        u_exact = exp(exponent);
        u_exact(exponent < -threshold) = 0;
        u_exact(X == 1) = 1;
    end
    if any(abs(X(:) - 1) < eps(class(X)) & X(:) ~= 1)
        u_exact(abs(X - 1) < eps(class(X))) = 1;
    end
end

function plot_solution_neurips_soft(x_pred, u_mean, std_model, ...
                               x_data, u_data, ...
                               u_exact_true, u_exact_hat, ...
                               nu_true, nu_hat, xs_hat, eps_hat, save_prefix)
    S = neurips_style();
    fig = figure('Color','w'); hold on;

    % 95% model band
    xb = [x_pred; flipud(x_pred)];
    yb = [u_mean + 2*std_model; flipud(u_mean - 2*std_model)];
    patch('XData', xb, 'YData', yb, ...
          'FaceColor', [0.85 0.88 1.00], 'EdgeColor', 'none');

    plot(x_pred, u_mean, 'r-', 'LineWidth', S.LineWidth+0.5);
    plot(x_pred, u_exact_true, 'k--', 'LineWidth', S.LineWidth);
    scatter(x_data, u_data, S.ScatterSize, 'filled','MarkerFaceColor', 'b');

    xlabel('$x$', 'FontSize', S.LabelFontSize);
    ylabel('$u(x)$', 'FontSize', S.LabelFontSize);
    title(sprintf(['Gated X-TFC: $\\hat{\\nu}=%.3g$, $x_s^*=%.3f$, ', ...
                   '$\\epsilon_{\\rm scale}^*=%.2f$'], ...
          nu_hat, xs_hat, eps_hat), 'FontSize', S.TitleFontSize);
    grid on; box on;
    set(gca, 'FontSize', S.AxisFontSize, 'LineWidth', S.AxesLineWidth, ...
             'XMinorGrid','on','YMinorGrid','on');
    lgd = legend({'$95\%$ band (model)', ...
                  'Predicted mean', ...
                  sprintf('Exact (true $\\nu$ = %.3g)', nu_true), ...
                  'Data'}, ...
        'FontSize', S.LegendFontSize, 'Location','best');
    set(lgd,'Interpreter','latex');

    save_figure(fig, [save_prefix '.pdf'], S.FigW, S.FigH, S.DPI);
end

function S = neurips_style()
    set(groot,'defaultTextInterpreter','latex');
    set(groot,'defaultAxesTickLabelInterpreter','latex');
    set(groot,'defaultLegendInterpreter','latex');

    S.FigW = 2*3.5;   % inches
    S.FigH = 2*2.6;

    S.AxisFontSize   = 12;
    S.LabelFontSize  = 14;
    S.TitleFontSize  = 14;
    S.LegendFontSize = 12;
    S.LineWidth      = 1.8;
    S.AxesLineWidth  = 1.0;
    S.MarkerSize     = 7;
    S.ScatterSize    = 24;

    S.DPI = 300;
end

function save_figure(fig, filename, W, H, dpi)
    set(fig, 'Units','inches'); pos = get(fig,'Position'); pos(3)=W; pos(4)=H; set(fig,'Position',pos);
    set(fig, 'PaperUnits','inches','PaperPosition',[0 0 W H], 'PaperSize',[W H]);
    try
        exportgraphics(fig, filename, 'Resolution', dpi);
    catch
        [~,~,ext] = fileparts(filename);
        if strcmpi(ext,'.pdf')
            print(fig, filename, '-dpdf', sprintf('-r%d',dpi));
        else
            print(fig, filename, '-dpng', sprintf('-r%d',dpi));
        end
    end
end
