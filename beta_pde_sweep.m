% INVERSE_betaPDE_SWEEP_SUPERPOSE.m
% Sweeps beta_pde = [10, 100, 500, 1000] in your inverse BO code and
% plots ONLY the final posterior mean solutions superposed (with zoom inset).
%
% Notes:
% - Uses the same BO settings (30 evals, EI+), but loops over beta_pde.
% - Keeps the same random seed for fair comparison across beta_pde.
% - Stores (nu_hat, xs_hat, eps_hat) per sweep value and plots u_mean(x).
% - Does NOT plot residuals, gates, or BO traces.

clc; clear; close all;
warning('off','all');

if exist('bayesopt','file')==0 || exist('optimizableVariable','file')==0
    error('This script requires the Statistics and Machine Learning Toolbox (bayesopt).');
end

tic;

%% -------------------- Sweep list --------------------
beta_pde_list = [10, 100, 500, 1000];

%% -------------------- Fixed problem + synthetic data (keep same across sweep) --------------------
rng(4);                          % keep fixed for reproducibility
nu_true = 0.005;
xL = 0; xR = 1;

exact = @(x,nu) EXACT_SOLUTION(x, nu);

N_data    = 50;
p_data    = 3;
x_data    = right_clustered_linspace(xL, xR, N_data, p_data);
noise_std = 1e-2;
u_data    = exact(x_data, nu_true) + noise_std*randn(N_data,1);

BL = 0; BR = 1;
gfun = @(x) (1 - x).*BL + x.*BR;

%% -------------------- Layout + physics width ----------------
Nc_per_block    = 2*200;        % PDE collocation per block (total = 2*Nc_per_block)
Nstar_per_block = 2*150;        % RBF centers per block    (total = 2*Nstar_per_block)
k_width         = 1.5;          % geometric width multiplier
kappa_nu        = 5.0;          % eps_phy = kappa_nu * nu

%% -------------------- Data precision (fixed) ----------------------------
beta_data = 1 / (noise_std^2 + 1e-12);

%% -------------------- BO search space (same) ----------------------------
vars = [ ...
    optimizableVariable('log10nu',   [-3, 1],        'Type','real')
    optimizableVariable('xs',        [0.85, 0.995],  'Type','real')
    optimizableVariable('eps_scale', [10, 100],      'Type','real')
];

%% -------------------- Prediction grid for superposed plot ----------------
N_pred = 800;
x_pred = linspace(xL, xR, N_pred).';

Umeans = zeros(numel(beta_pde_list), N_pred);
nu_hat_list  = zeros(size(beta_pde_list));
xs_hat_list  = zeros(size(beta_pde_list));
eps_hat_list = zeros(size(beta_pde_list));

fprintf('\n=== beta_pde sweep (inverse): superposed posterior mean solutions ===\n');

%% ==================== Sweep loop ====================
for ib = 1:numel(beta_pde_list)
    beta_pde = beta_pde_list(ib);

    fprintf('\n--- Running beta_pde = %g ---\n', beta_pde);

    % For fair comparison, reset RNG before each BO run (optional but recommended)
    rng(4);

    % Robust scalar objective (BO minimizes negative log-evidence)
    objFcn = @(T) bo_neglog_evidence_scalar_safe(T, ...
        xL, xR, Nc_per_block, Nstar_per_block, ...
        k_width, kappa_nu, ...
        x_data, u_data, beta_data, beta_pde, BL, BR, gfun);

    results = bayesopt(objFcn, vars, ...
        'IsObjectiveDeterministic', true, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'MaxObjectiveEvaluations', 30, ...
        'UseParallel', false, ...
        'Verbose', 1);

    best   = results.XAtMinObjective;
    nu_hat = 10.^best.log10nu;
    xs_hat = best.xs;
    eps_hat= best.eps_scale;

    nu_hat_list(ib)  = nu_hat;
    xs_hat_list(ib)  = xs_hat;
    eps_hat_list(ib) = eps_hat;

    % Rebuild at BO optimum and compute posterior mean coefficients
    [Phi_hat, y_hat, basis_opt] = build_whitened_xtfc_soft( ...
        nu_hat, xs_hat, eps_hat, ...
        xL, xR, Nc_per_block, Nstar_per_block, ...
        k_width, kappa_nu, ...
        x_data, u_data, beta_data, beta_pde, BL, BR, gfun);

    eta_bounds = [1e-12, 1e-2]; eta0 = 1e-7;
    [eta_hat, ~, Rchol, loge_hat] = auto_eta_for_Phi(Phi_hat, y_hat, eta0, eta_bounds);

    % Posterior mean
    mN = Rchol \ (Rchol.' \ (Phi_hat.'*y_hat));

    % Predicted posterior mean solution (ONLY)
    H_pred = build_rows_block(x_pred, basis_opt.psi_rows);
    u_mean = gfun(x_pred) + H_pred*mN;

    Umeans(ib,:) = u_mean(:).';

    fprintf('[beta_pde=%g] nu_hat=%.6g, xs_hat=%.4f, eps_hat=%.3f, eta_hat=%.3e, loge=%.3e\n', ...
        beta_pde, nu_hat, xs_hat, eps_hat, eta_hat, loge_hat);
end

%% ==================== Superposed plot ONLY (with zoom inset) ====================
fig = figure('Color','w');
ax1 = axes(fig); hold(ax1,'on');

LW = 2.2;

for ib = 1:numel(beta_pde_list)
    plot(ax1, x_pred, Umeans(ib,:), 'LineWidth', LW, ...
        'DisplayName', sprintf('\\beta_{\\rm pde}=%g, \\hat\\nu=%.3g', beta_pde_list(ib), nu_hat_list(ib)));
end

grid(ax1,'on'); box(ax1,'on');

FS_ax  = 20;
FS_lab = 24;
FS_tit = 24;
FS_leg = 16;

set(ax1, 'FontSize', FS_ax, 'LineWidth', 1.2);
xlabel(ax1, '$x$', 'Interpreter','latex', 'FontSize', FS_lab);
ylabel(ax1, '$u_{\rm mean}(x)$', 'Interpreter','latex', 'FontSize', FS_lab);

title(ax1, sprintf('Inverse Gated X--TFC: $\\beta_{\\rm pde}$ sweep ($\\nu_{\\rm true}=%.3g$)', nu_true), ...
    'Interpreter','latex', 'FontSize', FS_tit);

leg = legend(ax1, 'Interpreter','latex', 'Location','best', 'Box','off');
set(leg, 'FontSize', FS_leg);
xlim(ax1, [xL xR]);

%% --------- Zoom inset near x=1 ----------
x1 = 0.9; x2 = 1.0;
ix = (x_pred >= x1) & (x_pred <= x2);
ymin = min(Umeans(:,ix), [], 'all');
ymax = max(Umeans(:,ix), [], 'all');
pad  = 0.03 * max(1e-12, (ymax - ymin));
yl1  = ymin - pad;
yl2  = ymax + pad;

rectangle(ax1, 'Position', [x1, yl1, (x2-x1), (yl2-yl1)], ...
          'LineWidth', 1.2, 'LineStyle','--');

ax2 = axes('Position', [0.58 0.20 0.33 0.30]);
hold(ax2,'on'); box(ax2,'on'); grid(ax2,'on');

for ib = 1:numel(beta_pde_list)
    plot(ax2, x_pred, Umeans(ib,:), 'LineWidth', LW);
end

set(ax2, 'FontSize', 16, 'LineWidth', 1.1);
xlim(ax2, [x1 x2]);
ylim(ax2, [yl1 yl2]);
title(ax2, 'Zoom: $x\in[0.9,1]$', 'Interpreter','latex', 'FontSize', 16);

exportgraphics(fig, sprintf('FIG_INV_betaPDE_SWEEP_nuTrue_%g.pdf', nu_true), 'ContentType','vector');

toc;

%% ===================== BO objective (robust scalar) =====================
function f = bo_neglog_evidence_scalar_safe(T, ...
        xL, xR, Nc_pb, Ns_pb, k_width, kappa_nu, ...
        x_data, u_data, beta_data, beta_pde, BL, BR, gfun)
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

        eta_bounds = [1e-12, 1e-2]; eta0 = 1e-7;
        [~, ~, ~, loge] = auto_eta_for_Phi(Phi, y, eta0, eta_bounds);

        f = -loge;
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

    m  = 1./(sqrt(2)*sigma_x);
    b  = -m.*alpha_star;
    Ai = exp( - (b     ).^2 );
    Bi = exp( - (m + b ).^2 );

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
    f_g     = (BR - BL);
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

    xg = linspace(xL, xs, Nc+1)'; xg(end)=[];
    xl = linspace(xs, xR, Nc)'; 
    x_pde = [xg; xl];

    ag = linspace(xL, xs, Nstar+1)'; ag(end)=[];
    al = linspace(xs, xR, Nstar)'; 
    alpha_star = [ag; al];

    dL = (xs - xL) / max(Nstar,1);
    dR = (xR - xs) / max(Nstar,1);
    sigma_L = k*dL; sigma_R = k*dR;

    eps_geo = eps_scale * min(dL, dR);
    eps_phy = kappa_nu * nu;
    epsb    = max(eps_geo, eps_phy);

    t = (alpha_star - xs) / epsb;
    s = 1 ./ (1 + exp(-t));
    sigma_x = sigma_L * (1 - s) + sigma_R * s;
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

%% -------------------- Utilities --------------------
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
