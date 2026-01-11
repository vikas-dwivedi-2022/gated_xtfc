% META_COMPARE_FMINBND_ONLY.m
% Compare two strategies to tune (xs, eps_scale) for a fixed nu:
%   (A) Global nested fminbnd  on [0.8,0.999] x [10,100]
%   (B) Meta-bounded nested fminbnd using bounds predicted from meta models
%
% Requires: meta_gated_xtfc_model.mat (with model_xs, model_eps)

clc; clear; close all; warning('off','all');

%% -------------------- Load meta models & bind features -------------------
S = load('meta_gated_xtfc_model.mat');     % contains model_xs, model_eps
model_xs  = S.model_xs;
model_eps = S.model_eps;
% Rebind feature builders so they don't depend on external files
model_xs.Phi_fun  = @(xq) poly_features_local(xq, model_xs.deg);
model_eps.Phi_fun = @(xq) poly_features_local(xq, model_eps.deg);

%% -------------------- Choose operator & get meta bounds ------------------
nu  = 6.5e-4;                 % pick your test diffusion
xq  = log10(nu);
conf = 0.95;

[xs_mu, xs_lb_m, xs_ub_m] = predict_bounds_xs_local (model_xs,  xq, conf);
[e_mu,  e_lb_m,  e_ub_m ] = predict_bounds_eps_local(model_eps, xq, conf);

% Global safety box
XS_MIN=0.80; XS_MAX=0.999; E_MIN=10; E_MAX=100;
xs_lb_m = max(XS_MIN, xs_lb_m);  xs_ub_m = min(XS_MAX, xs_ub_m);
e_lb_m  = max(E_MIN,  e_lb_m );  e_ub_m  = min(E_MAX,  e_ub_m );

% If meta band is too tight/collapsed, pad a bit
if xs_ub_m - xs_lb_m < 1e-3
    c = 0.5*(xs_lb_m + xs_ub_m); w = 5e-3;
    xs_lb_m = max(XS_MIN, c - w); xs_ub_m = min(XS_MAX, c + w);
end
if e_ub_m - e_lb_m < 1e-2
    c = 0.5*(e_lb_m + e_ub_m); w = 0.5;
    e_lb_m = max(E_MIN, c - w); e_ub_m = min(E_MAX, c + w);
end

fprintf('Meta bounds at nu=%.3e:\n  xs in [%.4f, %.4f], eps in [%.2f, %.2f]\n', ...
        nu, xs_lb_m, xs_ub_m, e_lb_m, e_ub_m);

%% -------------------- PDE setup & objective (shared) ---------------------
% Domain & BCs
xL=0; xR=1; BL=0; BR=1;
% Layout
Nc=1000; Nstar=1000; k=1.5;

% Objective (validation residual energy)
Jfun = @(xs,eps) objective(xs, eps, xL, xR, Nc, Nstar, k, nu, BL, BR);

% fminbnd options (same tolerances)
opts_xs = optimset('TolX',1e-4,'Display','off');
opts_e  = optimset('TolX',1,'Display','off');

%% -------------------- Traces (globals) -----------------------------------
global TRACE_G BEST_G TRACE_MF BEST_MF
TRACE_G=[]; BEST_G=[];
TRACE_MF=[]; BEST_MF=[];

%% -------------------- (A) Global nested fminbnd --------------------------
tic;
outerG = @(xs) inner_min_over_eps_logged(xs, 10, 100, @Jfun_logG, opts_e, Jfun);
[xs_opt_G, ~, ~] = fminbnd(outerG, 0.80, 0.999, opts_xs);
[eps_opt_G, ~]   = inner_argmin_eps_logged(xs_opt_G, 10, 100, @Jfun_logG, opts_e, Jfun);
tG = toc;
JG = Jfun(xs_opt_G, eps_opt_G);
fprintf('(Global) xs=%.4f  eps=%.2f  J=%.3e  time=%.3fs  evals=%d\n', ...
        xs_opt_G, eps_opt_G, JG, tG, numel(TRACE_G));

%% -------------------- (B) Meta-bounded nested fminbnd --------------------
tic;
outerM = @(xs) inner_min_over_eps_logged(xs, e_lb_m, e_ub_m, @Jfun_logM, opts_e, Jfun);
[xs_opt_M, ~, ~] = fminbnd(outerM, xs_lb_m, xs_ub_m, opts_xs);
[eps_opt_M, ~]   = inner_argmin_eps_logged(xs_opt_M, e_lb_m, e_ub_m, @Jfun_logM, opts_e, Jfun);
tM = toc;
JM = Jfun(xs_opt_M, eps_opt_M);
fprintf('(Meta-bounded) xs=%.4f  eps=%.2f  J=%.3e  time=%.3fs  evals=%d\n', ...
        xs_opt_M, eps_opt_M, JM, tM, numel(TRACE_MF));

%% -------------------- Build best-so-far traces ---------------------------
bsfG  = cummin_vec(TRACE_G);
bsfM  = cummin_vec(TRACE_MF);

%% -------------------- Plot (publication quality) ------------------------
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');

fig = figure('Color','w'); fig.Units='inches'; fig.Position=[1 1 8.2 4.6];
tiledlayout(fig,1,1);
ax = nexttile; hold(ax,'on'); grid(ax,'on'); box(ax,'on');

% Observed vs Best-so-far (log10 J)
plot(ax, 1:numel(TRACE_G),  log10(max(TRACE_G,1e-300)),  ':', 'LineWidth',2.0, 'DisplayName','Observed (global)');
plot(ax, 1:numel(bsfG),     log10(max(bsfG, 1e-300)),    '-', 'LineWidth',3.0, 'DisplayName',sprintf('Best-so-far (global) [%.2fs]', tG));

plot(ax, 1:numel(TRACE_MF), log10(max(TRACE_MF,1e-300)), ':', 'LineWidth',2.0, 'DisplayName','Observed (meta)');
plot(ax, 1:numel(bsfM),     log10(max(bsfM, 1e-300)),    '-', 'LineWidth',3.0, 'DisplayName',sprintf('Best-so-far (meta) [%.2fs]', tM));

% Mark final best points
plot(ax, numel(bsfG), log10(max(bsfG(end),1e-300)), 'kp', 'MarkerFaceColor','k', 'MarkerSize',10, 'DisplayName','Optimum (global)');
plot(ax, numel(bsfM), log10(max(bsfM(end),1e-300)), 'rp', 'MarkerFaceColor','r', 'MarkerSize',10, 'DisplayName','Optimum (meta)');

pause(1)
xlabel(ax,'Objective evaluations','FontSize',14);
ylabel(ax,'$\log_{10} J(x_s,\varepsilon)$','FontSize',14);
title(ax, sprintf('fminbnd at $\\nu=%.2g$ : global vs meta bounds', nu), 'FontSize',14);
legend(ax,'Location','northeast','Box','off');

exportgraphics(fig,'FIG_TRACES_FMINBND_GLOBAL_vs_META.pdf','Resolution',300);

%% -------------------- Summary -------------------------------------------
fprintf('\n== Summary @ nu=%.3e ==\n', nu);
fprintf('Global:       xs=%.4f  eps=%.2f  J=%.3e  time=%.3fs  evals=%d\n', xs_opt_G,  eps_opt_G,  JG,  tG, numel(TRACE_G));
fprintf('Meta-bounded: xs=%.4f  eps=%.2f  J=%.3e  time=%.3fs  evals=%d\n', xs_opt_M, eps_opt_M, JM,  tM, numel(TRACE_MF));

%% ============================ Helpers ====================================
function Phi = poly_features_local(x, deg)
    x = x(:); Phi = ones(numel(x), deg+1);
    for d=1:deg, Phi(:,d+1) = x.^d; end
end

function c = cummin_vec(v)
    c = zeros(size(v)); m = inf;
    for i=1:numel(v), m = min(m, v(i)); c(i)=m; end
end

% --- Meta prediction (xs) with intervals ---
function [mu_xs, lb_xs, ub_xs] = predict_bounds_xs_local(model, xq, conf)
    if nargin<3, conf=0.95; end
    Phi_q   = model.Phi_fun(xq);
    mu_y    = Phi_q * model.beta;                 % latent (logit)
    var_mod = Phi_q * model.cov_beta * Phi_q.';
    var_n   = model.s2_fun(xq);
    var_tot = max(var_mod + var_n, 1e-12);
    z = sqrt(2) * erfinv(2*(0.5+conf/2)-1);
    zhat  = 1./(1+exp(-mu_y));
    mu_xs = model.a_xs + (model.b_xs - model.a_xs)*zhat;
    dxdY  = (model.b_xs - model.a_xs)*zhat.*(1-zhat);
    sY    = sqrt(var_tot); sX = abs(dxdY).*sY;
    lb_xs = max(model.a_xs, mu_xs - z*sX);
    ub_xs = min(model.b_xs, mu_xs + z*sX);
end

% --- Meta prediction (eps) with intervals ---
function [mu_eps, lb_eps, ub_eps] = predict_bounds_eps_local(model, xq, conf)
    if nargin<3, conf=0.95; end
    Phi_q   = model.Phi_fun(xq);
    mu_y    = Phi_q * model.beta;                 % latent (log10 eps)
    var_mod = Phi_q * model.cov_beta * Phi_q.';
    var_n   = model.s2_fun(xq);
    var_tot = max(var_mod + var_n, 1e-12);
    z = sqrt(2) * erfinv(2*(0.5+conf/2)-1);
    mu_eps = 10.^mu_y;
    sY     = sqrt(var_tot);
    sE     = log(10) * mu_eps .* sY;
    lb_eps = max(10,  mu_eps - z*sE);
    ub_eps = min(100, mu_eps + z*sE);
end

% --- fminbnd wrappers that log every evaluation ---
function val = inner_min_over_eps_logged(xs, e_lb, e_ub, Jlogger, opts_e, Jcore)
    obj_e = @(e) Jlogger(xs, e, Jcore);
    [~, val] = fminbnd(obj_e, e_lb, e_ub, opts_e);
end

function [e_star, fval] = inner_argmin_eps_logged(xs, e_lb, e_ub, Jlogger, opts_e, Jcore)
    obj_e = @(e) Jlogger(xs, e, Jcore);
    [e_star, fval] = fminbnd(obj_e, e_lb, e_ub, opts_e);
end

function J = Jfun_logG(xs, eps, Jcore)
    % Global logger
    global TRACE_G BEST_G
    J = Jcore(xs, eps);
    TRACE_G(end+1,1) = J;
    if isempty(BEST_G), BEST_G = J; else, BEST_G(end+1,1) = min(BEST_G(end), J); end
end

function J = Jfun_logM(xs, eps, Jcore)
    % Meta-bounded logger
    global TRACE_MF BEST_MF
    J = Jcore(xs, eps);
    TRACE_MF(end+1,1) = J;
    if isempty(BEST_MF), BEST_MF = J; else, BEST_MF(end+1,1) = min(BEST_MF(end), J); end
end

% --- Forward residual objective (validation energy) ---
function val = objective(xs, eps_scale, xL, xR, Nc, Nstar, k, nu, BL, BR)
    [x_pde, alpha_star, sig_x] = PDE_and_Kernel_Centers(xL, xR, xs, Nc, Nstar, k, eps_scale, nu);
    x  = x_pde.';  Nf = numel(x);
    w  = ones(1,Nf)/Nf;  Wsqrt = sqrt(w(:));
    Ns = numel(alpha_star);

    % RBF params & activations
    m = 1./(sqrt(2)*sig_x); b = -m.*alpha_star;
    phi=@(z)exp(-z.^2); dphi=@(z)-2*z.*exp(-z.^2); d2phi=@(z)(4*z.^2-2).*exp(-z.^2);
    Ai=phi(b); Bi=phi(m+b);
    Z=m*x + b; dPhi=dphi(Z); d2Phi=d2phi(Z);

    % Residual operator rows
    R_i=(Ai - Bi) + (m.*dPhi) - nu*((m.^2).*d2Phi);  % Ns x Nf
    R = R_i.';  f = (BR - BL)*ones(Nf,1);

    % Solve ridge LS
    Rw = R .* Wsqrt; fw = f .* Wsqrt; lambda = 1e-10;
    c = (Rw.'*Rw + lambda*eye(Ns)) \ (-Rw.'*fw);

    % Validation residual energy
    Nxv=800; xv=linspace(xL,xR,Nxv);
    Zp=m*xv + b; dPhip=dphi(Zp); d2Phip=d2phi(Zp);
    Rpsi_v=(Ai - Bi) + (m.*dPhip) - nu*((m.^2).*d2Phip);
    Rv = Rpsi_v.'; fv=(BR-BL)*ones(Nxv,1);
    resv = Rv*c + fv;
    val  = sqrt(trapz(xv, (resv.').^2));
end

function [x_pde, alpha_star, sigma_x] = PDE_and_Kernel_Centers(xL, xR, xs, Nc, Nstar, k, eps_scale, nu)
    assert(xL < xs && xs < xR, 'Require xL < xs < xR');
    xg = linspace(xL, xs, Nc+1)'; xg(end)=[];
    xl = linspace(xs, xR, Nc)'; 
    x_pde = [xg; xl];

    ag = linspace(xL, xs, Nstar+1)'; ag(end)=[];
    al = linspace(xs, xR, Nstar)'; 
    alpha_star = [ag; al];

    dL = (xs - xL)/max(Nstar,1);
    dR = (xR - xs)/max(Nstar,1);
    sigma_L = k*dL; sigma_R = k*dR;

    eps_geo = eps_scale * min(dL, dR);
    eps_phy = 5*nu;
    epsb    = max(eps_geo, eps_phy);

    t = (alpha_star - xs) / epsb;
    s = 1 ./ (1 + exp(-t));
    sigma_x = sigma_L*(1 - s) + sigma_R*s;
end
