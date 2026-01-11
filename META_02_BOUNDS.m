% FIT_META_GATED_XTFC_WLS.m
% Learn probabilistic maps nu -> xs_opt and nu -> eps_opt from training_data.mat
% using heteroskedastic weighted ridge (Bayesian WLS) with IRLS + GCV.
% Saves: meta_gated_xtfc_model.mat
%
% Usage after running:
%   load meta_gated_xtfc_model.mat
%   nu_new = 3e-3;
%   [xs_mu, xs_lb, xs_ub]    = predict_bounds_xs(model_xs,  log10(nu_new), 0.95);
%   [eps_mu, eps_lb, eps_ub] = predict_bounds_eps(model_eps, log10(nu_new), 0.95);

clc; clear; close all; rng(0);

%% -------------------- Load training data --------------------
S = load('training_data.mat');    % expects S.training_data: [nu, xs_opt, eps_opt]
if ~isfield(S,'training_data'), error('training_data.mat missing variable "training_data".'); end
T = S.training_data;
if size(T,2) ~= 3, error('training_data must be N x 3: [nu, xs_opt, eps_opt].'); end

nu      = T(:,1);
xs_opt  = T(:,2);
eps_opt = T(:,3);

% Basic sanity
nu = nu(:); xs_opt = xs_opt(:); eps_opt = eps_opt(:);
N = numel(nu);
if any(~isfinite(nu) | ~isfinite(xs_opt) | ~isfinite(eps_opt))
    error('Non-finite entries in training data.');
end

%% -------------------- Transforms & feature design --------------------
% Inputs: x = log10(nu) (better conditioning & near-linear)
x = log10(nu);

% Output transforms to unbounded space:
% For xs in (a,b) ~ [0.80,0.999], logit-scale after affine map
a_xs = min(0.80, min(xs_opt));   % safe anchors; we still clip in transform
b_xs = max(0.999, max(xs_opt));
xs2z = @(xs) min(max((xs - a_xs) ./ max(b_xs - a_xs,1e-12), 1e-6), 1-1e-6);
logit = @(z) log(z ./ (1 - z));
ilogit= @(t) 1 ./ (1 + exp(-t));
y_xs  = logit(xs2z(xs_opt));

% For eps_opt in [10,100]+, log-scale
y_eps = log10(max(eps_opt, 1e-12));

% Polynomial feature map (degree 3 is a good default)
deg = 3;
Phi  = build_poly_features(x, deg);   % N x (deg+1)

%% -------------------- Fit heteroskedastic weighted ridge (xs) -----------
model_xs = fit_wls_ridge_IRLS(x, y_xs, Phi, ...
    'lambda_grid', logspace(-8,2,60), ...
    'irls_iters',  3, ...
    'smooth_span', 0.2, ...
    'ridge_jitter', 1e-10);

% Store inverse transform & range (for back-transform & clipping)
model_xs.a_xs  = a_xs;
model_xs.b_xs  = b_xs;
model_xs.ilogit= ilogit;

%% -------------------- Fit heteroskedastic weighted ridge (eps) ----------
model_eps = fit_wls_ridge_IRLS(x, y_eps, Phi, ...
    'lambda_grid', logspace(-8,2,60), ...
    'irls_iters',  3, ...
    'smooth_span', 0.2, ...
    'ridge_jitter', 1e-10);

%% -------------------- Save models ---------------------------------------
save('meta_gated_xtfc_model.mat', 'model_xs', 'model_eps');
fprintf('Saved meta_gated_xtfc_model.mat (deg=%d, N=%d).\n', deg, N);

%% -------------------- Probabilistic Regression ---------------------------------------

%% -------------------- Visualization: meta fits with 95% bands (large fonts)
% Uses the fitted models (model_xs, model_eps) and the raw training data
% nu, xs_opt, eps_opt already loaded above.

% Global LaTeX & sizes
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');

LF = 20;   % Label font size (x/y)
AF = 18;   % Axis tick font size
TF = 20;   % Title font size (if used)
LG = 16;   % Legend font size
LW = 2.2;  % Line width
AXLW = 1.2;% Axes line width

% Prediction grid in log10(nu)
xg  = linspace(min(log10(nu)), max(log10(nu)), 400).';
nuG = 10.^xg;

% Predict bands
tic;
[xs_mu, xs_lb, xs_ub]     = predict_bounds_xs_vec(model_xs,  xg, 0.95);
[eps_mu, eps_lb, eps_ub]  = predict_bounds_eps_vec(model_eps, xg, 0.95);
toc;

% -------- Plot 1:  nu  vs  x_s^*  ----------
fig1 = figure('Color','w','Position',[100 100 1080 560]); hold on; box on; grid on;
set(gca,'XScale','log','FontSize',AF,'LineWidth',AXLW);

fill([nuG; flipud(nuG)], [xs_lb; flipud(xs_ub)], [0.95 0.55 0.15], ...
     'EdgeColor','none','FaceAlpha',0.40, 'DisplayName','95\% confidence band');
plot(nuG, xs_mu, 'LineWidth', LW, 'DisplayName','Mean');
plot(nu,  xs_opt, 'ko', 'MarkerFaceColor',[0.2 0.2 0.2], ...
     'MarkerSize',6, 'DisplayName','Data');

xlabel('$\nu$','FontSize',LF);
ylabel('$x_s^\ast$','FontSize',LF);
lg1 = legend('Location','best','Box','off'); set(lg1,'FontSize',LG);
exportgraphics(fig1, 'FIG_META_xs_vs_nu.pdf', 'Resolution', 400);

% -------- Plot 2:  nu  vs  \varepsilon_{\rm scale}^* ----------
fig2 = figure('Color','w','Position',[120 120 1080 560]); hold on; box on; grid on;
set(gca,'XScale','log','FontSize',AF,'LineWidth',AXLW);

fill([nuG; flipud(nuG)], [eps_lb; flipud(eps_ub)], [0.95 0.55 0.15], ...
     'EdgeColor','none','FaceAlpha',0.40, 'DisplayName','95\% confidence band');
plot(nuG, eps_mu, 'LineWidth', LW, 'DisplayName','Mean');
plot(nu,  eps_opt, 'ko', 'MarkerFaceColor',[0.2 0.2 0.2], ...
     'MarkerSize',6, 'DisplayName','Data');

xlabel('$\nu$','FontSize',LF);
ylabel('$\varepsilon_{\rm scale}^\ast$','FontSize',LF);
lg2 = legend('Location','best','Box','off'); set(lg2,'FontSize',LG);
exportgraphics(fig2, 'FIG_META_eps_vs_nu.pdf', 'Resolution', 400);

%% -------------------- Example: query bounds for new nu -------------------
% (You can comment out this demo block.)
nu_demo = [1e-2; 1e-3; 1e-4];
x_demo  = log10(nu_demo);
conf    = 0.95;

for i = 1:numel(nu_demo)
    [xs_mu, xs_lb, xs_ub]    = predict_bounds_xs(model_xs,  x_demo(i), conf);
    [eps_mu, eps_lb, eps_ub] = predict_bounds_eps(model_eps, x_demo(i), conf);
    fprintf('nu=%.3e  xs: %.4f [%.4f, %.4f]   eps: %.2f [%.2f, %.2f]\n', ...
        nu_demo(i), xs_mu, xs_lb, xs_ub, eps_mu, eps_lb, eps_ub);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            Helper functions                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Phi = build_poly_features(x, deg)
    % Return [1, x, x^2, ..., x^deg]
    x = x(:);
    Phi = ones(numel(x), deg+1);
    for d = 1:deg
        Phi(:,d+1) = x.^d;
    end
end

function model = fit_wls_ridge_IRLS(x, y, Phi, varargin)
% Heteroskedastic WLS ridge via IRLS + lambda chosen by GCV.
% No toolbox requirements: uses Gaussian kernel smoothing + erfinv.
% Returns model struct with fields:
%   beta, lambda, w (final), Phi_fun (handle), deg, x_train, y_train
%   s2_fun: local noise variance estimator handle
%   cov_beta: approximate coefficient covariance

    p = inputParser;
    addParameter(p, 'lambda_grid', logspace(-6,1,40));
    addParameter(p, 'irls_iters', 3);
    addParameter(p, 'smooth_span', 0.25);   % fraction of x-range used as bandwidth
    addParameter(p, 'ridge_jitter', 1e-10);
    parse(p, varargin{:});
    G    = p.Results.lambda_grid(:);
    K    = p.Results.irls_iters;
    span = p.Results.smooth_span;
    jit  = p.Results.ridge_jitter;

    N = size(Phi,1);
    I = eye(size(Phi,2));

    % ---------- IRLS loop: update weights from smoothed residual scale -----
    w = ones(N,1);
    for it = 1:K
        % Choose lambda by minimizing GCV under current weights
        [lambda, beta] = choose_lambda_gcv(Phi, y, w, G, I, jit);

        % Residuals & smooth their absolute value to estimate local scale
        yhat = Phi*beta;
        r    = y - yhat;
        rabs = abs(r);
        s    = gauss_smooth_1d(x, rabs, span);    % <- no toolbox
        s2   = max(s, median(rabs)*0.1).^2;       % stabilize a bit
        w    = 1 ./ max(s2, 1e-12);               % inverse variance
    end

    % Final refit with the selected lambda and weights
    [lambda, beta] = choose_lambda_gcv(Phi, y, w, G, I, jit);

    % Approximate coefficient covariance (sandwich-ish)
    W    = diag(w);
    A    = (Phi' * W * Phi) + lambda*I + jit*I;
    r    = y - Phi*beta;
    Sigma= diag(max(r.^2, 1e-12));
    covB = (A \ (Phi' * W * Sigma * W * Phi)) / A;   % (p x p)

    % Smooth residual variance vs x for predictive noise (no toolbox)
    r2   = r.^2;
    s2x  = gauss_smooth_1d(x, r2, span);
    s2x  = max(s2x, median(r2)*0.1);                 % floor

    model.beta      = beta;
    model.lambda    = lambda;
    model.w         = w;
    model.Phi_fun   = @(xq) build_poly_features(xq, size(Phi,2)-1);
    model.deg       = size(Phi,2)-1;
    model.x_train   = x;
    model.y_train   = y;
    model.cov_beta  = covB;
    model.s2_fun    = @(xq) interp1(x, s2x, xq, 'pchip', 'extrap');
end

function [lambda_best, beta_best] = choose_lambda_gcv(Phi, y, w, grid, I, jit)
% Weighted ridge, pick lambda by minimizing GCV = RSS / (N - df)^2
    N = size(Phi,1);
    W = diag(w);
    best = inf; lambda_best = grid(1); beta_best = zeros(size(Phi,2),1);

    for lam = grid.'
        A   = (Phi' * W * Phi) + lam*I + jit*I;
        rhs = Phi' * W * y;
        beta= A \ rhs;

        % RSS_w = sum(w .* (y - Phi*beta).^2)
        r      = y - Phi*beta;
        RSSw   = sum(w .* (r.^2));

        % df = trace(S), S = Phi * (Phi' W Phi + λI)^{-1} * Phi' W
        S   = Phi / A * (Phi' * W);
        df  = trace(S);
        GCV = RSSw / max((N - df)^2, 1e-12);

        if GCV < best
            best = GCV; lambda_best = lam; beta_best = beta;
        end
    end
end

function s = gauss_smooth_1d(x, y, span)
% Simple Gaussian kernel smoother without toolboxes.
% span is a fraction of the x-range used as bandwidth.
    x = x(:); y = y(:);
    N = numel(x);
    xr = max(x) - min(x);
    h  = max(span * xr, 1e-8);

    % Pairwise distances & Gaussian weights
    % (N is modest in this application; this is fine.)
    D = abs(x - x.');
    W = exp(-0.5*(D./h).^2);

    % Row-normalize weights and apply
    W = W ./ max(sum(W,2), 1e-12);
    s = W * y;
end

%% -------------------- Prediction & bounds for xs -------------------------
function [mu_xs, lb_xs, ub_xs] = predict_bounds_xs(model, xq, conf)
% Predict xs and an interval for a single log10(nu) query xq.
    if nargin < 3, conf = 0.95; end
    Phi_q = model.Phi_fun(xq);
    mu_y  = Phi_q * model.beta;

    % Model variance via coefficient covariance
    var_model = Phi_q * model.cov_beta * Phi_q.';

    % Noise variance via smoothed residual-variance map
    var_noise = model.s2_fun(xq);

    % Total predictive variance (approx)
    var_tot = max(var_model + var_noise, 1e-12);
    z = z_from_conf(conf);   % <- no toolbox

    % Back-transform from logit-scaled output to xs in [a_xs, b_xs]
    zhat   = 1 ./ (1 + exp(-mu_y));
    mu_xs  = model.a_xs + (model.b_xs - model.a_xs) * zhat;

    % Delta method: derivative of xs wrt y in logit space
    dxdY   = (model.b_xs - model.a_xs) * zhat .* (1 - zhat);
    sigmaY = sqrt(var_tot);
    sigmaX = abs(dxdY) .* sigmaY;

    lb_xs  = max(model.a_xs, mu_xs - z*sigmaX);
    ub_xs  = min(model.b_xs, mu_xs + z*sigmaX);
end

%% -------------------- Prediction & bounds for eps ------------------------
function [mu_eps, lb_eps, ub_eps] = predict_bounds_eps(model, xq, conf)
% Predict eps_scale and an interval for a single log10(nu) query xq.
    if nargin < 3, conf = 0.95; end
    Phi_q = model.Phi_fun(xq);
    mu_y  = Phi_q * model.beta;            % on log10 scale

    var_model = Phi_q * model.cov_beta * Phi_q.';
    var_noise = model.s2_fun(xq);
    var_tot   = max(var_model + var_noise, 1e-12);
    z = z_from_conf(conf);                 % <- no toolbox

    % Back to eps_scale
    mu_eps  = 10.^mu_y;

    % Delta method: d(10^y)/dy = ln(10) * 10^y
    sigmaY  = sqrt(var_tot);
    sigmaE  = log(10) * mu_eps .* sigmaY;

    % Clip to reasonable bounds if desired
    lb_eps  = max(10,  mu_eps - z*sigmaE);
    ub_eps  = min(1e6, mu_eps + z*sigmaE);
end

function z = z_from_conf(conf)
% Two-sided normal quantile without Statistics Toolbox.
% conf = 0.95 -> z ≈ 1.96
    p = 0.5 + conf/2;                 % upper tail probability
    z = sqrt(2) * erfinv(2*p - 1);    % Φ^{-1}(p) via erfinv
end


%% =================== Small helpers (vectorized predictors) ==============
function [mu, lb, ub] = predict_bounds_xs_vec(model, xq_vec, conf)
    if nargin < 3, conf = 0.95; end
    xq_vec = xq_vec(:);
    mu = zeros(size(xq_vec)); lb = mu; ub = mu;
    for i = 1:numel(xq_vec)
        [mu(i), lb(i), ub(i)] = predict_bounds_xs(model, xq_vec(i), conf);
    end
end

function [mu, lb, ub] = predict_bounds_eps_vec(model, xq_vec, conf)
    if nargin < 3, conf = 0.95; end
    xq_vec = xq_vec(:);
    mu = zeros(size(xq_vec)); lb = mu; ub = mu;
    for i = 1:numel(xq_vec)
        [mu(i), lb(i), ub(i)] = predict_bounds_eps(model, xq_vec(i), conf);
    end
end
