% FORWARD_epsPHY_SWEEP_SUPERPOSE.m
% Sweeps eps_phy = factor*nu for factor = [1,2,3,4,5]
% and plots ONLY the final forward solutions superposed (with zoom inset).
%
% Based on your FORWARD_k_SWEEP_SUPERPOSE.m (same bilevel optimization),
% but replaces eps_phy = 5*nu by eps_phy = factor*nu.

clc; clear; close all;
warning('off','all');

tic;

%% ------------------ Fixed problem setup ------------------
nu   = 1e-4;
BL   = 0; BR = 1;
xL   = 0; xR = 1;

Nc    = 1000;     % per block
Nstar = 1000;     % per block

% Keep k fixed (choose your default; paper uses 1.5 in many places)
k = 1.5;

% Bilevel fminbnd bounds (same as your script)
xs_lb  = 0.80;  xs_ub  = 0.999;
e_lb   = 10;    e_ub   = 100.00;

% Tolerances / display
tolx_xs = 1e-4;
tolx_e  = 100*1e-3;
opts_xs = optimset('TolX', tolx_xs, 'Display','iter');
opts_e  = optimset('TolX', tolx_e,  'Display','off');

% Dense grid for final plot
Nx = 20000;
xx = linspace(0,1,Nx);

factor_list = [1,2,3,4,5];

% Storage
UHs    = zeros(numel(factor_list), Nx);
xsopt  = zeros(size(factor_list));
epsopt = zeros(size(factor_list));

fprintf('\n=== eps_{phy} sweep: eps_phy = factor*nu (superposed final solutions) ===\n');

%% ------------------ Loop over eps_phy factor ------------------
for ik = 1:numel(factor_list)
    factor = factor_list(ik);
    fprintf('\n--- Running factor = %g (eps_phy = %g*nu) ---\n', factor, factor);

    % Outer optimization on xs
    outer_obj = @(xs) inner_min_over_eps(xs, xL, xR, Nc, Nstar, k, nu, BL, BR, ...
                                         e_lb, e_ub, opts_e, factor);
    [xs_opt, f_outer, ~, outp] = fminbnd(outer_obj, xs_lb, xs_ub, opts_xs);

    % Inner argmin on eps_scale at xs_opt
    [eps_opt, f_inner] = inner_argmin_eps(xs_opt, xL, xR, Nc, Nstar, k, nu, BL, BR, ...
                                          e_lb, e_ub, opts_e, factor);

    xsopt(ik)  = xs_opt;
    epsopt(ik) = eps_opt;

    fprintf('factor=%g: xs*=%.6f, eps_scale*=%.6f (outer=%.3e, inner=%.3e, iters=%d)\n', ...
        factor, xs_opt, eps_opt, f_outer, f_inner, outp.iterations);

    % Solve once at optimum
    [x_pde, alpha_star, sig_x] = PDE_and_Kernel_Centers(xL, xR, xs_opt, Nc, Nstar, ...
                                                       k, eps_opt, nu, factor);

    x     = x_pde.';                % 1 x Nf
    Nf    = numel(x);
    w     = ones(1,Nf)/Nf;
    Wsqrt = sqrt(w(:));
    Ns    = numel(alpha_star);

    m = 1 ./ (sqrt(2) * sig_x);     % Ns x 1
    b = -m .* alpha_star;           % Ns x 1

    phi   = @(z) exp(-z.^2);
    dphi  = @(z) -2*z.*exp(-z.^2);
    d2phi = @(z) (4*z.^2 - 2).*exp(-z.^2);

    Ai = phi(b);
    Bi = phi(m + b);

    Z     = m*x + b;                 % Ns x Nf
    dPhi  = dphi(Z);
    d2Phi = d2phi(Z);

    R_i = (Ai - Bi) + (m .* dPhi) - nu * ((m.^2) .* d2Phi);  % Ns x Nf
    R   = R_i.';                                             % Nf x Ns
    f   = (BR - BL) * ones(Nf,1);

    Rw = R .* Wsqrt;
    fw = f .* Wsqrt;

    lambda = 1e-10;
    c = (Rw.'*Rw + lambda*eye(Ns)) \ (-Rw.'*fw);

    % Predict ONLY final solution on xx (no gate/residual plots)
    Zp   = m*xx + b;                        % Ns x Nx
    Phip = phi(Zp);
    Psip = Phip - (Ai*(1 - xx)) - (Bi*xx);  % Ns x Nx
    g    = (1-xx)*BL + xx*BR;
    uh   = g + (c.'*Psip);                  % 1 x Nx

    UHs(ik,:) = uh;
end

%% ------------------ Superposed plot ONLY (with zoom inset) ------------------
f1 = figure('Color','w');
ax1 = axes(f1); hold(ax1,'on');

LW = 2.2;

for ik = 1:numel(factor_list)
    plot(ax1, xx, UHs(ik,:), 'LineWidth', LW, ...
        'DisplayName', sprintf('\\epsilon_{phy}=%g\\nu', factor_list(ik)));
end

grid(ax1,'on'); box(ax1,'on');

% ---- Bigger fonts everywhere ----
FS_ax   = 20;
FS_lab  = 24;
FS_tit  = 24;
FS_leg  = 18;

set(ax1, 'FontSize', FS_ax, 'LineWidth', 1.2);
xlabel(ax1, '$x$', 'Interpreter','latex', 'FontSize', FS_lab);
ylabel(ax1, '$u_h(x)$', 'Interpreter','latex', 'FontSize', FS_lab);

title(ax1, sprintf('Forward Gated X--TFC: $\\epsilon_{\\rm phy}$ sweep ($\\nu=10^{-4}$, $k=%.1f$)', k), ...
    'Interpreter','latex', 'FontSize', FS_tit);

leg = legend(ax1, 'Interpreter','latex', 'Location','best', 'Box','off');
set(leg, 'FontSize', FS_leg);

xlim(ax1, [0 1]);

%% --------- Zoom-in near x=1 inset ----------
x1 = 0.9; x2 = 1.0;

ix = (xx >= x1) & (xx <= x2);
ymin = min(UHs(:,ix), [], 'all');
ymax = max(UHs(:,ix), [], 'all');
pad  = 0.03 * max(1e-12, (ymax - ymin));
yl1  = ymin - pad;
yl2  = ymax + pad;

rectangle(ax1, 'Position', [x1, yl1, (x2-x1), (yl2-yl1)], ...
          'LineWidth', 1.2, 'LineStyle','--');

ax2 = axes('Position', [0.58 0.20 0.33 0.30]); % [left bottom width height]
hold(ax2,'on'); box(ax2,'on'); grid(ax2,'on');

for ik = 1:numel(factor_list)
    plot(ax2, xx, UHs(ik,:), 'LineWidth', LW);
end

set(ax2, 'FontSize', 16, 'LineWidth', 1.1);
xlim(ax2, [x1 x2]);
ylim(ax2, [yl1 yl2]);
title(ax2, 'Zoom: $x\in[0.9,1]$', 'Interpreter','latex', 'FontSize', 16);

exportgraphics(f1, sprintf('FIG_FORWARD_epsPHY_SWEEP_nu_%g.pdf', nu), 'ContentType','vector');

toc;

%% ======================= Helpers =======================

function val = inner_min_over_eps(xs, xL, xR, Nc, Nstar, k, nu, BL, BR, e_lb, e_ub, opts_e, factor)
    obj_e = @(e) objective(xs, e, xL, xR, Nc, Nstar, k, nu, BL, BR, factor);
    [~, val] = fminbnd(obj_e, e_lb, e_ub, opts_e);
end

function [e_star, fval] = inner_argmin_eps(xs, xL, xR, Nc, Nstar, k, nu, BL, BR, e_lb, e_ub, opts_e, factor)
    obj_e = @(e) objective(xs, e, xL, xR, Nc, Nstar, k, nu, BL, BR, factor);
    [e_star, fval] = fminbnd(obj_e, e_lb, e_ub, opts_e);
end

function val = objective(xs, eps_scale, xL, xR, Nc, Nstar, k, nu, BL, BR, factor)
    [x_pde, alpha_star, sig_x] = PDE_and_Kernel_Centers(xL, xR, xs, Nc, Nstar, k, eps_scale, nu, factor);

    x  = x_pde.';
    Nf = numel(x);
    w  = ones(1,Nf)/Nf;
    Wsqrt = sqrt(w(:));
    Ns = numel(alpha_star);

    m = 1./(sqrt(2)*sig_x);
    b = -m.*alpha_star;
    phi=@(z)exp(-z.^2);
    dphi=@(z)-2*z.*exp(-z.^2);
    d2phi=@(z)(4*z.^2-2).*exp(-z.^2);

    Ai=phi(b);
    Bi=phi(m+b);
    Z = m*x + b;
    dPhi  = dphi(Z);
    d2Phi = d2phi(Z);

    R_i = (Ai - Bi) + (m.*dPhi) - nu*((m.^2).*d2Phi);
    R   = R_i.';
    f   = (BR - BL)*ones(Nf,1);

    Rw=R.*Wsqrt; fw=f.*Wsqrt;
    lambda=1e-10;
    c = (Rw.'*Rw + lambda*eye(Ns)) \ (-Rw.'*fw);

    % Validation residual energy only
    Nxv=800; xv=linspace(xL,xR,Nxv);
    Zp     = m*xv + b;
    dPhip  = dphi(Zp);
    d2Phip = d2phi(Zp);
    Rpsi_v = (Ai - Bi) + (m.*dPhip) - nu*((m.^2).*d2Phip);
    Rv     = Rpsi_v.';
    fv     = (BR-BL)*ones(Nxv,1);
    resv   = Rv*c + fv;

    val  = sqrt(trapz(xv, (resv.').^2));
end

function [x_pde, alpha_star, sigma_x] = PDE_and_Kernel_Centers(xL, xR, xs, Nc, Nstar, k, eps_scale, nu, factor)
    xg = linspace(xL, xs, Nc+1)'; xg(end)=[];
    xl = linspace(xs, xR, Nc)';
    x_pde = [xg; xl];

    ag = linspace(xL, xs, Nstar+1)'; ag(end)=[];
    al = linspace(xs, xR, Nstar)';
    alpha_star = [ag; al];

    dL = (xs - xL) / max(Nstar,1);
    dR = (xR - xs) / max(Nstar,1);
    sigma_L = k*dL;
    sigma_R = k*dR;

    eps_geo = eps_scale * min(dL, dR);
    eps_phy = factor * nu;     % <--- sweep this
    epsb    = max(eps_geo, eps_phy);

    t = (alpha_star - xs) / epsb;
    s = 1 ./ (1 + exp(-t));
    sigma_x = sigma_L * (1 - s) + sigma_R * s;
end
