% FORWARD_k_SWEEP_SUPERPOSE.m
% Runs your forward gated X-TFC code for k = 0.5, 1.0, 1.5
% and plots ONLY the final solutions superposed.

clc; clear; close all;
warning('off','all');

tic;

%% ------------------ Fixed problem setup ------------------
nu   = 1e-4;
BL   = 0; BR = 1;
xL   = 0; xR = 1;

Nc    = 1000;     % per block
Nstar = 1000;     % per block

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

k_list = [0.5, 1.0, 1.5, 2.0, 2.5];

% Storage
UHs   = zeros(numel(k_list), Nx);
xsopt = zeros(size(k_list));
epsopt= zeros(size(k_list));

fprintf('\n=== k-sweep: superposed final forward solutions ===\n');

%% ------------------ Loop over k ------------------
for ik = 1:numel(k_list)
    k = k_list(ik);
    fprintf('\n--- Running k = %.2f ---\n', k);

    % Outer optimization on xs
    outer_obj = @(xs) inner_min_over_eps(xs, xL, xR, Nc, Nstar, k, nu, BL, BR, e_lb, e_ub, opts_e);
    [xs_opt, f_outer, ~, outp] = fminbnd(outer_obj, xs_lb, xs_ub, opts_xs);

    % Inner argmin on eps_scale at xs_opt
    [eps_opt, f_inner] = inner_argmin_eps(xs_opt, xL, xR, Nc, Nstar, k, nu, BL, BR, e_lb, e_ub, opts_e);

    xsopt(ik)  = xs_opt;
    epsopt(ik) = eps_opt;

    fprintf('k=%.2f: xs*=%.6f, eps*=%.6f (outer=%.3e, inner=%.3e, iters=%d)\n', ...
        k, xs_opt, eps_opt, f_outer, f_inner, outp.iterations);

    % Solve once at optimum
    [x_pde, alpha_star, sig_x] = PDE_and_Kernel_Centers(xL, xR, xs_opt, Nc, Nstar, k, eps_opt, nu);

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

    Z    = m*x + b;                 % Ns x Nf
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
    Zp   = m*xx + b;                       % Ns x Nx
    Phip = phi(Zp);
    Psip = Phip - (Ai*(1 - xx)) - (Bi*xx); % Ns x Nx
    g    = (1-xx)*BL + xx*BR;
    uh   = g + (c.'*Psip);                 % 1 x Nx

    UHs(ik,:) = uh;
end

%% ------------------ Superposed plot ONLY ------------------
%% ------------------ Superposed plot ONLY (with zoom inset) ------------------
f1 = figure('Color','w'); 
ax1 = axes(f1); hold(ax1,'on');

LW = 2.2;

for ik = 1:numel(k_list)
    uh = UHs(ik,:);
    plot(ax1, xx, uh, 'LineWidth', LW, ...
        'DisplayName', sprintf('k=%.1f', k_list(ik)));
end

grid(ax1,'on'); box(ax1,'on');

% ---- Bigger fonts everywhere ----
FS_ax   = 20;   % axes tick labels
FS_lab  = 24;   % x/y labels
FS_tit  = 24;   % title
FS_leg  = 18;   % legend

set(ax1, 'FontSize', FS_ax, 'LineWidth', 1.2);

xlabel(ax1, '$x$', 'Interpreter','latex', 'FontSize', FS_lab);
ylabel(ax1, '$u_h(x)$', 'Interpreter','latex', 'FontSize', FS_lab);
% title(ax1, sprintf('Forward Gated X--TFC: k-sweep (\\nu=%.1e)', nu), ...
%       'Interpreter','latex', 'FontSize', FS_tit);

title(ax1, ...
    sprintf('Forward Gated X--TFC: k-sweep ($\\nu = %.0f\\times10^{%d}$)', ...
    nu/10^floor(log10(nu)), floor(log10(nu))), ...
    'Interpreter','latex', 'FontSize', FS_tit);

leg = legend(ax1, 'Interpreter','latex', 'Location','best', 'Box','off');
set(leg, 'FontSize', FS_leg);

xlim(ax1, [0 1]);

%% --------- Zoom-in near x=1: inset axes "trick" ----------
x1 = 0.9; x2 = 1.0;

% Compute y-limits over the zoom window (robust for multiple curves)
ix = (xx >= x1) & (xx <= x2);
ymin = min(UHs(:,ix), [], 'all');
ymax = max(UHs(:,ix), [], 'all');
pad  = 0.03 * max(1e-12, (ymax - ymin));
yl1  = ymin - pad;
yl2  = ymax + pad;

% (Optional) draw a rectangle showing zoom region on main axes
rectangle(ax1, 'Position', [x1, yl1, (x2-x1), (yl2-yl1)], ...
          'LineWidth', 1.2, 'LineStyle','--');

% Create inset axes (position in normalized figure units)
ax2 = axes('Position', [0.58 0.20 0.33 0.30]);  % [left bottom width height]
hold(ax2,'on'); box(ax2,'on'); grid(ax2,'on');

for ik = 1:numel(k_list)
    plot(ax2, xx, UHs(ik,:), 'LineWidth', LW);
end

set(ax2, 'FontSize', 16, 'LineWidth', 1.1);
xlim(ax2, [x1 x2]);
ylim(ax2, [yl1 yl2]);

% Clean inset labels: usually best to avoid full labels inside inset
title(ax2, 'Zoom: $x\in[0.9,1]$', 'Interpreter','latex', 'FontSize', 16);

% Connect inset to main region (nice presentation)
% (Requires R2019b+ for annotation line positioning to look good)
annotation('textbox', [0.58 0.51 0.33 0.05], 'String','', ...
    'EdgeColor','none'); % placeholder (keeps layout stable in exports)

% Export
exportgraphics(f1, sprintf('FIG_FORWARD_k_SWEEP_nu_%g.pdf', nu), 'ContentType','vector');


toc;

%% ======================= Helpers (same as your script) =======================
function val = inner_min_over_eps(xs, xL, xR, Nc, Nstar, k, nu, BL, BR, e_lb, e_ub, opts_e)
    obj_e = @(e) objective(xs, e, xL, xR, Nc, Nstar, k, nu, BL, BR);
    [~, val] = fminbnd(obj_e, e_lb, e_ub, opts_e);
end

function [e_star, fval] = inner_argmin_eps(xs, xL, xR, Nc, Nstar, k, nu, BL, BR, e_lb, e_ub, opts_e)
    obj_e = @(e) objective(xs, e, xL, xR, Nc, Nstar, k, nu, BL, BR);
    [e_star, fval] = fminbnd(obj_e, e_lb, e_ub, opts_e);
end

function val = objective(xs, eps_scale, xL, xR, Nc, Nstar, k, nu, BL, BR)
    [x_pde, alpha_star, sig_x] = PDE_and_Kernel_Centers(xL, xR, xs, Nc, Nstar, k, eps_scale, nu);

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

function [x_pde, alpha_star, sigma_x] = PDE_and_Kernel_Centers(xL, xR, xs, Nc, Nstar, k, eps_scale, nu)
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
    eps_phy = 5*nu;
    epsb    = max(eps_geo, eps_phy);

    t = (alpha_star - xs) / epsb;
    s = 1 ./ (1 + exp(-t));
    sigma_x = sigma_L * (1 - s) + sigma_R * s;
end


function u_exact = EXACT_SOLUTION(X, nu)
% Stable exact solution for u' - nu u'' = 0, u(0)=0, u(1)=1 (plotting only)
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