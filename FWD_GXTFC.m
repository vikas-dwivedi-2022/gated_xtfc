% FORWARD_GATED_XTFC_PDE.m
% Forward solve for 1D steady convection-diffusion using kernel-adaptive X-TFC.
% Soft domain decomposition; tune (xs, eps_scale) by PDE residual ONLY.
% Exact solution is used ONLY for final visualization (not for optimization).

clc; clear; close all;

warning('off', 'all')

tic;

%% ------------------ User controls ------------------
nu   = 1e-4;            % diffusion
BL   = 0; BR = 1;       % boundary values
xL   = 0; xR = 1;

% Layout (per block)
Nc        = 1000;       % collocation per block (total Nf = 2*Nc)
Nstar     = 1000;       % centers per block (total Ns = 2*Nstar)
k         = 1;%1.5;        % width multiplier (shared L/R)

% Bounds for bilevel fminbnd
xs_lb  = 0.80;  xs_ub  = 0.999;    % reasonable xs window (layer near x=1)
e_lb   = 10;  e_ub   = 100.00;    % reasonable eps_scale window
% Tolerances / display
tolx_xs = 1e-4;   tolx_e = 100*1e-3;
opts_xs = optimset('TolX', tolx_xs, 'Display','iter');
opts_e  = optimset('TolX', tolx_e,  'Display','off');

fprintf('=== Outer fminbnd on xs; inner fminbnd on eps_scale (PDE residual only) ===\n');

%% ------------------ Outer optimization on xs ------------------
outer_obj = @(xs) inner_min_over_eps(xs, xL, xR, Nc, Nstar, k, nu, BL, BR, e_lb, e_ub, opts_e);

[xs_opt, f_outer, ~, outp] = fminbnd(outer_obj, xs_lb, xs_ub, opts_xs);
[eps_opt, f_inner] = inner_argmin_eps(xs_opt, xL, xR, Nc, Nstar, k, nu, BL, BR, e_lb, e_ub, opts_e);

fprintf('xs* = %.6f, eps_scale* = %.6f (outer obj = %.3e, inner obj = %.3e, iters=%d)\n', ...
         xs_opt, eps_opt, f_outer, f_inner, outp.iterations);

%% ------------------ Solve once at (xs_opt, eps_opt) ------------------
[x_pde, alpha_star, sig_x] = PDE_and_Kernel_Centers(xL, xR, xs_opt, Nc, Nstar, k, eps_opt, nu);

% Collocation + weights
x     = x_pde.';              % 1 x Nf
Nf    = numel(x);
w     = ones(1,Nf)/Nf;        % uniform quadrature
Wsqrt = sqrt(w(:));
Ns    = numel(alpha_star);

% RBF parameters
m = 1 ./ (sqrt(2) * sig_x);   % Ns x 1
b = -m .* alpha_star;         % Ns x 1

% Activations
phi   = @(z) exp(-z.^2);
dphi  = @(z) -2*z.*exp(-z.^2);
d2phi = @(z) (4*z.^2 - 2).*exp(-z.^2);

% Constrained basis pieces and residual design
Ai = phi(b);                  % Ns x 1  (phi at x=0)
Bi = phi(m + b);              % Ns x 1  (phi at x=1)
Z   = m*x + b;                % Ns x Nf
dPhi  = dphi(Z);              % Ns x Nf
d2Phi = d2phi(Z);             % Ns x Nf

% Residual operator L_\nu[psi_i] = (Ai-Bi) + m*dphi(z) - nu*m^2*d2phi(z)
R_i = (Ai - Bi) + (m .* dPhi) - nu * ((m.^2) .* d2Phi);   % Ns x Nf
R   = R_i.';                                                % Nf x Ns
f   = (BR - BL) * ones(Nf,1);                               % Nf x 1

% Solve (weighted normal equations)
Rw = R .* Wsqrt;  fw = f .* Wsqrt;
lambda = 1e-10;                                           % small ridge
c = (Rw.'*Rw + lambda*eye(Ns)) \ (-Rw.'*fw);              % Ns x 1

% Predict on a dense grid
Nx = 20000; xx = linspace(0,1,Nx);
Zp   = m*xx + b;                    % Ns x Nx
Phip = phi(Zp);                     % Ns x Nx
Psip = Phip - (Ai*(1 - xx)) - (Bi*xx);   % Ns x Nx
g    = (1-xx)*BL + xx*BR;
uh   = g + (c.'*Psip);              % 1 x Nx

% Residual energy on training grid (for info)
res = R*c + f;  
J   = mean(res.^2);
fprintf('At optimum: residual energy J ≈ %.3e\n', J);

% Residual on dense grid (for visualization)
dPhip   = dphi(Zp);
d2Phip  = d2phi(Zp);
Rpsi_dx = (Ai - Bi) + (m.*dPhip) - nu*((m.^2).*d2Phip);  % Ns x Nx
Rd      = Rpsi_dx.';                                     % Nx x Ns
fd      = (BR - BL)*ones(Nx,1);
res_dense = Rd*c + fd;                                   % Nx x 1

%% ------------------ Effective epsilon (for annotation) ------------------
% Recompute geometric spacings and effective transition width
dL = (xs_opt - xL) / max(Nstar,1);
dR = (xR - xs_opt) / max(Nstar,1);
sigma_L = k*dL; 
sigma_R = k*dR;
eps_geo = eps_opt * min(dL, dR);
eps_phy = 5*nu;
eps_eff = max(eps_geo, eps_phy);

% Gate profile (for plotting)
s_gate = 1 ./ (1 + exp(-(alpha_star - xs_opt)/eps_eff));  % Ns x 1

%% ------------------ Solution plot (context) + inset zoom ------------------
fs_axis=12; fs_label=14; lw=2.0;

% Exact solution ONLY for visualization (not used in optimization)
uex = EXACT_SOLUTION(xx, nu);
maxErr = max(abs(uh - uex));

f1 = figure('Color','w'); f1.Units='inches'; f1.Position=[1 1 8 3.5];

% Main axes
axMain = axes('Parent', f1);
plot(axMain, xx, uh, 'LineWidth', lw); hold(axMain,'on');
plot(axMain, xx, uex, '--', 'LineWidth', lw);
xline(axMain, xs_opt,'k--','$x_s^\ast$','LineWidth',1.5,'FontSize',16,'Interpreter','latex');
grid(axMain,'on'); hold(axMain,'off');
xlabel(axMain,'$x$','Interpreter','latex','FontSize',fs_label);
ylabel(axMain,'$u(x)$','Interpreter','latex','FontSize',fs_label);
legend(axMain, {'Gated X-TFC','Exact'}, 'Interpreter','latex','Location','best','Box','off');
axMain.LineWidth=.8; axMain.FontSize=fs_axis; axMain.TickLabelInterpreter='latex'; axMain.Box='on'; axMain.XLim=[0 1];
title(axMain, sprintf('$\\nu=%.3g$, $x_s^*=%.3f$, $\\varepsilon_{\\rm scale}^*=%.2f$, $\\max|e|=%.2e$', ...
      nu, xs_opt, eps_opt, maxErr), 'Interpreter','latex','FontSize',fs_axis);

% Inset axes (zoom near x=1)
axInset = axes('Parent', f1, 'Position',[0.58 0.20 0.30 0.30]); % [left bottom width height] in figure units
plot(axInset, xx, uh, '-', 'LineWidth', lw); hold(axInset,'on');
plot(axInset, xx, uex, '--', 'LineWidth', lw);
grid(axInset,'on'); box(axInset,'on');

% Choose zoom window (heuristic: 10*nu from the right, but not < 0.8)
xZoomStart = max(0.8, 1 - 10*nu);
xlim(axInset, [xZoomStart 1]);

% Y-limits from the data in the zoom window, with a small margin
idx = (xx >= xZoomStart) & (xx <= 1);
ymin = min( [uh(idx),  uex(idx)] );
ymax = max( [uh(idx),  uex(idx)] );
margin = 0.02 * max(1, ymax - ymin);
ylim(axInset, [ymin - margin, ymax + margin]);

set(axInset,'FontSize',12,'TickLabelInterpreter','latex');
title(axInset,'Zoom near $x=1$','Interpreter','latex','FontSize',12);
hold(axInset,'off');
pause (1)
% exportgraphics(f1,'FIG_SOLUTION_VS_EXACT.pdf','ContentType','vector');
filename = sprintf('FIG_SOLUTION_VS_EXACT_NU_%g.pdf', nu);
exportgraphics(f1, filename, 'ContentType','vector');

%% ------------------ 2×1 diagnostics: residual & abs error ------------------
f2 = figure('Color','w'); f2.Units='inches'; 
% f2.Position=[1 1 5.4 7.0];
f2.Position=[1 1 8 7];

% (a) Pointwise PDE residual
subplot(2,1,1);
abs_res = abs(res_dense(:)).';
if max(abs_res) / max(min(abs_res(abs_res>0)), eps) > 1e3
    semilogy(xx, abs_res, 'LineWidth', lw);
else
    plot(xx, abs_res, 'LineWidth', lw);
end
hold on; 
% xline(xs_opt,'k--','x_s^*','Interpreter','latex'); 
xline(xs_opt,'r--','$x_s^\ast$','LineWidth',2.5,'FontSize',16,'Interpreter','latex');

hold off;
grid on;
xlabel('$x$','Interpreter','latex','FontSize',fs_label);
ylabel('$|\mathcal{L}_\nu[u_h](x)|$','Interpreter','latex','FontSize',fs_label);
% title(sprintf('Pointwise PDE residual  ($x_s^*=%.3f$, $\\varepsilon_{\\rm scale}^*=%.2f$, $\\varepsilon_{\\rm eff}=%.2g$)', ...
%       xs_opt, eps_opt, eps_eff),'Interpreter','latex','FontSize',fs_axis);


title(sprintf('Pointwise PDE residual  ($x_s^*=%.3f$, $\\varepsilon_{\\rm scale}^*=%.2f$)', ...
      xs_opt, eps_opt),'Interpreter','latex','FontSize',fs_axis);

ax1=gca; ax1.LineWidth=.8; ax1.FontSize=fs_axis; ax1.TickLabelInterpreter='latex'; ax1.Box='on'; ax1.XLim=[0 1];

% (b) Pointwise absolute error (visualization only)
subplot(2,1,2);
abs_err = abs(uh(:).' - uex(:).');
if max(abs_err) / max(min(abs_err(abs_err>0)), eps) > 1e3
    semilogy(xx, abs_err, 'LineWidth', lw);
else
    plot(xx, abs_err, 'LineWidth', lw);
end
hold on; 
pause(1)
% xline(xs_opt,'k--','x_s^*','Interpreter','latex'); 
xline(xs_opt,'r--','$x_s^\ast$','LineWidth',2.5,'FontSize',16,'Interpreter','latex');
hold off;
grid on;
xlabel('$x$','Interpreter','latex','FontSize',fs_label);
ylabel('$|u_h(x)-u_{exact}(x)|$','Interpreter','latex','FontSize',fs_label);
% title(sprintf('Pointwise absolute error ($x_s^*=%.3f$, $\\varepsilon_{\\rm scale}^*=%.2f$, $\\varepsilon_{\\rm eff}=%.2g$)', ...
%       xs_opt, eps_opt, eps_eff),'Interpreter','latex','FontSize',fs_axis);

title(sprintf('Pointwise absolute error ($x_s^*=%.3f$, $\\varepsilon_{\\rm scale}^*=%.2f$)', ...
      xs_opt, eps_opt),'Interpreter','latex','FontSize',fs_axis);
ax2=gca; ax2.LineWidth=.8; ax2.FontSize=fs_axis; ax2.TickLabelInterpreter='latex'; ax2.Box='on'; ax2.XLim=[0 1];
pause(1)
% exportgraphics(f2,'FIG_RESIDUAL_AND_ERROR.pdf','ContentType','vector');
filename = sprintf('FIG_RESIDUAL_AND_ERROR_NU_%g.pdf', nu);
exportgraphics(f2, filename, 'ContentType','vector');
%% ------------------ Width/gate profile figure ------------------
f3 = figure('Color','w'); f3.Units='inches'; 
% f3.Position=[1 1 5.4 7.0];
f3.Position=[1 1 8 7];

% (a) Blended widths vs centers
subplot(2,1,1);
plot(alpha_star, sig_x, 'LineWidth', lw); hold on;
% yline(sigma_L,'--','$\sigma_L$','Interpreter','latex');
% yline(sigma_R,'--','$\sigma_R$','Interpreter','latex');
yline(sigma_L, '--', '$\sigma_L$', 'Interpreter', 'latex', ...
      'FontSize', 16, 'Color', 'k'); % blue-ish color
yline(sigma_R, '--', '$\sigma_R$', 'Interpreter', 'latex', ...
      'FontSize', 16, 'Color', 'k'); % orange-like color

xline(xs_opt,'r--','$x_s^\ast$','LineWidth',2.5,'FontSize',16,'Interpreter','latex');
hold off; grid on;
xlabel('$\alpha$','Interpreter','latex','FontSize',fs_label);
ylabel('$\sigma_x(\alpha)$','Interpreter','latex','FontSize',fs_label);
% title(sprintf('RBF width profile  ($x_s^*=%.3f$, $\\varepsilon_{\\rm scale}^*=%.2f$, $\\varepsilon_{\\rm eff}=%.2g$)', ...
%       xs_opt, eps_opt, eps_eff),'Interpreter','latex','FontSize',fs_axis);
title(sprintf('RBF width profile  ($x_s^*=%.3f$, $\\varepsilon_{\\rm scale}^*=%.2f$)', ...
      xs_opt, eps_opt),'Interpreter','latex','FontSize',fs_axis);
pause(1)
axw=gca; axw.LineWidth=.8; axw.FontSize=fs_axis; axw.TickLabelInterpreter='latex'; axw.Box='on'; axw.XLim=[0 1];

% (b) Logistic gate across the split
subplot(2,1,2);
plot(alpha_star, s_gate, 'LineWidth', lw); hold on;
xline(xs_opt,'r--','$x_s^\ast$','LineWidth',2.5,'FontSize',16,'Interpreter','latex');
hold off; grid on;
xlabel('$\alpha$','Interpreter','latex','FontSize',fs_label);
ylabel('$s(\alpha)$','Interpreter','latex','FontSize',fs_label);
% title(sprintf('Logistic gating  ($x_s^*=%.3f$, $\\varepsilon_{\\rm scale}^*=%.2f$, $\\varepsilon_{\\rm eff}=%.2g$)', ...
%       xs_opt, eps_opt, eps_eff),'Interpreter','latex','FontSize',fs_axis);
title(sprintf('Logistic gating  ($x_s^*=%.3f$, $\\varepsilon_{\\rm scale}^*=%.2f$)', ...
      xs_opt, eps_opt),'Interpreter','latex','FontSize',fs_axis);
axg=gca; axg.LineWidth=.8; axg.FontSize=fs_axis; axg.TickLabelInterpreter='latex'; axg.Box='on'; axg.XLim=[0 1];
pause(1)
% exportgraphics(f3,'FIG_WIDTH_GATE_PROFILES.pdf','ContentType','vector');
filename = sprintf('FIG_WIDTH_GATE_PROFILES_NU_%g.pdf', nu);
exportgraphics(f3, filename, 'ContentType','vector');
toc;

%% ======================= Helpers =======================
function val = inner_min_over_eps(xs, xL, xR, Nc, Nstar, k, nu, BL, BR, e_lb, e_ub, opts_e)
% For a fixed xs, minimize objective over eps_scale via fminbnd (PDE-only).
    obj_e = @(e) objective(xs, e, xL, xR, Nc, Nstar, k, nu, BL, BR);
    [~, val] = fminbnd(obj_e, e_lb, e_ub, opts_e);
end

function [e_star, fval] = inner_argmin_eps(xs, xL, xR, Nc, Nstar, k, nu, BL, BR, e_lb, e_ub, opts_e)
% Return the argmin eps_scale for a fixed xs (for reporting).
    obj_e = @(e) objective(xs, e, xL, xR, Nc, Nstar, k, nu, BL, BR);
    [e_star, fval] = fminbnd(obj_e, e_lb, e_ub, opts_e);
end

function val = objective(xs, eps_scale, xL, xR, Nc, Nstar, k, nu, BL, BR)
% PDE-only objective: L2 residual energy on an independent validation grid.
    [x_pde, alpha_star, sig_x] = PDE_and_Kernel_Centers(xL, xR, xs, Nc, Nstar, k, eps_scale, nu);

    % Training design
    x  = x_pde.';                 % 1 x Nf
    Nf = numel(x);
    w  = ones(1,Nf)/Nf;  Wsqrt = sqrt(w(:));
    Ns = numel(alpha_star);

    % RBF params and activations
    m = 1./(sqrt(2)*sig_x); 
    b = -m.*alpha_star;
    phi=@(z)exp(-z.^2); 
    dphi=@(z)-2*z.*exp(-z.^2); 
    d2phi=@(z)(4*z.^2-2).*exp(-z.^2);

    Ai=phi(b); 
    Bi=phi(m+b);
    Z = m*x + b;                    % Ns x Nf
    dPhi  = dphi(Z); 
    d2Phi = d2phi(Z);

    % Residual matrix and forcing on training grid
    R_i = (Ai - Bi) + (m.*dPhi) - nu*((m.^2).*d2Phi);   % Ns x Nf
    R   = R_i.';                                       % Nf x Ns
    f   = (BR - BL)*ones(Nf,1);

    % Solve for c (ridge LS)
    Rw=R.*Wsqrt; fw=f.*Wsqrt;
    lambda=1e-10;
    c = (Rw.'*Rw + lambda*eye(Ns)) \ (-Rw.'*fw);

    % Validation grid (independent) → residual energy only
    Nxv=800; xv=linspace(xL,xR,Nxv);
    Zp     = m*xv + b;                  % Ns x Nxv
    dPhip  = dphi(Zp); 
    d2Phip = d2phi(Zp);
    Rpsi_v = (Ai - Bi) + (m.*dPhip) - nu*((m.^2).*d2Phip); % Ns x Nxv
    Rv     = Rpsi_v.';                  % Nxv x Ns
    fv     = (BR-BL)*ones(Nxv,1);
    resv   = Rv*c + fv;                 % Nxv x 1

    % Objective value: L2 residual on validation grid
    val  = sqrt(trapz(xv, (resv.').^2));
end

function [x_pde, alpha_star, sigma_x] = PDE_and_Kernel_Centers(xL, xR, xs, Nc, Nstar, k, eps_scale, nu)
% Two-block partition + smooth width blend (physics-aware)
    assert(xL < xs && xs < xR, 'Require xL < xs < xR');
    assert(Nc > 0 && Nstar > 0, 'Nc and Nstar must be positive');

    % Collocation points (left and right blocks; omit duplicate xs)
    xg = linspace(xL, xs, Nc+1)'; xg(end)=[];
    xl = linspace(xs, xR, Nc)'; 
    x_pde = [xg; xl];

    % RBF centers (left and right blocks; omit duplicate xs)
    ag = linspace(xL, xs, Nstar+1)'; ag(end)=[];
    al = linspace(xs, xR, Nstar)'; 
    alpha_star = [ag; al];

    % Block spacings
    dL = (xs - xL) / max(Nstar,1);
    dR = (xR - xs) / max(Nstar,1);
    sigma_L = k*dL; 
    sigma_R = k*dR;

    % Smooth transition scale (geometry vs. physics)
    eps_geo = eps_scale * min(dL, dR);
    eps_phy = 5*nu;                     % a few boundary-layer widths
    epsb    = max(eps_geo, eps_phy);

    % Logistic gate and blended widths
    t = (alpha_star - xs) / epsb;
    s = 1 ./ (1 + exp(-t));
    sigma_x = sigma_L * (1 - s) + sigma_R * s;

    % Optional floor for stability (if needed):
    % sigma_min = 0.25*min(dL,dR);
    % sigma_x   = max(sigma_x, sigma_min);
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
