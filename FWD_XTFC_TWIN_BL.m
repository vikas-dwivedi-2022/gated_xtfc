% FORWARD_GATED_XTFC_PDE_2SPLITS_BO_TYPE2.m
% 1D steady convection-diffusion (Type-2) using kernel-adaptive X-TFC with TWO splits.
%
% --------- PDE (Type-2 in your table) ----------
%   L_ν[u] := 2(2x-1) u_x  - ν u_xx  + 4 u = 0,   x ∈ (0,1)
%   BCs:    u(0) = 1,  u(1) = 1
%   Exact:  u(x) = exp(-2 x (1-x) / ν)
%
% Changes vs. earlier Type-1 code:
%   • Boundary lift g(x) = 1 (since BCs are both 1)
%   • Residual operator uses a(x) = 2(2x-1) = 4x-2 everywhere it appears
%   • Forcing term for the lift: L_ν[g] = 4 (constant)
%   • Exact solution updated accordingly
%
% Geometry & optimization remain identical to the previous two-split BayesOpt code.

clc; clear; close all;
warning('off','all'); rng(1); tic;

%% ------------------ User controls ------------------
nu   = 1e-4;             % diffusion
BL   = 1;  BR = 1;       % Type-2 boundary values
xL   = 0;  xR = 1;

% Layout (per block)
Nc        = 1200;        % collocation per block (total Nf = 3*Nc)
Nstar     = 1200;        % centers per block (total Ns = 3*Nstar)
k         = 1.5;         % width multiplier per block

% BayesOpt variable ranges (reparameterized: optimize w1, w3; derive w2)
v_w1   = optimizableVariable('w1',   [0.01, 0.1], 'Type','real');
v_w3   = optimizableVariable('w3',   [0.01, 0.1], 'Type','real');
v_eps1 = optimizableVariable('eps1', [10.0, 100.0], 'Type','real');
v_eps2 = optimizableVariable('eps2', [10.0, 100.0], 'Type','real');
vars = [v_w1, v_w3, v_eps1, v_eps2];

MaxEvals = 60;   % change if you want more/less exploration

fprintf('=== BayesOpt on (w1, w3, eps1, eps2); w2 = 1 - w1 - w3 (auto in [0.60, 0.98]) ===\n');
fprintf('=== PDE: 2(2x-1) u_x - nu u_xx + 4 u = 0,  u(0)=u(1)=1 ===\n');

% Objective handle (table row X -> scalar value)
objFun = @(X) objective_BO_type2(X, xL, xR, Nc, Nstar, k, nu, BL, BR);

BOres = bayesopt(objFun, vars, ...
    'IsObjectiveDeterministic', true, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', MaxEvals, ...
    'PlotFcn', {}, ...
    'Verbose', 1);

best = bestPoint(BOres);
w1_opt   = best.w1;
w3_opt   = best.w3;
w2_opt   = 1 - w1_opt - w3_opt;     % ∈ [0.60, 0.98] by construction
eps1_opt = best.eps1;
eps2_opt = best.eps2;

xs1_opt = xL + w1_opt;
xs2_opt = xs1_opt + w2_opt;

fprintf('Best (BayesOpt): w1=%.5f, w2=%.5f, w3=%.5f  |  xs1=%.6f, xs2=%.6f\n', ...
        w1_opt, w2_opt, w3_opt, xs1_opt, xs2_opt);
fprintf('Best eps: eps1=%.6f, eps2=%.6f\n', eps1_opt, eps2_opt);

%% ------------------ Solve once at optimum ------------------
[x_pde, alpha_star, sig_x, sigma_blk, eps_eff, gates] = ...
    PDE_and_Kernel_Centers_3block(xL, xR, xs1_opt, xs2_opt, Nc, Nstar, k, eps1_opt, eps2_opt, nu);

% Collocation + weights
x     = x_pde.';                 % 1 x Nf
Nf    = numel(x);
wq    = ones(1,Nf)/Nf;           % uniform quadrature
Wsqrt = sqrt(wq(:));
Ns    = numel(alpha_star);

% RBF parameters
m = 1 ./ (sqrt(2) * sig_x);      % Ns x 1
b = -m .* alpha_star;            % Ns x 1

% Activations
phi   = @(z) exp(-z.^2);
dphi  = @(z) -2*z.*exp(-z.^2);
d2phi = @(z) (4*z.^2 - 2).*exp(-z.^2);

% Constrained basis pieces
Ai = phi(b);                     % Ns x 1  φ at x=0
Bi = phi(m + b);                 % Ns x 1  φ at x=1
Z   = m*x + b;                   % Ns x Nf
Phi  = phi(Z);                   % Ns x Nf
dPhi  = dphi(Z);                 % Ns x Nf
d2Phi = d2phi(Z);                % Ns x Nf
a_row = 4*x - 2;                 % 1 x Nf,  a(x) = 2(2x-1)=4x-2

% Residual operator for Type-2:
% ψ_i = φ_i - A_i(1-x) - B_i x
% ψ'_i = m_i dφ_i + (A_i - B_i)
% ψ''_i = m_i^2 d2φ_i
% Lν[ψ_i] = a(x)*(m_i dφ_i + (A_i - B_i)) - ν(m_i^2 d2φ_i) + 4*(φ_i - A_i(1-x) - B_i x)
AminusB = Ai - Bi;               % Ns x 1
term1 = (m .* dPhi) .* a_row;                      % Ns x Nf
term2 = (AminusB) * a_row;                         % Ns x Nf
term3 = - nu * ((m.^2) .* d2Phi);                  % Ns x Nf
term4 = 4*Phi;                                     % Ns x Nf
term5 = -4 * (Ai * (1 - x));                       % Ns x Nf
term6 = -4 * (Bi * x);                              % Ns x Nf
R_i = term1 + term2 + term3 + term4 + term5 + term6;  % Ns x Nf

R   = R_i.';                                       % Nf x Ns
f   = 4 * ones(Nf,1);                              % Lν[g=1] = 4

% Solve (weighted normal equations)
Rw = R .* Wsqrt;  fw = f .* Wsqrt;
lambda = 1e-10;                                    % small ridge
c = (Rw.'*Rw + lambda*eye(Ns)) \ (-Rw.'*fw);       % Ns x 1

% Predict on a dense grid
Nx = 20000; xx = linspace(0,1,Nx);
Zp   = m*xx + b;                     % Ns x Nx
Phip = phi(Zp);                      % Ns x Nx
Psip = Phip - (Ai*(1 - xx)) - (Bi*xx);  % Ns x Nx
g    = ones(1, Nx);                  % g(x) = 1 (Type-2 BCs)
uh   = g + (c.'*Psip);               % 1 x Nx

% Residual energy on training grid (for info)
res = R*c + f;  
J   = mean(res.^2);
fprintf('At optimum: residual energy J ≈ %.3e\n', J);

% Residual on dense grid (for visualization)
dPhip   = dphi(Zp);
d2Phip  = d2phi(Zp);
a_row_d = 4*xx - 2;                 % 1 x Nx
term1d = (m .* dPhip) .* a_row_d;                  % Ns x Nx
term2d = (AminusB) * a_row_d;                      % Ns x Nx
term3d = - nu * ((m.^2) .* d2Phip);                % Ns x Nx
term4d = 4*Phip;                                   % Ns x Nx
term5d = -4 * (Ai * (1 - xx));                     % Ns x Nx
term6d = -4 * (Bi * xx);                           % Ns x Nx
Rpsi_dx = term1d + term2d + term3d + term4d + term5d + term6d; % Ns x Nx
Rd      = Rpsi_dx.';                               % Nx x Ns
fd      = 4*ones(Nx,1);
res_dense = Rd*c + fd;                             % Nx x 1

%% ------------------ Solution plot (context) + inset zoom ------------------
fs_axis=12; fs_label=14; lw=2.0;

uex = EXACT_SOLUTION_TYPE2(xx, nu);                % visualization only
maxErr = max(abs(uh - uex));

f1 = figure('Color','w'); f1.Units='inches'; f1.Position=[1 1 8 3.5];
axMain = axes('Parent', f1);
plot(axMain, xx, uh, 'LineWidth', lw); hold(axMain,'on');
plot(axMain, xx, uex, '--', 'LineWidth', lw);
xline(axMain, xs1_opt,'k--','$x_{s1}^\ast$','LineWidth',1.5,'FontSize',16,'Interpreter','latex');
xline(axMain, xs2_opt,'k--','$x_{s2}^\ast$','LineWidth',1.5,'FontSize',16,'Interpreter','latex');
grid(axMain,'on'); hold(axMain,'off');
xlabel(axMain,'$x$','Interpreter','latex','FontSize',fs_label);
ylabel(axMain,'$u(x)$','Interpreter','latex','FontSize',fs_label);
legend(axMain, {'Gated X-TFC','Exact'}, 'Interpreter','latex','Location','best','Box','off');
axMain.LineWidth=.8; axMain.FontSize=fs_axis; axMain.TickLabelInterpreter='latex'; axMain.Box='on'; axMain.XLim=[0 1];
title(axMain, sprintf('Twin BL: $\\nu=%.3g$, $x_{s1}^*=%.3f$, $x_{s2}^*=%.3f$, $\\varepsilon_1^*=%.2f$, $\\varepsilon_2^*=%.2f$, $\\max|e|=%.2e$', ...
      nu, xs1_opt, xs2_opt, eps1_opt, eps2_opt, maxErr), 'Interpreter','latex','FontSize',fs_axis);

% Inset #1: zoom near x = 1
axInsetR = axes('Parent', f1, 'Position',[0.58 0.20 0.30 0.30]);
plot(axInsetR, xx, uh, '-', 'LineWidth', lw); hold(axInsetR,'on');
plot(axInsetR, xx, uex, '--', 'LineWidth', lw);
grid(axInsetR,'on'); box(axInsetR,'on');
xZoomStart = max(0.8, 1 - 10*nu);
xlim(axInsetR, [xZoomStart 1]);
idxR = (xx >= xZoomStart) & (xx <= 1);
yminR = min( [uh(idxR),  uex(idxR)] );
ymaxR = max( [uh(idxR),  uex(idxR)] );
marginR = 0.02 * max(1, ymaxR - yminR);
ylim(axInsetR, [yminR - marginR, ymaxR + marginR]);
set(axInsetR,'FontSize',12,'TickLabelInterpreter','latex');
title(axInsetR,'Zoom near $x=1$','Interpreter','latex','FontSize',12);

% Inset #2: zoom near x = 0
axInsetL = axes('Parent', f1, 'Position',[0.12 0.20 0.30 0.30]);
plot(axInsetL, xx, uh, '-', 'LineWidth', lw); hold(axInsetL,'on');
plot(axInsetL, xx, uex, '--', 'LineWidth', lw);
grid(axInsetL,'on'); box(axInsetL,'on');
xZoomEnd = min(0.001, max(0.02, 10*nu));    % small window near 0; never < 0.02
xlim(axInsetL, [0 xZoomEnd]);
idxL = (xx >= 0) & (xx <= xZoomEnd);
yminL = min( [uh(idxL),  uex(idxL)] );
ymaxL = max( [uh(idxL),  uex(idxL)] );
marginL = 0.02 * max(1, ymaxL - yminL);
ylim(axInsetL, [yminL - marginL, ymaxL + marginL]);
set(axInsetL,'FontSize',12,'TickLabelInterpreter','latex');
title(axInsetL,'Zoom near $x=0$','Interpreter','latex','FontSize',12);

% exportgraphics(f1,'FIG_SOLUTION_VS_EXACT_2SPLITS_BO_TYPE2.pdf','ContentType','vector');
filename = sprintf('FIG_SOLUTION_VS_EXACT_2SPLITS_BO_TYPE2_NU_%g.pdf', nu);
exportgraphics(f1, filename, 'ContentType','vector');
%% ------------------ 2×1 diagnostics: residual & abs error ------------------
f2 = figure('Color','w'); f2.Units='inches'; f2.Position=[1 1 8 7];

% (a) Pointwise PDE residual
subplot(2,1,1);
abs_res = abs(res_dense(:)).';
if max(abs_res) / max(min(abs_res(abs_res>0)), eps) > 1e3
    semilogy(xx, abs_res, 'LineWidth', lw);
else
    plot(xx, abs_res, 'LineWidth', lw);
end
hold on; 
xline(xs1_opt,'r--','$x_{s1}^\ast$','LineWidth',2.0,'FontSize',16,'Interpreter','latex');
xline(xs2_opt,'r--','$x_{s2}^\ast$','LineWidth',2.0,'FontSize',16,'Interpreter','latex');
hold off; grid on;
xlabel('$x$','Interpreter','latex','FontSize',fs_label);
ylabel('$|\mathcal{L}_\nu[u_h](x)|$','Interpreter','latex','FontSize',fs_axis);
title('PDE residual','Interpreter','latex','FontSize',fs_axis);
ax1=gca; ax1.LineWidth=.8; ax1.FontSize=fs_axis; ax1.TickLabelInterpreter='latex'; ax1.Box='on'; ax1.XLim=[0 1];

% (b) Pointwise absolute error
subplot(2,1,2);
abs_err = abs(uh(:).' - uex(:).');
if max(abs_err) / max(min(abs_err(abs_err>0)), eps) > 1e3
    semilogy(xx, abs_err, 'LineWidth', lw);
else
    plot(xx, abs_err, 'LineWidth', lw);
end
hold on; 
xline(xs1_opt,'r--','$x_{s1}^\ast$','LineWidth',2.0,'FontSize',16,'Interpreter','latex');
xline(xs2_opt,'r--','$x_{s2}^\ast$','LineWidth',2.0,'FontSize',16,'Interpreter','latex');
hold off; grid on;
xlabel('$x$','Interpreter','latex','FontSize',fs_label);
ylabel('$|u_h(x)-u_{exact}(x)|$','Interpreter','latex','FontSize',fs_axis);
title('Pointwise absolute error','Interpreter','latex','FontSize',fs_axis);
ax2=gca; ax2.LineWidth=.8; ax2.FontSize=fs_axis; ax2.TickLabelInterpreter='latex'; ax2.Box='on'; ax2.XLim=[0 1];
% exportgraphics(f2,'FIG_RESIDUAL_AND_ERROR_2SPLITS_BO_TYPE2.pdf','ContentType','vector');
filename = sprintf('FIG_RESIDUAL_AND_ERROR_2SPLITS_BO_TYPE2_NU_%g.pdf', nu);
exportgraphics(f2, filename, 'ContentType','vector');
%% ------------------ Width/gate profile figure ------------------
f3 = figure('Color','w'); f3.Units='inches'; f3.Position=[1 1 8 7];

% (a) Blended widths vs centers
subplot(2,1,1);
plot(alpha_star, sig_x, 'LineWidth', lw); hold on;
% yline(sigma_blk(1), '--', '$\sigma_1$','Interpreter','latex','FontSize',16,'Color','k');
% yline(sigma_blk(2), '--', '$\sigma_2$','Interpreter','latex','FontSize',16,'Color','k');
% yline(sigma_blk(3), '--', '$\sigma_3$','Interpreter','latex','FontSize',16,'Color','k');
xline(xs1_opt,'r--','$x_{s1}^\ast$','LineWidth',2.0,'FontSize',16,'Interpreter','latex');
xline(xs2_opt,'r--','$x_{s2}^\ast$','LineWidth',2.0,'FontSize',16,'Interpreter','latex');
hold off; grid on;
xlabel('$\alpha$','Interpreter','latex','FontSize',fs_label);
ylabel('$\sigma_x(\alpha)$','Interpreter','latex','FontSize',fs_label);
title(sprintf('RBF width profile  ($\\varepsilon_1^*=%.2f$, $\\varepsilon_2^*=%.2f$)', ...
      eps1_opt, eps2_opt),'Interpreter','latex','FontSize',fs_axis);
axw=gca; axw.LineWidth=.8; axw.FontSize=fs_axis; axw.TickLabelInterpreter='latex'; axw.Box='on'; axw.XLim=[0 1];

% (b) Logistic gates s1, s2 across the splits
subplot(2,1,2);
plot(alpha_star, gates.s1, 'LineWidth', lw); hold on;
plot(alpha_star, gates.s2, 'LineWidth', lw);
xline(xs1_opt,'r--','$x_{s1}^\ast$','LineWidth',2.0,'FontSize',16,'Interpreter','latex');
xline(xs2_opt,'r--','$x_{s2}^\ast$','LineWidth',2.0,'FontSize',16,'Interpreter','latex');
hold off; grid on;
xlabel('$\alpha$','Interpreter','latex','FontSize',fs_label);
ylabel('$s_1(\alpha),\, s_2(\alpha)$','Interpreter','latex','FontSize',fs_label);
legend({'$s_1$','$s_2$'},'Interpreter','latex','Location','best','Box','off');
title(sprintf('Logistic gates  ($\\varepsilon_1^*=%.2f$, $\\varepsilon_2^*=%.2f$)', ...
      eps1_opt, eps2_opt),'Interpreter','latex','FontSize',fs_axis);
axg=gca; axg.LineWidth=.8; axg.FontSize=fs_axis; axg.TickLabelInterpreter='latex'; axg.Box='on'; axg.XLim=[0 1];
% exportgraphics(f3,'FIG_WIDTH_GATE_PROFILES_2SPLITS_BO_TYPE2.pdf','ContentType','vector');
filename = sprintf('FIG_WIDTH_GATE_PROFILES_2SPLITS_BO_TYPE2_NU_%g.pdf', nu);
exportgraphics(f3, filename, 'ContentType','vector');
toc;

%% ======================= BayesOpt objective (Type-2 PDE) =======================
function val = objective_BO_type2(X, xL, xR, Nc, Nstar, k, nu, BL, BR)
    % X table has: w1, w3, eps1, eps2
    w1   = X.w1;
    w3   = X.w3;
    w2   = 1 - w1 - w3;           % ∈ [0.60, 0.98] by construction

    xs1  = xL + w1;
    xs2  = xs1 + w2;

    eps1 = X.eps1;
    eps2 = X.eps2;

    [x_pde, alpha_star, sig_x] = PDE_and_Kernel_Centers_3block( ...
        xL, xR, xs1, xs2, Nc, Nstar, k, eps1, eps2, nu);

    % ===== residual-only objective on an independent validation grid =====
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
    Phi  = phi(Z);
    dPhi = dphi(Z); 
    d2Phi= d2phi(Z);

    a_row = 4*x - 2;                % 1 x Nf
    AminusB = Ai - Bi;

    % Lν[ψ_i] for Type-2 on training grid
    term1 = (m .* dPhi) .* a_row;                   % Ns x Nf
    term2 = (AminusB) * a_row;                      % Ns x Nf
    term3 = - nu * ((m.^2) .* d2Phi);               % Ns x Nf
    term4 = 4*Phi;                                   % Ns x Nf
    term5 = -4 * (Ai * (1 - x));                     % Ns x Nf
    term6 = -4 * (Bi * x);                           % Ns x Nf
    R_i = term1 + term2 + term3 + term4 + term5 + term6; % Ns x Nf

    R   = R_i.';                                    % Nf x Ns
    f   = 4 * ones(Nf,1);                           % Lν[g=1] = 4

    % Solve for c (ridge LS)
    Rw=R.*Wsqrt; fw=f.*Wsqrt;
    lambda=1e-10;
    c = (Rw.'*Rw + lambda*eye(Ns)) \ (-Rw.'*fw);

    % Validation grid (independent) → residual energy only
    Nxv=800; xv=linspace(xL,xR,Nxv);
    Zp     = m*xv + b;                  % Ns x Nxv
    Phip   = phi(Zp);
    dPhip  = dphi(Zp); 
    d2Phip = d2phi(Zp);
    a_row_v = 4*xv - 2;                 % 1 x Nxv

    term1v = (m .* dPhip) .* a_row_v;
    term2v = (AminusB) * a_row_v;
    term3v = - nu * ((m.^2) .* d2Phip);
    term4v = 4*Phip;
    term5v = -4 * (Ai * (1 - xv));
    term6v = -4 * (Bi * xv);
    Rpsi_v = term1v + term2v + term3v + term4v + term5v + term6v;  % Ns x Nxv

    Rv     = Rpsi_v.';                  % Nxv x Ns
    fv     = 4*ones(Nxv,1);
    resv   = Rv*c + fv;                 % Nxv x 1

    % Objective value: L2 residual on validation grid
    val  = sqrt(trapz(xv, (resv.').^2));
end

%% ======================= Helpers (three-block geometry) =======================
function [x_pde, alpha_star, sigma_x, sigma_blk, eps_eff, gates] = ...
    PDE_and_Kernel_Centers_3block(xL, xR, xs1, xs2, Nc, Nstar, k, eps1, eps2, nu)
% Three-block partition + smooth width blends via TWO logistic gates.
    assert(xL < xs1 && xs1 < xs2 && xs2 < xR, 'Require xL < xs1 < xs2 < xR');
    assert(Nc > 0 && Nstar > 0, 'Nc and Nstar must be positive');

    % --- Collocation points (omit duplicates at splits)
    x1 = linspace(xL,  xs1, Nc+1)'; x1(end)=[];
    x2 = linspace(xs1, xs2, Nc+1)'; x2(end)=[];
    x3 = linspace(xs2, xR,  Nc)'; 
    x_pde = [x1; x2; x3];

    % --- RBF centers (omit duplicates at splits)
    a1 = linspace(xL,  xs1, Nstar+1)'; a1(end)=[];
    a2 = linspace(xs1, xs2, Nstar+1)'; a2(end)=[];
    a3 = linspace(xs2, xR,  Nstar)'; 
    alpha_star = [a1; a2; a3];

    % --- Block spacings and base widths
    d1 = (xs1 - xL)  / max(Nstar,1);
    d2 = (xs2 - xs1) / max(Nstar,1);
    d3 = (xR  - xs2) / max(Nstar,1);
    sigma1 = k*d1; 
    sigma2 = k*d2; 
    sigma3 = k*d3;
    sigma_blk = [sigma1; sigma2; sigma3];

    % --- Smooth transition scales (geometry vs physics)
    eps_geo1 = eps1 * min(d1, d2);
    eps_geo2 = eps2 * min(d2, d3);
    eps_phy1 = 5*nu; 
    eps_phy2 = 5*nu;
    epsb1    = max(eps_geo1, eps_phy1);
    epsb2    = max(eps_geo2, eps_phy2);
    eps_eff  = [epsb1; epsb2];

    % --- Two logistic gates centered at xs1 and xs2
    t1 = (alpha_star - xs1) / epsb1;
    t2 = (alpha_star - xs2) / epsb2;
    s1 = 1 ./ (1 + exp(-t1));   % rises across xs1
    s2 = 1 ./ (1 + exp(-t2));   % rises across xs2

    % Partition-of-unity weights: left/mid/right
    wL = (1 - s1);
    wM = s1 .* (1 - s2);
    wR = s1 .* s2;
    % wL + wM + wR == 1

    % Blended widths at each center
    sigma_x = sigma1 .* wL + sigma2 .* wM + sigma3 .* wR;

    if nargout >= 6
        gates.s1 = s1;
        gates.s2 = s2;
    end
end

%% ======================= Exact solution (Type-2) =======================
% function u_exact = EXACT_SOLUTION_TYPE2(X, nu)
% % Exact: u(x) = exp(-2 x (1-x) / nu), with branch protection.
%     exponent = -2 .* X .* (1 - X) ./ nu;
%     % Prevent huge negative underflow warnings; exp will safely go to 0
%     threshold = -log(realmax('double')); %#ok<NASGU> % not strictly needed
%     u_exact = exp(exponent);
% end

function u_exact = EXACT_SOLUTION_TYPE2(X, nu)
    % Numerically stable evaluation of u(x) = exp(-2x(1-x)/nu)
    % Handles small values of nu (e.g., 1e-15)

    % Compute the exponent
    exponent = -2 .* X .* (1 - X) ./ nu;

    % Use a threshold to suppress underflow
    % exp(-745) ~ realmin for double precision
    underflow_threshold = -745;

    % Evaluate safely
    u_exact = exp(exponent);

    % Set very small values (underflow) to 0
    u_exact(exponent < underflow_threshold) = 0;

    % Optional: force u(0) = u(1) = 1 when X is exactly 0 or 1
    u_exact(X == 0 | X == 1) = 1;
end
