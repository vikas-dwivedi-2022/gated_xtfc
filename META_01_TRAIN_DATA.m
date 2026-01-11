% BUILD_TRAINING_DATA_GATED_XTFC.m
% Sweep nu and record (xs_opt, eps_opt) from PDE-only KAX-TFC forward solver.
% Output: training_data.mat with [nu, xs_opt, eps_opt] per row.

clc; clear; close all; rng(0);
warning('off','all');

tic;

%% ------------------ Sweep over nu ------------------
nu_list = logspace(-3, -4, 500).';   % 120 x 1, from 1e-2 down to 1e-4

% Problem/domain
BL = 0; BR = 1;
xL = 0; xR = 1;

% Layout (per block) -- match your forward script
Nc    = 1000;      % collocation per block (total Nf = 2*Nc)
Nstar = 1000;      % centers per block    (total Ns = 2*Nstar)
k     = 1.5;       % width multiplier (shared L/R)

% Bounds for bilevel fminbnd
xs_lb  = 0.80;  xs_ub  = 0.999;    % reasonable xs window (layer near x=1)
e_lb   = 10;  e_ub   = 100.00;    % reasonable eps_scale window
% Tolerances / display
tolx_xs = 1e-4;   tolx_e = 100*1e-3;
opts_xs = optimset('TolX', tolx_xs, 'Display','off');
opts_e  = optimset('TolX', tolx_e,  'Display','off');

% xs_lb = 0.80;  xs_ub = 0.999;
% e_lb  = 10.0;  e_ub  = 100.0;
% 
% r = 1e-3;                              % target ~0.1% of range
% tolx_xs = r*(xs_ub - xs_lb) / (2*(1 + 0.5*(xs_lb+xs_ub)));   % ~5.2e-5
% tolx_e  = r*(e_ub  - e_lb ) / (2*(1 + 0.5*(e_lb +e_ub )));   % ~8.0e-4
% 
% opts_xs = optimset('TolX', tolx_xs, 'MaxIter',800,'MaxFunEvals',1600,'Display','off');
% opts_e  = optimset('TolX', tolx_e,  'MaxIter',800,'MaxFunEvals',1600,'Display','off');


% Storage
Nnu = numel(nu_list);
xs_opt_list  = nan(Nnu,1);
eps_opt_list = nan(Nnu,1);
J_outer_list = nan(Nnu,1);   % (optional) outer objective at xs*
J_inner_list = nan(Nnu,1);   % (optional) inner objective at eps*

fprintf('=== Building training set: optimizing (xs, eps_scale) for %d nu values ===\n', Nnu);

for i = 1:Nnu
    nu = nu_list(i);

    % ----- Outer objective: for fixed xs, minimize over eps_scale -----
    outer_obj = @(xs) inner_min_over_eps(xs, xL, xR, Nc, Nstar, k, nu, BL, BR, ...
                                         e_lb, e_ub, opts_e);

    try
        [xs_opt, f_outer, ~, ~] = fminbnd(outer_obj, xs_lb, xs_ub, opts_xs);
        [eps_opt, f_inner] = inner_argmin_eps(xs_opt, xL, xR, Nc, Nstar, k, nu, BL, BR, ...
                                              e_lb, e_ub, opts_e);
    catch ME
        warning('Optimization failed at nu=%.3e: %s', nu, ME.message);
        xs_opt  = NaN; eps_opt = NaN; f_outer = NaN; f_inner = NaN;
    end

    xs_opt_list(i)  = xs_opt;
    eps_opt_list(i) = eps_opt;
    J_outer_list(i) = f_outer;
    J_inner_list(i) = f_inner;

    fprintf('i=%3d/%3d  nu=%.3e  xs*=%.6f  eps*=%.3f  (J_out=%.3e, J_in=%.3e)\n', ...
            i, Nnu, nu, xs_opt, eps_opt, f_outer, f_inner);
end

% Pack and save (exactly 120x3 as requested)
training_data = [nu_list, xs_opt_list, eps_opt_list];
save('training_data.mat','training_data');

fprintf('\nSaved training_data.mat with size %dx%d [nu, xs_opt, eps_opt].\n', size(training_data,1), size(training_data,2));
toc;

%% ======================= Helpers (same formulation) =======================
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
