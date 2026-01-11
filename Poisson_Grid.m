clc; clear; close all;
w = [0.5, 0.5, 0.01];
alpha_grid = 0.8;

[Centers, Sigma] = generate_rbf_centers_sigma(w, alpha_grid);
plot_centers_sigma(Centers, Sigma, w, alpha_grid);


function [Centers, Sigma] = generate_rbf_centers_sigma(w, alpha)
% GENERATE_RBF_CENTERS_SIGMA  Build combined (global + local) RBF centers and widths.
% Inputs:
%   w      : [cx, cy, radius]  (tunable)
%   alpha  : alpha_grid for Chebyshev-like global grid (tunable)
% Outputs:
%   Centers: M-by-2 center coordinates in [0,1]^2
%   Sigma  : M-by-1 isotropic widths per center
%
% All other knobs are hard-coded below.

    % ---------------- Hard-coded design choices ----------------
    % Local (inside contour) sampling:
    N_inside   = 100;        % number of inside points
    nc         = 10;         % control knots for the periodic spline
    nl         = 200;        % samples/segment to draw a smooth polygon
    k_local    = 6;          % k-NN for local width
    alpha_loc  = 0.9;        % scale for local width from k-NN
    beta_cap   = 1.0;        % cap local sigma by beta * global sigma at that location

    % Global (structured) grid:
    nx = 31; ny = 31;        % resolution of global grid
    c_sigma   = 5.0;         % global sigma scale factor
    gamma_end = 0.8;         % end clustering warp

    % ---------------- Build spline & sample inside points ----------------
    center = w(1:2);
    radius = w(3);
    pre    = periodic_spline_precompute(nc, nl);
    [x0, y0] = control_knots_circle(center, radius, nc);
    Pin = sample_points_inside(pre, x0, y0, N_inside);  

    % ---------------- Global grid centers + widths -----------------------
    [Cg, sigma_g] = get_rbf_grid_sigma_vec(alpha, nx, ny, c_sigma, gamma_end);

    % ---------------- Local widths from k-NN, blended via cap ------------
    sigma_loc = local_sigma_knn(Pin, k_local, alpha_loc);
    Fsg = scatteredInterpolant(Cg(:,1), Cg(:,2), sigma_g, 'natural', 'nearest');
    sigma_cap = Fsg(Pin(:,1), Pin(:,2));
    sigma_loc = min(sigma_loc, beta_cap * sigma_cap);

    % ---------------- Combine -------------------------------------------
    Centers = [Cg; Pin];
    Sigma   = [sigma_g; sigma_loc];
end

% ========================= Subfunctions =========================

function [Pin] = sample_points_inside(pre, x0, y0, N)
    % Uniform N points inside spline via rejection sampling
    [xsi, ysi] = periodic_spline_eval(pre, x0, y0);
    areaA = polyarea(xsi, ysi);
    fracIn = max(min(areaA, 1-1e-12), 1e-12);
    Pin = zeros(N,2); nin = 0;
    while nin < N
        need = N - nin;
        K = max(1000, ceil(1.5*need/fracIn));
        U = rand(K,2);
        [in, on] = inpolygon(U(:,1), U(:,2), xsi, ysi);
        in = in | on;
        take = min(sum(in), need);
        if take > 0
            Pin(nin+(1:take), :) = U(find(in, take, 'first'), :);
            nin = nin + take;
        end
    end
end

function sigma = local_sigma_knn(P, k, alpha_loc)
    % Isotropic width = alpha_loc * (distance to k-th nearest neighbor)
    [N,dim] = size(P); if dim~=2, error('P must be N-by-2'); end
    k = max(1, min(k, N-1));
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

function [C, sigma_iso] = get_rbf_grid_sigma_vec(alpha_grid, nx, ny, c_sigma, gamma_end)
    % Chebyshev-like grid on [0,1]^2 with isotropic width from geometric mean spacing
    x1 = cheb_like_1d(nx, alpha_grid, gamma_end);
    y1 = cheb_like_1d(ny, alpha_grid, gamma_end);
    [X, Y] = meshgrid(x1, y1);
    dx = nodal_spacing(x1); dy = nodal_spacing(y1);
    [DX, DY] = meshgrid(dx, dy);
    H = sqrt(DX .* DY);
    sigma_iso = c_sigma * H(:);
    C = [X(:), Y(:)];
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

function [xsi, ysi] = periodic_spline_eval(pre, x0, y0)
    if numel(x0) ~= pre.nc || numel(y0) ~= pre.nc
        error('x0 and y0 must each have length nc=%d.', pre.nc);
    end
    x0 = x0(:); y0 = y0(:);
    coeff_x = pre.S * x0;
    coeff_y = pre.S * y0;
    xsi = pre.B * coeff_x; ysi = pre.B * coeff_y;
end

function pre = periodic_spline_precompute(nc, nl)
    s  = linspace(0,1,nc+1);
    M  = periodic_block(s);
    R  = spline_RHS_block(s);
    B  = interpolation_blocks(s, nl);
    pre = struct('nc', nc, 'nl', nl, 's', s, 'M', M, 'R', R, 'B', B);
    try
        pre.decM = decomposition(M);
        pre.S    = pre.decM \ R;
        pre.useDec = true;
    catch
        [L,U,P] = lu(M);
        pre.S = U \ (L \ (P*R));
        pre.useDec = false;
    end
end

function M=periodic_block(s)
    n = length(s)-1;
    M = zeros(4*n,4*n);
    for i=1:n
        j=1+4*(i-1);
        M(j,  j+(0:3)) = [s(i)^3,   s(i)^2,   s(i),   1];
        M(j+1,j+(0:3)) = [s(i+1)^3, s(i+1)^2, s(i+1), 1];
        if i==n
            M(j+2,j+(0:2)) = [3*s(i+1)^2, 2*s(i+1), 1];
            M(j+2,1:3)     = M(j+2,1:3) + [-3*s(1)^2, -2*s(1), -1];
            M(j+3,j+(0:1)) = [6*s(i+1), 2];
            M(j+3,1:2)     = M(j+3,1:2) + [-6*s(1), -2];
        else
            M(j+2,j+(0:2))   = [3*s(i+1)^2, 2*s(i+1), 1];
            M(j+2,j+4+(0:2)) = M(j+2,j+4+(0:2)) + [-3*s(i+1)^2, -2*s(i+1), -1];
            M(j+3,j+(0:1))   = [6*s(i+1), 2];
            M(j+3,j+4+(0:1)) = M(j+3,j+4+(0:1)) + [-6*s(i+1), -2];
        end
    end
end

function RHS=spline_RHS_block(s)
    n = length(s)-1;
    RHS = zeros(4*n,n);
    for i=1:n-1
        j=1+4*(i-1);
        RHS(j,i)     = 1;
        RHS(j+1,i+1) = 1;
    end
    RHS(4*(n-1)+1,n) = 1;
    RHS(4*(n-1)+2,1) = 1;
end

function B=interpolation_blocks(s,nl)
    nc = length(s)-1;
    B  = zeros(nc*nl,4*nc);
    for i=1:nc
        ss = linspace(s(i), s(i+1), nl);
        base = 4*(i-1);
        for j=1:nl
            row = (i-1)*nl + j;
            t = ss(j);
            B(row, base+(1:4)) = [t^3, t^2, t, 1];
        end
    end
end

function [x0, y0] = control_knots_circle(center, radius, nc)
    cx = center(1); cy = center(2);
    ang = linspace(0, 2*pi, nc+1); ang(end) = [];
    x0 = cx + radius*cos(ang);
    y0 = cy + radius*sin(ang);
    if any(x0 < 0 | x0 > 1 | y0 < 0 | y0 > 1)
        warning('Some control knots lie outside [0,1]^2. Adjust center/radius.');
    end
end

function plot_centers_sigma(Centers, Sigma, w, alpha_grid, outFile)
% PLOT_CENTERS_SIGMA  Publication-quality filled contour of Sigma over [0,1]^2
% with RBF centers and the spline contour (from w = [cx, cy, radius]).
%
%   plot_centers_sigma(Centers, Sigma, w, alpha_grid)
%   plot_centers_sigma(Centers, Sigma, w, alpha_grid, outFile)
%
% Saves a 300 dpi PNG (to outFile if provided, else auto-named).

    % ---------- settings (publication style) ----------
    figSizeIn   = [3.5, 3.5];  % inches, good half-width size
    fs_axes     = 12;          % tick font size
    fs_labels   = 14;          % axis label font size
    fs_title    = 12;          % title font size
    nLevels     = 30;          % contour levels
    G           = 300;         % grid for contour field (denser for smoothness)
    lw_box      = 0.9;
    lw_spline   = 1.8;
    ms_centers  = 8;

    if nargin < 5 || isempty(outFile)
        outFile = sprintf('FIG_RBF_SIGMA_w_%0.3f_%0.3f_%0.3f_a_%0.2f.png', ...
            w(1), w(2), w(3), alpha_grid);
    end

    % ---------- field on a grid ----------
    [X,Y] = meshgrid(linspace(0,1,G), linspace(0,1,G));
    F = scatteredInterpolant(Centers(:,1), Centers(:,2), Sigma, 'natural', 'nearest');
    Z = F(X,Y);

    % ---------- spline overlay from w ----------
    nc = 10; nl = 200;
    pre = periodic_spline_precompute(nc, nl);
    [x0, y0] = control_knots_circle(w(1:2), w(3), nc);
    [xsi, ysi] = periodic_spline_eval(pre, x0, y0);

    % ---------- figure/axes ----------
    f = figure('Color','w','Units','inches');
    f.Position(3:4) = figSizeIn;  % width x height in inches
    ax = axes('Parent',f); hold(ax,'on'); box(ax,'on');
    ax.FontSize = fs_axes;
    ax.TickLabelInterpreter = 'latex';
    ax.LineWidth = 0.8;

    % ---------- plot ----------
    contourf(ax, X, Y, Z, nLevels, 'LineStyle','none');
    colormap(ax, parula);  % or turbo, depending on your journal preferences
    cb = colorbar('peer',ax);
    cb.TickLabelInterpreter = 'latex';
    cb.Label.String = '$\sigma$';
    cb.Label.Interpreter = 'latex';
    cb.Label.FontSize = fs_labels;

    plot(ax, [0 1 1 0 0],[0 0 1 1 0], 'k-', 'LineWidth', lw_box); % domain box
    plot(ax, xsi, ysi, 'k-', 'LineWidth', lw_spline);              % spline
    plot(ax, Centers(:,1), Centers(:,2), 'k.', 'MarkerSize', ms_centers);
    pause (1)
    axis(ax,'equal'); xlim(ax,[0 1]); ylim(ax,[0 1]);
    ax.XLabel.String = '$x$'; ax.YLabel.String = '$y$';
    ax.XLabel.Interpreter = 'latex'; ax.YLabel.Interpreter = 'latex';
    ax.XLabel.FontSize = fs_labels;   ax.YLabel.FontSize = fs_labels;

    % Title per your convention
    title(ax, sprintf('$\\theta_H=[%.3f, %.3f, %.3f, %.3f]$', w, alpha_grid), ...
      'Interpreter','latex', 'FontSize', fs_title);

    % Tight-ish layout
    ax.LooseInset = max(ax.TightInset, 0.02);

    % ---------- save @ 300 dpi ----------
    try
        exportgraphics(f, outFile, 'Resolution', 300);
    catch
        % fallback for older MATLAB
        set(f, 'PaperPositionMode','auto');
        print(f, outFile, '-dpng', '-r300');
    end
end

