%% Ultrasound tomography simulation with k-Wave
% This script simulates 2-D transmission ultrasound tomography using k-Wave.
% It creates a heterogeneous sound-speed phantom, fires each element in a
% contour-following array, estimates first-arrival time shifts, and reconstructs a
% slowness perturbation map with a simple straight-ray least-squares model.
%
% Requirements:
%   - MATLAB
%   - k-Wave toolbox on the MATLAB path: http://www.k-wave.org/
%   - Signal Processing Toolbox for hilbert() is helpful, but a fallback
%     envelope estimator is included.

clear; close all; clc;

if exist('kspaceFirstOrder2D', 'file') ~= 2
    error('k-Wave was not found on the MATLAB path. Add it first, e.g. addpath(genpath(''C:\path\to\k-Wave''));');
end

%% Simulation controls
Nx = 192;                 % grid points in x
Ny = 192;                 % grid points in y
dx = 2.60416e-3;             % grid spacing [m]
dy = dx;
c0 = 1500;                % background sound speed [m/s]
rho0 = 1000;              % density [kg/m^3]
num_elements = 64;        % transducer count
array_standoff = 0;       % outward offset from abdomen boundary [m]
abdomen_mask_threshold = 1490; % sound-speed threshold used to find abdomen edge [m/s]
source_freq = 0.600e6;     % source centre frequency [Hz]
source_cycles = 3;        % tone burst length
ppw_min = c0 / source_freq / dx;

fprintf('Points per wavelength at c0: %.2f\n', ppw_min);
if ppw_min < 3
    warning('The grid is coarse for this frequency. Lower source_freq or reduce dx for higher accuracy.');
end

%% Grid and time axis
load ct_sound_speed_30000025112415592178200005404_192x192_m_s.mat;
Vq(135:end,:) = 1480.0;

kgrid = kWaveGrid(Nx, dx, Ny, dy);

medium.sound_speed = c0 * ones(Nx, Ny);
medium.density = rho0 * ones(Nx, Ny);
medium.alpha_coeff = 0.35;     % [dB/(MHz^y cm)]
medium.alpha_power = 1.5;

% Build a compact phantom: one fast inclusion, one slow inclusion, one ring.
[X, Y] = ndgrid(kgrid.x_vec, kgrid.y_vec);

medium.sound_speed = Vq;
%medium.density = density_kg_m3;

phantom_mask = medium.sound_speed;
phantom_mask(phantom_mask==1480.0) = 0.0;
phantom_mask(phantom_mask~=0.0) = 1.0;
phantom_mask(134,:) = 0.0;

%% Contour-following array coordinates
% The transducer elements are sampled uniformly by arc length along the
% outer edge of the abdomen phantom. Adjust abdomen_mask_threshold if the
% loaded sound-speed map uses a different background value.

% abdomen_mask = build_abdomen_mask(medium.sound_speed, abdomen_mask_threshold);
% [boundary_i, boundary_j] = outer_boundary_ordered(abdomen_mask);
% [element_i, element_j] = resample_closed_boundary(boundary_i, boundary_j, num_elements);
% 
% if array_standoff ~= 0
%     [element_i, element_j] = offset_boundary_points(kgrid, element_i, element_j, array_standoff);
% end
% 
% element_linear_index = sub2ind([Nx, Ny], element_i, element_j);
% [~, unique_idx] = unique(element_linear_index, 'stable');
% element_i = element_i(unique_idx);
% element_j = element_j(unique_idx);
% num_elements = numel(element_i);
% element_x = kgrid.x_vec(element_i).';
% element_y = kgrid.y_vec(element_j).';

element_x = [100.981 100.981 100.981 100.981 100.981 100.67 98.286 94.191 ...
    88.481 81.294 72.804 63.213 52.752 41.673 30.241 17.00 3.75 -9.158 -21.135 ...
    -32.752 -43.836 -54.221 -63.754 -72.292 -79.707 -85.889 -90.746 -94.206 ...
    -96.216 -95.981 -95.981 -95.981];
element_x = [element_x, flip(element_x)] * 1e-3 - 2*dx;
element_y = [-6.75 -20.00 -33.25 -46.50 -59.75 -71.699 -82.966 -93.729 ...
    -103.73 -112.729 -120.509 -126.884 -131.70 -134.842 -136.235 -135.981 ...
    -135.981 -136.019 -133.799 -130.136 -125.085 -118.721 -111.138 -102.45 ...
    -92.786 -82.291 -71.121 -59.442 -47.428 -35.00 -21.28 -7.56];
element_y = [element_y, flip(-element_y)] * 1e-3;

[element_i, element_j] = cart2grid_index(kgrid, element_x, element_y);
element_x = kgrid.x_vec(element_i).';
element_y = kgrid.y_vec(element_j).';

cfl = 0.25;
max_tx_rx_distance = max_pairwise_distance(element_x, element_y);
valid_sound_speed = medium.sound_speed(isfinite(medium.sound_speed) & medium.sound_speed > 0);
t_end = 1.35 * max_tx_rx_distance / min(valid_sound_speed(:));
kgrid.makeTime(max(valid_sound_speed(:)), cfl, t_end);

sensor.mask = [element_x; element_y];
sensor.record = {'p'};

source_signal = toneBurst(1 / kgrid.dt, source_freq, source_cycles);
input_args = { ...
    'PMLInside', false, ...
    'PMLSize', 20, ...
    'DataCast', 'single', ...
    'PlotSim', false};

%% Simulate all transmit events
num_time = numel(kgrid.t_array);
pressure_data = zeros(num_elements, num_elements, num_time, 'single');

fprintf('Running %d transmit events...\n', num_elements);
for tx = 1:num_elements
    source = struct();
    source.p_mask = zeros(Nx, Ny);
    source.p_mask(element_i(tx), element_j(tx)) = 1;
    source.p = source_signal;

    sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
    pressure_data(tx, :, :) = single(sensor_data.p);
    fprintf('  tx %02d/%02d complete\n', tx, num_elements);
end

%% Estimate first-arrival time shifts relative to a homogeneous background
arrival_time = nan(num_elements, num_elements);
baseline_time = nan(num_elements, num_elements);
min_separation = 3;       % ignore receivers too close along the contour
threshold_fraction = 0.18;

for tx = 1:num_elements
    for rx = 1:num_elements
        contour_gap = min(mod(rx - tx, num_elements), mod(tx - rx, num_elements));
        if contour_gap < min_separation
            continue;
        end

        trace = double(squeeze(pressure_data(tx, rx, :)));
        direct_distance = hypot(element_x(rx) - element_x(tx), element_y(rx) - element_y(tx));
        baseline_time(tx, rx) = direct_distance / c0;

        gate_start_time = max(0, baseline_time(tx, rx) - 6 / source_freq);
        gate_start_index = max(1, floor(gate_start_time / kgrid.dt));
        arrival_time(tx, rx) = first_arrival_time(trace, kgrid.t_array, gate_start_index, threshold_fraction);
    end
end

time_shift = arrival_time - baseline_time;

%% Straight-ray slowness reconstruction
recon_N = 80;
x_margin = 2 * dx;
y_margin = 2 * dy;
x_edges = linspace(min(element_x) - x_margin, max(element_x) + x_margin, recon_N + 1);
y_edges = linspace(min(element_y) - y_margin, max(element_y) + y_margin, recon_N + 1);
x_centres = 0.5 * (x_edges(1:end-1) + x_edges(2:end));
y_centres = 0.5 * (y_edges(1:end-1) + y_edges(2:end));
[XR, YR] = ndgrid(x_centres, y_centres);
recon_roi = hypot(X, Y) <= 1000e-3;

valid = isfinite(time_shift);
[tx_list, rx_list] = find(valid);
num_rays = numel(tx_list);
num_pixels = recon_N * recon_N;

fprintf('Building straight-ray system with %d rays...\n', num_rays);
ray_rows = cell(num_rays, 1);
ray_cols = cell(num_rays, 1);
ray_vals = cell(num_rays, 1);

for ray = 1:num_rays
    tx = tx_list(ray);
    rx = rx_list(ray);
    [cols, vals] = ray_pixel_lengths( ...
        element_x(tx), element_y(tx), element_x(rx), element_y(rx), ...
        x_edges, y_edges, recon_N, recon_roi);
    ray_rows{ray} = ray * ones(size(cols));
    ray_cols{ray} = cols;
    ray_vals{ray} = vals;
end

A = sparse(cell2mat(ray_rows), cell2mat(ray_cols), cell2mat(ray_vals), num_rays, num_pixels);
b = time_shift(valid);

% Mild Tikhonov regularisation stabilises the noisy, sparse inverse problem.
lambda = 1e-2;
A_aug = [A; lambda * speye(num_pixels)];
b_aug = [b; zeros(num_pixels, 1)];
slowness_delta = lsqr(A_aug, b_aug, 1e-14, 5000);

c_recon = c0 ./ (1 + c0 * reshape(slowness_delta, recon_N, recon_N));
c_recon(~recon_roi) = NaN;

%% Visualisation
figure('Color', 'w', 'Name', 'k-Wave ultrasound tomography');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
imagesc(kgrid.y_vec * 1e3, kgrid.x_vec * 1e3, medium.sound_speed);
axis image; colormap(gca, turbo); colorbar;
title('True sound speed [m/s]');
xlabel('y [mm]'); ylabel('x [mm]');
hold on; plot(element_y * 1e3, element_x * 1e3, 'w.', 'MarkerSize', 8);

nexttile;
imagesc(1:num_elements, 1:num_elements, time_shift * 1e6);
axis image; colorbar; colormap(gca, parula);
title('Measured time shift [\mus]');
xlabel('Receiver'); ylabel('Transmitter');

nexttile;
plot(kgrid.t_array * 1e6, squeeze(pressure_data(1, round(num_elements/2), :)), 'k');
grid on;
title('Example received waveform');
xlabel('Time [\mus]'); ylabel('Pressure [Pa]');

nexttile;
imagesc(y_centres * 1e3, x_centres * 1e3, c_recon);
axis image; colormap(gca, turbo); colorbar;
title('Straight-ray reconstructed sound speed [m/s]');
xlabel('y [mm]'); ylabel('x [mm]');

fprintf('Done. Reconstruction range inside ROI: %.1f to %.1f m/s\n', ...
    min(c_recon(recon_roi), [], 'omitnan'), max(c_recon(recon_roi), [], 'omitnan'));

%% Local functions
function mask = build_abdomen_mask(sound_speed, threshold)
    mask = isfinite(sound_speed) & sound_speed > threshold;
    mask = largest_component(mask);
    mask = fill_mask_holes(mask);
end

function mask = largest_component(mask)
    if exist('bwconncomp', 'file') == 2
        cc = bwconncomp(mask);
        if cc.NumObjects == 0
            error('No abdomen mask pixels were found. Lower abdomen_mask_threshold.');
        end
        [~, largest] = max(cellfun(@numel, cc.PixelIdxList));
        out = false(size(mask));
        out(cc.PixelIdxList{largest}) = true;
        mask = out;
        return;
    end

    % Toolbox-free fallback: keep the mask as-is. The boundary extraction
    % below will still work if the threshold leaves one dominant object.
    if ~any(mask(:))
        error('No abdomen mask pixels were found. Lower abdomen_mask_threshold.');
    end
end

function mask = fill_mask_holes(mask)
    if exist('imfill', 'file') == 2
        mask = imfill(mask, 'holes');
    end
end

function [boundary_i, boundary_j] = outer_boundary_ordered(mask)
    if exist('bwboundaries', 'file') == 2
        boundaries = bwboundaries(mask, 'noholes');
        if isempty(boundaries)
            error('No abdomen boundary points were found.');
        end
        [~, largest_boundary] = max(cellfun(@(b) size(b, 1), boundaries));
        boundary = boundaries{largest_boundary};
        boundary_i = boundary(:, 1);
        boundary_j = boundary(:, 2);
        return;
    end

    boundary = mask & (conv2(double(mask), ones(3), 'same') < 9);
    [ii, jj] = find(boundary);
    if isempty(ii)
        error('No abdomen boundary points were found.');
    end

    centroid_i = mean(ii);
    centroid_j = mean(jj);
    angle = atan2(ii - centroid_i, jj - centroid_j);
    [~, order] = sort(angle);
    boundary_i = ii(order);
    boundary_j = jj(order);
end

function [sample_i, sample_j] = resample_closed_boundary(boundary_i, boundary_j, num_samples)
    points = [boundary_i(:), boundary_j(:)];
    points(end + 1, :) = points(1, :);
    segment_length = sqrt(sum(diff(points, 1, 1).^2, 2));
    arc_length = [0; cumsum(segment_length)];

    keep = [true; diff(arc_length) > 0];
    points = points(keep, :);
    arc_length = arc_length(keep);

    sample_arc = linspace(0, arc_length(end), num_samples + 1).';
    sample_arc(end) = [];
    sample_i = round(interp1(arc_length, points(:, 1), sample_arc, 'linear'));
    sample_j = round(interp1(arc_length, points(:, 2), sample_arc, 'linear'));
end

function [offset_i, offset_j] = offset_boundary_points(kgrid, element_i, element_j, standoff)
    element_x = kgrid.x_vec(element_i).';
    element_y = kgrid.y_vec(element_j).';
    center_x = mean(element_x);
    center_y = mean(element_y);
    normal_x = element_x - center_x;
    normal_y = element_y - center_y;
    normal_length = hypot(normal_x, normal_y);
    normal_x = normal_x ./ max(normal_length, eps);
    normal_y = normal_y ./ max(normal_length, eps);

    offset_x = element_x + standoff * normal_x;
    offset_y = element_y + standoff * normal_y;
    [offset_i, offset_j] = cart2grid_index(kgrid, offset_x, offset_y);
end

function distance = max_pairwise_distance(x, y)
    distance = 0;
    for n = 1:numel(x)
        distance = max(distance, max(hypot(x(:) - x(n), y(:) - y(n))));
    end
end

function roi = roi_from_mask(kgrid, mask, xq, yq)
    ii = interp1(kgrid.x_vec, 1:numel(kgrid.x_vec), xq(:), 'nearest', 'extrap');
    jj = interp1(kgrid.y_vec, 1:numel(kgrid.y_vec), yq(:), 'nearest', 'extrap');
    roi = reshape(mask(sub2ind(size(mask), ii(:), jj(:))), size(xq));
end

function [ii, jj] = cart2grid_index(kgrid, x, y)
    ii = zeros(size(x));
    jj = zeros(size(y));
    for n = 1:numel(x)
        [~, ii(n)] = min(abs(kgrid.x_vec - x(n)));
        [~, jj(n)] = min(abs(kgrid.y_vec - y(n)));
    end
end

function t0 = first_arrival_time(trace, t_array, start_index, threshold_fraction)
    trace = trace(:);
    trace = trace - mean(trace(1:max(5, min(start_index, numel(trace)))));
    if exist('hilbert', 'file') == 2
        env = abs(hilbert(trace));
    else
        env = sqrt(movmean(trace.^2, 9));
    end

    env(1:start_index) = 0;
    threshold = threshold_fraction * max(env);
    hit = find(env >= threshold, 1, 'first');

    if isempty(hit) || hit == 1
        t0 = NaN;
        return;
    end

    % Linear interpolation around threshold crossing for sub-sample timing.
    y1 = env(hit - 1);
    y2 = env(hit);
    frac = (threshold - y1) / max(eps, y2 - y1);
    t0 = t_array(hit - 1) + frac * (t_array(hit) - t_array(hit - 1));
end

function [cols, vals] = ray_pixel_lengths(x1, y1, x2, y2, x_edges, y_edges, n, roi)
    ray_length = hypot(x2 - x1, y2 - y1);
    sample_step = min(diff(x_edges(1:2)), diff(y_edges(1:2))) / 2;
    num_samples = max(2, ceil(ray_length / sample_step));

    s = linspace(0, 1, num_samples);
    xs = x1 + s * (x2 - x1);
    ys = y1 + s * (y2 - y1);
    segment_length = ray_length / (num_samples - 1);

    ix = discretize(xs, x_edges);
    iy = discretize(ys, y_edges);
    inside = ~isnan(ix) & ~isnan(iy);
    ix = ix(inside);
    iy = iy(inside);

    keep = roi(sub2ind([n, n], ix, iy));
    ix = ix(keep);
    iy = iy(keep);

    cols = sub2ind([n, n], ix(:), iy(:));
    if isempty(cols)
        vals = [];
        return;
    end

    [cols, ~, groups] = unique(cols);
    vals = accumarray(groups, segment_length, [], @sum);
end
