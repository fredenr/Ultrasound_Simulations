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
transducer_azimuth_deg = 0;   % CCW global rotation of all transducer positions [deg]
transducer_rotation_center = [0, 0];  % [x, y] global rotation center [m]

element_x = [96.981 96.981 96.981 96.981 94.981 94.721 92.530 88.765 ...
    83.517 76.910 69.105 60.289 50.670 40.488 29.979 17.00 3.75 -8.183 -19.199 ...
    -29.885 -40.080 -49.633 -58.401 -66.254 -73.075 -78.761 -83.228 -86.411 ...
    -88.260 -87.981 -87.981 -87.981];
element_x = [element_x, flip(element_x)] * 1e-3 - 2*dx;
element_y = [-6.750 -20.000 -33.250 -46.500 -59.750 -70.916 -81.273 -91.167 ...
    -100.361 -108.633 -115.785 -121.645 -126.072 -128.961 -130.241 -129.981 ...
    -129.981 -128.078 -126.037 -122.668 -118.021 -112.167 -105.193 -97.202 ...
    -88.313 -78.659 -68.385 -57.642 -46.592 -35.000 -21.280 -7.560];
element_y = [element_y, flip(-element_y)] * 1e-3;

% Optional per-transducer manual rotations. Use these when an individual
% transducer axis does not pass through the phantom center. Angles are CCW
% in degrees. Centers are [x, y] in metres for each transducer. Leave all
% zeros to keep the current geometry.
transducer_element_rotation_deg = [-4.086 -11.952 -19.388 -26.205 -32.866 -30.059 ...
    -25.685 -21.305 -16.912 -12.534 -8.146 -3.756 9.523 5.025 9.414 6.365 0.551 ...
    2.232 4.231 6.211 14.907 10.072 11.931 13.727 15.444 17.068 18.579 19.956 ...
    21.174 21.148 13.235 4.776];
transducer_element_rotation_deg = [transducer_element_rotation_deg, flip(-transducer_element_rotation_deg)];
transducer_element_rotation_centers = [element_x', element_y'];
% Example:
% transducer_element_rotation_deg(7) = -3.5;
% transducer_element_rotation_centers(7,:) = [0.012, -0.004];
abdomen_mask_threshold = 1490; % sound-speed threshold used to find abdomen edge [m/s]
source_freq = 0.600e6;     % source centre frequency [Hz]
source_cycles = 3;        % tone burst length

% Synthetic k-Wave transmitter model. A finite normal-velocity aperture is
% used instead of a single isotropic pressure point when use_measured_data=false.
piston_aperture_width_m = 5e-3;
piston_velocity_scale = 1.0;

% Direction-aware first-arrival timing. The baseline time and ray path use
% effective points on the finite piston faces rather than element centres.
use_directional_time_shifts = true;
directional_time_shift_aperture_width_m = piston_aperture_width_m;
directional_min_abs_cos = 0.0;  % set >0 to reject very grazing TX/RX pairs

% Use measured RF data exported by UTA64_LEMO_AcquireRF_2D_Grid_Array_64_Line.m.
% Set use_measured_data = false to run the original synthetic k-Wave acquisition.
use_measured_data = true;
measured_rf_file = 'C:\Users\Administrator\Documents\Vantage-4.9.6-2502061500\Example_Scripts\CustomScripts\measured_rf\UTA64_LEMO_RF_latest_FWI.mat';
experimental_time_step_s = 4.1667e-07;  % experimental RF sample interval [s]
measured_sample_rate_hz = 1 / experimental_time_step_s;
measured_time_zero_s = 0;

ppw_min = c0 / source_freq / dx;

fprintf('Points per wavelength at c0: %.2f\n', ppw_min);
if ppw_min < 3
    warning('The grid is coarse for this frequency. Lower source_freq or reduce dx for higher accuracy.');
end

%% Grid and time axis
load ct_sound_speed_1.3.12.2.1107.5.1.4.83567.30000025112415592178200005306_192x192_m_s.mat;
sound_speed(135:end,:) = 1480.0;

kgrid = kWaveGrid(Nx, dx, Ny, dy);

medium.sound_speed = c0 * ones(Nx, Ny);
medium.density = rho0 * ones(Nx, Ny);
medium.alpha_coeff = 0.35;     % [dB/(MHz^y cm)]
medium.alpha_power = 1.5;

% Build a compact phantom: one fast inclusion, one slow inclusion, one ring.
[X, Y] = ndgrid(kgrid.x_vec, kgrid.y_vec);

medium.sound_speed = sound_speed;
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


% Apply global and per-element rotations before snapping coordinates to the k-Wave grid.
[element_x, element_y] = rotate_transducer_azimuth( ...
    element_x, element_y, transducer_azimuth_deg, transducer_rotation_center);
[element_x, element_y] = rotate_individual_transducers( ...
    element_x, element_y, transducer_element_rotation_deg, transducer_element_rotation_centers);

[element_i, element_j] = cart2grid_index(kgrid, element_x, element_y);
element_x = kgrid.x_vec(element_i).';
element_y = kgrid.y_vec(element_j).';
[tx_normal_x, tx_normal_y] = compute_transducer_normals( ...
    element_x, element_y, transducer_element_rotation_deg, ...
    transducer_element_rotation_centers, transducer_rotation_center, min(dx, dy));

cfl = 0.25;
max_tx_rx_distance = max_pairwise_distance(element_x, element_y);
valid_sound_speed = medium.sound_speed(isfinite(medium.sound_speed) & medium.sound_speed > 0);
t_end = 1.35 * max_tx_rx_distance / min(valid_sound_speed(:));
% Force the simulation time grid to match the experimental sampling interval.
simulation_num_time = floor(t_end / experimental_time_step_s) + 1;
kgrid.setTime(simulation_num_time, experimental_time_step_s);
fprintf('Using shared simulation/experimental dt = %.4e s, Nt = %d\n', kgrid.dt, numel(kgrid.t_array));

sensor.mask = [element_x; element_y];
sensor.record = {'p'};

source_signal = toneBurst(1 / kgrid.dt, source_freq, source_cycles);
input_args = { ...
    'PMLInside', false, ...
    'PMLSize', 20, ...
    'DataCast', 'single', ...
    'PlotSim', false};

%% Load measured RF data or simulate synthetic transmit events
if use_measured_data
    fprintf('Loading measured Verasonics RF data from %s\n', measured_rf_file);
    [pressure_data, data_t_array, rf_meta] = load_verasonics_rf_cube( ...
        measured_rf_file, num_elements, measured_sample_rate_hz, measured_time_zero_s);
    pressure_data = pressure_data(:,:,1:619);
    pressure_data(:,:,1:20) = 0;
    num_time = size(pressure_data, 3);
    data_t_array = measured_time_zero_s + (0:num_time-1) * experimental_time_step_s;
    fprintf('Loaded measured RF cube: %d tx x %d rx x %d samples, fs = %.3f MHz\n', ...
        size(pressure_data, 1), size(pressure_data, 2), num_time, rf_meta.sample_rate_hz / 1e6);
else
    num_time = numel(kgrid.t_array);
    data_t_array = measured_time_zero_s + (0:num_time-1) * experimental_time_step_s;
    pressure_data = zeros(num_elements, num_elements, num_time, 'single');

    fprintf('Running %d synthetic k-Wave transmit events...\n', num_elements);
    for tx = 1:num_elements
        source = build_piston_velocity_source( ...
            kgrid, Nx, Ny, element_x(tx), element_y(tx), ...
            tx_normal_x(tx), tx_normal_y(tx), piston_aperture_width_m, ...
            piston_velocity_scale * source_signal);

        sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
        pressure_data(tx, :, :) = single(sensor_data.p);
        fprintf('  tx %02d/%02d complete\n', tx, num_elements);
    end
end

%% Estimate first-arrival time shifts relative to a homogeneous background
arrival_time = nan(num_elements, num_elements);
baseline_time = nan(num_elements, num_elements);
ray_start_x = nan(num_elements, num_elements);
ray_start_y = nan(num_elements, num_elements);
ray_end_x = nan(num_elements, num_elements);
ray_end_y = nan(num_elements, num_elements);
directional_cos_tx = nan(num_elements, num_elements);
directional_cos_rx = nan(num_elements, num_elements);
min_separation = 3;       % ignore receivers too close along the contour
threshold_fraction = 0.05;
data_dt = experimental_time_step_s;

for tx = 1:num_elements
    for rx = 1:num_elements
        contour_gap = min(mod(rx - tx, num_elements), mod(tx - rx, num_elements));
        if contour_gap < min_separation
            continue;
        end

        trace = double(squeeze(pressure_data(tx, rx, :)));
        if use_directional_time_shifts
            [x1_eff, y1_eff, x2_eff, y2_eff, direct_distance, tx_cos, rx_cos] = directional_piston_ray( ...
                element_x(tx), element_y(tx), tx_normal_x(tx), tx_normal_y(tx), ...
                element_x(rx), element_y(rx), tx_normal_x(rx), tx_normal_y(rx), ...
                directional_time_shift_aperture_width_m);
        else
            x1_eff = element_x(tx);
            y1_eff = element_y(tx);
            x2_eff = element_x(rx);
            y2_eff = element_y(rx);
            direct_distance = hypot(x2_eff - x1_eff, y2_eff - y1_eff);
            tx_cos = 1;
            rx_cos = 1;
        end

        if tx_cos < directional_min_abs_cos || rx_cos < directional_min_abs_cos
            continue;
        end

        ray_start_x(tx, rx) = x1_eff;
        ray_start_y(tx, rx) = y1_eff;
        ray_end_x(tx, rx) = x2_eff;
        ray_end_y(tx, rx) = y2_eff;
        directional_cos_tx(tx, rx) = tx_cos;
        directional_cos_rx(tx, rx) = rx_cos;
        baseline_time(tx, rx) = direct_distance / c0;

        gate_start_time = max(0, baseline_time(tx, rx) - 6 / source_freq);
        gate_start_index = max(1, floor((gate_start_time - data_t_array(1)) / data_dt));
        arrival_time(tx, rx) = first_arrival_time(trace, data_t_array, gate_start_index, threshold_fraction);
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
        ray_start_x(tx, rx), ray_start_y(tx, rx), ray_end_x(tx, rx), ray_end_y(tx, rx), ...
        x_edges, y_edges, recon_N, recon_roi);
    ray_rows{ray} = ray * ones(size(cols));
    ray_cols{ray} = cols;
    ray_vals{ray} = vals;
end

A = sparse(cell2mat(ray_rows), cell2mat(ray_cols), cell2mat(ray_vals), num_rays, num_pixels);
b = time_shift(valid);

% Mild Tikhonov regularisation stabilises the noisy, sparse inverse problem.
% Solve the regularised normal equations directly to avoid using lsqr().
lambda = 1e-2;
normal_matrix = A.' * A + (lambda^2) * speye(num_pixels);
normal_rhs = A.' * b;
slowness_delta = normal_matrix \ normal_rhs;

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
plot(data_t_array * 1e6, squeeze(pressure_data(37, 34, :)), 'k');
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
    ix = uniform_bin_index(xs, x_edges);
    iy = uniform_bin_index(ys, y_edges);
    inside = ix >= 1 & ix <= n & iy >= 1 & iy <= n;
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
function idx = uniform_bin_index(values, edges)
    % Replacement for discretize(values, edges) on uniformly spaced bins.
    step = edges(2) - edges(1);
    idx = floor((values - edges(1)) ./ step) + 1;
    idx(values < edges(1) | values > edges(end)) = 0;
    idx(values == edges(end)) = numel(edges) - 1;
end

function [x1_eff, y1_eff, x2_eff, y2_eff, path_length, tx_cos, rx_cos] = directional_piston_ray( ...
        tx_x, tx_y, tx_nx, tx_ny, rx_x, rx_y, rx_nx, rx_ny, aperture_width)
    tx_nlen = max(hypot(tx_nx, tx_ny), eps);
    rx_nlen = max(hypot(rx_nx, rx_ny), eps);
    tx_nx = tx_nx / tx_nlen;
    tx_ny = tx_ny / tx_nlen;
    rx_nx = rx_nx / rx_nlen;
    rx_ny = rx_ny / rx_nlen;

    tx_tx = -tx_ny;
    tx_ty = tx_nx;
    rx_tx = -rx_ny;
    rx_ty = rx_nx;

    delta_x = tx_x - rx_x;
    delta_y = tx_y - rx_y;
    G = [tx_tx, -rx_tx; tx_ty, -rx_ty];
    if abs(det(G)) > 1e-12
        s = -G \ [delta_x; delta_y];
        tx_offset = s(1);
        rx_offset = s(2);
    else
        tx_offset = 0;
        rx_offset = 0;
    end

    half_width = max(aperture_width, 0) / 2;
    tx_offset = min(max(tx_offset, -half_width), half_width);
    rx_offset = min(max(rx_offset, -half_width), half_width);

    x1_eff = tx_x + tx_offset * tx_tx;
    y1_eff = tx_y + tx_offset * tx_ty;
    x2_eff = rx_x + rx_offset * rx_tx;
    y2_eff = rx_y + rx_offset * rx_ty;

    ray_x = x2_eff - x1_eff;
    ray_y = y2_eff - y1_eff;
    path_length = hypot(ray_x, ray_y);
    if path_length <= eps
        tx_cos = 0;
        rx_cos = 0;
        return;
    end

    ray_x = ray_x / path_length;
    ray_y = ray_y / path_length;
    tx_cos = abs(ray_x * tx_nx + ray_y * tx_ny);
    rx_cos = abs(ray_x * rx_nx + ray_y * rx_ny);
end

function [pressure_data, t_array, meta] = load_verasonics_rf_cube(file_name, num_elements, sample_rate_hz, time_zero_s)
    if ~isfile(file_name)
        error('Measured RF file was not found: %s', file_name);
    end

    S = load(file_name);
    cube = [];
    candidate_names = {'rf_data', 'rfCube', 'RFData', 'RData'};
    for k = 1:numel(candidate_names)
        name = candidate_names{k};
        if isfield(S, name) && isnumeric(S.(name)) && ndims(S.(name)) == 3
            cube = S.(name);
            break;
        end
    end

    if isempty(cube)
        vars = fieldnames(S);
        for k = 1:numel(vars)
            value = S.(vars{k});
            if isnumeric(value) && ndims(value) == 3
                cube = value;
                break;
            end
        end
    end

    if isempty(cube)
        error('No 3-D numeric RF cube found in %s. Expected samples x rx x tx or tx x rx x samples.', file_name);
    end

    cube = single(cube);
    sz = size(cube);
    if sz(2) == num_elements && sz(3) == num_elements
        pressure_data = permute(cube, [3, 2, 1]);       % samples x rx x tx -> tx x rx x samples
    elseif sz(1) == num_elements && sz(2) == num_elements
        pressure_data = cube;                           % tx x rx x samples
    else
        error('RF cube has size [%s], but expected samples x %d rx x %d tx or %d tx x %d rx x samples.', ...
            num2str(sz), num_elements, num_elements, num_elements, num_elements);
    end

    if isempty(sample_rate_hz)
        if isfield(S, 'sample_rate_hz')
            sample_rate_hz = double(S.sample_rate_hz);
        elseif isfield(S, 'sampleRateHz')
            sample_rate_hz = double(S.sampleRateHz);
        elseif isfield(S, 'fs')
            sample_rate_hz = double(S.fs);
        else
            error('Sample rate not found in %s. Set measured_sample_rate_hz in the tomography script.', file_name);
        end
    end

    num_samples = size(pressure_data, 3);
    t_array = time_zero_s + (0:num_samples-1) / sample_rate_hz;
    meta = struct('sample_rate_hz', sample_rate_hz, 'num_samples', num_samples, 'source_file', file_name);
end

function [x_rot, y_rot] = rotate_transducer_azimuth(x, y, azimuth_deg, center_xy)
    if isempty(azimuth_deg)
        azimuth_deg = 0;
    end
    if isempty(center_xy)
        center_xy = [0, 0];
    end
    if numel(center_xy) ~= 2
        error('transducer_rotation_center must be [x, y] in metres.');
    end
    if ~isscalar(azimuth_deg)
        error('transducer_azimuth_deg must be a scalar array rotation angle in degrees.');
    end

    theta = deg2rad(azimuth_deg);
    x0 = center_xy(1);
    y0 = center_xy(2);
    xr = x - x0;
    yr = y - y0;
    x_rot = x0 + xr * cos(theta) - yr * sin(theta);
    y_rot = y0 + xr * sin(theta) + yr * cos(theta);
end

function [x_out, y_out] = rotate_individual_transducers(x, y, angle_deg, centers_xy)
    num_elements = numel(x);
    if isempty(angle_deg)
        angle_deg = zeros(1, num_elements);
    end
    if isscalar(angle_deg)
        angle_deg = repmat(angle_deg, 1, num_elements);
    end
    if numel(angle_deg) ~= num_elements
        error('transducer_element_rotation_deg must be scalar or have one angle per transducer.');
    end
    if isempty(centers_xy)
        centers_xy = zeros(num_elements, 2);
    end
    if isequal(size(centers_xy), [1, 2])
        centers_xy = repmat(centers_xy, num_elements, 1);
    end
    if ~isequal(size(centers_xy), [num_elements, 2])
        error('transducer_element_rotation_centers must be [1 x 2] or [num_elements x 2] in metres.');
    end

    x_out = x;
    y_out = y;
    for idx = 1:num_elements
        if angle_deg(idx) == 0
            continue;
        end
        theta = deg2rad(angle_deg(idx));
        x0 = centers_xy(idx, 1);
        y0 = centers_xy(idx, 2);
        xr = x(idx) - x0;
        yr = y(idx) - y0;
        x_out(idx) = x0 + xr * cos(theta) - yr * sin(theta);
        y_out(idx) = y0 + xr * sin(theta) + yr * cos(theta);
    end
end

function source = build_piston_velocity_source(kgrid, Nx, Ny, x0, y0, normal_x, normal_y, aperture_width, source_signal)
    normal_length = hypot(normal_x, normal_y);
    if normal_length <= eps
        error('Transducer normal vector has zero length. Check rotation centers and angles.');
    end
    normal_x = normal_x / normal_length;
    normal_y = normal_y / normal_length;
    tangent_x = -normal_y;
    tangent_y = normal_x;

    grid_step = min(abs(kgrid.x_vec(2) - kgrid.x_vec(1)), abs(kgrid.y_vec(2) - kgrid.y_vec(1)));
    num_points = max(1, 2 * floor(max(aperture_width, grid_step) / grid_step / 2) + 1);
    aperture_s = linspace(-aperture_width / 2, aperture_width / 2, num_points);
    aperture_x = x0 + aperture_s * tangent_x;
    aperture_y = y0 + aperture_s * tangent_y;
    [aperture_i, aperture_j] = cart2grid_index(kgrid, aperture_x, aperture_y);
    aperture_i = min(max(aperture_i, 1), Nx);
    aperture_j = min(max(aperture_j, 1), Ny);
    aperture_index = unique(sub2ind([Nx, Ny], aperture_i(:), aperture_j(:)), 'stable');

    source = struct();
    source.u_mask = zeros(Nx, Ny);
    source.u_mask(aperture_index) = 1;
    num_source_points = numel(aperture_index);
    source.ux = repmat(normal_x * source_signal(:).', num_source_points, 1);
    source.uy = repmat(normal_y * source_signal(:).', num_source_points, 1);
end

function [normal_x, normal_y] = compute_transducer_normals(x, y, angle_deg, centers_xy, fallback_center, center_tolerance)
    if nargin < 6 || isempty(center_tolerance)
        center_tolerance = eps;
    end
    num_elements = numel(x);
    if isempty(angle_deg)
        angle_deg = zeros(1, num_elements);
    end
    if isscalar(angle_deg)
        angle_deg = repmat(angle_deg, 1, num_elements);
    end
    if numel(angle_deg) ~= num_elements
        error('transducer_element_rotation_deg must be scalar or have one angle per transducer.');
    end
    if isempty(centers_xy)
        centers_xy = repmat(fallback_center, num_elements, 1);
    end
    if isequal(size(centers_xy), [1, 2])
        centers_xy = repmat(centers_xy, num_elements, 1);
    end
    if ~isequal(size(centers_xy), [num_elements, 2])
        error('transducer_element_rotation_centers must be [1 x 2] or [num_elements x 2] in metres.');
    end

    normal_x = zeros(1, num_elements);
    normal_y = zeros(1, num_elements);
    for idx = 1:num_elements
        vx = x(idx) - centers_xy(idx, 1);
        vy = y(idx) - centers_xy(idx, 2);
        if hypot(vx, vy) <= center_tolerance
            vx = x(idx) - fallback_center(1);
            vy = y(idx) - fallback_center(2);
        end
        if hypot(vx, vy) <= eps
            vx = 1;
            vy = 0;
        end

        theta = deg2rad(angle_deg(idx));
        nx = vx * cos(theta) - vy * sin(theta);
        ny = vx * sin(theta) + vy * cos(theta);
        nlen = max(hypot(nx, ny), eps);
        normal_x(idx) = nx / nlen;
        normal_y(idx) = ny / nlen;
    end
end
