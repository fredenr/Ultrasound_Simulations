%% Ultrasound tomography simulation with k-Wave
% This script simulates 2-D transmission ultrasound tomography using k-Wave.
% It creates a heterogeneous sound-speed phantom, fires each element in a
% contour-following array, estimates first-arrival time shifts, and reconstructs a
% sound-speed map with a compact full-waveform inversion loop.
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

%% GPU controls
use_gpu = true;
gpu_index = 1;
if use_gpu && exist('gpuDeviceCount', 'file') == 2 && gpuDeviceCount > 0
    gpuDevice(gpu_index);
    data_cast = 'gpuArray-single';
    fprintf('Using GPU %d for k-Wave simulations.\n', gpu_index);
else
    if use_gpu
        warning('GPU requested, but no compatible MATLAB GPU device was found. Falling back to CPU.');
    end
    use_gpu = false;
    data_cast = 'single';
end

%% Simulation controls
Nx = 192;                 % grid points in x
Ny = 192;                 % grid points in y
dx = 2.60416e-3;             % grid spacing [m]
dy = dx;
c0 = 1500;                % background sound speed [m/s]
rho0 = 1000;              % density [kg/m^3]
num_elements = 64;        % circular transducer count
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
array_radius = 150e-3;     % radius of circular transducer array [m]
source_freq = 0.600e6;     % source centre frequency [Hz]
source_cycles = 3;        % tone burst length

% Synthetic k-Wave transmitter model. A finite normal-velocity aperture is
% used instead of a single isotropic pressure point when use_measured_data=false.
piston_aperture_width_m = 5e-3;
piston_velocity_scale = 1.0;

% Direction-aware first-arrival timing. The baseline time uses effective
% points on the finite piston faces rather than element centres.
use_directional_time_shifts = true;
directional_time_shift_aperture_width_m = piston_aperture_width_m;
directional_min_abs_cos = 0.0;  % set >0 to reject very grazing TX/RX pairs

% Use measured RF data exported by UTA64_LEMO_AcquireRF_Planar_Abdomen_Array_64_Save_RFData.m.
% Set use_measured_data = false to run synthetic k-Wave acquisition.
use_measured_data = true;
measured_rf_file = 'C:\Users\Administrator\Documents\Vantage-4.9.6-2502061500\Example_Scripts\CustomScripts\measured_rf\UTA64_LEMO_RF_latest_FWI.mat';
experimental_time_step_s = 4.1667e-07;  % experimental RF sample interval [s]
measured_sample_rate_hz = 1 / experimental_time_step_s;
measured_time_zero_s = 0;
normalize_measured_channels = true;

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
% phantom boundary. If phantom_mask is replaced with a segmented abdomen
% mask, the array will automatically follow that outer contour.

% [boundary_i, boundary_j] = outer_boundary_ordered(phantom_mask);
% [element_i, element_j] = resample_closed_boundary(boundary_i, boundary_j, num_elements);
%
% if array_standoff ~= 0
%     [element_i, element_j] = offset_boundary_points(kgrid, element_i, element_j, array_standoff);
% end
%
% element_linear_index = sub2ind([Nx, Ny], element_i, element_j);
% [element_linear_index, unique_idx] = unique(element_linear_index, 'stable');
% element_i = element_i(unique_idx);
% element_j = element_j(unique_idx);
% num_elements = numel(element_i);
% element_x = kgrid.x_vec(element_i).';
% element_y = kgrid.y_vec(element_j).';

[element_i, element_j] = cart2grid_index(kgrid, element_x, element_y);
element_linear_index = sub2ind([Nx, Ny], element_i, element_j);
[element_linear_index, unique_idx] = unique(element_linear_index, 'stable');
element_x = kgrid.x_vec(element_i).';
element_y = kgrid.y_vec(element_j).';

[tx_normal_x, tx_normal_y] = compute_transducer_normals( ...
    element_x, element_y, transducer_element_rotation_deg, ...
    transducer_element_rotation_centers, transducer_rotation_center, min(dx, dy));

cfl = 0.25;
max_tx_rx_distance = max_pairwise_distance(element_x, element_y);
t_end = 1.35 * max_tx_rx_distance / min(medium.sound_speed(:));
% Force the simulation time grid to match the experimental sampling interval.
simulation_num_time = floor(t_end / experimental_time_step_s) + 1;
kgrid.setTime(simulation_num_time, experimental_time_step_s);
data_t_array = measured_time_zero_s + (0:simulation_num_time-1) * experimental_time_step_s;
fprintf('Using shared simulation/experimental dt = %.4e s, Nt = %d\n', kgrid.dt, numel(kgrid.t_array));

sensor.mask = [element_x; element_y];
sensor.record = {'p'};

source_signal = toneBurst(1 / kgrid.dt, source_freq, source_cycles);
input_args = { ...
    'PMLInside', false, ...
    'PMLSize', 20, ...
    'DataCast', data_cast, ...
    'DataRecast', true, ...
    'PlotSim', false};

%% Load measured RF data or simulate synthetic transmit events
num_time = numel(data_t_array);
if use_measured_data
    fprintf('Loading measured Verasonics RF data from %s\n', measured_rf_file);
    [pressure_data, measured_t_array, rf_meta] = load_verasonics_rf_cube( ...
        measured_rf_file, num_elements, measured_sample_rate_hz, measured_time_zero_s, data_t_array, normalize_measured_channels);
    pressure_data = pressure_data(:,:,1:577);
    pressure_data(:,:,1:20) = 0;
    fprintf('Loaded measured RF cube: %d tx x %d rx x %d samples, fs = %.3f MHz; resampled to k-Wave dt = %.3f ns\n', ...
        size(pressure_data, 1), size(pressure_data, 2), size(pressure_data, 3), ...
        rf_meta.sample_rate_hz / 1e6, experimental_time_step_s * 1e9);
else
    pressure_data = zeros(num_elements, num_elements, num_time, 'single');

    fprintf('Running %d synthetic k-Wave transmit events...\n', num_elements);
    for tx = 1:num_elements
        source = build_piston_velocity_source( ...
            kgrid, Nx, Ny, element_x(tx), element_y(tx), ...
            tx_normal_x(tx), tx_normal_y(tx), piston_aperture_width_m, ...
            piston_velocity_scale * source_signal);

        sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
        pressure_data(tx, :, :) = single(to_cpu(sensor_data.p));
        fprintf('  tx %02d/%02d complete\n', tx, num_elements);
    end
end

%% Estimate first-arrival time shifts relative to a homogeneous background
arrival_time = nan(num_elements, num_elements);
baseline_time = nan(num_elements, num_elements);
directional_ray_start_x = nan(num_elements, num_elements);
directional_ray_start_y = nan(num_elements, num_elements);
directional_ray_end_x = nan(num_elements, num_elements);
directional_ray_end_y = nan(num_elements, num_elements);
directional_cos_tx = nan(num_elements, num_elements);
directional_cos_rx = nan(num_elements, num_elements);
min_separation = 3;       % ignore receivers too close along the contour
threshold_fraction = 0.05;

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

        directional_ray_start_x(tx, rx) = x1_eff;
        directional_ray_start_y(tx, rx) = y1_eff;
        directional_ray_end_x(tx, rx) = x2_eff;
        directional_ray_end_y(tx, rx) = y2_eff;
        directional_cos_tx(tx, rx) = tx_cos;
        directional_cos_rx(tx, rx) = rx_cos;
        baseline_time(tx, rx) = direct_distance / c0;

        gate_start_time = max(0, baseline_time(tx, rx) - 6 / source_freq);
        gate_start_index = max(1, floor((gate_start_time - data_t_array(1)) / experimental_time_step_s));
        arrival_time(tx, rx) = first_arrival_time(trace, data_t_array, gate_start_index, threshold_fraction);
    end
end

time_shift = arrival_time - baseline_time;

%% Full-waveform inversion reconstruction
fwi_iterations = 3;
fwi_tx_list = 1:num_elements;    % use 1:num_elements for a slower, fuller inversion
fwi_step = 18;                     % sound-speed update step [m/s] after normalisation
c_min = 1350;
c_max = 1650;
imaging_roi = hypot(X, Y) <= 1000e-3;
full_sensor_mask = true(Nx, Ny);
[adjoint_source_order, adjoint_source_to_element] = sort(element_linear_index);

c_recon = c0 * ones(Nx, Ny);
c_recon(~imaging_roi) = c0;
misfit_history = zeros(fwi_iterations, 1);

fwi_sensor = struct();
fwi_sensor.mask = full_sensor_mask;
fwi_sensor.record = {'p'};
misfit_sensor = sensor;

fprintf('Running full-waveform inversion with %d transmitters and %d iterations...\n', ...
    numel(fwi_tx_list), fwi_iterations);

for iter = 1:fwi_iterations
    gradient_c = zeros(Nx, Ny);
    iter_misfit = 0;
    iter_samples = 0;

    for tx = fwi_tx_list
        current_medium = medium;
        current_medium.sound_speed = c_recon;

        source = build_piston_velocity_source( ...
            kgrid, Nx, Ny, element_x(tx), element_y(tx), ...
            tx_normal_x(tx), tx_normal_y(tx), piston_aperture_width_m, ...
            piston_velocity_scale * source_signal);

        forward_data = kspaceFirstOrder2D(kgrid, current_medium, source, fwi_sensor, input_args{:});
        forward_p = to_cpu(forward_data.p);
        predicted = double(forward_p(element_linear_index, :));
        forward_wavefield = reshape(double(forward_p), Nx, Ny, num_time);
        observed = double(squeeze(pressure_data(tx, :, :)));

        residual = predicted - observed;
        residual(tx, :) = 0;       % ignore the active source element
        residual = remove_receiver_mean(residual);

        residual_energy = sum(residual(:).^2) * kgrid.dt;
        iter_misfit = iter_misfit + 0.5 * residual_energy;
        iter_samples = iter_samples + numel(residual);

        adjoint_source = struct();
        adjoint_source.p_mask = zeros(Nx, Ny);
        adjoint_source.p_mask(adjoint_source_order) = 1;
        adjoint_source.p = fliplr(residual(adjoint_source_to_element, :));

        adjoint_data = kspaceFirstOrder2D(kgrid, current_medium, adjoint_source, fwi_sensor, input_args{:});
        adjoint_p = to_cpu(adjoint_data.p);
        adjoint_wavefield = reshape(double(adjoint_p), Nx, Ny, num_time);

        forward_accel = second_time_derivative(forward_wavefield, kgrid.dt);
        adjoint_wavefield = flip(adjoint_wavefield, 3);

        gradient_c = gradient_c - sum(forward_accel .* adjoint_wavefield, 3) * kgrid.dt;
        fprintf('  iter %d/%d tx %02d: residual energy %.3e\n', ...
            iter, fwi_iterations, tx, residual_energy);
    end

    gradient_c(~imaging_roi) = 0;
    gradient_c = (2 ./ max(c_recon, eps).^3) .* gradient_c;
    gradient_c = smooth_map(gradient_c, imaging_roi);
    gradient_c = gradient_c / max(abs(gradient_c(:)) + eps);

    c_trial_plus = clamp_model(c_recon + fwi_step * gradient_c, imaging_roi, c0, c_min, c_max);
    c_trial_minus = clamp_model(c_recon - fwi_step * gradient_c, imaging_roi, c0, c_min, c_max);

    plus_misfit = waveform_misfit(kgrid, medium, c_trial_plus, source_signal, pressure_data, ...
        element_i, element_j, element_linear_index, fwi_tx_list, misfit_sensor, input_args, ...
        element_x, element_y, tx_normal_x, tx_normal_y, piston_aperture_width_m, piston_velocity_scale);
    minus_misfit = waveform_misfit(kgrid, medium, c_trial_minus, source_signal, pressure_data, ...
        element_i, element_j, element_linear_index, fwi_tx_list, misfit_sensor, input_args, ...
        element_x, element_y, tx_normal_x, tx_normal_y, piston_aperture_width_m, piston_velocity_scale);

    if plus_misfit <= minus_misfit
        c_recon = c_trial_plus;
        chosen_misfit = plus_misfit;
        step_sign = 1;
    else
        c_recon = c_trial_minus;
        chosen_misfit = minus_misfit;
        step_sign = -1;
    end

    c_recon = smooth_map(c_recon, imaging_roi);
    c_recon = clamp_model(c_recon, imaging_roi, c0, c_min, c_max);
    misfit_history(iter) = chosen_misfit / max(1, iter_samples);

    fprintf('  FWI iter %d/%d complete: sign %+d, normalised misfit %.3e\n', ...
        iter, fwi_iterations, step_sign, misfit_history(iter));
end

c_recon(~imaging_roi) = NaN;

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
plot(data_t_array * 1e6, squeeze(pressure_data(1, round(num_elements/2), :)), 'k');
grid on;
title('Example received waveform');
xlabel('Time [\mus]'); ylabel('Pressure [Pa]');

% nexttile;
% plot(1:fwi_iterations, misfit_history, 'ko-', 'LineWidth', 1.2);
% grid on;
% title('FWI waveform misfit');
% xlabel('Iteration'); ylabel('Normalised misfit');

nexttile;
imagesc(kgrid.y_vec * 1e3, kgrid.x_vec * 1e3, c_recon);
axis image; colormap(gca, turbo); colorbar;
title('FWI reconstructed sound speed [m/s]');
xlabel('y [mm]'); ylabel('x [mm]');

fprintf('Done. Reconstruction range inside ROI: %.1f to %.1f m/s\n', ...
    min(c_recon(imaging_roi), [], 'omitnan'), max(c_recon(imaging_roi), [], 'omitnan'));

%% Local functions
function [boundary_i, boundary_j] = outer_boundary_ordered(mask)
    if exist('bwboundaries', 'file') == 2
        boundaries = bwboundaries(mask, 'noholes');
        if isempty(boundaries)
            error('No boundary points were found. Check that phantom_mask is non-empty.');
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
        error('No boundary points were found. Check that phantom_mask is non-empty.');
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

function eroded = erode_mask(mask, radius)
    eroded = mask;
    for n = 1:radius
        eroded = eroded & (conv2(double(eroded), ones(3), 'same') == 9);
    end
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

function residual = remove_receiver_mean(residual)
    residual = residual - mean(residual, 2);
end

function d2pdt2 = second_time_derivative(p, dt)
    d2pdt2 = zeros(size(p));
    d2pdt2(:, :, 2:end-1) = (p(:, :, 3:end) - 2 * p(:, :, 2:end-1) + p(:, :, 1:end-2)) / dt^2;
end

function model = clamp_model(model, roi, background_c, c_min, c_max)
    model = min(max(model, c_min), c_max);
    model(~roi) = background_c;
end

function misfit = waveform_misfit(kgrid, base_medium, c_model, source_signal, observed_data, ...
        element_i, element_j, element_linear_index, tx_list, sensor, input_args, ...
        element_x, element_y, tx_normal_x, tx_normal_y, piston_aperture_width_m, piston_velocity_scale)
    misfit = 0;
    test_medium = base_medium;
    test_medium.sound_speed = c_model;

    for tx = tx_list
        source = build_piston_velocity_source( ...
            kgrid, size(c_model, 1), size(c_model, 2), element_x(tx), element_y(tx), ...
            tx_normal_x(tx), tx_normal_y(tx), piston_aperture_width_m, ...
            piston_velocity_scale * source_signal);

        sensor_data = kspaceFirstOrder2D(kgrid, test_medium, source, sensor, input_args{:});
        sensor_p = to_cpu(sensor_data.p);
        if islogical(sensor.mask) && isequal(size(sensor.mask), size(c_model))
            predicted = double(sensor_p(element_linear_index, :));
        else
            predicted = double(sensor_p);
        end
        observed = double(squeeze(observed_data(tx, :, :)));
        residual = predicted - observed;
        residual(tx, :) = 0;
        residual = remove_receiver_mean(residual);
        misfit = misfit + 0.5 * sum(residual(:).^2) * kgrid.dt;
    end
end

function x = to_cpu(x)
    if isa(x, 'gpuArray')
        x = gather(x);
    end
end

function smoothed = smooth_map(map, roi)
    kernel = [1 2 1; 2 4 2; 1 2 1] / 16;
    numerator = conv2(map .* roi, kernel, 'same');
    denominator = conv2(double(roi), kernel, 'same');
    smoothed = map;
    smoothed(roi) = numerator(roi) ./ max(denominator(roi), eps);
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

function [pressure_data, measured_t_array, meta] = load_verasonics_rf_cube(file_name, num_elements, sample_rate_hz, time_zero_s, target_t_array, normalize_channels)
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
        rf_tx_rx_t = permute(cube, [3, 2, 1]);       % samples x rx x tx -> tx x rx x samples
    elseif sz(1) == num_elements && sz(2) == num_elements
        rf_tx_rx_t = cube;                           % tx x rx x samples
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
            error('Sample rate not found in %s. Set measured_sample_rate_hz in the FWI script.', file_name);
        end
    end

    num_measured_samples = size(rf_tx_rx_t, 3);
    measured_t_array = time_zero_s + (0:num_measured_samples-1) / sample_rate_hz;
    pressure_data = zeros(num_elements, num_elements, numel(target_t_array), 'single');

    for tx = 1:num_elements
        traces = double(squeeze(rf_tx_rx_t(tx, :, :))).';  % samples x rx
        traces = traces - mean(traces(1:min(32, size(traces, 1)), :), 1);
        if normalize_channels
            scale = median(abs(traces), 1) / 0.6745;
            scale(~isfinite(scale) | scale <= 0) = median(scale(isfinite(scale) & scale > 0));
            if isempty(scale) || any(~isfinite(scale))
                scale = ones(1, size(traces, 2));
            end
            traces = traces ./ max(scale, eps);
        end
        traces_resampled = interp1(measured_t_array(:), traces, target_t_array(:), 'linear', 0);
        pressure_data(tx, :, :) = single(traces_resampled.');
    end

    meta = struct('sample_rate_hz', sample_rate_hz, ...
        'num_measured_samples', num_measured_samples, ...
        'source_file', file_name, ...
        'resampled_to_kwave_time', true, ...
        'normalize_channels', normalize_channels);
end

