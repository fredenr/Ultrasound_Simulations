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
array_radius = 150e-3;     % radius of circular transducer array [m]
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
element_linear_index = sub2ind([Nx, Ny], element_i, element_j);
[element_linear_index, unique_idx] = unique(element_linear_index, 'stable');
element_x = kgrid.x_vec(element_i).';
element_y = kgrid.y_vec(element_j).';

cfl = 0.25;
max_tx_rx_distance = max_pairwise_distance(element_x, element_y);
t_end = 1.35 * max_tx_rx_distance / min(medium.sound_speed(:));
kgrid.makeTime(max(medium.sound_speed(:)), cfl, t_end);

sensor.mask = [element_x; element_y];
sensor.record = {'p'};

source_signal = toneBurst(1 / kgrid.dt, source_freq, source_cycles);
input_args = { ...
    'PMLInside', false, ...
    'PMLSize', 20, ...
    'DataCast', data_cast, ...
    'DataRecast', true, ...
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
    pressure_data(tx, :, :) = single(to_cpu(sensor_data.p));
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

        source = struct();
        source.p_mask = zeros(Nx, Ny);
        source.p_mask(element_i(tx), element_j(tx)) = 1;
        source.p = source_signal;

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
        element_i, element_j, element_linear_index, fwi_tx_list, misfit_sensor, input_args);
    minus_misfit = waveform_misfit(kgrid, medium, c_trial_minus, source_signal, pressure_data, ...
        element_i, element_j, element_linear_index, fwi_tx_list, misfit_sensor, input_args);

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
plot(kgrid.t_array * 1e6, squeeze(pressure_data(1, round(num_elements/2), :)), 'k');
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
        element_i, element_j, element_linear_index, tx_list, sensor, input_args)
    misfit = 0;
    test_medium = base_medium;
    test_medium.sound_speed = c_model;

    for tx = tx_list
        source = struct();
        source.p_mask = zeros(size(c_model));
        source.p_mask(element_i(tx), element_j(tx)) = 1;
        source.p = source_signal;

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
