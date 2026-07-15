
%% Convert CT Number / Hounsfield units to sound speed
% Use this with true CT numbers whenever possible. A display PNG is usually
% windowed grayscale, not quantitative HU, so the PNG branch below is only
% an approximation unless you know the CT window used to export it.

clear; close all; clc;

%% Input
input_file = 'C:\Users\Administrator\Documents\RB-QUST-main\r-Wave\1.3.12.2.1107.5.1.4.83567.30000025112415592178200005243.png';

% If input_file is a display PNG, set the CT window used during export.
% If you have the original DICOM, use that instead and these are ignored.
png_window_center_hu = 0;
png_window_width_hu = 2000;

% Acoustic calibration. Replace these with measured values for your phantom
% materials if you have them. For water-bath UST simulations, map air/outside
% image regions to water rather than physical air.
hu_cal = [-1000, -100,    0,   50,  100,  300, 1000, 2000];
c_cal  = [ 1480, 1450, 1480, 1540, 1580, 1650, 2200, 3000]; % [m/s]

c_min = 1400;
c_max = 3000;
water_bath_speed = 1480;

%% Load CT numbers
[~, ~, ext] = fileparts(input_file);
switch lower(ext)
    case '.dcm'
        info = dicominfo(input_file);
        raw = double(dicomread(info));
        if isfield(info, 'RescaleSlope')
            slope = double(info.RescaleSlope);
        else
            slope = 1;
        end
        if isfield(info, 'RescaleIntercept')
            intercept = double(info.RescaleIntercept);
        else
            intercept = 0;
        end
        ct_hu = slope * raw + intercept;

    case {'.png', '.tif', '.tiff', '.jpg', '.jpeg'}
        img = imread(input_file);
        if ndims(img) == 3
            img = rgb2gray(img(:, :, 1:3));
        end
        img = double(img);
        img = (img - min(img(:))) / max(eps, max(img(:)) - min(img(:)));

        hu_low = png_window_center_hu - png_window_width_hu / 2;
        hu_high = png_window_center_hu + png_window_width_hu / 2;
        ct_hu = hu_low + img * (hu_high - hu_low);
        warning(['PNG/JPEG/TIFF input was converted from display grayscale to HU using ', ...
            'window center %.1f HU and width %.1f HU. Use DICOM for quantitative conversion.'], ...
            png_window_center_hu, png_window_width_hu);

    otherwise
        error('Unsupported input extension: %s', ext);
end

%% HU to sound speed
sound_speed = interp1(hu_cal, c_cal, ct_hu, 'linear', 'extrap');
sound_speed = min(max(sound_speed, c_min), c_max);

% Treat very low HU/background as water bath for k-Wave UST. Change this to
% 343 m/s only if you intentionally want acoustic air in the model.
background_mask = ct_hu <= -900;
sound_speed(background_mask) = water_bath_speed;

%% Save outputs
out_dir = 'C:\Users\Administrator\Documents\RB-QUST-main\r-Wave';
save(fullfile(out_dir, 'ct_to_sound_speed_result.mat'), ...
    'ct_hu', 'sound_speed','Vq', 'hu_cal', 'c_cal', 'png_window_center_hu', 'png_window_width_hu');

speed_png = uint16(65535 * mat2gray(sound_speed, [c_min, c_max]));
imwrite(speed_png, fullfile(out_dir, 'sound_speed_map.png'));

%% Display
figure('Color', 'w', 'Name', 'CT Number to Sound Speed');
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
imagesc(ct_hu); axis image off; colormap(gca, gray); colorbar;
title('CT number [HU]');

nexttile;
imagesc(sound_speed, [c_min, c_max]); axis image off; colormap(gca, turbo); colorbar;
title('Sound speed [m/s]');

fprintf('Saved: %s\n', fullfile(out_dir, 'ct_to_sound_speed_result.mat'));
fprintf('Saved: %s\n', fullfile(out_dir, 'sound_speed_map.png'));
fprintf('Sound speed range: %.1f to %.1f m/s\n', min(sound_speed(:)), max(sound_speed(:)));
