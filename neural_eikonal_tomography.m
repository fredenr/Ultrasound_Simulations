function [sound_speed, result] = neural_eikonal_tomography( ...
        time_shift, source_x, source_y, receiver_x, receiver_y, ...
        x_centres, y_centres, roi, background_speed, options)
%NEURAL_EIKONAL_TOMOGRAPHY Reconstruct sound speed from first-arrival shifts.
% Requires Deep Learning Toolbox; no labeled speed image is used. One neural
% field represents c(x,y), another represents positive T(x,y;source).
% Training fits arrivals and c*|grad T|=1. This first-arrival approximation
% allows refraction but does not invert diffraction, reflections or amplitudes.
% Coordinates are metres, shifts seconds, output rows x and columns y.
% OFFLINE ONLY: the saved coordinate fields describe this scan. For fast
% reconstruction of new scans use train_usct_image_model/predict_usct_image.

    if nargin < 11, options = struct(); end
    defaults = struct('iterations',8000, 'hidden_units',128, 'hidden_layers',4, ...
        'batch_size',512, 'collocation_points',8192, 'learn_rate',1e-3, ...
        'min_sound_speed',1000, 'max_sound_speed',7000, 'physics_weight',1, ...
        'smoothness_weight',0.001, 'background_weight',1e-4, ...
        'contrast_scale',0.05, 'gradient_clip',5, 'smooth_step',0.02, ...
        'position_frequencies',[1 2 4 8 16], 'use_gpu',true, ...
        'physics_chunk_size',1024, 'boundary_sample_fraction',0.3, ...
        'adaptive_sample_fraction',0.25, 'huber_delta',1, ...
        'pair_weights',ones(size(time_shift)), 'feature_weights',[], ...
        'run_independent_validation',true, 'validation_max_pairs',64, ...
        'display_interval',100, 'random_seed',7, 'arrival_tolerance_s',0);
    names = fieldnames(defaults);
    for k = 1:numel(names)
        if ~isfield(options,names{k}), options.(names{k}) = defaults.(names{k}); end
    end
    if exist('dlnetwork','class') ~= 8 && exist('dlnetwork','file') ~= 2
        error('neuralUST:MissingToolbox','Deep Learning Toolbox is required.');
    end
    validateattributes(background_speed,{'numeric'},{'scalar','finite','positive'});
    validateattributes(x_centres,{'numeric'},{'vector','finite','increasing'});
    validateattributes(y_centres,{'numeric'},{'vector','finite','increasing'});
    if numel(x_centres)<2 || numel(y_centres)<2 || ...
            ~isequal(size(roi),[numel(x_centres),numel(y_centres)]) || ~any(roi(:))
        error('neuralUST:InvalidROI','roi must be a nonempty x-by-y mask.');
    end
    if ~isequal(size(time_shift),size(source_x),size(source_y),size(receiver_x),size(receiver_y))
        error('neuralUST:GeometrySize','All pair matrices must have identical sizes.');
    end
    if ~(options.min_sound_speed<background_speed && background_speed<options.max_sound_speed)
        error('neuralUST:SpeedBounds','Background speed must lie strictly inside the speed bounds.');
    end
    for name = {'iterations','hidden_units','hidden_layers','batch_size','collocation_points','display_interval','physics_chunk_size'}
        validateattributes(options.(name{1}),{'numeric'},{'scalar','integer','positive'});
    end
    for name = {'learn_rate','contrast_scale','gradient_clip','smooth_step','huber_delta'}
        validateattributes(options.(name{1}),{'numeric'},{'scalar','finite','positive'});
    end
    for name = {'physics_weight','smoothness_weight','background_weight','arrival_tolerance_s'}
        validateattributes(options.(name{1}),{'numeric'},{'scalar','finite','nonnegative'});
    end
    if options.physics_weight<=0
        error('neuralUST:PhysicsWeight','A positive physics weight is required.');
    end
    validateattributes(options.pair_weights,{'numeric'},{'finite','nonnegative'});
    if ~isequal(size(options.pair_weights),size(time_shift))
        error('neuralUST:Weights','pair_weights must match the time-shift matrix.');
    end
    options.position_frequencies=reshape(options.position_frequencies,1,[]);
    if any(options.position_frequencies<=0 | ~isfinite(options.position_frequencies))
        error('neuralUST:Frequencies','Fourier frequencies must be positive and finite.');
    end
    for name={'boundary_sample_fraction','adaptive_sample_fraction'}
        validateattributes(options.(name{1}),{'numeric'},{'scalar','>=',0,'<=',0.5});
    end
    rng_state = rng;
    restore_rng = onCleanup(@() rng(rng_state)); %#ok<NASGU>
    rng(options.random_seed);
    [grid_x,grid_y] = ndgrid(x_centres,y_centres);
    geometry.c0 = background_speed;
    geometry.centre = [mean(x_centres([1 end]));mean(y_centres([1 end]))];
    geometry.scale = max([x_centres(end)-x_centres(1),y_centres(end)-y_centres(1)])/2;
    geometry.lower = ([x_centres(1);y_centres(1)]-geometry.centre)/geometry.scale;
    geometry.upper = ([x_centres(end);y_centres(end)]-geometry.centre)/geometry.scale;
    geometry.x_centres = x_centres;
    geometry.y_centres = y_centres;
    geometry.roi = logical(roi);
    edge=geometry.roi & conv2(double(geometry.roi),ones(3),'same')<9;
    [bi,bj]=find(edge);
    geometry.boundary_coordinates=bsxfun(@minus, ...
        [reshape(x_centres(bi),1,[]);reshape(y_centres(bj),1,[])],geometry.centre)/geometry.scale;
    geometry.speed_bias = log((background_speed-options.min_sound_speed)/ ...
        (options.max_sound_speed-background_speed));
    geometry.factor_min = background_speed/options.max_sound_speed;
    geometry.factor_max = background_speed/options.min_sound_speed;
    geometry.time_bias = log((1-geometry.factor_min)/(geometry.factor_max-1));

    distance = hypot(receiver_x-source_x,receiver_y-source_y);
    arrival = distance/background_speed+time_shift;
    valid = isfinite(time_shift) & isfinite(source_x) & isfinite(source_y) & ...
        isfinite(receiver_x) & isfinite(receiver_y) & distance>0 & options.pair_weights>0;
    outside_bounds = valid & (arrival<=0 | ...
        arrival<distance/options.max_sound_speed-options.arrival_tolerance_s | ...
        arrival>distance/options.min_sound_speed+options.arrival_tolerance_s);
    if any(outside_bounds(:))
        warning('neuralUST:ArrivalBounds', ...
            '%d arrivals exceed speed bounds and were excluded; check calibration/picking.',nnz(outside_bounds));
    end
    valid(outside_bounds) = false;
    if nnz(valid)<8
        error('neuralUST:InsufficientData','At least eight usable transmitter/receiver arrivals are required.');
    end
    sources = [source_x(valid).';source_y(valid).'];
    receivers = [receiver_x(valid).';receiver_y(valid).'];
    sources = bsxfun(@minus,sources,geometry.centre)/geometry.scale;
    receivers = bsxfun(@minus,receivers,geometry.centre)/geometry.scale;
    if any(any(bsxfun(@lt,[sources receivers],geometry.lower-1e-6))) || ...
            any(any(bsxfun(@gt,[sources receivers],geometry.upper+1e-6)))
        error('neuralUST:Domain','Output coordinate domain must contain every effective source and receiver.');
    end
    observations.coordinates = single([receivers;sources]);
    observations.baseline = single(distance(valid).'/background_speed);
    observations.shift = single(time_shift(valid).');
    observations.weights=single(options.pair_weights(valid).');
    geometry.time_scale = max(median(double(observations.baseline))*options.contrast_scale,1e-8);
    use_gpu = true;
    if options.use_gpu
        try
            use_gpu = gpuDeviceCount>0;
        catch
            use_gpu = false;
        end
        if ~use_gpu, warning('neuralUST:CPU','No usable MATLAB GPU found; training on CPU.'); end
    end
    speed_net = coordinate_network(2+4*numel(options.position_frequencies),options);
    time_net = coordinate_network(4+8*numel(options.position_frequencies),options);
    if use_gpu
        speed_net = dlupdate(@gpuArray,speed_net);
        time_net = dlupdate(@gpuArray,time_net);
    end
    speed_average = []; speed_average_sq = [];
    time_average = []; time_average_sq = [];
    monitor = make_batch(observations,sources,geometry,options,true,use_gpu);
    options.feature_weights=feature_schedule(options.position_frequencies,0);
    hard_points=[];
    history = nan(options.iterations,5);
    feature_weight_history=nan(options.iterations,numel(options.position_frequencies));
    best_iteration = 0;
    best_speed_net = speed_net;
    best_time_net = time_net;
    fprintf('Neural eikonal training: %d pairs, %d iterations, GPU=%d.\n', ...
        nnz(valid),options.iterations,use_gpu);
    % Model selection uses a fixed monitoring batch, not the stochastic loss.
    [initial_loss,~,~,initial_parts] = evaluate_batch( ...
        speed_net,time_net,monitor,geometry,options,false);
    best_loss = scalar_cpu(initial_loss);
    best_parts = cpu_values(initial_parts);
    monitor_history = [0 best_loss cpu_values(initial_parts)];
    % Compare candidate models only after every Fourier band is enabled.
    best_loss=Inf;
    for iteration = 1:options.iterations
        options.feature_weights=feature_schedule(options.position_frequencies,iteration/options.iterations);
        feature_weight_history(iteration,:)=options.feature_weights;
        batch = make_batch(observations,sources,geometry,options,false,use_gpu);
        if ~isempty(hard_points)
            count=min(size(hard_points,2),round(options.adaptive_sample_fraction*options.collocation_points));
            ids=randperm(size(hard_points,2),count);
            batch.collocation(:,1:count)=to_dl(hard_points(:,ids),use_gpu);
            batch.roi(:,1:count)=to_dl(roi_weights(hard_points(1:2,ids),geometry),use_gpu);
            nearby=hard_points(1:2,ids)+options.smooth_step*randn(2,count);
            nearby=bsxfun(@max,geometry.lower,bsxfun(@min,geometry.upper,nearby));
            batch.nearby(:,1:count)=to_dl(nearby,use_gpu);
            batch.nearby_roi(:,1:count)=to_dl(roi_weights(nearby,geometry),use_gpu);
        end
        [loss,speed_gradient,time_gradient,parts,residuals] = evaluate_batch( ...
            speed_net,time_net,batch,geometry,options,true);
        if mod(iteration,options.display_interval)==0
            [~,order]=sort(abs(residuals),'descend');
            ids=order(1:max(1,round(options.adaptive_sample_fraction*numel(order))));
            hard_points=double(cpu_parameter(extractdata(batch.collocation(:,ids))));
        end
        value = scalar_cpu(loss);
        if ~isfinite(value), error('neuralUST:NonfiniteLoss','Non-finite loss at iteration %d.',iteration); end
        speed_gradient = clip_gradients(speed_gradient,options.gradient_clip);
        time_gradient = clip_gradients(time_gradient,options.gradient_clip);
        rate = options.learn_rate*(0.1+0.9*(1-iteration/options.iterations));
        [speed_net,speed_average,speed_average_sq] = adamupdate( ...
            speed_net,speed_gradient,speed_average,speed_average_sq,iteration,rate);
        [time_net,time_average,time_average_sq] = adamupdate( ...
            time_net,time_gradient,time_average,time_average_sq,iteration,rate);
        history(iteration,:) = [value cpu_values(parts)];
        if iteration==1 || mod(iteration,options.display_interval)==0 || iteration==options.iterations
            [monitor_loss,~,~,monitor_parts] = evaluate_batch( ...
                speed_net,time_net,monitor,geometry,options,false);
            value = scalar_cpu(monitor_loss);
            if ~isfinite(value), error('neuralUST:NonfiniteMonitor','Non-finite monitoring loss.'); end
            monitor_history(end+1,:) = [iteration value cpu_values(monitor_parts)]; %#ok<AGROW>
            if all(options.feature_weights>=1) && value<best_loss
                best_loss = value;
                best_speed_net = speed_net;
                best_time_net = time_net;
                best_iteration = iteration;
                best_parts = cpu_values(monitor_parts);
            end
            fprintf('  neural %5d/%d: data %.3e, eikonal %.3e, monitor %.3e\n', ...
                iteration,options.iterations,scalar_cpu(monitor_parts(1)), ...
                scalar_cpu(monitor_parts(2)),value);
        end
    end
    speed_net = dlupdate(@cpu_parameter,best_speed_net);
    time_net = dlupdate(@cpu_parameter,best_time_net);
    coordinates = bsxfun(@minus,[grid_x(:).';grid_y(:).'],geometry.centre)/geometry.scale;
    values = zeros(1,numel(grid_x));
    for first = 1:4096:numel(grid_x)
        indices = first:min(first+4095,numel(grid_x));
        z = dlarray(single(coordinates(:,indices)),'CB');
        values(indices) = cpu_values(speed_field(speed_net,z, ...
            single(reshape(roi(indices),1,[])),geometry,options));
    end
    sound_speed = reshape(values,size(roi));
    sound_speed(~roi) = NaN;
    predicted_time = travel_time(time_net,dlarray(observations.coordinates,'CB'),geometry,options);
    fitted_shift = cpu_values(predicted_time)-double(observations.baseline);
    predicted_shift = nan(size(time_shift));
    predicted_shift(valid) = fitted_shift;
    rmse_s = sqrt(mean((fitted_shift-double(observations.shift)).^2));
    eikonal_rms = options.contrast_scale*sqrt(best_parts(2));
    fprintf('Returned neural iteration %d: arrival fit RMSE %.3f microseconds.\n',best_iteration,rmse_s*1e6);
    fprintf('Fixed monitoring points: RMS eikonal residual %.4f (target 0).\n',eikonal_rms);
    if best_iteration==0
        warning('neuralUST:NoImprovement','Training did not improve the initial homogeneous model.');
    end
    result = struct('method','source-conditioned neural eikonal first-arrival inversion', ...
        'model_kind','per_scan_eikonal_fields','reusable_for_new_scan',false, ...
        'speed_network',speed_net, 'travel_time_network',time_net, ...
        'options',options, 'geometry',geometry, 'used_gpu',use_gpu, ...
        'valid_pairs',valid, 'excluded_arrival_bounds',outside_bounds, ...
        'predicted_time_shift_s',predicted_shift, 'arrival_rmse_s',rmse_s, ...
        'eikonal_rms_relative',eikonal_rms, 'best_monitor_components',best_parts, ...
        'best_iteration',best_iteration, 'best_monitor_loss',best_loss, ...
        'training_history',history, 'monitor_history',monitor_history, ...
        'feature_weight_history',feature_weight_history, ...
        'history_columns',{{'total','arrival','eikonal','smoothness','background'}}, ...
        'monitor_columns',{{'iteration','total','arrival','eikonal','smoothness','background'}});
    if options.run_independent_validation
        result.independent_validation=validate_neural_speed_map(sound_speed,x_centres,y_centres, ...
            roi,background_speed,time_shift,source_x,source_y,receiver_x,receiver_y,valid, ...
            struct('max_pairs',options.validation_max_pairs));
        checked=result.independent_validation.selected_pairs;
        disagreement=predicted_shift(checked)- ...
            result.independent_validation.predicted_time_shift_s(checked);
        result.network_vs_speed_map_rmse_s=sqrt(mean(disagreement.^2));
        fprintf('Travel-time network vs independent speed-map RMSE: %.3f us.\n', ...
            result.network_vs_speed_map_rmse_s*1e6);
    end
end

function weights=feature_schedule(frequencies,progress)
    % Activate fine bands gradually; all bands are fully active by 80%.
    start=max(0,(log2(frequencies)-1)*0.2);
    weights=min(1,max(0,(progress-start)/0.2));
    weights(frequencies<=2)=1;
    if progress>=0.8, weights(:)=1; end
end

function [loss,gs,gt,parts,residuals]=evaluate_batch(sn,tn,batch,geometry,options,differentiate)
    % Accumulate physics microbatches to bound higher-derivative GPU memory.
    count=size(batch.collocation,2);
    gs=[]; gt=[]; loss=0; parts=zeros(1,4); residuals=zeros(1,count);
    for first=1:options.physics_chunk_size:count
        ids=first:min(first+options.physics_chunk_size-1,count);
        small=batch;
        for field={'collocation','nearby','roi','nearby_roi'}
            small.(field{1})=batch.(field{1})(:,ids);
        end
        [l,a,b,p,r]=dlfeval(@model_loss,sn,tn,small,geometry,options,differentiate);
        fraction=numel(ids)/count;
        loss=loss+fraction*scalar_cpu(l);
        parts=parts+fraction*cpu_values(p);
        residuals(ids)=cpu_values(r);
        if differentiate
            if isempty(gs)
                gs=a; gt=b;
                for k=1:height(gs)
                    gs.Value{k}=fraction*a.Value{k};
                    gt.Value{k}=fraction*b.Value{k};
                end
            else
                for k=1:height(gs)
                    gs.Value{k}=gs.Value{k}+fraction*a.Value{k};
                    gt.Value{k}=gt.Value{k}+fraction*b.Value{k};
                end
            end
        end
    end
end

function net = coordinate_network(channels,options)
    layers = featureInputLayer(channels,'Normalization','none','Name','coordinates');
    for k = 1:options.hidden_layers
        layers = [layers; fullyConnectedLayer(options.hidden_units,'Name',sprintf('hidden_%d',k)); ...
            tanhLayer('Name',sprintf('tanh_%d',k))]; %#ok<AGROW>
    end
    % Zero final output initializes c and T to homogeneous water.
    layers = [layers; fullyConnectedLayer(1,'Name','output', ...
        'WeightsInitializer','zeros','BiasInitializer','zeros')];
    net = dlnetwork(layers);
end

function batch = make_batch(observations,sources,geometry,options,monitor,use_gpu)
    n = size(observations.coordinates,2);
    if monitor, indices = 1:n; else, indices = randperm(n,min(options.batch_size,n)); end
    batch.observations = to_dl(observations.coordinates(:,indices),use_gpu);
    batch.baseline = to_dl(observations.baseline(indices),use_gpu);
    batch.shift = to_dl(observations.shift(indices),use_gpu);
    batch.weights=to_dl(observations.weights(indices),use_gpu);
    count = options.collocation_points;
    xy = bsxfun(@plus,geometry.lower,bsxfun(@times,geometry.upper-geometry.lower,rand(2,count)));
    boundary_count=round(options.boundary_sample_fraction*count);
    if ~isempty(geometry.boundary_coordinates) && boundary_count>0
        ids=randi(size(geometry.boundary_coordinates,2),1,boundary_count);
        xy(:,1:boundary_count)=geometry.boundary_coordinates(:,ids)+0.03*randn(2,boundary_count);
        xy=bsxfun(@max,geometry.lower,bsxfun(@min,geometry.upper,xy));
    end
    source = sources(:,randi(n,1,count));
    % Exclude the non-differentiable point-source singularity from PDE samples.
    near_source = sum((xy-source).^2,1)<1e-6;
    while any(near_source)
        xy(:,near_source) = bsxfun(@plus,geometry.lower, ...
            bsxfun(@times,geometry.upper-geometry.lower,rand(2,nnz(near_source))));
        near_source = sum((xy-source).^2,1)<1e-6;
    end
    angle = 2*pi*rand(1,count);
    nearby = xy+options.smooth_step*[cos(angle);sin(angle)];
    nearby = bsxfun(@max,geometry.lower,bsxfun(@min,geometry.upper,nearby));
    batch.collocation = to_dl([xy;source],use_gpu);
    batch.nearby = to_dl(nearby,use_gpu);
    batch.roi = to_dl(roi_weights(xy,geometry),use_gpu);
    batch.nearby_roi = to_dl(roi_weights(nearby,geometry),use_gpu);
end

function mask = roi_weights(xy,geometry)
    physical = bsxfun(@plus,xy*geometry.scale,geometry.centre);
    mask = single(interp2(geometry.y_centres,geometry.x_centres,double(geometry.roi), ...
        physical(2,:),physical(1,:),'nearest',0));
end

function [loss,speed_gradient,time_gradient,parts,residual] = model_loss( ...
        speed_net,time_net,batch,geometry,options,differentiate)
    predicted = travel_time(time_net,batch.observations,geometry,options);
    error=(predicted-batch.baseline-batch.shift)/geometry.time_scale;
    robust=2*options.huber_delta^2*(sqrt(1+(error/options.huber_delta).^2)-1);
    data_loss = sum(batch.weights.*robust,'all')/max(sum(batch.weights,'all'),eps);
    z = batch.collocation;
    t = travel_time(time_net,z,geometry,options);
    derivative = dlgradient(sum(t,'all'),z,'EnableHigherDerivatives',differentiate);
    speed = speed_field(speed_net,z(1:2,:),batch.roi,geometry,options);
    residual = (speed/geometry.scale).*sqrt(sum(derivative(1:2,:).^2,1)+1e-20)-1;
    physics_loss = mean((residual/options.contrast_scale).^2,'all');
    nearby_speed = speed_field(speed_net,batch.nearby,batch.nearby_roi,geometry,options);
    common_roi = batch.roi.*batch.nearby_roi;
    smooth_loss = mean(common_roi.*((speed-nearby_speed)/ ...
        (geometry.c0*options.contrast_scale)).^2,'all');
    background_loss = mean(((speed-geometry.c0)/(geometry.c0*options.contrast_scale)).^2,'all');
    loss = data_loss+options.physics_weight*physics_loss+ ...
        options.smoothness_weight*smooth_loss+options.background_weight*background_loss;
    parts = [data_loss physics_loss smooth_loss background_loss];
    if differentiate
        [speed_gradient,time_gradient] = dlgradient(loss,speed_net.Learnables,time_net.Learnables);
    else
        speed_gradient = []; time_gradient = [];
    end
end

function speed = speed_field(net,xy,roi,geometry,options)
    features = xy;
    for k=1:numel(options.position_frequencies)
        frequency=options.position_frequencies(k); weight=options.feature_weights(k);
        features = [features;weight*sin(pi*frequency*xy);weight*cos(pi*frequency*xy)]; %#ok<AGROW>
    end
    raw = forward(net,features);
    fraction = 1./(1+exp(-(raw+geometry.speed_bias)));
    bounded = options.min_sound_speed+(options.max_sound_speed-options.min_sound_speed)*fraction;
    speed = geometry.c0+roi.*(bounded-geometry.c0);
end

function t = travel_time(net,z,geometry,options)
    features=z;
    for k=1:numel(options.position_frequencies)
        frequency=options.position_frequencies(k); weight=options.feature_weights(k);
        features=[features;weight*sin(pi*frequency*z);weight*cos(pi*frequency*z)]; %#ok<AGROW>
    end
    raw = forward(net,features);
    fraction = 1./(1+exp(-(raw+geometry.time_bias)));
    factor = geometry.factor_min+(geometry.factor_max-geometry.factor_min)*fraction;
    distance = geometry.scale*sqrt(sum((z(1:2,:)-z(3:4,:)).^2,1));
    t = (distance/geometry.c0).*factor;
end

function gradient = clip_gradients(gradient,limit)
    for k = 1:height(gradient)
        g = gradient.Value{k};
        magnitude = scalar_cpu(sqrt(sum(g.^2,'all')));
        if ~isfinite(magnitude), error('neuralUST:NonfiniteGradient','Non-finite neural parameter gradient.'); end
        gradient.Value{k} = g*min(1,limit/max(magnitude,eps));
    end
end

function value = to_dl(value,use_gpu)
    value = single(value);
    if use_gpu, value = gpuArray(value); end
    value = dlarray(value,'CB');
end

function value = cpu_parameter(value)
    if isa(value,'dlarray')
        if isa(extractdata(value),'gpuArray'), value = gather(value); end
    elseif isa(value,'gpuArray')
        value = gather(value);
    end
end

function value = scalar_cpu(value)
    if isa(value,'dlarray'), value=extractdata(value); end
    value = double(cpu_parameter(value));
end

function values = cpu_values(values)
    if isa(values,'dlarray'), values=extractdata(values); end
    values = reshape(double(cpu_parameter(values)),1,[]);
end
