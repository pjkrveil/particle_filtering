function [] = particle_filter_tracking( seq_path, result_file_path )
%PARTICLE_FILTER_TRACKING Track the given target with color-based particle filter algorithm

dbstop error; % Stop the code when error occurs
% If you stuck in debug-mode due to error, you can take an action in below.
    % type *dbquit* to exit debug mode.
    % type *dbup* and *dbdown* move former/later called function
    % type *dbcont* to execute remaining code

global display_figure
display_figure = 1;
%display_figure = 0; % If you don't want figure window

%% parameters for tracker
% Tune parameter for better tracking result
p.M = 100; % the number of particles
p.Nh = 10;
p.Ns = 10;
p.Nv = 10;
p.lambda = 20;
p.Nbin = p.Nh * p.Ns + p.Nv;

p.sigma_x = 0.2;
p.sigma_y = 0.2;
p.sigma_s = 0.1;

%% load sequence file information
% load paths of image files
img_files = dir([seq_path '/img/*.jpg']);

% the number of frames
Nf = length(img_files);

gt_file = [seq_path '/groundtruth_rect.txt'];
Ybox = load_ground_truth(gt_file);

% variable for saving tracking result
X = cell(Nf,1); % use cell data-type for readability

% compute state vector at first frame
% [x1 y1 width height]
% -->  [x_center y_center scale(width)], aspect ratio (height/width)
[X1, p.A] = bbox2xys(Ybox{1});
X{1} = [X1 1];

%% intialize entire pipeline for tracker
initialize_display();

I1 = imread([img_files(1).folder '/' img_files(1).name]);
p.W = size(I1, 2); % image width
p.H = size(I1, 1); % image height

F = initialize_observation_model(I1, X{1}, p);

P = cell(Nf,1);
P{1} = initialize_particles(X{1}, p.M);

ROI_HIST = zeros(Nf,1);

%% main body for tracker
for t=2:min(Nf, size(Ybox,1))
    
    img_file_path = [img_files(t).folder '/' img_files(t).name];
    It = imread(img_file_path);
    
    P{t} = iterate_particle_filter(X{t-1}, P{t-1}, It, F, p, @compute_likelihood);
    
    X{t} = estimate_current_state(P{t});
    
    ROI_HIST(t) = display_result(It, X{t}, Ybox{t}, P{t}, p);
end

%% save result
Xmat = cell2mat(X);
Xbox = xys2bbox(Xmat, p.A);
save_tracking_result(Xbox, result_file_path);
histogram(ROI_HIST)
end

%% YOU NEED TO IMPLEMENT THOSE FUNCTIONS
function [Ht] = color_histogram( Rt, It, p )
    Ih = rgb2hsv(It);
    %% cellfun for alternative to a for-loop
    Rt_cell = mat2cell(Rt, ones(1, size(Rt,1)), [size(Rt,2)]);
    function [Hist_HSV] = hist_HS_V(Rti)
        % transforming coordinate for cropping image.
        width = Rti(3);
        height = Rti(3) * p.A;
        
        x1 = Rti(1) - width/2;
        y1 = Rti(2) - height/2;
        x2 = x1 + width - 1;
        y2 = y1 + height - 1;
        
        % handling out-of-image coordinate
        x1 = max(1, ceil(x1));
        y1 = max(1, ceil(y1));
        x2 = min(p.W, ceil(x2));
        y2 = min(p.H, ceil(y2));
        
        r_x1 = min(x1, x2);
        r_x2 = max(x1, x2);
        r_y1 = min(y1, y2);
        r_y2 = max(y1, y2);
        
        % make the color model in ref [1]
        Crop_I = Ih(r_y1:r_y2, r_x1:r_x2, :);
        
        H = discard_Value(Crop_I(:, :, 1), 0.1);
        S = discard_Value(Crop_I(:, :, 2), 0.2);
        V = Crop_I(:, :, 3);
        
        % make the 2D-histogram for HS space and 1D-histogram for V space
        [Hist_HS, ~, ~] = histcounts2(H, S, p.Nh);
        [Hist_V, ~] = histcounts(V, p.Nv);

        % vectorize the above histogram matrix and concatenate the two color histogram vector
        Hist_HS = reshape(Hist_HS, [p.Nh * p.Nv, 1]);
        Hist_V = reshape(Hist_V, [p.Nv, 1]);
        Hist_HSV = cat(1, Hist_HS, Hist_V);
        
        % normalize the color histogram vector
        Hist_HSV = Hist_HSV / sum(Hist_HSV);
    end

    %% function for discarding hue(H) and saturation(S) dimension with threshold
    function [Hist] = discard_Value(Hist, thres)
        Hist = Hist.*(Hist > thres);
    end

    Ht_cell = cellfun(@hist_HS_V, Rt_cell, 'UniformOutput', false);
    Ht = cell2mat(Ht_cell);
    % if you don't believe, check the time of for-loop.
end

% compute likelihood using distance
function [Lt] = compute_likelihood(Rt, It, F, p)
    % F : observation model at initial frame
    Lt = zeros(p.M, 1);     
    
    for i = 1 : p.M
        Q = color_histogram(Rt(i, :), It, p);       % observation model
        D = 1 - sum(sqrt(F .* Q));
        Lt(i) = exp(- p.lambda * D);
    end
end

function [Xt] = estimate_current_state(Pt)
[~, max_idx] = max(Pt(:, 4));
Xt = Pt(max_idx, :);
end

%% ALREADY IMPLEMENTED FUNCTIONS
function [F] = initialize_observation_model(I1, Y1, p)
    F = color_histogram(Y1, I1, p);
end

function [X3d, A] = bbox2xys(Xbox)
    X3d = zeros(size(Xbox,1),3);
    
    % x_center
    X3d(:, 1) = (Xbox(:, 1)*2+Xbox(:,3)-1)/2;
    
    % y_center
    X3d(:, 2) = (Xbox(:, 2)*2+Xbox(:,4)-1)/2;
    
    % scale = width
    X3d(:, 3) = Xbox(:, 3);
    
    % Aspect ratio = height/width
    A = Xbox(:, 4) / Xbox(:, 3);
end

function [Xbox] = xys2bbox(X3d, A)
   W = X3d(:, 3);
   H = X3d(:, 3) .* A;
   X1 = X3d(:, 1) - W/2;
   Y1 = X3d(:, 2) - H/2;
   Xbox = ceil([X1 Y1 W H]);
end

function [P1] = initialize_particles(Y1, M)
    P0 = [Y1 1]; % only ground truth
    P1 = repmat(P0, M, 1);
    P1(:,4) = P1(:,4)/M;
end

function [Ycell] = load_ground_truth(gt_file)
    Ybox = dlmread(gt_file);
    Ycell = mat2cell(Ybox, ones(1, size(Ybox,1)), [4]);
end

function [] = save_tracking_result(X, result_file_path)
    dlmwrite(result_file_path, X, ',');
end

function [] = initialize_display()
global display_figure
if(display_figure)
    figure(1)
end
end

function [ROI_HIST] = display_result(It, Xt, Ybox, Pt, p)
global display_figure
if(display_figure)
    
    Xbox = xys2bbox(Xt, p.A);
    Pbox = xys2bbox(Pt, p.A);
    
    imshow(It);
    hold on
    for i = 1:min(size(Pbox, 1), 20)
        rectangle('Position', Pbox(i, :), 'EdgeColor', 'y', 'LineStyle', '--');
    end
    rectangle('Position', Ybox, 'EdgeColor', 'g');
    rectangle('Position', Xbox, 'EdgeColor', 'r');
    ROI_HIST = bboxOverlapRatio(Ybox, Xbox);
    
    hold off
    drawnow;
end
end