function [ Pt ] = iterate_particle_filter( Xt_1, Pt_1, It, Ft_1, p, likelihood_func )
%ITERATE_PARTICLE_FILTER Summary of this function goes here
% Detailed explanation goes here

Rt_1 = resampling(Pt_1, p);
Rt = state_transion(Rt_1, Xt_1, p);
Lt = likelihood_func(Rt, It, Ft_1, p);
Pt = [Rt Lt];

end

function [Rt_1] = resampling(Pt_1, p)
%% Resample a new particle set from particles
% which represents posterior distribution at previous time step p(x_(t-1) | z_(1:t-1))

rsam_vec = mnrnd(p.M, Pt_1(:, 4));

cnt = 1;
Rt_1 = zeros(p.M, 3);

for i = 1 : p.M
    if rsam_vec(i) ~= 0
        for j = 1 : rsam_vec(i)
            Rt_1(cnt, 1:3) = Pt_1(i, 1:3);
            cnt = cnt + 1;
        end
    end
end

end

function [Rt] = state_transion(Rt_1, Xt_1, p)
%% State transition by random walk 
% p(x_(t-1) | z_(1:t-1)) --> p(x_t | z_(1:t-1))
% random walk for each particle
% move state of each particle using random Gaussian samples

est_s = mean(Rt_1(:, 3));
SIGMA = diag([p.sigma_x * est_s, p.sigma_y * est_s * p.A, p.sigma_s * est_s]);
SIGMA = SIGMA .* SIGMA;

mu = zeros(1, length(SIGMA));

% generate zero-mean Gaussian samples
noise = mvnrnd(mu, SIGMA, p.M);

% move state of each particle using the above noise
Xt_1 = repmat(Xt_1, p.M);
Xt_1 = Xt_1(:, 1:3);
Rt = Xt_1 + noise;

end