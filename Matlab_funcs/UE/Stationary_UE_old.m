function [locs, centers_xy] = Stationary_UE_old(params, N, max_xy, centers_xy)
% Return randomly placed UEs over the entire region, drawn from wide
% variance, multiple gaussian distribution

% Allocate users densely around 2D gaussians
% In a dense urban region, usually centered around buildings every ~300m 
var_range = 250;

if nargin < 3
    N_gaussians = floor(2*max_xy / var_range);
    n_sets = 1:N_gaussians;
    centers = var_range/2 + -max_xy + var_range*(n_sets-1);

    centers_xy = combvec(centers, centers);
else
    [~, N_gaussians] = size(centers_xy);
end


    
assignments = discretize(1:N, numel(centers_xy(1, :)));
randomness = sqrt(6*var_range) * randn(2, N);

locs = zeros(N, 3);
for i=1:N
    locs(i, 1:2) = centers_xy(:, assignments(i)) + randomness(:, i);
    if any(max_xy - abs(locs(i, :)) < max_xy/100)
        locs(i, 1:2) = centers_xy(:, assignments(i)) + randomness(:, i)/2;
    end
end
locs(:, 3) = 1.5; % 1.5m tall UEs for stationary


% scatter(locs(:, 1), locs(:, 2), 'filled');
% hold on;
% scatter(centers_xy(1, :), centers_xy(2, :), 'filled')
% hold off;

end

