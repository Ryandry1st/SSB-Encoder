function [locs, bs_locs] = Stationary_UE(params, N, bs_locs)
% Return randomly placed UEs over the entire region, drawn from uniform
% spacing

radii = params.rmin + (params.rmax-params.rmin) * sqrt(rand(N, 1));
angles = params.theta_min + (params.theta_max-params.theta_min) * rand(N, 1);

assignments = discretize(1:N, numel(bs_locs(1, :)));

for i=1:N
    angles(i) = angles(i) + deg2rad(params.orientations(assignments(i), 1)) - pi/2; % Offset for different oriented BS sectors
end
x_locs = radii .* cos(angles);
y_locs = radii .* sin(angles);

locs = zeros(N, 3);
for i=1:N
    locs(i, 1:2) = bs_locs(:, assignments(i)) + [x_locs(i); y_locs(i)];
end
locs(:, 3) = 1.5; % 1.5m tall UEs for stationary


% scatter(locs(:, 1), locs(:, 2), 'filled');
% hold on;
% scatter(bs_locs(1, :), bs_locs(2, :), 'filled')
% hold off;

end

