function [locs, velocities] = Walking_UE(params, N, bs_locs)
%WALKING_UE generate locations and velocities for walking users.
% Velocities come from normal distribution speeds and uniform direction
% while initial positions are uniform over the space

speeds = 1.4 + 0.15*randn(N, 1);
headings = 2*pi*rand(N, 1);
x_vel = speeds .* cos(headings);
y_vel = speeds .* sin(headings);
velocities = zeros(N, 3);
velocities(:, 1) = x_vel;
velocities(:, 2) = y_vel;

[locs, bs_locs] = Stationary_UE(params, N, bs_locs);

end

