function [locs, velocities] = Driving_UE(N, roads)
%DRIVING_UE generate locations and velocities for driving UEs.
% Driving follows two different distributions for general driving and
% highway driving

% roads are defined as line segments of [Road([(x1, y1), (x2, y2)]), ... ]
%                                         ^----------------^
%                                                road 1
% and UE are distributed evenly by length
tot_length = 0;
lengths = zeros(1, length(roads));
for i=1:length(roads)
    lengths(i) = roads(i).get_length();
    tot_length = tot_length + lengths(i);
end

percent_lengths = lengths ./ tot_length;
N_i = round(N*percent_lengths);

while sum(N_i) > N
    [~, i] = min(N_i);
    N_i(i) = N_i(i) - 1;
end

while sum(N_i) < N
    [~, i] = max(N_i);
    N_i(i) = N_i(i) + 1;
end

locs = [];
velocities = [];
for i=1:length(roads)
    [new_locs, new_vels] = roads(i).assign(N_i(i));
    locs = [locs; new_locs];
    velocities = [velocities; new_vels];
end

% figure;
% scatter(locs(:, 1), locs(:, 2), 'x');
% hold on;
% r1.plot();
% r2.plot();
% r3.plot();
% hold off;
end

