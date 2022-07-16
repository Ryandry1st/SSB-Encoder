function [h_layout] = UE_dist(params, opt, max_xy, h_layout, p_stationary, p_walking, p_driving, p_flying, ttime)
% Defines and generates UE distributions
% opt can be [1, 2, 3, 4, 5, 6]
%   1) All stationary/walking UE
%   2) All driving UE
%   3) Mixture of 1, 2 for a given region layout
%   4) Mixture of 1, 2 for a different region layout
%   5) Mixture 1, 2, UAVs for the first region
%   6) Mixture 1, 2, UAVs for the second region

% Quick defaults
if nargin == 0
    params.fs = 200;
    params.Nx = 8;
    params.Ny = 8;
    params.Nr = 4;
    params.fc = 3.5e9;
    params.total_time = 2;
    ttime = params.total_time;
end
if nargin <= 1
    opt = 7; % case 7 is for single sector M-MIMO
    max_xy = 300;
    h_layout = qd_layout;
    p_stationary = 0.0;
    p_walking = 0.5;
    p_driving = 0.4;
    p_flying = 0.1;
    ttime = params.total_time;

else
    if nargin < 10
    ttime = 2; % 2 sec default
    end
    if nargin < 9
        p_flying = 0.00001;
    end
    if nargin < 8
        p_driving = 0.1;
    end
    if nargin < 7
        p_walking = 0.2;
    end
    if nargin < 6
        p_stationary = 0.7;
    end
end

approx_no_UE = params.UEs;
road_set = 0;
switch(opt)
    case 1
        p_flying = 0;
        p_driving = 0;
        p_stationary = p_stationary/(p_startionary + p_walking);
        p_walking = p_walking/(p_stationary + p_walking);
    case 2
        p_flying = 0;
        p_stationary = 0;
        p_walking = 0;
        p_driving = 1;
    case 3
        road_set = 1;
        p_flying = 0;
    case 4
        road_set = 2;
        p_flying = 0;
    case 5
        if p_flying < 0.05
            p_flying = 0.05;
        end
        p_flying = p_flying / (p_stationary + p_walking + p_driving + p_flying);
        p_driving = p_driving / (p_stationary + p_walking + p_driving + p_flying);
        p_walking = p_walking / (p_stationary + p_walking + p_driving + p_flying);
        p_stationary = p_startionary / (p_stationary + p_walking + p_driving + p_flying);
        road_set = 1;
    case 6
        if p_flying < 0.05
            p_flying = 0.05;
        end
        p_flying = p_flying / (p_stationary + p_walking + p_driving + p_flying);
        p_driving = p_driving / (p_stationary + p_walking + p_driving + p_flying);
        p_walking = p_walking / (p_stationary + p_walking + p_driving + p_flying);
        p_stationary = p_startionary / (p_stationary + p_walking + p_driving + p_flying);
        road_set = 2;
        
    case 7
        road_set = 3;
        p_stationary = 0.00;
        p_walking = 0.60;
        p_driving = 0.40;
        p_flying = 0;
    case 8
        road_set = 4;
        p_stationary = 0.00;
        p_walking = 0.4;
        p_driving = 0.60;
        p_flying = 0;
end
if (p_stationary + p_walking + p_driving + p_flying) < 0.99 || (p_stationary + p_walking + p_driving + p_flying) > 1.01
    warn("Your probabilities dont add up to 1!!");
end
    
s = qd_simulation_parameters;
s.center_frequency = params.fc;
h_layout.simpar = s;
if road_set == 1
    % road_set 1 with isd = 300
    centers_xy = [0, 0; sqrt(2)/2*300, sqrt(2)/2*300; -sqrt(2)/2*300, sqrt(2)/2*300]';

%     centers_xy = [0, 0];
    r1 = Road([-100, 200], [200, 200]);
    r2 = Road([200, 200], [200, 0]);
    r3 = Road([-100, 0], [100, 200]);
    r4 = Road([0, 100], [100, 0], 1); % highway = 1
    roads = [r1, r2, r3, r4];
elseif road_set == 2
    centers_xy = [0, 0; sqrt(2)/2*300, sqrt(2)/2*300; -sqrt(2)/2*300, sqrt(2)/2*300]';
%     centers_xy = [0, 0];
          
    r1 = Road([-150, 40], [150, 40]);
    r2 = Road([-150, 175], [150, 200]);
    r3 = Road([100, 250], [50, 75]);
    roads = [r1, r2, r3];
    
elseif road_set == 3
    centers_xy = [0; 0];
    r1 = Road([-75, 200], [50, 200]);
    r2 = Road([-50, 100], [50, 100]);
    roads = [r1, r2];
    
elseif road_set == 4
    centers_xy = [0; 0];
    r1 = Road([-75, 200], [50, 70]);
    r2 = Road([0, 40], [0, 80]);
    roads = [r1, r2];
    
else
    % three roads randomly scattered over the primary square of 
    % [(-150, 300), (200, 0)] rectangle
    x_range = -150:10:200;
    y_range = 0:10:300;
    x_set = randsample(x_range, 6);
    y_set = randsample(y_range, 6);
    roads = [];
    for i=1:3
        roads = [roads, Road([x_set(i), y_set(i)], [x_set(i+3), y_set(i+3)])];
    end
end

% Get number of each unit
no_stat_UE = sum(rand(approx_no_UE, 1) < p_stationary);
if no_stat_UE == 0
    no_stat_UE = 1;
end

no_walk_UE = sum(rand(approx_no_UE, 1) < p_walking);
no_drive_UE = sum(rand(approx_no_UE, 1) < p_driving);
no_fly_UE = sum(rand(approx_no_UE, 1) < p_flying);
tot_UEs = no_stat_UE + no_walk_UE + no_drive_UE + no_fly_UE;

total_area = (params.theta_max-params.theta_min)*(params.rmax-params.rmin);  % (θ/360º) × πr2
density = 100*tot_UEs/total_area;

fprintf("Laying %i UEs with %.2e UEs/m^2 over %.1f m^2 area", tot_UEs, density, total_area);
fprintf(" with %i, stationary, %i walking, %i driving, and %i flying.\n", no_stat_UE, no_walk_UE, no_drive_UE, no_fly_UE);

locs = zeros(tot_UEs, 3);
velocities = zeros(tot_UEs, 3);
running_sum = 0;

%% Generate UE distributions
% Stationary UEs
[locs(1:no_stat_UE, :), centers] = Stationary_UE(params, no_stat_UE, centers_xy);
running_sum = running_sum + no_stat_UE;

% Walking UEs
[locs(running_sum+1 : running_sum+no_walk_UE, :), velocities(running_sum+1 : running_sum+no_walk_UE, :)] = Walking_UE(params, no_walk_UE, centers_xy);
running_sum = running_sum + no_walk_UE;

% Driving UEs
[locs(running_sum+1 : running_sum + no_drive_UE, :), velocities(running_sum+1 : running_sum + no_drive_UE, :)] = Driving_UE(no_drive_UE, roads);
running_sum = running_sum + no_drive_UE;

% Flying UEs -- just using walking UE for now with higher velocity
[locs(running_sum+1 : running_sum+no_fly_UE, :), velocities(running_sum+1 : running_sum+no_fly_UE, :)] = Walking_UE(params, no_fly_UE, centers_xy);
velocities(running_sum+1 : running_sum+no_fly_UE, :) = velocities(running_sum+1 : running_sum+no_fly_UE, :) * 10;
% running_sum = running_sum + no_fly_UE;


%% Add UEs to a layout
h_layout.no_rx = tot_UEs;
h_layout.no_tx = params.no_tx;
h_layout.tx_position = [0, 0, 25; sqrt(2)/2*300, sqrt(2)/2*300, 25; -sqrt(2)/2*300, sqrt(2)/2*300, 25]';

for k=1:no_stat_UE
    t = qd_track('linear'); 
    t.name = ['Rx', sprintf( '%05d', k ) ;];
    t.initial_position = locs(k, :)';
    t.calc_orientation;
    h_layout.rx_track(k) = t.copy;
end

if no_stat_UE > 0
    for k=no_stat_UE:tot_UEs
        t = qd_track('linear', ttime * norm(velocities(k, :)), atan2(velocities(k, 2), velocities(k, 1))); 
        t.initial_position = locs(k, :)';
        t.name = ['Rx', num2str(k)];
        t.movement_profile = [0, ttime; 0, norm(velocities(k, :))*ttime];
        t.calc_orientation;
        [~, h_layout.rx_track(k)] = interpolate(t.copy, 'time', 1/params.fs);
        calc_orientation(h_layout.rx_track(k));
    end
end

%% Create BS
% orientations = [10, 225, -45];
h_layout.no_tx = params.no_tx;
for i=1:params.no_tx
    j=1;
    h_layout.tx_array(i) = qd_arrayant('3gpp-3d', params.Ny, params.Nx, params.fc, 1, params.downtilt);
%     h_layout.tx_array(i)= qd_arrayant('3gpp-macro', 60, 10, 30, 25);

    h_layout.tx_array(i).rotate_pattern(params.orientations(j, i), 'z');

    for j=2:params.no_sectors
        a = qd_arrayant('3gpp-3d', 8, 8, params.fc, 1, params.downtilt);
%         a = qd_arrayant('3gpp-macro', 60, 10, 30, 25);
        a.rotate_pattern(params.orientations(j, i), 'z');
        h_layout.tx_array(i).append_array(a);
    end
    h_layout.tx_array(i).center_frequency = params.fc;
end
for i=1:h_layout.no_rx
   h_layout.rx_array(i) = qd_arrayant('3gpp-3d', params.Nr/2, 2, params.fc, 1);
%     h_layout.rx_array(i) = qd_arrayant('ula8');
%     h_layout.rx_array(i) = qd_arrayant('3gpp-macro', 90, 160/params.Nr, 30, -10);
    h_layout.rx_array(i).center_frequency = params.fc;
end

h_layout.set_scenario(params.scen, [], []);
end

