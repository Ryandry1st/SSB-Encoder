params.total_time = 4;           % Total simulated time
params.fs = 200;                 % Sampling frequency of channels (not 5G sampling freq)
params.fc = 3.5e9;               % Carrier frequency
params.BW = 100e6;                 % Bandwidth
params.batch = 0;
params.n_FFT = 270;
params.scs = 30e3;
params.Ts = 1/(params.n_FFT * params.scs);
params.output_rsrp = 0;
params.L = 192;
params.save_H_D = 0;
params.save_folder_r = 'data/';
directories = dir(params.save_folder_r);
num_dir = numel(directories([directories(:).isdir]))-2;
params.save_folder_r = [params.save_folder_r, 'M-MIMO ', num2str(num_dir), '/'];
mkdir(params.save_folder_r);
mkdir([params.save_folder_r, 'channels/']);

params.downtilt = 5;
% params.orientations = [0, 5; 135, 5; -135, 5;
%                        0, 5; 135, 5; -135, 5;
%                        0, 5; 135, 5; -135, 5];
params.orientations = [90, params.downtilt; 210, params.downtilt; 330, params.downtilt];

params.Nr = 4;
params.Nx = 8;
params.Ny = 8;
params.theta_min = 5*pi/24;
params.theta_max = 19*pi/24;
params.phi_min = 5*pi/24;
params.phi_max = 12*pi/24;

params.no_sectors = 1;
params.no_tx = 1;
params.Tx_P_dBm(1, 1) = 62 + 10*log10(params.BW);
params.bs_height = 25;
params.UEs = 150; % Approximate number of UEs

params.rmin = params.bs_height * tan(params.phi_min + deg2rad(params.downtilt));
params.rmax = 400;
params.scen = '3GPP_38.901_UMa'; % '3GPP_3D_UMi', 'Freespace', '3GPP_38.901_UMa', '3GPP_3D_UMa', 'TwoRayGR'
