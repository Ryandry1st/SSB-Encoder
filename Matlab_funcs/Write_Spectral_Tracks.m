%% Store Spectral Information
% This script is used to take the channel information from a simulation and
% convert it into a file format for NS3. There is no encoding and the data
% should be ready to be loaded in right away.
show_plot = 0;

% See
% https://home.zhaw.ch/kunr/NTM1/literatur/LTE%20in%20a%20Nutshell%20-%20Physical%20Layer.pdf
% for details of LTE PHY options
% BW_options   = [1.25, 2.5, 5, 10, 15, 20, 40, 100]; % requested bandwidth should be one of these
% FFT_Sampling = [1.92, 3.84, 7.68, 15.36, 23.04, 30.72]*10e6;
% FFT_options  = [128, 256, 1024, 1536, 2048];
% RB_options   = [6, 12, 25, 50, 75, 100];
% SC_options   = 12 * RB_options;


% choice = find(BW_options == params.BW);
% if isempty(choice)
%     choice = find(BW_options*10e6 == params.BW);
%     if isempty(choice)
%         error("No valid BW selected, try again");
%     end
% else
%     params.BW = 10e6*params.BW;
% end
% 
% num_RBs = RB_options(choice);
% fft_size = FFT_options(choice);
% fft_freq = FFT_Sampling(choice);
% RB_BW = 180e3;
% subcarrier_spacing_Hz = 15e3;

% naively assign the center most values as the useful frequencies, and
% throw away edge values (broad assumption on the overhead usage)

% useful_BW = RB_BW * num_RBs; %[1.14, 2.265, 4.515, 9.015, 13.515, 18.015]MHz
% useful_fft_points = floor(useful_BW/subcarrier_spacing_Hz);


Tx_P = 10.^(0.1 * params.Tx_P_dBm) / 1000;
% mimo_no_tx = l.tx_array(1, 1).no_elements / params.no_sectors;
% mimo_no_rx = l.rx_array(1, 1).no_elements;
% no_mimo_links = mimo_no_tx * mimo_no_rx; % Get the number of MIMO sub-channels in the channel matrix
% pwr_mW_perSC_perMIMOtx = (10^(0.1 * params.Tx_P_dBm(1, 1)) / useful_fft_points / mimo_no_tx);


% difference = fft_size - useful_fft_points;
% range_of_interest = floor(difference/2) + 1:fft_size - floor(difference/2) - 1;
% 
% if length(range_of_interest) > useful_fft_points
%     range_of_interest(end) = [];
% 
% elseif length(range_of_interest) < useful_fft_points
%     range_of_interest = [range_of_interest, max(range_of_interest) + 1];
% 
% end

%% Start calculations
tic
% WidebandRSRP; % Calculates RSRP values
% Y_save = zeros([num_RBs, size(rsrp_p0)]);

running_index = ones(l.no_tx, params.no_sectors); % track which AoA correspond
paths = max([c(:).no_path]);



for tx_k = 1:l.no_tx
    for sector = 1:params.no_sectors
        chan = {};
        if params.save_H_D
            H = zeros(l.no_rx, params.Nr, params.Nx*params.Ny, paths, params.total_time*params.fs+1);
            D = zeros(l.no_rx, params.Nr, params.Nx*params.Ny, paths, params.total_time*params.fs+1);
        end
        for rx_k = 1:l.no_rx
            tx_sec_index = (tx_k-1)*params.no_sectors+sector;
            % Get the frequency response values
            % Create virtual beamspace channel
            % H = sum alpha_i .* a(AoA, EoA) * a(AoD, EoD)'
%             no_segments = l.rx_track(rx_k).no_segments;
%             index_start = running_index(tx_k, sector);
%             index_end = index_start + no_segments;
%             if index_end == p(1).no_rx_positions+1
%                 index_end = index_end - 1;
%             end
% 
%             alpha_i = c(rx_k, tx_sec_index).coeff;
%             angles = get_angles(p(tx_k)); % AoD, AoA, EoD, EoA
%             angles = angles(:, index_start:index_end);
% 
%             H = zeros(params.Nr, params.Nx, params.Ny, no_segments);
%             for t=1:no_segments
%                 for path_l=1:c(rx_k, tx_sec_index).no_path
%                     AoDEoD = VanderVect(angles(1, t), params.Nx, params.fc)' * ( VanderVect(angles(3, t), params.Ny, params.fc));
%                     % VanderVect(angles(2, t), params.Nr, params.fc)',
%                     H(:, :, :, t) = reshape(alpha_i(:, :, path_l, t), [params.Nr, params.Nx, params.Ny]) .* einsum(VanderVect(angles(2, t), params.Nr, params.fc)', permute(AoDEoD(:, :, 1), [3, 1, 2]), 'ij,jkl->ikl');
%                 end
%             end
%             if no_segments < params.total_time * params.fs
%                % Need to duplicate the channel since the UE is stationary
%                H = repmat(H, [1, 1, 1, params.fs+1]);
%             end
%             running_index(tx_k, sector) = index_end;
            a = c(rx_k, tx_sec_index).coeff;
            dims = size(a);
            if ~params.save_H_D
                clear a
            else
                b = c(rx_k, tx_sec_index).delay;
                c_paths = c(rx_k, tx_sec_index).no_path;
            end
            freq_response = c(rx_k, tx_sec_index).fr(params.BW, params.n_FFT, [], 2); % passing in 2 instead of 1 means use gpu but with single-precision
            fr = zeros(params.Nr, params.Nx*params.Ny, params.n_FFT, params.total_time*params.fs+1);
            % if stationary, last dim of a needs to be tiled
            if dims(end) == 2
                if params.save_H_D
                    H(rx_k, :, :, 1:c_paths, :) = repmat(a(:, :, :, end), [1, 1, 1, params.fs*params.total_time+1]);
                    D(rx_k, :, :, 1:c_paths, :) = repmat(b(:, :, :, end), [1, 1, 1, params.fs*params.total_time+1]);
                end
                fr(:, :, :, :) = repmat(freq_response(:, :, :, end), [1, 1, 1, params.fs*params.total_time+1]);
            else
                if params.save_H_D
                    H(rx_k, :, :, 1:c_paths, :) = a;
                    D(rx_k, :, :, 1:c_paths, :) = b;
                end
                fr(:, :, :, :) = freq_response;
            end
            save([params.save_folder_r, '/channels/', 'TX_', num2str(tx_k), '_Sector_', num2str(sector), '_UE_', num2str(rx_k), ], '-v7.3', 'fr', '-nocompression'); % '-nocompression' increases file size by ~6x but ~15x faster write speed
        end
        if params.save_H_D
            chan.H = H;
            chan.D = D;
            save([params.save_folder_r, '/channels/', 'TX_', num2str(tx_k), '_Sector_', num2str(sector), '_Channel'], '-v7.3', 'chan');
        end
        
    end
end

