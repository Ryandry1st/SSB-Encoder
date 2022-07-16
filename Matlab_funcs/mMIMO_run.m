addpath(genpath([pwd, '/functions']));
clear all;
close all;
for step=47:49
    seed = step;
    rng(seed);
    clear all;
    close all;
% step = 3; % 2 results in 31 UEs vs 25
    %% Setup


    tstart = tic;
    init_xmimo;
    l = UE_layout(params);
%     for i=1:l.no_rx
%         l.rx_track(i).no_segments = l.rx_track(i).no_snapshots;
%     end

    max_xy = 400;
    %% Channels
    p = l.init_builder;                                       % Create channel builders
    gen_parameters( p );                                      % Generate small-scale fading
    cn = merge( get_channels( p ) ); 

    c = reshape(cn, l.no_rx, l.no_tx, []);
%     for i=1:l.no_rx
%         for j=1:l.no_tx
%             c(i, j) = c(i, j).quantize_delays(params.Ts, params.L, [], [], [], 0);
%         end
%     end
    clear p cn a b fr freq_response % needed for memory room for the frequency response
    used_time = toc(tstart);
    fprintf("Time taken for simulation = %3.1f minutes ", used_time/60);
    l.visualize([],[],0);                                     % Show BS and MT positions on the map
    saveas(gcf, strcat(params.save_folder_r, 'Layout.png'))
    %% Outputs

    [ map,x_coords,y_coords] = l.power_map('3GPP_3D_UMa_NLOS', 'sf',15,-300,300,-300,300,1.5, params.Tx_P_dBm(1, 1));
    % scenario FB_UMa_NLOS, type 'quick', sample distance, x,y min/max, rx
    % height; type can be 'quick', 'sf', 'detailed', 'phase'

    P = 10*log10(sum(abs(map{:}), 3:4));        % Simplified Total received power; Assumed W and converted to dBm

    l.visualize([],[],0);
    hold on;
    imagesc( x_coords, y_coords, P);          % Plot the received power
    axis([-max_xy max_xy -max_xy max_xy])                               % Plot size
    caxis( max(P(:)) + [-50 -5] )
    colmap = colormap;
    colbar = colorbar;
    colbar.Label.String = "Receive Power [dBm]";
    set(gca,'layer','top')                                    % Show grid on top of the map
    hold on;
    set(0,'DefaultFigurePaperSize',[14.5 7.3])                % Adjust paper size for plot                                  % Show BS and MT positions on the map

    saveas(gcf, strcat(params.save_folder_r, 'Rough_RSRP_Map.png'))
    
    close all
    clear map x_coords y_coords P 

    Write_Spectral_Tracks;

    total_time = toc(tstart);
    fprintf("Time taken for simulation+writing = %3.1f minutes ", total_time/60);
end