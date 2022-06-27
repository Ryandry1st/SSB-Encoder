classdef Road
    %ROAD Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        start
        stop
        isHighway
        heading
    end
    
    methods
        function obj = Road(start, stop, highway)
            %ROAD Construct an instance of this class
            if nargin < 2
                obj.start = [0, 0];
                obj.stop = [0, 0];
            else
                obj.start = start;
                obj.stop = stop;
            end
            if nargin < 3
                obj.isHighway = 0;
            else
                obj.isHighway = highway;
            end
            obj.heading = angle(obj.stop(1) + 1j*obj.stop(2) - obj.start(1) - 1j*obj.start(2));
        end
        
        function length = get_length(obj)
            %METHOD1 Get the length of the road
            length = norm(obj.stop - obj.start);
        end
        
        function [locs, vels] = assign(obj, N)
           % get a location and velocity assignment for the road
            if obj.isHighway
              speeds = 30 + 5*randn(1, N);
            else
               speeds = 25 + 5*randn(1, N);
            end
            x_vel = speeds .* cos(obj.heading);
            y_vel = speeds .* sin(obj.heading);
            vels = zeros(N, 3);
            vels(:, 1) = x_vel;
            vels(:, 2) = y_vel;
            
            locs = 1.5*ones(N, 3);
            locs(:, 1) = (obj.stop(1) - obj.start(1)) * rand(N, 1) + obj.start(1);  %(x2-x1)*(n random number)+x1
            if obj.start(1) == obj.stop(1)
                locs(:, 2) = (obj.stop(2) - obj.start(2)) * rand(N, 1) + obj.start(2);
            else
                locs(:, 2) = (obj.start(2)-obj.stop(2)) / (obj.start(1) - obj.stop(1)) .* (locs(:, 1) - obj.start(1)) + obj.start(2); % (y1-y2)/(x1-x2)*(x-x1) + y1
            end
        end
        
        function plot(obj)
            fig = gcf();
            plot([obj.start(1), obj.stop(1)], [obj.start(2), obj.stop(2)]);
        end
    end
end

