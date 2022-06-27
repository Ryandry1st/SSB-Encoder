function isOk = CheckLocBounds(params, xloc, yloc)
% Checks the location to verify it is within the bounds of the problem
% only checks against BS at (0,0)
isOk = 1;
point_ang = atan2(yloc, xloc);
point_rad = sqrt(xloc^2 + yloc^2);
if point_ang < params.theta_min || point_ang > params.theta_max
    isOk = 0;
elseif point_rad < params.rmin || point_rad > params.rmax
    isOk = 0;
end

end