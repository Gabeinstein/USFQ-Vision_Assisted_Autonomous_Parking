function DF = nextMove(x0,y0,theta,vx,wz,ICC_x,ICC_y,t)
    if wz == 0
        theta_n = theta;
        x_n = x0 + vx*t*cos(theta);
        y_n = y0 + vx*t*sin(theta);
    else
        dtheta = wz*t;
        x_n = cos(dtheta)*(x0 - ICC_x) - sin(dtheta)*(y0 - ICC_y) + ICC_x;
        y_n = sin(dtheta)*(x0 - ICC_x) + cos(dtheta)*(y0 - ICC_y) + ICC_y;
        theta_n = theta + dtheta;
    end
    DF = [x_n y_n theta_n];
end