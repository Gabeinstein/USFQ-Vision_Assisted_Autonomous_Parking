clear variables;
close all;
clc;

%%% Path Critical Points
puntos = [0,0; 0.41,0; 0.18, -0.06; -0.025, -0.25];

%%% Transbot Drawing
transbot = [-0.12,-0.15; 0.12,-0.15; 0.12, 0.15; -0.12, 0.15]';
phi = pi/2;
Rot = [cos(phi) -sin(phi); sin(phi) cos(phi)];
P = Rot*transbot;

v = 0.05;   %m/s
w = 0.05;   %rad/s
x = 0;
y = 0;
theta = 0;

data = [inf,0.2,1,0,0; 0.5308,0.9846,-1,0.4255,-0.5305; 0,-0.9846,-1,-0.025,-0.2499];

%%%0,-0.9846,-1,-0.2675,-0.36375

%%%[radio de curvatura, angulo, sentido, ICCx, ICCy] mov circular
%%%[radio de curvatura (inf) , distancia lineal, sentido, 0, 0]

trayectoria = zeros(length(data(:,1))*100,2);
velocity_profile = zeros(length(data(:,1))*100,1);
angular_profile = zeros(length(data(:,1))*100,1);
time_vector = zeros(length(data(:,1)*100),1);
iterator = 1;

total_time = 0;

for k = 1:length(data(:,1))
    if data(k,1) == inf
        vx = data(k,3)*v;
        wz = 0;
        tf = abs(data(k,2)/v)
        total_time = total_time + tf;
        t1 = linspace(0,tf);
        ICC_x = 0;
        ICC_y = 0;
    else
        if data(k,1) ~= 0
            vx = data(k,3)*v;
            wz = abs(vx/data(k,1));
        else
            vx = 0;
            wz = data(k,3)*w;
        end
        
        tf = abs(data(k,2)/wz);
        total_time = total_time + tf;
        t1 = linspace(0,tf);
        ICC_x = data(k,4);
        ICC_y = data(k,5);
    end
    for i = 1:length(t1)
        Pos = nextMove(x,y,theta,vx,wz,ICC_x,ICC_y,t1(i));
        Pos_A_B_orig = [Pos(1); Pos(2)];
        Rot_AB = [cos(Pos(3)) -sin(Pos(3)); sin(Pos(3)) cos(Pos(3))];
        P_B = P;
        P_A = Rot_AB*P_B + Pos_A_B_orig;
        fill(P_A(1,:), P_A(2,:), 'g');
        hold on;
        scatter(puntos(:,1),puntos(:,2))
        axis equal;
        axis([-1,1,-1,1])
        grid on;
        trayectoria(iterator,:) = [Pos(1), Pos(2)];
        velocity_profile(iterator) = vx;
        angular_profile(iterator) = wz;
        iterator = iterator + 1;
        hold off;
        pause(0.001);
    end
    x = Pos(1);
    y = Pos(2);
    theta = Pos(3);
end
tiempo = linspace(0,total_time,length(data(:,1))*100);
time_vector = tiempo';
figure(2)
plot(trayectoria(:,1),trayectoria(:,2),'--')
axis equal;
axis([-0.5,0.5,-0.5,0.5])
grid on;
xlabel("Distancia x [m]");
ylabel("Distancia y [m]");
title("Trayectoria")

figure(3)
plot(tiempo,velocity_profile)
grid on;
xlabel("Tiempo [s]");
ylabel("Velocidad Lineal X [m/s]");
axis([-5,45,-0.1,0.1])
title("Perfil de velocidad lineal del eje X")

figure(4)
plot(tiempo,angular_profile)
grid on;
axis([-5,45,-0.1,0.1])
xlabel("Tiempo [s]");
ylabel("Velocidad Angular Z [rad/s]");
title("Perfil de velocidad angular del eje Z")