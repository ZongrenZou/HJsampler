close all

x1 = 0:0.1:40;
y1 = 4.*cos(x1)./(x1+2);
x2 = 1:0.2:20;
y2 = x2.^2./x2.^3;


t = tiledlayout(1,1);
ax1 = axes(t);
h1 = plot(ax1,x1,y1,'-r');
hold on
h2 = plot(ax1, x1, y1, "--k");


ax2 = axes(t);
plot(ax2,x2,y2,'-k')
ax2.XAxisLocation = 'bottom';
ax2.YAxisLocation = 'right';
ax2.Color = 'none';
ax1.Box = 'off';
ax2.Box = 'off';