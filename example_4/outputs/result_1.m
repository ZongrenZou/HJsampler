% for figures
close all
clear all


%% Figure 1
data = load("result_1.mat");

x = data.x;
mus = data.mus;
sds = data.sds;
y0 = data.y0;
yt = data.yt;
ind = linspace(1, 100, 100);


%% t = 90
dt = 90;

t = tiledlayout(1,1);
ax1 = axes(t);


h1 = fill(ax1, [x, fliplr(x)], ...
    [mus(end-dt, :) + 2*sds(end-dt, :), ...
    fliplr(mus(end-dt, :) - 2*sds(end-dt, :))], ...
    "c", "edgecolor", "none");
hold on
h3 = scatter(ax1, x, yt, 50, "b", "filled");
h3.MarkerFaceAlpha = 0.5;
h2 = plot(ax1, x, mus(end-dt, :), "r:", "linewidth", 2.0);
h4 = scatter(ax1, x, mus(end-dt, :), 50, "+", "k", "linewidth", 2.0);
% h3 = plot(x, yt, "bo", "markersize", 5, "LineWidth", 2.0);
ax1.XColor = "r";
axis square
ax2 = axes(t);

% plot(ax2, ind, mus(end-dt, :), "r:", "linewidth", 2.0);
ax2.XAxisLocation = 'top';
ax2.YAxis.Visible = 'off'; % remove x-axis
xlim([1, 100])

ax2.Color = "none";
ax1.Box = 'off';
ax2.Box = 'off';

axis square

h = legend([h4, h2, h1, h3], ["Mean", "Mean", "2 SD", "$y_1$"], "interpreter", "latex");
set(h, "fontsize", 14, "interpreter", "latex", "Location", "northeast")
% ylabel("$f$", "fontsize", 14, "interpreter", "latex")
% xlabel("$x$", "fontsize", 14, "interpreter", "latex")
xlabel(ax1, "$x$", "fontsize", 10, "interpreter", "latex")
xlabel(ax2, "Index", "fontsize", 10, "interpreter", "latex")
title("$P(Y_{0.9}|Y_T = y_1)$", "fontsize", 12, "interpreter", "latex")
% xlim([0, 1])
ylim([-0.4, 1])
box(ax1, "on")
box(ax2, "on")
legend(ax1, "boxoff")
% legend boxoff
exportgraphics(gca, "fig1_1.png", 'Resolution', 300)


%% t = 50
dt = 50;
figure;
hold on
h1 = fill([x, fliplr(x)], ...
    [mus(end-dt, :) + 2*sds(end-dt, :), ...
    fliplr(mus(end-dt, :) - 2*sds(end-dt, :))], ...
    "c", "edgecolor", "none");
h3 = scatter(x, yt, 50, "b", "filled");
h3.MarkerFaceAlpha = 0.5;
h2 = plot(x, mus(end-dt, :), "r:", "linewidth", 2.0);
% h3 = plot(x, yt, "bo", "markersize", 5, "LineWidth", 2.0);
axis square

% h = legend([h2, h3, h1], ["Reference", "Mean", "2 SD"], "interpreter", "latex");
% set(h, "fontsize", 14, "interpreter", "latex", "Location", "northwest")
% ylabel("$f$", "fontsize", 14, "interpreter", "latex")
xlabel("$x$", "fontsize", 14, "interpreter", "latex")
title("$P(Y_{0.5}|Y_T = y_1)$", "fontsize", 14, "interpreter", "latex")
xlim([0, 1])
ylim([-0.4, 1])
box on
% legend boxoff
exportgraphics(gca, "fig1_2.png", 'Resolution', 300)


%% t = 20
dt = 20;
figure;
hold on
h1 = fill([x, fliplr(x)], ...
    [mus(end-dt, :) + 2*sds(end-dt, :), ...
    fliplr(mus(end-dt, :) - 2*sds(end-dt, :))], ...
    "c", "edgecolor", "none");
h3 = scatter(x, yt, 50, "b", "filled");
h3.MarkerFaceAlpha = 0.5;
h2 = plot(x, mus(end-dt, :), "r:", "linewidth", 2.0);
% h3 = plot(x, yt, "bo", "markersize", 5, "LineWidth", 2.0);
axis square

% h = legend([h2, h3, h1], ["Reference", "Mean", "2 SD"], "interpreter", "latex");
% set(h, "fontsize", 14, "interpreter", "latex", "Location", "northwest")
% ylabel("$f$", "fontsize", 14, "interpreter", "latex")
xlabel("$x$", "fontsize", 14, "interpreter", "latex")
title("$P(Y_{0.2}|Y_T = y_1)$", "fontsize", 14, "interpreter", "latex")
xlim([0, 1])
ylim([-0.4, 1])
box on
% legend boxoff
exportgraphics(gca, "fig1_3.png", 'Resolution', 300)


%% t = 10
dt = 10;
figure;
hold on
h1 = fill([x, fliplr(x)], ...
    [mus(end-dt, :) + 2*sds(end-dt, :), ...
    fliplr(mus(end-dt, :) - 2*sds(end-dt, :))], ...
    "c", "edgecolor", "none");
h3 = scatter(x, yt, 50, "b", "filled");
h3.MarkerFaceAlpha = 0.5;
h2 = plot(x, mus(end-dt, :), "r:", "linewidth", 2.0);
% h3 = plot(x, yt, "bo", "markersize", 5, "LineWidth", 2.0);
axis square

% h = legend([h2, h3, h1], ["Reference", "Mean", "2 SD"], "interpreter", "latex");
% set(h, "fontsize", 14, "interpreter", "latex", "Location", "northwest")
% ylabel("$f$", "fontsize", 14, "interpreter", "latex")
xlabel("$x$", "fontsize", 14, "interpreter", "latex")
title("$P(Y_{0.1}|Y_T = y_1)$", "fontsize", 14, "interpreter", "latex")
xlim([0, 1])
ylim([-0.4, 1])
box on
% legend boxoff
exportgraphics(gca, "fig1_4.png", 'Resolution', 300)


%% t = 0
dt = 0;
figure;
hold on
h1 = fill([x, fliplr(x)], ...
    [mus(end-dt, :) + 2*sds(end-dt, :), ...
    fliplr(mus(end-dt, :) - 2*sds(end-dt, :))], ...
    "c", "edgecolor", "none");
h3 = scatter(x, yt, 50, "b", "filled");
h3.MarkerFaceAlpha = 0.5;
h4 = plot(x, y0, "k-", "linewidth", 2.0);
h2 = plot(x, mus(end-dt, :), "r:", "linewidth", 2.0);
% h3 = plot(x, yt, "bo", "markersize", 5, "LineWidth", 2.0);
axis square

% h = legend([h2, h3, h1], ["Reference", "Mean", "2 SD"], "interpreter", "latex");
% set(h, "fontsize", 14, "interpreter", "latex", "Location", "northwest")
% ylabel("$f$", "fontsize", 14, "interpreter", "latex")
xlabel("$x$", "fontsize", 14, "interpreter", "latex")
title("$P(Y_{0}|Y_T = y_1)$", "fontsize", 14, "interpreter", "latex")
xlim([0, 1])
ylim([-0.4, 1])
box on
% legend boxoff
exportgraphics(gca, "fig1_5.png", 'Resolution', 300)
