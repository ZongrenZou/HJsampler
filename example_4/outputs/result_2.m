% for figures
close all
clear all


%% Figure 1
data = load("result_2.mat");

x = data.x;
mus = data.mus;
sds = data.sds;
y0 = data.y0;
yt = data.yt;


%% t = 90
dt = 90;
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
title("$P(Y_{0.9}|Y_T = y_2)$", "fontsize", 14, "interpreter", "latex")
xlim([0, 1])
ylim([-0.4, 1])
box on
% legend boxoff
exportgraphics(gca, "fig2_1.png", 'Resolution', 300)


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
title("$P(Y_{0.5}|Y_T = y_2)$", "fontsize", 14, "interpreter", "latex")
xlim([0, 1])
ylim([-0.4, 1])
box on
% legend boxoff
exportgraphics(gca, "fig2_2.png", 'Resolution', 300)


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
title("$P(Y_{0.2}|Y_T = y_2)$", "fontsize", 14, "interpreter", "latex")
xlim([0, 1])
ylim([-0.4, 1])
box on
% legend boxoff
exportgraphics(gca, "fig2_3.png", 'Resolution', 300)


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
title("$P(Y_{0.1}|Y_T = y_2)$", "fontsize", 14, "interpreter", "latex")
xlim([0, 1])
ylim([-0.4, 1])
box on
% legend boxoff
exportgraphics(gca, "fig2_4.png", 'Resolution', 300)


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
title("$P(Y_{0}|Y_T = y_2)$", "fontsize", 14, "interpreter", "latex")
xlim([0, 1])
ylim([-0.4, 1])
box on
% legend boxoff
exportgraphics(gca, "fig2_5.png", 'Resolution', 300)
