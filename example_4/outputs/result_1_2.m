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
h2 = plot(ax1, x, mus(end-dt, :), "r:x", "linewidth", 2.0, "MarkerSize", 7);
axis(ax1, "square")

ax2 = axes(t);
% plot(ax2, ind-1, mus(end-dt, :), "w:x", "linewidth", 2.0, "MarkerSize", 10);
ax2.XAxisLocation = 'top';
ax2.YAxis.Visible = 'off'; % remove x-axis
ax2.Color = "none";
axis(ax2, "square")
xlim(ax2, [0, 100])
xticks(ax2, [0 20 40 60 80 100])
xticklabels(ax2, {'1','20','40', '60', '80', '100'})

% ax1.Box = 'off';
% ax2.Box = 'off';


h = legend([h2, h1, h3], ["Mean", "2 SD", "$y_1$"], "interpreter", "latex");
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

exportgraphics(gcf, "fig1_1.png", 'Resolution', 300)
close


%% t = 50
dt = 50;

t = tiledlayout(1,1);

ax1 = axes(t);
h1 = fill(ax1, [x, fliplr(x)], ...
    [mus(end-dt, :) + 2*sds(end-dt, :), ...
    fliplr(mus(end-dt, :) - 2*sds(end-dt, :))], ...
    "c", "edgecolor", "none");
hold on
h3 = scatter(ax1, x, yt, 50, "b", "filled");
h3.MarkerFaceAlpha = 0.5;
h2 = plot(ax1, x, mus(end-dt, :), "r:x", "linewidth", 2.0, "MarkerSize", 7);
axis(ax1, "square")

ax2 = axes(t);
% plot(ax2, ind-1, mus(end-dt, :), "w:x", "linewidth", 2.0, "MarkerSize", 10);
ax2.XAxisLocation = 'top';
ax2.YAxis.Visible = 'off'; % remove x-axis
ax2.Color = "none";
axis(ax2, "square")
xlim(ax2, [0, 100])
xticks(ax2, [0 20 40 60 80 100])
xticklabels(ax2, {'1','20','40', '60', '80', '100'})

xlabel(ax1, "$x$", "fontsize", 10, "interpreter", "latex")
xlabel(ax2, "Index", "fontsize", 10, "interpreter", "latex")
title("$P(Y_{0.5}|Y_T = y_1)$", "fontsize", 12, "interpreter", "latex")
% xlim([0, 1])
ylim([-0.4, 1])
box(ax1, "on")
box(ax2, "on")

exportgraphics(gcf, "fig1_2.png", 'Resolution', 300)
close



%% t = 20
dt = 20;
t = tiledlayout(1,1);

ax1 = axes(t);
h1 = fill(ax1, [x, fliplr(x)], ...
    [mus(end-dt, :) + 2*sds(end-dt, :), ...
    fliplr(mus(end-dt, :) - 2*sds(end-dt, :))], ...
    "c", "edgecolor", "none");
hold on
h3 = scatter(ax1, x, yt, 50, "b", "filled");
h3.MarkerFaceAlpha = 0.5;
h2 = plot(ax1, x, mus(end-dt, :), "r:x", "linewidth", 2.0, "MarkerSize", 7);
axis(ax1, "square")

ax2 = axes(t);
% plot(ax2, ind-1, mus(end-dt, :), "w:x", "linewidth", 2.0, "MarkerSize", 10);
ax2.XAxisLocation = 'top';
ax2.YAxis.Visible = 'off'; % remove x-axis
ax2.Color = "none";
axis(ax2, "square")
xlim(ax2, [0, 100])
xticks(ax2, [0 20 40 60 80 100])
xticklabels(ax2, {'1','20','40', '60', '80', '100'})

xlabel(ax1, "$x$", "fontsize", 10, "interpreter", "latex")
xlabel(ax2, "Index", "fontsize", 10, "interpreter", "latex")
title("$P(Y_{0.2}|Y_T = y_1)$", "fontsize", 12, "interpreter", "latex")
% xlim([0, 1])
ylim([-0.4, 1])
box(ax1, "on")
box(ax2, "on")

exportgraphics(gcf, "fig1_3.png", 'Resolution', 300)
close


%% t = 10
dt = 10;
t = tiledlayout(1,1);

ax1 = axes(t);
h1 = fill(ax1, [x, fliplr(x)], ...
    [mus(end-dt, :) + 2*sds(end-dt, :), ...
    fliplr(mus(end-dt, :) - 2*sds(end-dt, :))], ...
    "c", "edgecolor", "none");
hold on
h3 = scatter(ax1, x, yt, 50, "b", "filled");
h3.MarkerFaceAlpha = 0.5;
h2 = plot(ax1, x, mus(end-dt, :), "r:x", "linewidth", 2.0, "MarkerSize", 7);
axis(ax1, "square")

ax2 = axes(t);
% plot(ax2, ind-1, mus(end-dt, :), "w:x", "linewidth", 2.0, "MarkerSize", 10);
ax2.XAxisLocation = 'top';
ax2.YAxis.Visible = 'off'; % remove x-axis
ax2.Color = "none";
axis(ax2, "square")
xlim(ax2, [0, 100])
xticks(ax2, [0 20 40 60 80 100])
xticklabels(ax2, {'1','20','40', '60', '80', '100'})

xlabel(ax1, "$x$", "fontsize", 10, "interpreter", "latex")
xlabel(ax2, "Index", "fontsize", 10, "interpreter", "latex")
title("$P(Y_{0.1}|Y_T = y_1)$", "fontsize", 12, "interpreter", "latex")
% xlim([0, 1])
ylim([-0.4, 1])
box(ax1, "on")
box(ax2, "on")

exportgraphics(gcf, "fig1_4.png", 'Resolution', 300)
close

%% t = 0
dt = 0;

t = tiledlayout(1,1);

ax1 = axes(t);
h1 = fill(ax1, [x, fliplr(x)], ...
    [mus(end-dt, :) + 2*sds(end-dt, :), ...
    fliplr(mus(end-dt, :) - 2*sds(end-dt, :))], ...
    "c", "edgecolor", "none");
hold on
h4 = plot(ax1, x, y0, "k-x", "linewidth", 2.0, "MarkerSize", 7);
h3 = scatter(ax1, x, yt, 50, "b", "filled");
h3.MarkerFaceAlpha = 0.5;
h2 = plot(ax1, x, mus(end-dt, :), "r:x", "linewidth", 2.0, "MarkerSize", 7);
axis(ax1, "square")

ax2 = axes(t);
% plot(ax2, ind-1, mus(end-dt, :), "w:x", "linewidth", 2.0, "MarkerSize", 10);
ax2.XAxisLocation = 'top';
ax2.YAxis.Visible = 'off'; % remove x-axis
ax2.Color = "none";
axis(ax2, "square")
xlim(ax2, [0, 100])
xticks(ax2, [0 20 40 60 80 100])
xticklabels(ax2, {'1','20','40', '60', '80', '100'})


xlabel(ax1, "$x$", "fontsize", 10, "interpreter", "latex")
xlabel(ax2, "Index", "fontsize", 10, "interpreter", "latex")
title("$P(Y_{0}|Y_T = y_1)$", "fontsize", 12, "interpreter", "latex")
% xlim([0, 1])
ylim([-0.4, 1])
box(ax1, "on")
box(ax2, "on")

exportgraphics(gcf, "fig1_5.png", 'Resolution', 300)

