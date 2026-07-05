% for figures
close all
clear all


%% Figure 1
data = load("sgm1.mat");

t = data.t;
mu = data.mu;
sd = data.sd;
sols = data.sols;


figure;
hold on
h1 = fill([t, fliplr(t)], ...
    [mu' + 2*sd', ...
    fliplr(mu' - 2*sd')], ...
    "c", "edgecolor", "none");
h2 = plot(t, sols, "k-", "linewidth", 2.0);
h3 = plot(t, mu, "r--", "linewidth", 2.0);

h = legend([h2, h3, h1], ["Reference", "Mean", "2 SD"], "interpreter", "latex");
set(h, "fontsize", 14, "interpreter", "latex", "Location", "northwest")
% ylabel("$f$", "fontsize", 14, "interpreter", "latex")
xlabel("$t$", "fontsize", 14, "interpreter", "latex")
title("$\epsilon=1\times10^{-3}$", "fontsize", 14, "interpreter", "latex")
xlim([0, 1])
ylim([0.0, 0.5])
box on
legend boxoff
exportgraphics(gca, "fig1.png", 'Resolution', 300)


%% Figure 2
data = load("sgm2.mat");

t = data.t;
mu = data.mu;
sd = data.sd;
sols = data.sols;


figure;
hold on
h1 = fill([t, fliplr(t)], ...
    [mu' + 2*sd', ...
    fliplr(mu' - 2*sd')], ...
    "c", "edgecolor", "none");
h2 = plot(t, sols, "k-", "linewidth", 2.0);
h3 = plot(t, mu, "r--", "linewidth", 2.0);

h = legend([h2, h3, h1], ["Reference", "Mean", "2 SD"], "interpreter", "latex");
set(h, "fontsize", 14, "interpreter", "latex", "Location", "northwest")
% ylabel("$f$", "fontsize", 14, "interpreter", "latex")
xlabel("$t$", "fontsize", 14, "interpreter", "latex")
title("$\epsilon=5\times10^{-3}$", "fontsize", 14, "interpreter", "latex")
xlim([0, 1])
ylim([0.0, 0.5])
box on
legend boxoff
exportgraphics(gca, "fig2.png", 'Resolution', 300)




%% Figure 3
data = load("sgm3.mat");

t = data.t;
mu = data.mu;
sd = data.sd;
sols = data.sols;


figure;
hold on
h1 = fill([t, fliplr(t)], ...
    [mu' + 2*sd', ...
    fliplr(mu' - 2*sd')], ...
    "c", "edgecolor", "none");
h2 = plot(t, sols, "k-", "linewidth", 2.0);
h3 = plot(t, mu, "r--", "linewidth", 2.0);

h = legend([h2, h3, h1], ["Reference", "Mean", "2 SD"], "interpreter", "latex");
set(h, "fontsize", 14, "interpreter", "latex", "Location", "northwest")
% ylabel("$f$", "fontsize", 14, "interpreter", "latex")
xlabel("$t$", "fontsize", 14, "interpreter", "latex")
title("$\epsilon=1\times10^{-2}$", "fontsize", 14, "interpreter", "latex")
xlim([0, 1])
ylim([0.0, 0.5])
box on
legend boxoff
exportgraphics(gca, "fig3.png", 'Resolution', 300)


%% Figure 0
data = load("sgm2.mat");

t = data.t;
sols = data.sols;
z_sols = flip(data.z_sols);


figure;
hold on
h1 = plot(t, sols, "k-", "linewidth", 2.0);
h2 = plot(t, z_sols, "r--", "linewidth", 2.0);

h = legend([h1, h2], ["Reference", "Inference"], "interpreter", "latex");
set(h, "fontsize", 14, "interpreter", "latex", "Location", "northeast")
% ylabel("$f$", "fontsize", 14, "interpreter", "latex")
xlabel("$t$", "fontsize", 14, "interpreter", "latex")
title("Consequence of model misspecification", "fontsize", 14, "interpreter", "latex")
xlim([0, 1])
ylim([0.0, 0.5])
box on
legend boxoff
exportgraphics(gca, "fig0.png", 'Resolution', 300)

