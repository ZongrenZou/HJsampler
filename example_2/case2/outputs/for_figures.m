% for figures
close all
clear all


%% Figure 1
data = load("sgm4.mat");

t = data.t;
mu = data.mu;
sd = data.sd;
sols = data.sols;
mu = flip(mu);
sd = flip(sd);


figure;
hold on
h1 = fill([t, fliplr(t)], ...
    [mu(:, 1)' + 2*sd(:, 1)', ...
    fliplr(mu(:, 1)' - 2*sd(:, 1)')], ...
    "c", "edgecolor", "none", "FaceAlpha", 0.8);
h2 = fill([t, fliplr(t)], ...
    [mu(:, 2)' + 2*sd(:, 2)', ...
    fliplr(mu(:, 2)' - 2*sd(:, 2)')], ...
    "r", "edgecolor", "none", "FaceAlpha", 0.3);
h3 = plot(t, sols(:, 1), "k-", "linewidth", 2.0);
h4 = plot(t, sols(:, 2), "b-", "linewidth", 2.0);
h5 = plot(t, mu(:, 1)', "r--", "linewidth", 2.0);
h6 = plot(t, mu(:, 2)', "m--", "linewidth", 2.0);

h = legend([h3, h4, h5, h1, h6, h2], ...
    ["Reference of $y_1$", "Reference of $y_2$", ...
    "Mean of inferred $y_1$", "2 SD of inferred $y_1$", ...
    "Mean of inferred $y_2$", "2 SD of inferred $y_2$"], ...
    "interpreter", "latex");
set(h, "fontsize", 14, "interpreter", "latex", "Location", "northeast")
% ylabel("$f$", "fontsize", 14, "interpreter", "latex")
xlabel("$t$", "fontsize", 14, "interpreter", "latex")
title("$\epsilon=1\times10^{-5}$", "fontsize", 14, "interpreter", "latex")
xlim([0, 5])
ylim([-0.2, 0.4])
box on
legend boxoff
exportgraphics(gca, "fig1.png", 'Resolution', 300)


%% Figure 2
data = load("sgm1.mat");

t = data.t;
mu = data.mu;
sd = data.sd;
sols = data.sols;
mu = flip(mu);
sd = flip(sd);


figure;
hold on
h1 = fill([t, fliplr(t)], ...
    [mu(:, 1)' + 2*sd(:, 1)', ...
    fliplr(mu(:, 1)' - 2*sd(:, 1)')], ...
    "c", "edgecolor", "none", "FaceAlpha", 0.8);
h2 = fill([t, fliplr(t)], ...
    [mu(:, 2)' + 2*sd(:, 2)', ...
    fliplr(mu(:, 2)' - 2*sd(:, 2)')], ...
    "r", "edgecolor", "none", "FaceAlpha", 0.3);
h3 = plot(t, sols(:, 1), "k-", "linewidth", 2.0);
h4 = plot(t, sols(:, 2), "b-", "linewidth", 2.0);
h5 = plot(t, mu(:, 1)', "r--", "linewidth", 2.0);
h6 = plot(t, mu(:, 2)', "m--", "linewidth", 2.0);

h = legend([h3, h4, h5, h1, h6, h2], ...
    ["Reference of $y_1$", "Reference of $y_2$", ...
    "Mean of inferred $y_1$", "2 SD of inferred $y_1$", ...
    "Mean of inferred $y_2$", "2 SD of inferred $y_2$"], ...
    "interpreter", "latex");
set(h, "fontsize", 14, "interpreter", "latex", "Location", "northeast")
% ylabel("$f$", "fontsize", 14, "interpreter", "latex")
xlabel("$t$", "fontsize", 14, "interpreter", "latex")
title("$\epsilon=1\times10^{-4}$", "fontsize", 14, "interpreter", "latex")
xlim([0, 5])
ylim([-0.2, 0.4])
box on
legend boxoff
exportgraphics(gca, "fig2.png", 'Resolution', 300)




%% Figure 3
data = load("sgm3.mat");

t = data.t;
mu = data.mu;
sd = data.sd;
sols = data.sols;
mu = flip(mu);
sd = flip(sd);


figure;
hold on
h1 = fill([t, fliplr(t)], ...
    [mu(:, 1)' + 2*sd(:, 1)', ...
    fliplr(mu(:, 1)' - 2*sd(:, 1)')], ...
    "c", "edgecolor", "none", "FaceAlpha", 0.8);
h2 = fill([t, fliplr(t)], ...
    [mu(:, 2)' + 2*sd(:, 2)', ...
    fliplr(mu(:, 2)' - 2*sd(:, 2)')], ...
    "r", "edgecolor", "none", "FaceAlpha", 0.3);
h3 = plot(t, sols(:, 1), "k-", "linewidth", 2.0);
h4 = plot(t, sols(:, 2), "b-", "linewidth", 2.0);
h5 = plot(t, mu(:, 1)', "r--", "linewidth", 2.0);
h6 = plot(t, mu(:, 2)', "m--", "linewidth", 2.0);

h = legend([h3, h4, h5, h1, h6, h2], ...
    ["Reference of $y_1$", "Reference of $y_2$", ...
    "Mean of inferred $y_1$", "2 SD of inferred $y_1$", ...
    "Mean of inferred $y_2$", "2 SD of inferred $y_2$"], ...
    "interpreter", "latex");
set(h, "fontsize", 14, "interpreter", "latex", "Location", "northeast")
% ylabel("$f$", "fontsize", 14, "interpreter", "latex")
xlabel("$t$", "fontsize", 14, "interpreter", "latex")
title("$\epsilon=1\times10^{-3}$", "fontsize", 14, "interpreter", "latex")
xlim([0, 5])
ylim([-0.2, 0.4])
box on
legend boxoff
exportgraphics(gca, "fig3.png", 'Resolution', 300)


%% Figure 0
data = load("sgm2.mat");

t = data.t;
sols = data.sols;
z_sols = data.z_sols;
z_sols = flip(z_sols);


figure;
hold on
h3 = plot(t, sols(:, 1), "k-", "linewidth", 2.0);
h4 = plot(t, sols(:, 2), "b-", "linewidth", 2.0);
h5 = plot(t, z_sols(:, 1)', "r--", "linewidth", 2.0);
h6 = plot(t, z_sols(:, 2)', "m--", "linewidth", 2.0);

h = legend([h3, h4, h5, h6], ...
    ["Reference of $y_1$", "Reference of $y_2$", ...
    "Inferred $y_1$", ...
    "Inferred $y_2$"], ...
    "interpreter", "latex");
set(h, "fontsize", 14, "interpreter", "latex", "Location", "northeast")
% ylabel("$f$", "fontsize", 14, "interpreter", "latex")
xlabel("$t$", "fontsize", 14, "interpreter", "latex")
title("Consequence of model misspecification", "fontsize", 14, "interpreter", "latex")
xlim([0, 5])
ylim([-0.2, 0.4])
box on
legend boxoff
exportgraphics(gca, "fig0.png", 'Resolution', 300)

