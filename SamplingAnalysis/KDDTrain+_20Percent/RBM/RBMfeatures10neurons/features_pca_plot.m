clc, clear all, close all;

%Normal features
normal_feats = load('features_normal_data.txt','-ascii');
proj_normal_features_coeff = pca(normal_feats, 'NumComponents', 2);
proj_normal_feats = normal_feats * proj_normal_features_coeff;
%Dos features
dos_feats = load('features_dos_data.txt','-ascii');
proj_dos_features_coeff = pca(dos_feats, 'NumComponents', 2);
proj_dos_feats = dos_feats * proj_dos_features_coeff;
%R2L features
r2l_feats = load('features_r2l_data.txt','-ascii');
proj_r2l_features_coeff = pca(r2l_feats, 'NumComponents', 2);
proj_r2l_feats = r2l_feats * proj_r2l_features_coeff;
%U2R features
u2r_feats = load('features_u2r_data.txt','-ascii');
proj_u2r_features_coeff = pca(u2r_feats, 'NumComponents', 2);
proj_u2r_feats = u2r_feats * proj_u2r_features_coeff;
%Probe features
probe_feats = load('features_probe_data.txt','-ascii');
proj_probe_features_coeff = pca(probe_feats, 'NumComponents', 2);
proj_probe_feats = probe_feats * proj_probe_features_coeff;

scatter(proj_normal_feats(1 : end, 1), proj_normal_feats(1 : end, 2), '+r'), hold on,
scatter(proj_dos_feats(1 : end, 1), proj_dos_feats(1 : end, 2), '>c'),
scatter(proj_r2l_feats(1 : end, 1), proj_r2l_feats(1 : end, 2), 'db'),
scatter(proj_u2r_feats(1 : end, 1), proj_u2r_feats(1 : end, 2), 'og'),
scatter(proj_probe_feats(1 : end, 1), proj_probe_feats(1 : end, 2), 'xy'), hold off;
title('Projeção PCA 2D base KDDTrain+\_20Percent com features da RBM');
legend({'Normal','Dos', 'R2L', 'U2R', 'Probe'},'Location','northeast')
xlabel('Primeira componente principal')
ylabel('Segunda componente principal')
saveas(gcf, 'KDDTrain+_20Percent using RBM features','png')