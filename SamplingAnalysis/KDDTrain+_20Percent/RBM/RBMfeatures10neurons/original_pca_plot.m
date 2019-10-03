clc, clear all, close all;

%Normal data
normal_data = load('original_normal_data.txt','-ascii');
proj_normal_coeff = pca(normal_data, 'NumComponents', 2);
proj_normal_data = normal_data * proj_normal_coeff;
%Dos data
dos_data = load('original_dos_data.txt','-ascii');
proj_dos_coeff = pca(dos_data, 'NumComponents', 2);
proj_dos_data = dos_data * proj_dos_coeff;
%R2L data
r2l_data = load('original_r2l_data.txt','-ascii');
proj_r2l_coeff = pca(r2l_data, 'NumComponents', 2);
proj_r2l_data = r2l_data * proj_r2l_coeff;
%U2R data
u2r_data = load('original_u2r_data.txt','-ascii');
proj_u2r_coeff = pca(u2r_data, 'NumComponents', 2);
proj_u2r_data = u2r_data * proj_u2r_coeff;
%Probe data
probe_data = load('original_probe_data.txt','-ascii');
proj_probe_coeff = pca(probe_data, 'NumComponents', 2);
proj_probe_data = probe_data * proj_probe_coeff;

scatter(proj_normal_data(1 : end, 1), proj_normal_data(1 : end, 2), '+r'), hold on,
scatter(proj_dos_data(1 : end, 1), proj_dos_data(1 : end, 2), '>c'),
scatter(proj_r2l_data(1 : end, 1), proj_r2l_data(1 : end, 2), 'db'),
scatter(proj_u2r_data(1 : end, 1), proj_u2r_data(1 : end, 2), 'og'),
scatter(proj_probe_data(1 : end, 1), proj_probe_data(1 : end, 2), 'xy'), hold off;
title('Projeção PCA 2D base KDDTrain+\_20Percent Sem RBM');
legend({'Normal','Dos', 'R2L', 'U2R', 'Probe'},'Location','northeast')
xlabel('Primeira componente principal')
ylabel('Segunda componente principal')
saveas(gcf, 'KDDTrain+_20Percent','png')
