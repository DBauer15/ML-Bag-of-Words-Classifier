%% Setup
close all
clear
clc

folder = "./ass5_data/train";
num_clusters = 50;

%% Build vocabulary
C = BuildVocabulary(folder, num_clusters);
fprintf("Successfully built vocabulary\n");

%% Build feature representation
[training, group] = BuildKNN(folder,C);
fprintf("Successfully built feature representation (KNN)\n");

%% Classify images
conf_matrix = ClassifyImages(folder,C,training,group);
disp(conf_matrix);
fprintf("Successfully classified images\n");

%% Display results
disp(conf_matrix);
conf_sum = sum(sum(conf_matrix));
conf_correct = trace(conf_matrix);
fprintf("Evaluation: %d/%d correct (%f%%)\n", conf_correct, conf_sum, (conf_correct/conf_sum)*100);