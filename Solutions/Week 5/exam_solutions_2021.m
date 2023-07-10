%% QUESTION 1
disp(['Question 1: ', num2str(12*(5.12 + 7.16))])


%% QUESTION 3
L = [28.9, 19.9, 13.7, 9.8, 7.0];
t = [1, 2, 3, 4, 5];
[~, i] = max(L.*t);
disp(['Question 3: ', num2str(sqrt(2*t(i)))])


%% QUESTION 4
d = dlmread('../data/distances.txt');
lab = dlmread('../data/labels.txt');
[~, i] = min(d);
lab_c2 = lab(i==2);
disp(['Question 4: ', num2str(sum(lab_c2==1)/numel(lab_c2))])
 
 
%% QUESTION 6 *
s = 1.7;
t = [36, 13]';
theta = 140/180*pi;
R = [cos(theta), -sin(theta); sin(theta), cos(theta)];

p = dlmread('../data/points_p.txt');
q = dlmread('../data/points_q.txt');
p_ = R'*(q-t)/s;

% visualization
figure
subplot(1,2,1), hold on
plot(p(1,:), p(2,:), 'r.', q(1,:), q(2,:), 'b.')
plot([p(1,:); q(1,:)], [p(2,:); q(2,:)] , 'k', 'LineWidth', 0.5)
axis('equal')
subplot(1,2,2), hold on
plot(p(1,:), p(2,:), 'r.', p_(1,:), p_(2,:), 'b.')
plot([p(1,:); p_(1,:)], [p(2,:); p_(2,:)] , 'k', 'LineWidth', 0.5)
axis('equal')

d = sqrt(sum((p - p_).^2));
disp(['Question 6: ', num2str(sum(d>2))])
 
 
%% QUESTION 9
disp(['Question 9: ', num2str(sqrt((209-147)^2+(158-215)^2))])
 
 
%% QUESTION 10
disp(['Question 10: ', num2str((52-30)^2-(52-20)^2+(3-1)*125)])
 

%% QUESTION 11 *
% Solution based on:
% https://lab.compute.dtu.dk/aia02506/weekly-solutions/-/blob/master/Week05/week05_mcode/quiz_solutiton.m
I = double(imread('../data/circly.png'));
mu = [70, 120, 180];
beta = 100;
 
U = (I - reshape(mu,[1,1,3])).^2;
[~, S0] = min(U, [], 3);
 
prior = beta * (sum(sum(S0(2:end,:)~=S0(1:end-1,:))) + sum(sum(S0(:,2:end)~=S0(:,1:end-1))));
likelihood = sum(sum((mu(S0)-I).^2));
 
figure
subplot(1,2,1), imagesc(I), colormap(gca, 'gray'), axis image
subplot(1,2,2), imagesc(S0), colormap(gca, 'jet'), axis image
disp(['Question 11: ', num2str(prior+likelihood)])

 
%% QUESTION 12 *
% Solution based on:
% https://lab.compute.dtu.dk/aia02506/weekly-solutions/-/blob/master/Week05/week05_mcode/DTU_binary.m
% Requires GraophCut code used in the exercises, which can be found here:
% https://lab.compute.dtu.dk/aia02506/weekly-solutions/-/tree/master/Week05/week05_mcode/GraphCut
addpath('GraphCut')
I = double(imread('C:\Users\hujo8\OneDrive\Advanced image analysis\Old exam/bony.png'));
mu = [130, 190];
beta  = 3000;
 
% Graph with internal and external edges
U = (I(:) - mu).^2;
indices = reshape(1:numel(I), size(I));
edge_x = indices(1:end-1,:);
edge_y = indices(:,1:end-1);
edge_n = [edge_x(:), edge_x(:)+1, beta*ones(numel(edge_x),2);
    edge_y(:),edge_y(:)+size(I,1), beta*ones(numel(edge_y),2)];
edge_t = [(1:numel(I))', U]; 
S = false(size(I));
S(GraphCutMex(numel(I),edge_t,edge_n)) = true;
%S_old = S;
 
% Visualization
figure
imagesc(S)
disp(['Question 12: ', num2str(sum(S(:)))])
 
 
%% QUESTION 14 *
% Solution based on
% https://lab.compute.dtu.dk/aia02506/weekly-solutions/-/blob/master/Week06/week06_mcode/quiz_solution.m
% and
% https://lab.compute.dtu.dk/aia02506/weekly-solutions/-/blob/master/Week06/week06_mcode/plusplus_segmentation.m
I = double(imread('../data/frame.png'))/255;
mask = false(size(I));
mask(size(I,1)/2-39:size(I,1)/2+40, size(I,2)/2-39:size(I,2)/2+40) = 1;
 
m_in = mean(I(mask));
m_out = mean(I(~mask));
 
p = [size(I,1)/2+40.5, size(I,2)/2-39.5];
I_p = I(round(p(1)),round(p(2)));
f_ext = (m_in - m_out) * (2*I_p - m_in - m_out);
 
% Visualization
figure, hold on
rgb = 0.5*(cat(3,I,I,I) + cat(3,mask,mask,0*mask));
imagesc(rgb), axis image ij
plot(p(2), p(1), 'co', 'MarkerSize', 10)
disp(['Question 14: ', num2str(f_ext)])
 
 
%% QUESTION 15
S = [0.1, 2.9; 1.2, 5.4; 3.3, 7.1; 3.5, 0.2; 1.4, 1.1];
P = S(1,:) + 0.05*(S(2,:)+S(end,:)-2*S(1,:)) + 0.1*(-S(3,:)-S(end-1,:)+4*S(2,:)+4*S(end,:)-6*S(1,:));
% Visualization
figure, hold on
plot(S(:,1), S(:,2), 'b-o', S([1,end],1), S([1,end],2), 'b:')
plot(S(1,1), S(1,2), 'ro', P(1), P(2), 'co')
disp(['Question 15: ', num2str(P)])
 
 
%% QUESTION 17
I = dlmread('../data/layers.txt');
bright = sum(20-I, 2);
dark = sum(I, 2);
cost = cumsum(bright) + cumsum([dark(2:end); 0], 'reverse');
disp(['Question 17: ', num2str(min(cost))])
 
 
%% QUESTION 18 
yhat = [0.5, 8.2, 6.9, -0.1, 0.3];
y = exp(yhat);
y = y/sum(y);
disp(['Question 18: ', num2str(-log(y(2)))])
 
 
%% QUESTION 19
W1 = [0.2, -1.3; -0.3, 1.8; -1.7, 1.6];
W2 = [-1.4, 1.5, -0.5, 0.9; 0.2, 1.2, -0.9, 1.7];
hp  = W1 * [1; 2.5];
yhat = W2 * [1; max(hp,0)];
y = exp(yhat);
y = y/sum(y);
disp(['Question 19: ',num2str(y(2))])
 
 
%% QUESTIION 20
disp(['Question 20: ', num2str((5*5+1)*8 + (200+1)*10)])

