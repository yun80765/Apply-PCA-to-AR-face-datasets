clc;
clear;
close all


TRAIN_DATASET_ROOT = 'dataset\training\';
TEST_DATASET_ROOT = 'dataset\test\';
imgfile = dir(['dataset\training\','*.bmp']);
testFile = dir(['dataset\test\','*.bmp']);
IMG_TRAINFILE_LENGTH = length(imgfile);
IMG_TESTFILE_LENGTH = length(testFile);
PICTURE_SIZE = 120*165;
PEOPLE_SIZE = 99;
dim_size = 9;

% Step 1: Get some data
% load train datas
for i = 1 : IMG_TRAINFILE_LENGTH
    trainDatas(:,i) = double(reshape(imread(strcat(TRAIN_DATASET_ROOT,imgfile(i).name)),PICTURE_SIZE,1))/255;
end

% Step 2: Subtract the mean
m = mean(trainDatas,2);
A = trainDatas - m;

% Step 3: Calculate the covariance matrix
Cov = A * A' / 1286;

% Step 4: Calculate the eigenvectors and eigenvalues of the covariance matrix
if ~exist('m.mat') 
    [V,D] = eig(Cov);
    D = diag(D);
    save('m.mat','V','D','-v7.3');
else
    eigen = load ('m.mat');
    V = eigen.V;
    D = eigen.D;
end
d = sort(-D);

% Step 5: Choosing components and forming a feature vector
% dim = number of Components (eigenvectors) to be taken
for i = 1 : dim_size 
    n(i) = find(D == -d(i));
end

% save img
for i = 1 : dim_size
    eigenFaces{i} = V(:,n(i));
end

% Foolproof
if ~exist('a(i)\')
    mkdir a(i)
end

for i = 1 : dim_size 
    subplot(3,3,i),imshow(mat2gray(reshape(eigenFaces{i},165,120))),title(num2str(i));
    if(i == 1) 
        saveas(gcf,['a(i)\','1','.png']);
    end
    if(i == 5)
        saveas(gcf,['a(i)\','5','.png']);
    end
    if(i == 9)
        saveas(gcf,['a(i)\','9','.png']);
    end
end

% loading data
% r = randomly choose which picture from 99 people of test set 
r = randi([1,13],1,PEOPLE_SIZE);
for i = 1 : PEOPLE_SIZE
    random_test_imgs{i} = double(reshape(imread(strcat(TEST_DATASET_ROOT,testFile(13*(i-1) + r(i)).name)),PICTURE_SIZE,1))/255;
end

% Representation of Face Images using Eigenfaces
% The coefficient of linear combination can be obtained by :
% a = eigenFace' * u
% u : EigenFaces
% M = test image
for i = 1 : dim_size
    u{i} = double(eigenFaces{i});
end

% dim = 1
% Foolproof
if ~exist('a(ii)\d=1')
    mkdir a(ii)\d=1
end

u1 = u{1};
for i = 1 : PEOPLE_SIZE
    M = random_test_imgs{i}' * u1 * u1';
    subplot(1,2,1),imshow(mat2gray(reshape(random_test_imgs{i},165,120))),title('original');
    subplot(1,2,2),imshow(mat2gray(reshape(M,165,120))),title('linear combination of the eigenfaces');
    saveas(gcf,['a(ii)\d=1\',int2str(i),'.png']);
end

% dim = 5
% Foolproof
if ~exist('a(ii)\d=5')
    mkdir a(ii)\d=5
end

u5 = [u{1} u{2} u{3} u{4} u{5}];
for i = 1 : PEOPLE_SIZE
    M = random_test_imgs{i}' * u5 * u5';
    subplot(1,2,1),imshow(mat2gray(reshape(random_test_imgs{i},165,120))),title('original');
    subplot(1,2,2),imshow(mat2gray(reshape(M,165,120))),title('linear combination of the eigenfaces');
    saveas(gcf,['a(ii)\d=5\',int2str(i),'.png']);
end

% dim = 9
% Foolproof
if ~exist('a(ii)\d=9')
    mkdir a(ii)\d=9
end

u9 = [u{:}];
for i = 1 : PEOPLE_SIZE
    M = random_test_imgs{i}' * u9 * u9';
    subplot(1,2,1),imshow(mat2gray(reshape(random_test_imgs{i},165,120))),title('original');
    subplot(1,2,2),imshow(mat2gray(reshape(M,165,120))),title('linear combination of the eigenfaces');
    saveas(gcf,['a(ii)\d=9\',int2str(i),'.png']);
end

% Project the training images to the k-dimensional Eigenspace
train_project_data1 = trainDatas' * u1;
train_project_data5 = trainDatas' * u5;
train_project_data9 = trainDatas' * u9;


% Recognition (testing)
% loading all the test datas
for i = 1 : IMG_TESTFILE_LENGTH
    test_imgs(:,i) = double(reshape(imread(strcat(TEST_DATASET_ROOT,testFile(i).name)),PICTURE_SIZE,1))/255;
end

% Project the test images to the k-dimensional Eigenspace
test_project_data1 = test_imgs' * u1;
test_project_data5 = test_imgs' * u5;
test_project_data9 = test_imgs' * u9;

% Nearest Neighbor Classification
% Calcute distance : SAD – sum of absolute distance
for i = 1 : IMG_TESTFILE_LENGTH
    for j = 1 : IMG_TRAINFILE_LENGTH
        s1(i,j) = sqrt(test_project_data1(i,:) * test_project_data1(i,:)' + train_project_data1(j,:) * train_project_data1(j,:)' - 2 * test_project_data1(i,:) * train_project_data1(j,:)');
        s5(i,j) = sqrt(test_project_data5(i,:) * test_project_data5(i,:)' + train_project_data5(j,:) * train_project_data5(j,:)' - 2 * test_project_data5(i,:) * train_project_data5(j,:)');
        s9(i,j) = sqrt(test_project_data9(i,:) * test_project_data9(i,:)' + train_project_data9(j,:) * train_project_data9(j,:)' - 2 * test_project_data9(i,:) * train_project_data9(j,:)');
    end
end

% Assign the label of the NN to the test sample
% dim = 1
[M,I] = min (s1,[],2);
error1 = 0;
for i = 1 : IMG_TESTFILE_LENGTH
    if ( fix( I(i)/13 ) ~= fix( i/13 ) )
        error1 = error1 + 1;
    end
end

% dim = 5
[M,I] = min (s5,[],2);
error5 = 0;
for i = 1 : IMG_TESTFILE_LENGTH
    if ( fix( I(i)/13 ) ~= fix( i/13 ) )
        error5 = error5 + 1;
    end
end

% d = 9
[M,I] = min (s9,[],2);
error9 = 0;
for i = 1 : IMG_TESTFILE_LENGTH
    if ( fix( I(i)/13 ) ~= fix( i/13 ) )
        error9 = error9 + 1;
    end
end

error_rate1 = error1 / IMG_TESTFILE_LENGTH ;
error_rate5 = error5 / IMG_TESTFILE_LENGTH ;
error_rate9 = error9 / IMG_TESTFILE_LENGTH ;

disp('error rate :');
disp('dim = 1 :');
disp(error_rate1);
disp('dim = 5 :');
disp(error_rate5);
disp('dim = 9 :');
disp(error_rate9);







































