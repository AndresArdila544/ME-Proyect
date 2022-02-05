%% LOAD DATASET
images_tr = loadMNISTImages('./dataset/train-images.idx3-ubyte');
labels_tr = loadMNISTLabels('./dataset/train-labels.idx1-ubyte');
images_ts = loadMNISTImages('./dataset/t10k-images.idx3-ubyte');
labels_ts = loadMNISTLabels('./dataset/t10k-labels.idx1-ubyte');

images_tr = images_tr';
images_ts = images_ts';

labelsNoHot= labels_tr;
labels_tr = one_hot(labels_tr, 10);
labels_ts = one_hot(labels_ts, 10);

Nd = size(images_tr, 1);
Ni = size(images_tr, 2);

%% INIT RBM
nhidden = 100; % number of hidden units
[M, b, c] = rbm_init(Ni, nhidden);

%% TRAIN RBM
max_epochs = 10; % number of training epochs
eta = 0.001; % learning rate
alpha = 0.5; % momentum
lambda = 1e-5; % regularization
k = 1; % contrastive-divergence steps
[M, b, c, errors] = rbm_train(images_tr, M, b, c, k, eta, alpha, lambda, max_epochs);

%% ENCODE DIGITS
img_codes_tr = rbm_encode(images_tr, M, b, c);
img_codes_ts = rbm_encode(images_ts, M, b, c);

disp(size(labels_tr'));

%TRAIN WITH ONE VS ALL
lambda = 0.1;
[all_theta] = oneVsAll(img_codes_tr, labelsNoHot , 10 , 0.1);
y_pred_tr = predictOneVsAll(all_theta, img_codes_tr);
y_pred_ts= predictOneVsAll(all_theta, img_codes_ts);



%% TRAIN SOFTMAX LAYER
%%softmax = trainSoftmaxLayer(img_codes_tr', labels_tr', 'MaxEpochs', 1000);
%%y_pred_tr = softmax(img_codes_tr');
%%y_pred_ts = softmax(img_codes_ts');

%% PLOT ERROR
figure
plot(1:size(errors, 2), errors);
title(sprintf('training error, alpha: %f lambda: %f', alpha, lambda));
xlabel('epoch');
ylabel('error');

fprintf('\nTraining Set Accuracy: %f\n', mean(double(y_pred_tr == labelsNoHot)) * 100);

confmat=ConfusionMatrix(y_pred_tr,labelsNoHot);
plotConfMat(confmat, {'0', '1','2','3','4','5','6','7','8','9'}, 16);