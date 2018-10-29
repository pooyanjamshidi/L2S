function score = rmse(actual, prediction)
%RMSE   Computes the root mean-squared error between actual and prediction

% Authors: Pooyan Jamshidi (pooyan.jamshidi@gmail.com)
% The code is released under the FreeBSD License.
% Copyright (C) 2016 Pooyan Jamshidi, Imperial College London
score = sqrt(mse(actual, prediction));
