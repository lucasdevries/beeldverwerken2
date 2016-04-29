function main() 
%% Onno Hartveldt 10972935 en Lucas de Vries 10650881
%% Q1
% Read images.
image = imread('kids.tif');
image = im2double(image);
image = image / max(max(image));

% Unscharp Mask
G = 1/9.0* [1,1,1;1,1,1;1,1,1];
blurred = imfilter(image, G, 'conv', 'replicate');
masked = 2* image - blurred;
subplot(2,4,1);
imshow(image);
title('Original');
subplot(2,4,2);
imshow(masked);
title('Unsharp Mask');

% Motion Blur
G =fspecial('motion', 5, 0);
motionblurred = imfilter(image, G, 'conv', 'replicate');
subplot(2,4,3);
imshow(motionblurred);
title('Motion Blur');

% Optical Blur
G = fspecial('gaussian', [5 5], 4);
optblur = imfilter(image, G, 'conv', 'replicate');
subplot(2,4,4);
imshow(optblur);
title('Gaussian optical blur, sigma = 4');
%% Q2
%% 2.1
G = gauss(3);
%% 2.2
sum_over_gauss = sum(sum(gauss(5)))
B = sum(sum(gauss(9)));
% A and B gave values a bit lower than one. We solved this by taking
% different ranges for x and y. First we used M or N = 2S, which covers
% about 98 percent. With ranges 4S, we get a sum of 0.9999, which is close
% enough to the expected one. 
%% 2.3 
figure
mesh(gauss(3));
title('Mesh plot gaussian kernel size 3');
%% 2.4
% The physical unit of the scale factor (standard deviation) is the number
% of pixels
%% 2.5
data = sigmatimetest(image, 1,20,20);
figure
scatter(data(:,1),data(:,2),15,linspace(1,20,length(data(:,1))))
title('Elapsed computation time vs. sigma')
%% 2.6
% The computational complexity for an MxM kernel is O(M^2). For
% gauss(sigma) this is O(8*S*8*S) = O(S^2).
%% 2.7
%% 2.8
% See gauss1(sigma) function below. 
size(gauss1(5));
sum_over_gauss1 = sum(sum(gauss1(5)))
%% 2.9
data = sigmatimetestseparate(image, 1,20,20);
figure
scatter(data(:,1),data(:,2),15,linspace(1,20,length(data(:,1))))
title('Elapsed computation time vs. sigma (seperable)')
% The separable calculation takes less time, even for higher scale factors
% (it stays constant). 
% The order of complexity now is O(S).
%% 2.10/11
figure
gD(image, 5, 2, 2);
%% 3.1
% Analytical derivatives:
% fx = V*A*cos(Vx)
% fxx = -V^2*A*sin(Vx)
% fy = -B*W*sin(Wy)
% fyy = -B*W^2*cos(Wy)
% fxy = 0
%% 3.2
x = -100:100;
y = -100:100;
[X, Y] = meshgrid(x, y);
A = 1; B = 2; V = 6*pi/201; W = 4*pi/201;
F = A * sin(V*X) + B * cos(W*Y);
figure;
imshow(F, [], 'xData', x, 'yData', y);
Fx = V.*A*cos(V.*X);
Fy = -B.*W.*sin(W.*Y);

figure
imshow(Fx, []);
title('Fx')
figure
imshow(Fy, []);
title('Fy')
%% 3.3
xx = -100:10:100;
yy = -100:10:100;
[XX, YY] = meshgrid(xx, yy);
Fx = V.*A*cos(V.*XX);
Fy = -B.*W.*sin(W.*YY);
figure
imshow(F, [], 'xData', x, 'yData', y); hold on;
quiver(xx, yy, Fx, Fy, 'r');
hold off;
%% 3.4

% xx = -100:10:100;
% yy = -100:10:100;
% [XX, YY] = meshgrid(xx, yy);
% Gx = imfilter(F, gD(F, 3, 1, 1), 'conv', 'replicate');
% Gy = imfilter(F, gD(F, 3, 1, 1), 'conv', 'replicate');
% figure
% imshow(Gx)
% figure
% imshow(Gy)

%% Functions

function G = gauss(S)
% 2 simga covers about 98% of the gaussian, to get sum(gauss(sigma)) = 1,
% we have to take 4 sigma, to get 0.9999. 
M = 4*S;
N = 4*S;
x = -M : M ;
y = -N : N ;
% create a sampling grid
[X , Y ] = meshgrid (x , y );
% determine the scale
sigma = S ;
% calculate the Gaussian function
G = (1/((sqrt(2*pi)*sigma)^2))*exp(-(X.^2+Y.^2)/(2*(sigma^2)));
end

function data = sigmatimetest(image, minSigma,maxSigma,N)
% FUnction sigmatimetest takes a maximum sigma and a minimum sigma,
% an image and the number of measurements (N) per sigma. 
% It than computes the time N times per sigma and plots the times vs the
% scalefactor to get a nice view of the computation time, for different
% scale factore. 
data = zeros((maxSigma-minSigma+1)*N,2);
count = 0;
for sigma=minSigma:maxSigma,

    for j=1:N,
        tic;
        imfilter(image, gauss(sigma), 'conv','replicate');
        time = toc;
        data(count*N+j,:) = [sigma,time];
    end
    count = count + 1;
end
end
function G = gauss1(S)
% create appropriate ranges for x and y
M = 4*S;
N = 4*S;
X = -M : M ;
% create a sampling grid
% determine the scale
sigma = S ;
% calculate the Gaussian function
G = (1/(sqrt(2*pi)*sigma))*exp(-(X.^2)/(2*(sigma^2)));
end
function data = sigmatimetestseparate(image, minSigma,maxSigma,N)
% FUnction sigmatimetestseparate takes a maximum sigma and a minimum sigma,
% an image and the number of measurements (N) per sigma. 
% It than computes the time N times per sigma and plots the times vs the
% scalefactor to get a nice view of the computation time, for different
% scale factore. Here, we use separate convolution kernels. 
data = zeros((maxSigma-minSigma+1)*N,2);
count = 0;
for sigma=minSigma:maxSigma,

    for j=1:N,
        tic;
        imfilter(image, gauss1(sigma)*gauss1(sigma)', 'conv','replicate');
        time = toc;
        data(count*N+j,:) = [sigma,time];
    end
    count = count + 1;
end
     
end


function G = gD(image, sigma, xorder, yorder)
M = 4*sigma;
X = -M : M ;
% The 0th, 1st and 2nd order derivatives of the gaussian.
gD = [gauss1(sigma); (-X./(sigma^2)).* gauss1(sigma); ((X.^2) / (sigma^4) - 1/(sigma^2)) .* gauss1(sigma)];
% First the x derivative, then the y derivative. 
G = imfilter(image, gD(xorder+1,:), 'conv', 'replicate');
G = imfilter(G, gD(yorder+1,:), 'conv', 'replicate');
imshow(G, []);
end


end


