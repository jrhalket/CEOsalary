function h = silvermanRoTBand(x)
%Silverman's rule of thumb for data x where we want to kernal estimate
%the pdf of the differences x_i-x_j for j~=i

n = length(x);
X = x - transpose(x);
for i=1:n
    X(i,i)=NaN;
end
sigma = std(X(:));
Iqr = iqr(X(:));

h = 0.9*min(sigma,Iqr/1.34)*(n*(n-1))^(.2);
end


