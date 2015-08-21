% dist = cosineDistance(x,y)
% This function computes the cosine distance between feature vectors 
% x and y. This distance is frequently used for text classification. It
% varies between 0 and 1. The distance is 0 iff x==y. 

function dist = cosineDistance(x,y)
denom = sqrt(sum(x.^2)*sum(y.^2));
dist = 1-(x*y')/denom;
