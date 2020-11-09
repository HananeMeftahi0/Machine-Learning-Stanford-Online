function J = computeCost(X, y, theta)
  
      m =length(y);
      h=X*theta ;
      J=1/(2*m)*sum((h-y).^2);

end
