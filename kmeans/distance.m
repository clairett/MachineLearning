function dis = distance(x, mu, k)
    dis = sum((repmat(x, k, 1) - mu).^2, 2);
end
