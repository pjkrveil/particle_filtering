mu = 62.43;
sigma = 13.5;

mu2 = 30;
sigma2 = 12.5;

l_bnd = 2.4;
u_bnd = 97.5;

A = normrnd(mu, sigma, 5089, 1);

for i = 1: 5089
    temp = A(i);
    
    if (temp < l_bnd || temp > u_bnd)
        while (temp < l_bnd || temp > u_bnd)
            temp = abs(temp);
            noise = mvnrnd(mu2, sigma2);
            
            if (temp > u_bnd)
                temp = temp - noise;
            else
                temp = temp + noise;
            end
        end
    end
    
    A(i) = temp;
end