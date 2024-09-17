# Gibbs Sampler Fortran Implementation

## Initialization
- $p_c^\kappa = 1/25$
- $p_g^\kappa = 1/25$
- $p_h^\kappa = 1/25$
- $F = 0$
- $S_m = 0$ 
- For $ir = 1$ to $n$:
    - Do sth about $X$
    - $C_{ir} = X_{ir}-F$
- $Y^0 = 0$
- $p_c^\theta = 1/100$
- $p_g^\theta = 1/100$
- $p_h^\theta = 1/100$
- $\sigma_F^2 = [10^{-6}, \frac{0.03^2}{2.198}]$
- $\rho_m = \rho^1$
- $\mu_c = 0$
- $\omega^2 = 1$
- $\kappa_{c,i}^2 = 1$
- $\kappa_{g,j}^2 = 1$
- $\kappa_{h,k}^2 = 1$
- $\lambda_{c,i} = 0$
- $\lambda_{g,j} = 0$
- For $i = 1$ to $25$:
    - $cindCF(i) = (i-1) mod100 + 1$
    - $G_i = 0$
    - $K(i) = (i-1)mod10 + 1$
- For $ir = 1$ to $n$:
    - $J(ir) = (ir-1) mod25+1$
    - $cindu(ir) = cindCF(J(ir))$
- For $k=1$ to $10$:
    - $cindCOC(k) = (k-1)mod100+1$
    - $H_k = 0$
- $p_c^\lambda = 1/25$
- $p_g^\lambda = 1/25$


## Step 1: $\{X_i\}_{i=1}^n$ 
- For $ir=1$ to $n$:
    - $i = J(ir)$
    - $s^2 = (1-\lambda_{c,ir}^2)*\kappa_{c,ir}^2*\omega^2$
    - $\mu_C = \lambda_{c,ir}*G_i$
    - $\mu_C[0] += \mu_c$
    - $u \sim N(0, I)$ **(length of $u$: 32)**
    - $u = s*(cholesky(\Sigma_U(\theta_{c,ir}^u)) * u)+\mu_C$
    - $C_{ir} = u - (SuAA(:,:,cindu(ir), ir)*(u-C_{ir}))$
    - If $fwheights(ir) > 0$:
        - $fhat = fhat +fweights(ir)*C_{ir}$
        - $sVs = sVs+fweights(ir)^2*s^2*SuAAS(:,:,cindu(ir), ir)$
- $e \sim N(0,I)$ **(length of $e$: 32)**
- $e = Y^0 + cholesky(Delta)*e$
- $fhat=(sVs+Delta)^{-1}*(fhat-e)$
- For $ir = 1$ to $n$:
    - If $fweigths(ir) > 0$:
        - $s^2 = (1-\lambda_{c,ir}^2)*\kappa_{c,ir}^2*\omega^2$
        - $C_{ir} = C_{ir}-fweights(ir)*s^2*SuAAS(:,:,cindu(ir),ir)* fhat$
    - $X_{ir} = C_{ir}+F$

## Step 2: $\{G_j\}_{j=1}^{25}$
- For $i=1$ to $m$:
    - $k = K(i)$
    - $V_{g,i} = \frac{\Sigma_U^{-1}(\theta_{g,i}^u) }{(1-\lambda_{g,i}^2)*\omega^2*\kappa_{g,i}^2}$
    - $ms(:,i) = V_{g,i}*\lambda_{g,i}*H_k$
- For $ir=1$ to $n$:
    - $i = J(ir)$
    - $u = C_{ir}$
    - $u[0] = u[0] - \mu_c$
    - $s^2 = \omega^2*(1-\lambda_{c,ir}^2)*\kappa_{c,ir}^2$
    - $V_{g,i} = V_{g,i}+\lambda_{c,i}^2*\frac{\Sigma_U^{-1}(\theta_{c,ir}^u)}{s^2}$
    - $ms(:,i) = ms(:,ir)+\lambda_{c,ir}*\frac{\Sigma_U^{-1}(\theta_{c,ir}^u)*u}{s^2}$
- For $i=1$ to $25$:
    - $V_{g,i} = V_{g,i}^{-1}$
    - $G_i \sim N(0,I)$
    - $G_i = cholesky(V_{g,i})*G_i+V_{g,i}*ms(:,i)$

## Step 3: $\{H_k\}_{k=1}^{10}$
- For $k=1$ to $10$:
    - $V_{h,k} = \frac{\Sigma_U^{-1}(\theta_{h,k}^u)}{\omega^2*\kappa_{h,k}^2}$
    - $ms(:,k) = 0$
- For $i=1$ to $25$:
    - $k=K(i)$
    - $u = G_i$
    - $s^2 = \omega^2*(1-\lambda_{g,i}^2)*\kappa_{g,i}^2$
    - $V_{h,k} = V_{h,k}+\lambda_{g,i}^2*\frac{\Sigma_U^{-1}(\theta_{g,i}^u)}{s^2}$
    - $ms(:,k) = ms(:,k)+\lambda_{g,i}*\frac{\Sigma_U^{-1}(\theta_{g,i}^u)*u}{s^2}$
- For $k=1$ to $10$:
    - $V_{h,k} = V_{h,k}^{-1}$
    - $H_k \sim N(0,I)$
    - $H_k = cholesky(V_{h,k})*H_k+V_{h,k}*ms(:,k)$

## Step 4: $\{\lambda_{c,i}\}_{i=1}^n$
- For $ir=1$ to $n$:
    - $i=J(ir)$
    - For $l=1$ to $25$:
        - $u = C_{ir}-\lambda^l*G_i$
        - $u[0] = u[0]-\mu_c$
        - $s^2 = (1-(\lambda^l)^2)*\kappa_{c,ir}^2*\omega^2$
        - $p_l = -0.5*\frac{u^T*\Sigma_U^{-1}(\theta_{c,ir}^u)*u}{s^2}-0.5*(q+1)*ln(s^2)+ln(p_{c,l}^\lambda)$
    - $p = e^{p-max(p)}$
    - For $l=2$ to $25$:
        - $p_l = p_l+p_{l-1}$
    - $p=p/p(25)$
    - $unif \sim U(0,1)$
    - $ind = count(unif>p)+1$
    - $\lambda_{c,ir} = \lambda^{ind}$

## Step 5: $\{\lambda_{g,j}\}_{j=1}^{25}$
- For $i=1$ to $25$:
    - $k=K(i)$
    - For $l=1$ to $25$:
        - $u = G_i-\lambda^l*H_k$
        - $s^2 = (1-(\lambda^l)^2)*\kappa_{g,i}^2*\omega^2$
        - $p_l = -0.5*\frac{u^T*\Sigma_U^{-1}(\theta_{g,i}^u)*u}{s^2}-0.5*(q+1)*ln(s^2)+ln(p_{g,l}^\lambda)$
    - $p = e^{p-max(p)}$
    - For $l=2$ to $25$:
        - $p_l = p_l+p_{l-1}$
    - $p=p/p(25)$
    - $unif \sim U(0,1)$
    - $ind = count(unif>p)+1$
    - $\lambda_{g,i} = \lambda^{ind}$
    
## Step 6: $p_c^\lambda$
- $a = 20/25$ **(size of a: 25)**
- For $ir=1$ to $n$:
    - $j = round(\frac{24*\lambda_{c,ir}}{max(\lambda^l)})+1$
    - $a(j) = a(j)+1$
- For $j=1$ to $25$:
    - $p_{c,j}^\lambda \sim \Gamma(a(j))$
- $p_c^\lambda = \frac{p_c^\lambda}{\Sigma p_c^\lambda}$

## Step 7: $p_g^\lambda$
- $a = 20/25$ **(size of a: 25)**
- For $i=1$ to $25$:
    - $j = round(\frac{24*\lambda_{g,i}}{max(\lambda^l)})+1$
    - $a(j) = a(j)+1$
- For $j=1$ to $25$:
    - $p_{g,j}^\lambda \sim \Gamma(a(j))$
- $p_g^\lambda = \frac{p_g^\lambda}{\Sigma p_g^\lambda}$

## Step 8: $\{\kappa_{c,i}\}_{i=1}^n$
- For $l=1$ to $25$:
    - $pbase(l) = -0.5*(q+1)*ln(\kappa^l)+ln(p_{c,l}^\kappa)$
- For $ir=1$ to $n$:
    - $i = J(ir)$
    - $u = C_{ir}-\lambda_{c,ir}*G_i$
    - $u[0] = u[0]-\mu_c$
    - $s^2 = \omega^2*(1-\lambda_{c,ir}^2)$
    - $usu = u^T*\Sigma_U^{-1}(\theta_{c,ir}^u)*u$
    - For $l=1$ to $25$:
        - $p_l = -0.5*\frac{usu}{s^2*(\kappa^l)^2}$
    - $p = p+pbase$
    - $p = e^{p-max(p)}$
    - For $l=2$ to $25$:
        - $p_l = p_l+p_{l-1}$
    - $p=p/p(25)$
    - $unif \sim U(0,1)$
    - $ind = count(unif>p)+1$
    - $\kappa_{c,ir} = \kappa^{ind}$

## Step 9: $\{\kappa_{g,j}\}_{j=1}^{25}$
- For $l=1$ to $25$:
    - $pbase(l) = -0.5*(q+1)*ln(\kappa^l)+ln(p_{g,l}^\kappa)$
- For $i=1$ to $25$:
    - $k = K(i)$
    - $u = G_i-\lambda_{g,i}*H_k$
    - $s^2 = \omega^2*(1-\lambda_{g,i}^2)$
    - $usu = u^T*\Sigma_U^{-1}(\theta_{g,i}^u)*u$
    - For $l=1$ to $25$:
        - $p_l = -0.5*\frac{usu}{s^2*(\kappa^l)^2}$
    - $p = p+pbase$
    - $p = e^{p-max(p)}$
    - For $l=2$ to $25$:
        - $p_l = p_l+p_{l-1}$
    - $p=p/p(25)$
    - $unif \sim U(0,1)$
    - $ind = count(unif>p)+1$
    - $\kappa_{g,i} = \kappa^{ind}$

## Step 10: $\{\kappa_{h,k}\}_{k=1}^{10}$
- For $l=1$ to $25$:
    - $pbase(l) = -0.5*(q+1)*ln(\kappa^l)+ln(p_{h,l}^\kappa)$
- For $k=1$ to $10$:
    - $u = H_k$
    - $s^2 = \omega^2$
    - $usu = u^T*\Sigma_U^{-1}(\theta_{h,k}^u)*u$
    - For $l=1$ to $25$:
        - $p_l = -0.5*\frac{usu}{s^2*(\kappa^l)^2}$
    - $p = p+pbase$
    - $p = e^{p-max(p)}$
    - For $l=2$ to $25$:
        - $p_l = p_l+p_{l-1}$
    - $p=p/p(25)$
    - $unif \sim U(0,1)$
    - $ind = count(unif>p)+1$
    - $\kappa_{h,k} = \kappa^{ind}$

## Step 11: $p_c^\kappa$
- $a=20/25$ **(size of a: 25)**
- For $j=1$ to $n$:
    - $ind = count(\kappa_{c,j} >= \kappa^l)+1$
    - $a(ind) = a(ind) + 1$
- For $ind=1$ to $25$:
    - $p_{c,ind}^\kappa \sim \Gamma(a(ind))$
- $p_c^\kappa = \frac{p_c^\kappa}{\Sigma p_c^\kappa}$

## Step 12: $p_g^\kappa$
- $a=20/25$ **(size of a: 25)**
- For $j=1$ to $25$:
    - $ind = count(\kappa_{g,j} >= \kappa^l)+1$
    - $a(ind) = a(ind) + 1$
- For $ind=1$ to $25$:
    - $p_{g,ind}^\kappa \sim \Gamma(a(ind))$
- $p_g^\kappa = \frac{p_g^\kappa}{\Sigma p_g^\kappa}$

## Step 13: $p_h^\kappa$
- $a=20/25$ **(size of a: 25)**
- For $j=1$ to $10$:
    - $ind = count(\kappa_{h,j} >= \kappa^l)+1$
    - $a(ind) = a(ind) + 1$
- For $ind=1$ to $25$:
    - $p_{h,ind}^\kappa \sim \Gamma(a(ind))$
- $p_h^\kappa = \frac{p_h^\kappa}{\Sigma p_h^\kappa}$

## Step 14: $\{K(j)\}_{j=1}^{25}$
- For $i=1$ to $25$:
    - $v = G_i$
    - $s^2 = (1-\lambda_{g,i}^2)*\kappa_{g,i}^2*\omega^2$
    - For $k=1$ to $10$:
        - $u = v-\lambda_{g,i}*H_k$
        - $p_k = -0.5*\frac{u^T*\Sigma_U^{-1}(\theta_{g,i}^u)*u}{s^2}$
    - $p = e^{p-max(p)}$
    - For $k=2$ to $10$:
        - $p_k = p_k+p_{k-1}$
    - $p = p/p(10)$
    - $unif \sim U(0,1)$
    - $K(i) = count(unif>p)+1$

## Step 15: $\{J(i)\}_{i=1}^n$
- For $ir=1$ to $n$:
    - $v = C_{ir}$
    - $v[0] = v[0]-\mu_c$
    - $s^2 = (1-\lambda{c,ir}^2)*\kappa_{c,ir}^2*\omega^2$
    - For $i=1$ to $25$:
        - $u = v-\lambda_{c,ir}*G_i$
        - $p_i = -0.5*\frac{u^T*\Sigma_U^{-1}(\theta_{c,ir}^u)*u}{s^2}$
    - $p = e^{p-max(p)}$
    - For $i=2$ to $25$:
        - $p_i = p_i+p_{i-1}$
    - $p = p/p(25)$
    - $unif \sim U(0,1)$
    - $J(ir) = count(unif>p)+1$

## Step 16: $\{\theta_{c,i}^u\}_{i=1}^n$
- For $ir=1$ to $n$:
    - $i=J(ir)$
    - $meas(:,ir) = C_{ir}-\lambda_{c,ir}*G_i$
    - $meas[0] = meas[0]-\mu_c$
- For $i=1$ to $n$:
    - $u=meas(:,i)$
    - $s^2 = \omega^2*\kappa_{c,i}^2*(1-\lambda_{c,i}^2)$
    - For $j=1$ to $100$:
        - $p_j = -0.5*\frac{u^T*\Sigma_U^{-1}(\theta^j)*u}{s^2}+ln(\frac{p_{c,j}^\theta}{\sqrt{|det(\Sigma_U(\theta^j))|}})$
    - $p=e^{p-max(p)}$
    - For $j=2$ to $100$:
        - $p_j = p_{j-1}+p_j$
    - $p = p/p(100)$
    - $unif \sim U(0,1)$
    - $ind = count(unif(1)>p)+1$
    - $\theta_{c,i}^u = \theta^{ind}$

## Step 17: $\{\theta_{g,j}^u\}_{j=1}^{25}$
- For $i=1$ to $25$:
    - $k=K(i)$
    - $meas(:,i) = G_i-\lambda_{g,i}*H_k$
- For $i=1$ to $25$:
    - $u=meas(:,i)$
    - $s^2 = \omega^2*\kappa_{g,i}^2*(1-\lambda_{g,i}^2)$
    - For $j=1$ to $100$:
        - $p_j = -0.5*\frac{u^T*\Sigma_U^{-1}(\theta^j)*u}{s^2}+ln(\frac{p_{g,j}^\theta}{\sqrt{|det(\Sigma_U(\theta^j))|}})$
    - $p=e^{p-max(p)}$
    - For $j=2$ to $100$:
        - $p_j = p_{j-1}+p_j$
    - $p = p/p(100)$
    - $unif \sim U(0,1)$
    - $ind = count(unif(1)>p)+1$
    - $\theta_{g,i}^u = \theta^{ind}$

## Step 18: $\{\theta_{h,k}^u\}_{k=1}^{10}$
- $meas = \{H_k\}_{k=1}^{10}$
- For $i=1$ to $10$:
    - $u=meas(:,i)$
    - $s^2 = \omega^2*\kappa_{h,i}^2$
    - For $j=1$ to $100$:
        - $p_j = -0.5*\frac{u^T*\Sigma_U^{-1}(\theta^j)*u}{s^2}+ln(\frac{p_{h,j}^\theta}{\sqrt{|det(\Sigma_U(\theta^j))|}})$
    - $p=e^{p-max(p)}$
    - For $j=2$ to $100$:
        - $p_j = p_{j-1}+p_j$
    - $p = p/p(100)$
    - $unif \sim U(0,1)$
    - $ind = count(unif(1)>p)+1$
    - $\theta_{h,i}^u = \theta^{ind}$

## Step 19: $p_c^\theta$
- $a = 20/100$ **(size of a: $100$)**
- For $i=1$ to $n$:
    - $ind = x$ such that $\theta_{c,i}^u = \theta^x$ 
    - $a(ind) = alpha(ind)+1$
- For $i=1$ to $100$:
    - $p_{c,i}^\theta \sim \Gamma(a(i))$
- $p_c^\theta = \frac{p_c^\theta}{\Sigma p_c^\theta}$

## Step 20: $p_g^\theta$
- $a = 20/100$ **(size of a: $100$)**
- For $i=1$ to $25$:
    - $ind = x$ such that $\theta_{g,i}^u = \theta^x$ 
    - $a(ind) = alpha(ind)+1$
- For $i=1$ to $100$:
    - $p_{g,i}^\theta \sim \Gamma(a(i))$
- $p_g^\theta = \frac{p_g^\theta}{\Sigma p_g^\theta}$

## Step 21: $p_h^\theta$
- $a = 20/100$ **(size of a: $100$)**
- For $i=1$ to $10$:
    - $ind = x$ such that $\theta_{h,i}^u = \theta^x$ 
    - $a(ind) = alpha(ind)+1$
- For $i=1$ to $100$:
    - $p_{c,i}^\theta \sim \Gamma(a(i))$
- $p_h^\theta = \frac{p_h^\theta}{\Sigma p_h^\theta}$

## Step 22: $\mu_c$
- $prec = 0$
- $m = 0$
- For $ir=1$ to $n$:
    - $i = J(ir)$
    - $s^2 = \omega^2*(1-\lambda_{c,ir}^2)*\kappa_{c,ir}^2$
    - $u = C_{ir}-\lambda_{c,ir}*G_i$
    - $m = m+\frac{\iota_1^T*\Sigma_U^{-1}(\theta_{c,ir}^u)*u}{s^2}$
    - $prec = prec+\frac{\iota_1^T*\Sigma_U^{-1}*\iota_1}{s^2}$
- $v \sim N(0,1)$
- $\mu_c = \frac{m}{prec}+\frac{v}{\sqrt{prec}}$

## Step 23: $\omega^2$
- $ssum = 1/2.198$
- $snu = 1$
- For $ir=1$ to $n$:
    - $i = J(ir)$
    - $u = C_{ir}-\lambda_{c,ir}*G_i$
    - $s^2 = (1-\lambda_{c,ir}^2)*\kappa_{c,ir}^2$
    - $ssum = ssum + \frac{u^T*\Sigma_U^{-1}(\theta_{c,ir}^u)*u}{s^2}$
    - $snu = snu + q+1$
- For $i=1$ to $25$:
    - $k = K(i)$
    - $u = G_i-\lambda_{g,i}*H_k$
    - $s^2 = (1-\lambda_{g,i}^2)*\kappa_{g,i}^2$
    - $ssum = ssum + \frac{u^T*\Sigma_U^{-1}(\theta_{g,i}^u)*u}{s^2}$
    - $snu = snu + q+1$
- For $k=1$ to $10$:
    - $u = H_k$
    - $s^2 = \kappa_{h,k}^2$
    - $ssum = ssum + \frac{u^T*\Sigma_U^{-1}(\theta_{h,k}^u)*u}{s^2}$
    - $snu = snu + q+1$
- $v \sim \chi_{snu}^2$
- $\omega^2 = \frac{ssum}{v}$

## Step 24: $(f_0, \mu_m)$
- $\Sigma_F = \sigma_m^2*\Sigma_m(\rho_m)+\sigma_{\Delta a}^2*\Sigma_a$
- $prec = (\iota_{1:2}^T*\Sigma_F^{-1}*\iota_{1:2})^{-1}$
- $m = \iota_{1:2}^T*\Sigma_F^{-1}*F$
- $(f_0, \mu_m) \sim N(0,I)$
- $(f_0, \mu_m) = prec*m+cholesky(prec)*(f_0,\mu_m)$

## Step 25: $F$
- $m=0$ **(size of $m$: $32$)**
- $fhat = 0 $ **(size of $fhat$: $32$)**
- $\Sigma_F = \sigma_m^2*\Sigma_m(\rho_m)+\sigma_{\Delta a}^2*\Sigma_a$
- For $ir=1$ to $n$:
    - $i = J(ir)$
    - $s^2 = (1-\lambda_{c,ir}^2)*\kappa_{c,ir}^2*\omega^2$
    - $u = X_{ir}-\lambda_{c,ir}*G_i$
    - $u[0] = u[0] - \mu_c - f_0$
    - $u[1] = u[1] - \mu_m$
    - $m = m + \frac{\Sigma_U^{-1}(\theta_{c,ir}^u)*u}{s^2}$
    - $\Sigma_F = \Sigma_F+\frac{\Sigma_U^{-1}(\theta_{c,ir}^u)}{s^2}$
    - If $fweights(ir) > 0$:
        - $fhat = fhat+fweights(ir)*X_{ir}$
- $V_F = (\Sigma_F^{-1}+\Delta^{-1})^{-1}$
- $fhat[0:1] = fhat[0:1]-(f_0,\mu_m)$
- $m = m+\Delta^{-1}*(fhat-Y^0)$
- $F = \sim N(0,I)$
- $F = V_F*m+cholesky(V_F)*F$
- $F[0:1] = F[0:1] + (f_0,\mu_m)$
- For $ir=1$ to $n$:
    - $C_{ir} = X_{ir} - F$

## Step 26: $S_m$
- $\Sigma_m = \sigma_m^2*\Sigma_m(\rho_m)$
- $\Sigma_s = \Sigma_m +\sigma_{\Delta a}^2*\Sigma_a$
- $u = F$
- $u[0:1] = u[0:1]-(f_0,\mu_m)$
- $mfm = \Sigma_m*\Sigma_s^{-1}$
- $S_m \sim N(0,I)$
- $S_m = mfm*u+cholesky(\Sigma_m-mfm*\Sigma_m)*S_m$

## Step 27: $(\sigma_m^2, \sigma_{\Delta a}^2)$
- $usu = S_m^T*\Sigma_m^{-1}(\rho_m)*S_m$
- For $l=1$ to $25$:
    - $p_l = -0.5*\frac{usu}{(\sigma^l)^2}-0.5*(q+1)*ln((\sigma^l)^2)$
- $p = e^{p-max(p)}$
- For $l=1$ to 25:
    - If $2*l <=25$:
        - $p_l = p_l*l$
    - Else:
        - $p_l = p_l*(26-l)$
- For $l=2$ to $25$:
    - $p_l = p_l+p_{l-1}$
- $p = p/p(25)$
- $unif \sim U(0,1)$
- $ind = count(unif>p)+1$
- $\sigma_m = \sigma^{ind}$
- $u = F-S_m$
- $u[0:1] = u[0:1]-(f_0,\mu_m)$
- $ssum = \frac{0.03^2}{2.198}+u^T*\Sigma_A^{-1}*u$
- $snu = 1+q+1$
- $v \sim \chi_{snu}^2$
- $\sigma_{\Delta a}^2 = \frac{ssum}{v}$

## Step 28: $\rho_m$
- $u=S_m$
- For $i=1$ to $25$:
    - $p_i = \frac{e^{-\frac{0.5}{\sigma_m^2}*u^T*\Sigma_m^{-1}(\rho^i)}}{det(\Sigma_m(\rho^i))^2}$
- For $i=2$ to $25$:
    - $p_i = p_{i-1}+p_i$
- $p=p/p(25)$
- $unif \sim U(0,1)$
- $ind = count(unif>p)+1$
- $\rho_m = \rho^{ind}$