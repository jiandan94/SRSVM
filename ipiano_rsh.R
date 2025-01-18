ipiano_rsh <- function(trainx, trainy, lam0, eta0){
  # proximal operator of l1 penalty
  prox_l1 <- function(u, a = a0, lam = lam0){
    aa <- ifelse(abs(u) - a*lam > 0, abs(u) - a*lam, 0)
    return(aa*sign(u))
  }
  
  # rsh loss function
  rsh <- function(u, eta = eta0){
    aa <- ifelse(u > 0, u, 0)
    return(1 - exp(-eta*aa^2))
  }
  
  # the derivative of rsh loss function
  d_rsh <- function(u, eta = eta0){
    aa <- 2*eta*u*exp(-eta*u^2)
    return(ifelse(aa > 0, aa, 0))
  }
  
  # the derivative of loss part with respect to w
  d_f <- function(x = trainx, y = trainy, w, eta = eta0){
    drsh <- d_rsh(c(1 - diag(y)%*%x%*%w), eta)# length-n vector
    dl <- -apply(diag(drsh*y)%*%x, 2, mean)# length-p vector
    return(dl)
  }
  
  # initializing
  n <- nrow(trainx)
  trainx <- cbind(trainx, rep(1, n))
  p <- ncol(trainx)
  b0 <- 0.75
  lip_const <- mean(sqrt(apply(trainx^2, 1, sum)))*sqrt(2*eta0/exp(1))
  a0 <- 2*(1 - b0)/lip_const
  
  # main iteration
  w0 <- rep(1, p)
  w1 <- w0 + 0.5
  t <- 0
  while(sum(abs(w0)) > 0 && sum(abs(w1-w0))/sum(abs(w0)) >= 0.001 && t <= 5000){
    ww <- w1
    w1 <- prox_l1(w1 - a0*d_f(w = w1) + b0*(w1 - w0))
    w0 <- ww
    t <- t + 1
  }
  
  s <- length(which(w1[1:(p-1)] == 0))
  svmic <- sum(rsh(c(1 - diag(trainy)%*%trainx%*%w1))) + log(n)*s/n
  
  rsh_out <- list(w = w1[-p], b = w1[p], svmic = svmic)
  return(rsh_out)
}
