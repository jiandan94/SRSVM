ipiano_stls <- function(trainx, trainy, lam0, eta0){
  # proximal operator of l1 penalty
  prox_l1 <- function(u, a = a0, lam = lam0){
    aa <- ifelse(abs(u) - a*lam > 0, abs(u) - a*lam, 0)
    return(aa*sign(u))
  }
  
  # stls loss function
  stls <- function(u, eta = eta0){
    main <- function(u, a){
      if(u < a){
        return(max(u, 0)^2)
      }else if(a <= u && u < 1/a){
        return(-a^2/(1 - a^2)*(u - 1/a)^2 + 1)
      }else{
        return(1)
      }
    }
    return(sapply(u, function(s){return(main(s, a = eta))}))
  }
  
  # the derivative of stls loss function
  d_stls <- function(u, eta = eta0){
    main <- function(u, a){
      if(u >= 0 && u < a){
        return(2*u)
      }else if(u >= a && u < 1/a){
        return(-a^2/(1 - a^2)*(u - 1/a))
      }else{
        return(0)
      }
    }
    return(sapply(u, function(s){return(main(s, a = eta))}))
  }
  
  # the derivative of loss part with respect to w
  d_f <- function(x = trainx, y = trainy, w, eta = eta0){
    dstls <- d_stls(c(1 - diag(y)%*%x%*%w), eta)# length-n vector
    dl <- -apply(diag(dstls*y)%*%x, 2, mean)# length-p vector
    return(dl)
  }
  
  # initializing
  n <- nrow(trainx)
  trainx <- cbind(trainx, rep(1, n))
  p <- ncol(trainx)
  b0 <- 0.75
  lip_const <- mean(sqrt(apply(trainx^2, 1, sum)))*(2*eta0)
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
  svmic <- sum(stls(c(1 - diag(trainy)%*%trainx%*%w1))) + log(n)*s/n
  
  stls_out <- list(w = w1[1:(p-1)], b = w1[p], svmic = svmic)
  return(stls_out)
}
