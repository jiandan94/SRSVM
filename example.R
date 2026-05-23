rm(list = ls())

source("ipiano_rsh.R")
source("ipiano_stls.R")

library(MASS)

set.seed(968)
n <- 250
sig <- diag(c(0.02, 0.03))
xpos <- mvrnorm(0.5*n, mu = c(-0.2, 0.7), Sigma = sig)
xneg <- mvrnorm(0.5*n, mu = c(-0.7, 0.3), Sigma = sig)
x_main <- rbind(xpos, xneg)
x_res <- matrix(runif(n*50, -0.1,0.1), nrow = n)

x <- cbind(x_main, x_res)
y <- rep(c(1,-1), c(0.5*n, 0.5*n))

# stls
op_stls <- ipiano_stls(x, y, lam0 = 0.01, eta0 = 0.2)
stls_fity <- sign(x%*%op_stls$w + op_stls$b)
sum(stls_fity == y)/n

# rsh
op_rsh <- ipiano_rsh(x, y, lam0 = 0.01, eta0 = 0.2)
rsh_fity <- sign(x%*%op_rsh$w + op_rsh$b)
sum(rsh_fity == y)/n
