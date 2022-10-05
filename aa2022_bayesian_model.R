remove(list=ls())
library(INLA)
library(hexbin)
require(hexbin)
require(lattice)
require(RColorBrewer)
library(vioplot)

###################################
### DEFINE MODEL with R-generic ###
###################################

test.mu.model = function(cmd = c("graph", "Q", "mu", "initial", "log.norm.const",
                                 "log.prior", "quit"), theta=NULLL){
  envir <-parent.env(environment())
  prec.high = exp(15)
  
  interpret.theta <- function(){
    return(list(b1 = theta[1L], b2 = theta[2L], 
                c1 = exp(theta[3L]), c2 = exp(theta[4L]),
                v1 = exp(theta[5L])))  
  }
  
  graph <-function(){
    G <- Diagonal(n = length(vt), x=1)
    return(G)
  }
  
  Q <- function(){
    #prec.high <- interpret.theta()$prec
    Q <- prec.high*graph()
    return(Q)
  }
  
  mu <- function(){
    par = interpret.theta()
    return(log(( (((vr-par$v1)^2+(vt-(12*0.75/r))^2)^0.5)/50 )^(par$b1)*r^(par$b2)*par$c1 + par$c2))
  }
  
  log.norm.const <-function(){
    return(numeric(0))
  }
  
  # Log-prior for thetas
  log.prior <- function(){
    par = interpret.theta()
    
    #nice priors
    val <- (dnorm(par$b1, mean = 2.2, sd = 0.2, log=TRUE) +
              dnorm(par$b2, mean = -1.8, sd = 0.2, log=TRUE) + 
              dgamma(par$c1, shape = 3, rate = 1, log=TRUE) + theta[3L]+
              dgamma(par$c2, shape = 3, rate = 1, log=TRUE) + theta[4L]+
              dgamma(par$v1, shape = 10, rate = 0.2, log=TRUE) + theta[5L])
    
    #more vague
    #val <- (dnorm(par$b1, mean = 2.2, sd = 0.5, log=TRUE) +
    #          dnorm(par$b2, mean = -1.8, sd = 0.5, log=TRUE) + 
    #          dgamma(par$c1, shape = 2, rate = 1.5, log=TRUE) + theta[3L]+
    #          dgamma(par$c2, shape = 2, rate = 1.5, log=TRUE) + theta[4L]+
    #          dgamma(par$v1, shape = 5, rate = 0.1, log=TRUE) + theta[5L])
    
    #lower values
    #val <- (dnorm(par$b1, mean = 1.9, sd = 0.2, log=TRUE) +
    #          dnorm(par$b2, mean = -2.1, sd = 0.2, log=TRUE) + 
    #          dgamma(par$c1, shape = 3, rate = 2, log=TRUE) + theta[3L]+
    #          dgamma(par$c2, shape = 3, rate = 2, log=TRUE) + theta[4L]+
    #          dgamma(par$v1, shape = 10, rate = 0.4, log=TRUE) + theta[5L])
    
    return(val)
  }
  
  # Initial values of theta
  initial <- function(){
    #return(c(3,-3,1.099,1.099,3.401))
    return(c(2.2,-1.8,0.693,0.693,3.807))   #mode
    #return(c(2.2,-1.8,1.099,1.099,3.912))   #mean
    #return(c(2.2,-1.8,0.7,0.7,3.9))
  }
  
  quit <-function(){
    return(invisible())
  }
  
  val <- do.call(match.arg(cmd), args = list())
  return(val)
}


mydata = read.csv(file = 'flux_to_fit_cnn_full.csv')
names(mydata)[c(2,3,4,5,6)] = c("flux","vr","vt","r","exposure")
n = length(mydata$vr)
mydata$idx = 1:n 


rgen = inla.rgeneric.define(model = test.mu.model, vr = mydata$vr, vt = mydata$vt, r = mydata$r)
result = inla(flux ~ -1 + f(idx, model = rgen),
              data = mydata, family = "poisson", E = exposure, 
              control.compute = list(cpo=TRUE, dic=TRUE, config = TRUE),      #new part
              safe = TRUE, verbose = TRUE)


summary(result)

hist(result$cpo$pit)     # ok 
result$cpo$failure       # also OK
pit = result$cpo$pit
save(pit, file = "pit.RData")
  
plot(mydata$flux/mydata$exposure, ylab="counts/E")
lines(result$summary.fitted.values$mean, col=2, lwd=3)
  
# Posterior means of the hyperparameters
result$summary.hyperpar$mean[1]
result$summary.hyperpar$mean[2]
inla.emarginal(function(x) exp(x), result$marginals.hyperpar$`Theta3 for idx`)
inla.emarginal(function(x) exp(x), result$marginals.hyperpar$`Theta4 for idx`)
inla.emarginal(function(x) exp(x), result$marginals.hyperpar$`Theta5 for idx`)

plot((result$marginals.hyperpar$`Theta1 for idx`[1:43]),result$marginals.hyperpar$`Theta1 for idx`[44:86],main = "b1")
plot((result$marginals.hyperpar$`Theta2 for idx`[1:43]),result$marginals.hyperpar$`Theta2 for idx`[44:86],main = "b2")
plot(exp(result$marginals.hyperpar$`Theta3 for idx`[1:43]),result$marginals.hyperpar$`Theta3 for idx`[44:86],main = "c1")
plot(exp(result$marginals.hyperpar$`Theta4 for idx`[1:43]),result$marginals.hyperpar$`Theta4 for idx`[44:86],main = "c2")
plot(exp(result$marginals.hyperpar$`Theta5 for idx`[1:43]),result$marginals.hyperpar$`Theta5 for idx`[44:86],main = "v1")

s = inla.hyperpar.sample(1000000, result)

b1=s[,1]
b2=s[,2]
c1=exp(s[,3])
c2=exp(s[,4])
v1=exp(s[,5])


hexbinplot(v1~c2, 
           data=data.frame(v1,c2), 
           colramp=colorRampPalette(c("grey", "yellow")),
           main="joint histogram b1, c2" ,  
           xlab="c2", 
           ylab="v1" ,
           panel=function(x, y, ...)
           {
             panel.hexbinplot(x, y, ...)
             panel.abline(v=c(mean(c2)), h=c(mean(v1)), col="black", lwd=2, lty=3)
           }
)

save(b1, b2, c1, c2, v1, file = "sample.RData")






