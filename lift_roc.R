# funzione per il calcolo e i grafici delle curve lift e roc

lift.roc<- function(previsti, g, type="bin", plot.it=TRUE)
{
  library(sm)
  if(!is.numeric(g)) stop("g not numeric")
  ind <- rev(order(previsti))
  n <- length(g)
  x1 <-  (1:n)/n
  x2 <- cumsum(g[ind])/(mean(g)*(1:n))
  if(type=="crude" & plot.it) 
    plot(x1, x2, type="l", col=2,
         xlab="frazione di soggetti previsti", ylab="lift")
  if(type=="sm") {
    a<- sm.regression(x1, x2, h=0.1, display="none")
    if(plot.it)
      plot(a$eval, a$estimate, type="l",xlim=c(0,1), col=2,
           xlab="frazione di soggetti previsti", ylab="lift")
  }
  if(type=="bin") {
    b <-  binning(x1,x2, breaks=(-0.001:10)/9.999)
    x <- c(0,seq(0.05,0.95, by=0.1),1)
    if(plot.it) plot(x, c(x2[1],b$means,1), type="b", xlim=c(0,1),
                     ylim=c(1,max(x2)), cex=0.75, col=2,
                     xlab="frazione di soggetti previsti",
                     ylab="fattore di miglioramento")
    x1<- x
    x2<- c(x2[1],b$means,1)
  }
  if(plot.it) {cat("premere <cr>"); readline()}
  u1<- cumsum(1-g[ind])/sum(1-g)
  u2<- cumsum(g[ind])/sum(g)
  if(type=="crude" & plot.it)
    plot(u1, u2, type="l", xlim=c(0,1), ylim=c(0,1), col=2,
         xlab="1-specificita`", ylab="sensibilita`")
  if(type=="sm") {
    # browser()
    eps<- 0.00001
    a<- sm.regression(u1,log((u2+eps)/(1-u2+2*eps)), h=0.1, display="none")
    q<- exp(a$estimate)/(1+exp(a$estimate))
    if(plot.it) plot(a$eval, q, type="l", xlim=c(0,1), ylim=c(0,1),
                     xlab="1-specificita`", ylab="sensibilita`", col=2)
  }
  if(type=="bin") {
    b <- binning(u1,u2, breaks=(-0.001:10)/9.999)
    x <- c(0,seq(0.05,0.95, by=0.1),1)
    y<- c(0,b$means,1)
    if(plot.it)
      plot(x, y, type="b", xlim=c(0,1),
           ylim=c(0,1),cex=0.75, xlab="1-specificita`",
           ylab="sensibilita`", col=2)
    u1<- x
    u2<- y
  }                      
  if(plot.it) {
    abline(0,1, lty=2, col=3)
  }
  invisible(list(x1,x2,u1,u2))
}
#
# funzione che calcola matrice di confusione e gli errori di classificazione
#
tabella.sommario <- function(previsti, osservati){
  n <-  table(previsti,osservati)
  err.tot <- 1-sum(diag(n))/sum(n)
  fn <- n[1,2]/(n[1,2]+n[2,2])
  fp <- n[2,1]/(n[1,1]+n[2,1])
  print(n)
  cat("errore totale: ", format(err.tot),"\n")
  cat("falsi positivi & falsi negativi: ",format(c(fp, fn)),"\n")
  invisible(n)
}