# CHAPTER 0
# 4.3.0.1 선형회귀, 다항회귀
library(ISLR)
fit <- lm(mpg~horsepower, data = Auto) # 선형회귀, 연비~마력
summary(fit)
fit2 <- lm(mpg~poly(horsepower,2, raw=T), data = Auto) # 다항회귀, raw는 orthogonal polynomial를 계산하는지 여부
summary(fit2)

with(Auto, plot(horsepower,mpg))
abline(fit, col=2,lwd=2) # fit 의 추세선
curve(coef(fit2)[1] + coef(fit2)[2] *x + coef(fit2)[3] * x^2,
      add = T, col=3, lwd =2) #  add : TRUE일 경우 이전 그림에 겹쳐서
# fit2 is better
 
# 4.1.1.1 KNN 회귀(비모수)
install.packages("caret")
library(caret)
fit <- knnreg(data.frame(horsepower = Auto$horsepower), Auto$mpg,k=10) 
# knnreg(trainx, (testx), trainy, k)
xt <- seq(46,230, by=0.001)
yhat <- predict(fit, data.frame(horsepower = xt)) # predict(모델, 데이터, 간격)
plot(xt, yhat, type = "l", col = "red", lwd =2)
with(Auto, points(horsepower, mpg))


# CHAPTER 1

#4.2.1. bootstrap CI  

library(boot)
head(city)
# confidence intervals for the city data 
ratio <- function(d,w) sum(d$x * w)/sum(d$u * w) # 사용자정의함수 정의
sum(city$x)/sum(city$u)
city.boot <- boot(city, ratio, R=999, stype= "w", sim="ordinary") # boot(dataset, statistic, 반복횟수) 
boot.ci(city.boot, conf = c(0.9,0.95), type = c("norm","basic","perc","bca")) # 네가지 방법으로 구한 CI

#nonpara CI for mean failure time of the air conditioning data as in ex 5.4
head(aircondit)
mean.fun <- function(d,i) {
  m <-mean(d$hours[i])
  n <- length(i)
  v <- (n-1) * var(d$hours[i])/n^2
  c(m,v)
}
air.boot <- boot(aircondit, mean.fun, R  =999)
boot.ci(air.boot, type = c("norm","basic","perc","stud"))
mean.fun(aircondit,1:12) # 모든 ci가 실제 mean을 포함


# 4.3.3.1 bootstrap variance    
library(ISLR)
library(boot)
alpha.fn = function(data, index) {
    X = data$X[index]
    Y = data$Y[index]
    return((var(Y) - cov(X,Y))/(var(X) + var(Y) - 2 * cov(X,Y)))
}  # alpha*X +(1-alpha)Y 의 분산

alpha.fn(Portfolio, sample(100,100,replace = T)) # sample(후보군, 뽑을 개수, 복원함수인지)
boot(Portfolio, alpha.fn, R=1000)


# 6.2.1. validation set(hold-out) approach

library(ISLR)
str(Auto)
n<-dim(Auto) # 데이터 수 확인 
train <- sample(1:n, n/2)
train # indices that are included in the training set
length(train)
# we apply the validation set approach 10 times with distinct seeds

for (j in 1:10){
  set.seed(10*j)
  vs.error = rep(0:10)
  for (i in 1:10){
    fit = lm(mpg~poly(horsepower,i), data=Auto, subset = train) # subset : data 안의 어떤 데이터 원소를 모델에 적합할 것인지 
    vs.error[i] = mean((Auto$mpg[-train]-predict(fit,Auto[-train,]))^2) # i가 몇일때 vs.error가 가장 작아지는지 확인하기 위함
    }
  if (j<2) {
    plot(vs.error, type="l", ylim = c(10,30), ylab="MSE", xlab="Degrees of polynomial")
  } else {lines(vs.error, type="l",col=j)}
} 


# 6.3.1 COOCV 
library(ISLR)
glm.fit = glm(mpg~horsepower,data=Auto)
coef(glm.fit)
# glm(일반화선형모델) : 함수의 사용방법은 lm()함수와 유사하나 추가로 family라는 인수를 지정해준다. family 인자에는 종속변수의 분포를 지정
lm.fit = lm(mpg~horsepower, data=Auto)
coef(lm.fit)
mean(((Auto$mpg - fitted(lm.fit))/(1-influence(lm.fit)$hat))^2) 
# influence(lm.fit)$hat = (i,i) element of the hat matrix
library(boot)
glm.fit = glm(mpg~horsepower, data = Auto)
cv.err = cv.glm(Auto, glm.fit) # cv.glm : glm에 대한 k-fold cv predict error 
# call : 호출 / k : k-fold cv 에 사용되는 k값 / delta : raw cv estimate of prediction error, adjusted value / seed : seed값
cv.err
cv.err$delta
cv.error = rep(0,10)
library(doParallel)
nc<- detectCores()
registerDoParallel(nc)

cve <- foreach(i = 1:10) %dopar% {
  library(ISLR)
  library(boot)
  glm.fit = glm(mpg~poly(horsepower,i), data = Auto)
  cv.glm(Auto, glm.fit)$delta[1] # degree of polynomial이 i 일 때 prediction error의 estimate
}
cv.error <- unlist(cve) # list를 벡터로
cv.error 
plot(c(1:!0), cv.error, type = "o", xlab ="Index", col = "darkcyan")


# 6.4.1 10-Fold CV
# k-fold cv for linear model

set.seed(17)
k=10
kfcv.error = rep(0,10)
for (i in 1:k) {
  glm.fit = glm(mpg~poly(horsepower, i), data=Auto)
  kfcv.error[i] = cv.glm(Auto, glm.fit, K=k)$delta[1]  # degree of polynomial이 i 일 때 prediction error의 estimate
}
kfcv.error
plot(kfcv.error, type = "o")

# manual CV
k<-10
n<-dim(Auto)[1]
folds <- sample(1:k,n, replace =T , prob =rep(1/k,k)) # sample(후보군, 뽑을 개수, 복원함수인지)

kmse <- matrix(0, ncol = 10, nrow = k)
nd<-10
library(doParallel)
nc <-detectCores()
registerDoParallel(nc)
cvp<-foreach( i =1:k ) %dopar% {
  library(ISLR)
  cvmse <- numeric(nd) #numaric() 함수는 0을 함수에 지정하는 숫자만큼 생성하여 벡터(vector)를 만들어 준다.
  for (j in 1:nd) {
    fit <- lm(mpg~poly(horsepower, j), data = Auto[folds != i,]) # i번째 fold를 제외한 fold로 training
    pred <- predict(fit, newdata = Auto[folds == i, ]) # i 번째 fold로 validaion 
    cvmse[j] <- mean((pred-Auto$mpg[folds==i])^2) # j번째 시행에서의 mse 평균
  }
  cvmse
}
for (i in 1:k) kmse[i, ]<-cvp[[i]]
apply(kmse,2,mean)



