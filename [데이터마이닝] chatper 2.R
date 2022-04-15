# CHAPTER2. Regularization Methods 

# 1.0.1. ridge regression for boston data
library(MASS)
?Boston
x = model.matrix(medv~., Boston)[,-1]
y = Boston$medv
grid = 10^seq(10,-2,length = 100) #10 ~ -2까지 등간격으로 자른 후 10의 지수로 쓰겠다.

head(Boston)

library(glmnet) # ridge와 lasso 를 위한 library
ridge.mod = glmnet(x,y,alpha=0,lambda=grid) # alpha = 0 for ridge, alpha = 1 for Lasso
dim(coef(ridge.mod)) # 변수 15개, 100개의 lamba값

ridge.mod
ridge.mod$lambda[50]
coef(ridge.mod)[,50] #50번째 lamda에서 coef 

coef(ridge.mod)[-1,50]
coef(ridge.mod)[-1,50]^2
sqrt(sum(coef(ridge.mod)[-1,50]^2)) # 원소 제거(인터셉트 제거) 후 나머지 값에 제곱에 합을 취하고 루트(축소하기 위함)

ridge.mod$lambda[60]
coef(ridge.mod)[,60]
sqrt(sum(coef(ridge.mod)[-1,60]^2))

# ridge estimates when s=lambda=50 (50번째 lambda의 결과가 더 좋다)

predict(ridge.mod,s=50, type="coefficients")[1:15, ] # lambda 50번째일때 coefficient 

# validation set approach to  tune the hyperparameter 
set.seed(100)
nrow(x)
train = sample(1:nrow(x), 0.7*nrow(x))
test = (-train)
y.test = y[test]
ridge.mod = glmnet(x[train,], y[train], alpha=0,lambda=grid)
ridge.pred = predict(ridge.mod, s=4, newx = x[test,])
mean((ridge.pred-y.test)^2)



lm(y~x, subset=train)
predict(ridge.mod,s=0, type="coefficients")[1:14, ]
# s = lambda = 0일때는 linear regression과 매우 유사

set.seed(1)
cv.out = cv.glmnet(x[train,],y[train],alpha=0) # glmnet에 대한 cross validation
plot(cv.out) # 람다에 따른 MSE
cv.out


plot(ridge.mod, xvar = "lambda", label = T)  
plot(ridge.mod, xvar = "norm", label = T)

bestlam = cv.out$lambda.min
bestlam
ridge.pred = predict(ridge.mod, s=bestlam, newx = x[test,])
mean((ridge.pred-y.test)^2)
out = glmnet(x,y,alpha = 0)
predict(out,type="coefficients", s=bestlam)[1:14,]





# 2.0.1 LASSO regression
library(MASS)
?Boston
x = model.matrix(medv~., Boston)[,-1]
y = Boston$medv


nrow(x)
train = sample(1:nrow(x), 0.7*nrow(x))
test = (-train)
y.test = y[test]
LASSO.mod = glmnet(x[train,], y[train], alpha=1,lambda=grid)
LASSO.mod
LASSO.pred = predict(LASSO.mod, s=4, newx = x[test,])
LASSO.pred
y.test
mean((LASSO.pred-y.test)^2) #lambda = 4



cv.out1 = cv.glmnet(x[train,],y[train],alpha=1) # cv for LASSO
plot(cv.out)
cv.out1
plot(LASSO.mod, xvar = "lambda", label = T)
plot(LASSO.mod, xvar = "norm", label = T)


bestlam1 = cv.out$lambda.min
bestlam1
LASSO.pred = predict(LASSO.mod, s=bestlam, newx = x[test,])
mean((LASSO.pred-y.test)^2)
out = glmnet(x,y,alpha = 0)
predict(out,type="coefficients", s=bestlam1)[1:14,]





# 3.0.1. Elastic net

enet = glmnet(x[train,],y[train], alpha=0.5, lambda = grid) # elastic net
set.seed(1)
cv.outnet = cv.glmnet(x[train,],y[train], alpha=0.5) # cross validation for elastic net 
cv.outnet

bestlamenet = cv.outnet$lambda.min
enet.pred = predict(enet,s=bestlamenet,newx=x[test,])
mean((enet.pred-y.test)^2)


out.enet = glmnet(x,y,alpha = 0.5,lambda=grid) 
enet.coef=predict(out.enet,type = "coefficients",s=bestlam)[1:14,] # type : 예측 결과의 유형을. "link" 일 경우 log-odds/"response"의 경우 확률 $p$
enet.coef
enet.coef[enet.coef !=0] # 변수 선택 효과



# ????
set.seed(100)
train = sample(nrow(x), 0.7*nrow(x))
test = (-train)
x.train = x[train,]
x.test = x[test,]
y.train = y[train]
y.test = y[test]

library(doParallel)

nc = detectCores()
registerDoParallel(nc)
library(caret) # classification and Regression Train

myControl = trainControl(method = "cv", number =10, allowParallel = T,savePredictions = "final" ) 
# traincontrol(데이터 샘플링 방법, 교차 검증을 몇 겹으로 할것인지 또는 부트스트래핑을 몇 번 시행할지)

set.seed(10)
fit = train(x.train, y.train, method = "glmnet", trControl = myControl,tuneLength =3,  preProcess = c("center","scale"))
# tuneLength : 조율모수(알파)의 후보 값 변경 ,preprocess : 중심화(center)와 척도화(scale)를 수행

fit
fit $bestTune
bt=fit$bestTune

pred = predict.train(fit,x.test)
mean((y.test-pred)^2)

fitb = glmnet(x.train, y.train, alpha=bt$alpha, lambda = bt$lambda)
predict(fitb, type = "coefficient")[,1]




# 4.0.1. regularization for logistic regression

hdata=read.csv(file = "C:/Users/qazpl/Desktop/과제/대3/데이터마이닝/Heart.csv")
head(hdata)
str(hdata)
hdata$X = NULL # remove variable
sum(is.na(hdata))
hdata[!complete.cases(hdata),] # 결측치 listwise view
apply(hdata,2,function(x) sum(is.na(x))) # 결측치 개수

hdata = na.omit(hdata) # 결측치 제거
x= model.matrix(AHD ~ ., data=hdata)[,-1]
y = hdata$AHD
n = nrow(x)
n

set.seed(1)
train = sample(n,n/2)
x.train = x[train, ]
x.test = x[-train,]
y.train = y[train]
y.test = y[-train]

# ridge 
f = glmnet(x.train,y.train,alpha=0,lambda=1,family="binomial")
f.out = cv.glmnet(x.train,y.train,alpha=0,family="binomial")
f.out
f.out$lambda.min
fit0 = glmnet(x.train,y.train,alpha=0,family="binomial")
predict(fit0,type="coefficients",s=f.out$lambda.min)[1:17,]

pred = predict(fit0, newx=x.test, s=f.out$lambda.min, type="response")
pcl = ifelse(pred>0.5,"Yes","No")
mean(y.test != pcl)





# 5.0.1. grid and random search

library(caret)
library(glmnet)
library(doParallel)
nc=detectCores()
registerDoParallel(nc)

hdata=read.csv(file = "C:/Users/qazpl/Desktop/과제/대3/데이터마이닝/Heart.csv")
hdata$X = NULL
hdata = na.omit(hdata)
x= model.matrix(AHD ~ ., data=hdata)[,-1]
y = hdata$AHD
n = nrow(x)
n

set.seed(1)
train = sample(n,n/2)
x.train = x[train, ]
x.test = x[-train,]
y.train = y[train]
y.test = y[-train]

