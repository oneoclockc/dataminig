# CHAPTER 4

# 1.1 regression tree

library(MASS)
library(tree)
n = dim(Boston)[1] # data size 확인인
set.seed(100)
train = sample(n,n*0.7)
b.train = Boston[train,]
b.test = Boston[-train,]
bt = tree(medv~., data=b.train) # regression tree 
plot(bt) 
text(bt)

pred.bt = predict(bt, b.test) # predict
mean((pred.bt - b.test$medv)^2) # mse

cor(pred.bt, b.test$medv) # 상관계수
cor(pred.bt, b.test$medv)^2 # 결정계수


f0 = lm(medv~., data = b.train) # linear model과 비교
pred0 = predict(f0, b.test) # linear model prediction 
mean((pred0 - b.test$medv)^2) # linear model mse

cor(pred0, b.test$medv) # 상관계수
cor(pred0, b.test$medv)^2 # 결정계수

# linear regression is better



# 2.1 classification tree

x1 = rep(1:4, 5)
x1
x2 = rep(1:5, each = 4)
x2
y = c(0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0)

d = data.frame(y = factor(y), x1= x1, x2=x2)
d
f = tree(y~., data=d)
par(mfrow = c(1,2)) # mfrow -> 한 화면에 그래프 여러개

plot(x1,x2,col = y+1, pch = y+1)
abline(v=2.5)
abline(v=1.5,lty=2)
abline(v=3.5,lty=2)

plot(f)
text(f)



# heart( data
library(tree)
hdata <- read.csv("C:/Users/qazpl/Desktop/과제/대3/데이터마이닝/Heart.csv")
str(hdata)
hdata$X <- NULL # index 제거
str(hdata)
hdata <- na.omit(hdata)
n <-dim(hdata)[1]
set.seed(12)
train <- sample(n,n/2)
hdata$AHD <- factor(hdata$AHD)
hdata$ChestPain <- factor(hdata$ChestPain)
hdata$Thal <- factor(hdata$Thal)

htree <- tree(AHD ~. ,data=hdata[train,])
plot(htree)
text(htree, pretty = T)
htree
