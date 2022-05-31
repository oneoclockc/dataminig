# chapter 6

# 2.4.1. bagging & random forest
# classification 
# bagging
hdata <- read.csv("C:/Users/qazpl/Desktop/과제/대3/데이터마이닝/Heart.csv", header = TRUE, stringsAsFactors = T)
hdata$X <- NULL # 인덱스 제거
hdata <- na.omit(hdata)
n <- dim(hdata)[1]
install.packages("randomForest")
library(randomForest)

set.seed(10)
train <- sample(n,n/2)
htrain <- hdata[train,]
htest <- hdata[-train,]
set.seed(1)

bag.h <- randomForest(AHD~., data = htrain, mtry = 13, importance = T) 
# mtry : Number of variables randomly sampled as candidates at each split.
# importance : 각 변수의 중요성을 평가하는지
bag.h

yhat.bag <- predict(bag.h, newdata = htest)
mean(yhat.bag != htest$AHD)

importance(bag.h)
varImpPlot(bag.h)

# randomforest
set.seed(1)
rf.h <- randomForest(AHD~.,data=htrain, mtry = 4, importance = T)
rf.h
yhat.rf <- predict(bag.h, newdata = htest)
yhat.rf <- predict(rf.h, newdata = htest)
mean(yhat.rf != htest$AHD) # bagging과 비교하여 error가 약간 줄음
importance(rf.h)
varImpPlot(rf.h)


# regression
# bagging
library(MASS)
n <- dim(Boston)[1]
set.seed(1)
train <- sample(n,n/2)
boston.test <- Boston$medv[-train]
boston.test
btrain <- Boston[train,]
btest <- Boston[-train,]
set.seed(1)
bag.boston = randomForest(medv~., data=btrain, mtry = 13, importance = TRUE)
bag.boston
yhat.bag = predict(bag.boston, newdata=btest)
plot(yhat.bag, boston.test)
abline(0,1)
mean((yhat.bag - boston.test)^2)
importance(bag.boston)
varImpPlot(bag.boston)

# randomforest
set.seed(1)
rf.boston = randomForest(medv~., data=btrain, mtry = 6, importance = TRUE)
rf.boston
yhat.rf = predict(rf.boston, newdata=btest)
plot(yhat.rf, boston.test)
abline(0,1)
mean((yhat.rf - boston.test)^2)
importance(rf.boston)
varImpPlot(rf.boston)




# boosting
# gbm
install.packages("gbm")
library(gbm)
set.seed(1)
n<-dim(iris)[1]
train <- sample(n,n/2)
fitgbm <- gbm(Species ~., data = iris[train,], cv.folds = 10,
              n.tree = 100, distribution = "multinomial")
best.iter = gbm.perf(fitgbm, method = "cv")
best.iter # 최적의 부스팅 반복 횟수

predgbm <- predict(fitgbm, iris[-train,], best.iter, type = "response")
predgbm
pclgbm <- apply(predgbm, 1, which.max)
pclgbm
table(pclgbm, iris$Species[-train])


install.packages("caret")
library(caret)
mycontrol = trainControl(method = "cv", number =10, savePrediction = "final",
                         classProbs = T, allowParallel = T)
set.seed(12)
fit_gbm = train(Species~., data = iris[train,], method = "gbm", trControl = mycontrol, tuneLength = 3, preProc = c("center", "scale"), verbose = F)
pred_gbm = predict(fit_gbm, iris[-train,])             
pred_gbm
table(pred_gbm, iris$Species[-train])                
mean(pred_gbm == iris$Species[-train])
