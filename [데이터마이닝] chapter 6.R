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
fitgbm <- gbm(Species ~., data = iris[train,], cv.folds = 10, n.tree = 100, distribution = "multinomial")
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





# stacking

library(MASS)
library(caret)
library(caretEnsemble)
library(doParallel)
nc = detectCores()
registerDoParallel(nc)
nc

n = dim(Boston)[1]
n

set.seed(1)
train = sample(n,n/2)
x_train = Boston[train,-14]
y_train = Boston[train,14]
x_test = Boston[-train,-14]
y_test = Boston[-train,14]

str(Boston)
my_control = trainControl(method = "cv", number = 10, savePredictions = "final", allowParallel = TRUE)
set.seed(2)
model_list = caretList(x_train, y_train, methodList = c("lm","rf","xgbTree","xgbLinear", "svmRadial"), preProcess = c("center","scale"), trControl = my_control,verbosity = 0)

model_list$lm
model_list$rf
model_list$xgbTree
model_list$xgbLinear
model_list$svmRadial

rs = resamples(model_list)
rs$values

dotplot(rs, metric = "RMSE")
modelCor(rs) # model correlations if high then not so much improvement by ensemble


# test RMSE of base learners
p_lm = predict.train(model_list$lm, newdata=x_test)
p_rf = predict.train(model_list$rf, newdata=x_test)
p_xgbTree = predict.train(model_list$xgbTree, newdata=x_test)
p_xgbLinear = predict.train(model_list$xgbLinear, newdata=x_test)
p_svmRadial = predict.train(model_list$svmRadial, newdata=x_test)

RMSE(p_lm,y_test)
RMSE(p_rf,y_test)
RMSE(p_xgbTree,y_test)
RMSE(p_xgbLinear,y_test)
RMSE(p_svmRadial,y_test)


# stacking regression uning a linear meta learner
set.seed(3)
ens1 = caretEnsemble(model_list, metric = "RMSE", trControl = my_control)  
ens1
summary(ens1)
p_ens1 = predict(ens1, newdata= x_test)
p_ens1
results = data.frame(lm = RMSE(p_lm,y_test),
  rf = RMSE(p_rf,y_test),
  xgbTree = RMSE(p_xgbTree,y_test),
  xgbLinear = RMSE(p_xgbLinear,y_test),
  svmRadial = RMSE(p_svmRadial,y_test),
  stacked_regression = RMSE(p_ens1,y_test))
results  

# other meta learners such as elastic net can be used through caretStack
set.seed(3)
ens2 = caretStack(model_list, method = "glmnet", trControl = my_control, metric = "RMSE")
ens2
p_ens2 = predict(ens2, newdata= x_test)
RMSE(p_ens2, y_test)
results$stacking_elastic_net = RMSE(p_ens2, y_test)
results


set.seed(3)
ens3 = caretStack(model_list, method = "rf", trControl = my_control, metric = "RMSE")
ens3
p_ens3 = predict(ens3, newdata= x_test)
RMSE(p_ens3,y_test)



#level-1 data
z1 = model_list$lm$pred[order(model_list$lm$pred$rowIndex),]$pred
z2 = model_list$rf$pred[order(model_list$rf$pred$rowIndex),]$pred
z3 = model_list$xgbTree$pred[order(model_list$xgbTree$pred$rowIndex),]$pred
z4 = model_list$xgbLinear$pred[order(model_list$xgbLinear$pred$rowIndex),]$pred
z5 = model_list$svmRadial$pred[order(model_list$svmRadial$pred$rowIndex),]$pred

z_train = data.frame(z1 = z1, z2 = z2, z3 = z3, z4 = z4, z5 = z5)
head(z_train)
fit_ens1 = lm(y_train~., data = z_train)
summary(fit_ens1)
coef(fit_ens1)
summary(ens1) # 위와 결과 동일!




# stacking for classification with heart data
hdata <- read.csv("C:/Users/qazpl/Desktop/과제/대3/데이터마이닝/Heart.csv", header = TRUE, stringsAsFactors = T)
head(hdata)
hdata$X <- NULL # 인덱스 제거
str(hdata)
sum(is.na(hdata))
hdata <- na.omit(hdata)
n <- dim(hdata)[1]
n

set.seed(1)
train <- sample(n,n/2)
htrain <- hdata[train,]
htest <- hdata[-train,]
dim(hdata)

x.train = htrain[,-14]
x.test = htest[,-14]
y.train = htrain[,14]
y.test = htest[,14]

my.control = trainControl(method = "cv", number = 10, savePredictions = "final", allowParallel = TRUE, classProbs = TRUE)
set.seed(2)
hmodel = caretList(AHD~., data = htrain, methodList = c("rf", "glm","naive_bayes"), metric ="Accuracy", trControl = my.control)
hmodel$rf
hmodel$glm
hmodel$naive_bayes

pc_hrf = predict.train(hmodel$rf, x.test)
pc_hglm = predict.train(hmodel$glm, x.test)
pc_hnb = predict.train(hmodel$naive_bayes, x.test)

acc = function(x,y) mean(x ==y)
acc(pc_hrf, y.test)
acc(pc_hglm, y.test)
acc(pc_hnb, y.test)

head(hmodel$rf$pred)
hens1 = caretEnsemble(hmodel, metric = "Accuracy", trControl = my.control) # stacking 
summary(hens1)

pc_hens1 = predict(hens1, x.test)
acc(pc_hens1, y.test) # stacking model accuracy

res = data.frame(rf = acc(pc_hrf, y.test), glm = acc(pc_hglm, y.test), naive_bayes = acc(pc_hnb, y.test), stacking1 = acc(pc_hens1, y.test))
res

set.seed(2222)
hens2 = caretStack(hmodel, method = "glmnet", metric = "Accuracy", trControl = my.control, tuneLength = 5)
hens2

pc_hens2 = predict(hens2, x.test)
acc(pc_hens2, y.test)
res$stack_elastic_net = acc(pc_hens2, y.test)
res
