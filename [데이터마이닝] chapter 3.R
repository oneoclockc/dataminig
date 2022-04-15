# CHAPTER 3. Classification

# 5.1.1. Bayes classifier

head(iris)
str(iris)
n = dim(iris)[1]
set.seed(1)
train <- sample(1:n, n/2)
iris.train <- iris[train,]
iris.test <- iris[-train,]

# bayes' classifier under multivariate normal distribution can be simply performed by qda fun in MASS package.

library(MASS)
fitq<- qda(Species ~ ., data = iris)
predq <-predict(fitq, newdata = iris.test)

predq$class

mean(predq$class != iris.test$Species)


# discretized iris data

n<- dim(iris)[1]
set.seed(1)
train <-sample(1:n, n/2)

fsl <- rep("VL", n) # sepal length very long (6.4~)
fsl[iris[,1]<=6.4] <- "L" # long (3.8~6.4]
fsl[iris[,1]<=5.8] <- "S" # short (5.1~5.8]
fsl[iris[,1]<=5.1] <- "VS" # very short (~5.1]
fsl <- factor(fsl) # vector -> factor


fsw <- rep("L", n) # sepal width long (3.3~)
fsw[iris[,2]<=3.3] <- "M" # long (2.8~3.3]
fsw[iris[,2]<=2.8] <- "S" # short (~2.8]
fsw <-factor(fsw)
fsw

irisf.train <-data.frame(fsl = fsl[train], fsw = fsw[train], Y=iris[train,5])
head(irisf.train)
irisf.test <-data.frame(fsl = fsl[-train], fsw = fsw[-train], Y=iris[-train,5])

library(nnet)
fitm <- multinom(Y~fsl*fsw, data=irisf.train)

predm <- predict(fitm,newdata = irisf.test)
predm
mean(predm != irisf.test$Y)




# 5.2.3. iris data : LDA and QDA

# use Sepal length and Sepat width to classify species

library(MASS)
iris12 <- iris[c(1,2,5)]
head(iris12)
attach(iris12)
plot(iris12, col = iris12[,3])
plot(Sepal.Length, Sepal.Width, col=iris12[,3])


n<-dim(iris12)[1]
set.seed(1)
train <- sample(n,n/2)

# LDA 
fit <- lda(Species~., data= iris12, subset = train)
pred.lda <- predict(fit, iris12[-train,])$class
pred.lda
table(pred.lda, iris12[-train,3])
mean(pred.lda != iris12[-train,3])


# QDA
fit2 <- qda(Species~., data=iris12, subset=train)
pred.qda <- predict(fit2, iris12[-train,])$class
pred.qda
table(pred.qda, iris12[-train,3])
mean(pred.qda != iris12[-train,3])



# model selection
# LOOCV 
fitloocv<-lda(Species~., data=iris12[train,], cv=T)
predloocv<-predict(fitloocv, iris12[train,])$class
table(predloocv, iris12[train,3])
mean(predloocv!=iris12[train,3])

mean(predloocv!=iris12[train,3])
fit2loocv<-qda(Species~., data=iris12[train,], cv=T)
pred2loocv<-predict(fit2loocv,iris12[train,])$class
table(pred2loocv, iris12[train,3])
mean(pred2loocv!=iris12[train,3])


# model selection and model assessment based on k-fold cv
iris12.train <-iris12[train,]
library(caret)
set.seed(10)
fold=createFolds(iris12.train[,3],k=10, returnTrain = T)
myControl = trainControl(method="cv", number = 10, allowParallel = T, index=fold)
fit.lda=train(iris12.train[,c(1,2)], iris12.train[,3],method="lda", trControl=myControl)
fit.qda=train(iris12.train[,c(1,2)], iris12.train[,3],method="qda", trControl=myControl)
fit.lda
fit.qda

# lda is better
fit <- lda(Spcieds~., data=iris12.train)
pred<-predict(fit, newdata=iris12[-train,])$class
table(pred,iris12[-train,3])
mean(pred!=iris12[-train,3])
