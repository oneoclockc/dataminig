# 2.0.1. Exmaple 
library(ISLR)
data(Default)
str(Default)
n<-dim(Default)[1]
set.seed(1)
train<-sample(1:n,n/2)

fit <- glm(default~., data=Default, subset=train, family = binomial)
pred <- predict(fit, Default[-train,], type = "response")
predclass <- rep("No", length(pred))
predclass[pred >= 0.5] <- "Yes"
predclass

library(caret)
confusionMatrix(factor(predclass), Default$default[-train], positive="Yes", mode = "everything")


# 3.4.1. ROC CURVE 
require("ROCR")
cls = c("P","P","N","P","P","P","N","N","P",
        "N","P","N","P","N","N","N","P","N","P","N")
score = c(0.9,0.8,0.7,0.6,0.55,0.54,0.53,0.52,0.52,0.5,0.4,0.39,0.38,0.37,0.36,0.35,0.34,0.3,0.2,0.1)
pred = prediction(score, cls)
roc = performance(pred, "tpr","fpr")

plot(roc, lwd=2, colorize = TRUE)
lines(x=c(0,1), y=c(0,1), col = 'black', lwd=1)

auc = performance(pred, "auc")
auc = unlist(auc@y.values)
auc # auc값 
acc = performance(pred,"acc")

acc@x.values
acc@y.values
ac.val = max(unlist(acc@y.values))
ac.val

th = unlist(acc@x.values)[unlist(acc@y.values) == ac.val] # optimal threshold 대체 뭘 절사한다는거???????
th


# logistic regression example with Default
plot(acc)
abline(v=th,col="grey", lty=2)

library(ISLR)
attach(Default)
str(Default)
dim(Default)

set.seed(1234)
train = sample(1:10000,5000)
test = Default[-train,]
g = glm(default ~ .,data=Default, family=binomial(link=logit), subset = train)
summary(g)


score<- predict(g, newdata=Default, type = "response")[-train] # dafault에 대한 prediction 
cls <- Default[-train,1] # 실제 default
pred = prediction(score, cls) 
roc = performance(pred, "tpr","fpr") 
plot(roc, lwd = 2, colorsize=T)
plot(roc, lwd = 2, col = "darkcyan")
lines(x=c(0,1), y=c(0,1), col = "black", lwd=1)


auc = performance(pred, "auc")
auc = unlist(auc@y.values)
auc

acc = performance(pred,"acc")
ac.val = max(unlist(acc@y.values))
th = unlist(acc@x.values)[unlist(acc@y.values) == ac.val]
th

plot(acc)
abline(v=th, col="grey",lty=2)

# select optimal threshold in k-fold cv

library(ISLR)
n = dim(Default)[1]
set.seed(1)
train = sample(n,n/2)
library(doParallel)
nc = detectCores(nc)
registerDoParallel(nc)
library(caret)

Default$default = relevel(Default$default, ref= "Yes") 
Default$default
# The levels of a factor are re-ordered so that the level specified by 'ref' is first and the others are moved down.
# to set positive="yes"
mycontrol = trainControl(method = "cv", number = 10, classProbs = T, savePredictions = "all", summaryFunction = twoClassSummary)
set.seed(10)
f = train(default~., data=Default[train,], method = "glmnet", trControl = mycontrol, metric = "ROC", preProc = c("center","scale"))
f
# metric = "ROC" -> optimal tuning parameter for maximum AUC

fth_res = thresholder(f, threshold = seq(0.05, 0.95, by=0.05), final = TRUE)
head(fth_res)
fth_res[which.max(fth_res$`Balanced Accuracy`),] # balanced accuracy를 최대로 하는 tuning parameter
fth_res[which.max(fth_res$F1),] # F1를 최대로 하는 tuning parameter



ff = train(default~., data=Default[train,], method = "glmnet", trControl = trainControl(method = "none", classProbs = TRUE,summaryFunction = twoClassSummary), preProc = c("center", "scale"))
ff
preddf = predict(ff, Default[-train,], type = "prob")
ffcls = ifelse(preddf[,1]>fth_res[which.max(fth_res$`Balanced Accuracy`),]$prob_thresh,"Yes","No")
ffcls = factor(ffcls, levels= c("Yes","No"))
# "yes"의 확률이 위에서 구한 절사점보다 큰 경우  "yes"로, 아니면 "no"로 코딩

confusionMatrix(ffcls, Default$default[-train])



# Imbalanced Data
# undersampling, oversampling, and SMOTE on default data

# logistic regression with default data
library(ISLR)
n = dim(Default)[1]
n
sum(Default$default =="Yes")/n # yes 가 0.0333, imbalanced data
set.seed(1234)
train = sample(n,n/2)
test.data = Default[-train,]
g = glm(default~student + income + balance, family = binomial(link=logit), data = Default, subset = train)
summary(g)
pred = predict(g,test.data, type ="response")
pcl = ifelse(pred>0.5,"Yes","No")
head(pcl)
library(caret)
confusionMatrix(factor(pcl), test.data$default, positive = 'Yes') # 실제 no인데 yes로 예측한 게 많음, accuracy 0.0254로 매우 낮음

# undersampling
n1 = sum(Default[train,]$default == "Yes")
n1 # yes는 161개
train.data = Default[train,]
train.data
train.data_no = train.data[train.data$default == "No",]
train.data_no
dim(train.data_no) # no는 4839개


n0 = dim(train.data_no)[1]
n0
set.seed(2)
us = sample(n0,n1) # no에서 161개 추출 (4839 -> 161 undersampling)
us

a = which(train.data$default== "Yes")
a

train.new = train.data[c(a,us),]
g1 = glm(default ~ student+income+balance, family = binomial(lin = logit), data = train.new)
summary(g1)

pred1 = predict(g1, test.data, type = "response")
pcl1 = ifelse(pred1>0.5, "Yes","No")
library(caret)
confusionMatrix(factor(pcl1), test.data$default, positive = "Yes") # accuracy 0.1368로 향상되었으나 여전히 낮음


# oversampling
b = which(train.data$default == "No")
set.seed(3)
n0
os = sample(a,n0, replace = T) # yes에서 4839번 복원추출 (161 -> 4839 oversampling)
train.os = train.data[c(b,os),]
g2 = glm(default ~ student+income+balance, family = binomial(link = logit), data = train.os)
summary(g2)
pred2 = predict(g2, test.data, type = "response")
pcl2 = ifelse(pred2>0.5, "Yes","No")
library(caret)
confusionMatrix(factor(pcl2), test.data$default, positive = "Yes") # accuracy 0.1388로 undersampling보다 낮음



# SMOTE
Defaultn = Default
Defaultn$student <- ifelse(Default$student == "Yes", 1,0)
str(Defaultn)
install.packages("smotefamily")
library(smotefamily)
sum(Defaultn$default[train] == "Yes")
sum(Defaultn$default[train] == "No")
sum(Defaultn$default[train] == "No")/sum(Defaultn$default[train] == "Yes")
set.seed(1234)
strain = SMOTE(Defaultn[train, 2:4], Defaultn[train,1], dup_size = 30)$data 
mean(strain$class == "Yes")

names(strain) = c(names(strain)[1:3], "default")
strain$default = factor(strain$default)
g3 = glm(default~student+income+balance, family = binomial(link = logit), data = strain)
summary(g3)

pred3 = predict(g3,Defaultn[-train, ], type = "response")
pcl3 = ifelse(pred3>0.5, "Yes","No")
library(caret)
confusionMatrix(factor(pcl3), Defaultn[-train, ]$default, positive = "Yes") # accuracy 0.8796


# catet function
# downsampling
set.seed(1)
down_train = downSample(x = Default[train, -1], y = Default[train,"default"])
table(down_train$Class)

# oversampling
up_train = upSample(x = Default[train, -1], y = Default[train,"default"])
table(up_train$Class)

# subsamples inside CV for tuning
# downsampling
dctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, 
                      summaryFunction = twoClassSummary, allowParallel = TRUE, 
                      sampling = "down") # "up","smote" 도 사용 가능
set.seed(1)
f_d <- train(default~., data = Default[train,], method = "glmnet", metric = "ROC", preProc = c("center", "scale"),trControl = dctrl)
f_d

# upsampling 
uctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, 
                      summaryFunction = twoClassSummary, allowParallel = TRUE, 
                      sampling = "up") 
set.seed(1)
f_u <- train(default~., data = Default[train,], method = "glmnet", metric = "ROC", preProc = c("center", "scale"),trControl = uctrl)
f_u


# smote
sctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, 
                      summaryFunction = twoClassSummary, allowParallel = TRUE, 
                      sampling = "smote") 
set.seed(1)
f_s <- train(default~., data = Default[train,], method = "glmnet", metric = "ROC", preProc = c("center", "scale"),trControl = uctrl)
f_s
