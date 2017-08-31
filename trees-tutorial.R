library(rpart)
library(caret)

# load some data
mydata <- read.table("data/aloedichotoma.csv",sep=",",header=T)
mydata <- mydata[complete.cases(mydata),]

# set response variable
yf <- log(mydata$tottrees)
# set predictor variables
xf <- mydata[,-c(1)]

# make training and test datasets (70% in training, 30% in test)
train_id = sample(1:length(yf),0.7*length(yf),replace=F)
xtrain <- data.frame(y=yf[train_id],x=xf[train_id,])
xtest <- data.frame(y=yf[-train_id],x=xf[-train_id,])

### fit classification tree with rpart

# fit full tree
fit <- rpart(y ~ ., xtrain)

# some results
printcp(fit) # display the results
plotcp(fit) # visualize cross-validation results
summary(fit) # detailed summary of splits

# plot full tree
plot(fit)
text(fit, use.n=TRUE, all=TRUE, cex=.8)

# accuracy in training dataset
fittedtrain <- predict(fit,type="class")
predtrain <- table(xtrain$y,fittedtrain)
predtrain
sum(diag(predtrain))/sum(predtrain) # training accuracy

# accuracy in test dataset
fittedtest <- predict(fit,newdata=xtest,type="class")
predtest <- table(xtest$y,fittedtest)
predtest
sum(diag(predtest))/sum(predtest) # test accuracy
# training accuracy >> test accuracy => over-fitting

# prune full tree
pfit<- prune(fit, cp = fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])

# plot pruned tree
plot(pfit, main="Pruned Classification Tree")
text(pfit, use.n=TRUE, all=TRUE, cex=.8)

# accuracy in training dataset
fittedtrain <- predict(pfit,type="class")
predtrain <- table(xtrain$y,fittedtrain)
predtrain
sum(diag(predtrain))/sum(predtrain) # training accuracy

# accuracy in test dataset
fittedtest <- predict(pfit,newdata=xtest,type="class")
predtest <- table(xtest$y,fittedtest)
predtest
sum(diag(predtest))/sum(predtest) # test accuracy




### fit classification tree with caret

# fit tree with 10-fold CV
ctrl <- trainControl(method = "cv", number = 10, savePred=T, classProb=F)
mod <- train(factor(y) ~ ., 
             data=xtrain, 
             method = "rpart",
             tuneLength = 10,
             trControl = ctrl)

# show model results
mod

# plot best (via cross-validation) classification tree
plot(mod$finalModel)
text(mod$finalModel, use.n=TRUE, all=TRUE, cex=.8)

# get cross-validated accuracy
max(mod$results$Accuracy)

# accuracy in training dataset
fittedtrain <- predict(mod)
predtrain <- table(mod$finalModel$y,fittedtrain)
predtrain
sum(diag(predtrain))/sum(predtrain) # training accuracy

# accuracy in test dataset
fittedtest <- predict(mod$finalModel,newdata=xtest,type="class")
predtest <- table(xtest$y,fittedtest)
predtest
sum(diag(predtest))/sum(predtest) # test accuracy

### fit random forest with caret

# fit tree with 10-fold CV
ctrl <- trainControl(method = "cv", number = 10, savePred=T, classProb=F)
mod <- train(factor(y) ~ ., 
             data=xtrain, 
             method = "rf", 
             trControl = ctrl)

# variable importance (no tree plot any more)
varImpPlot(mod$finalModel)

# accuracy in training dataset
predtrain <- table(mod$finalModel$y,mod$finalModel$predicted)
predtrain
sum(diag(predtrain))/sum(predtrain) # training accuracy

# accuracy in test dataset
fittedtest <- predict(mod$finalModel,newdata=xtest,type="class")
predtest <- table(xtest$y,fittedtest)
predtest
sum(diag(predtest))/sum(predtest) # test accuracy

