
trainset <- read.csv("titanic-train.csv")
testset <- read.csv("titanic-test.csv")

trainset$Survived=factor(trainset$Survived)
trainset$Pclass=ordered(trainset$Pclass)
testset$Survived=factor(testset$Survived)
testset$Pclass=ordered(testset$Pclass)

library(RWeka)
MS <- make_Weka_filter("weka/filters/unsupervised/attribute/ReplaceMissingValues") 
trainset_nomissing <-MS(data=trainset, na.action = NULL)
testset_nomissing <-MS(data=testset, na.action = NULL)


library("RWeka")
InfoGainAttributeEval(Survived ~ . , data = trainset_nomissing)

myVars=c("Pclass", "Sex", "Age", "SibSp", "Fare", "Survived")
newtrain=trainset_nomissing[myVars]
newtest=testset_nomissing[myVars]
str(newtrain)
str(newtest)




install.packages("infotheo")
library(infotheo)
data <- rbind(newtrain, newtest) 
dData <- discretize(data[, 2:4], disc = "equalwidth", nbins=10)
dData <- lapply(dData, as.factor)
dData <- cbind(data[, c(1,6)], dData)
dlabel <- data$Survived
dData <- cbind(dData, dlabel)
# separate train (1-891) and test
train_index <- 1:891
train1<- dData[train_index,]
test1<- dData[-train_index,]


# train and test naive Bayes model



# kNN in the "class" package


install.packages("class")
library(class)
train_labels = newtrain$Survived
sex=as.numeric(newtrain$Sex)
pclass=as.numeric(newtrain$Pclass)
dtrain=cbind(sex, newtrain[, c(3,4)] )
dtrain=cbind(dtrain, pclass)

sex=as.numeric(newtest$Sex)
pclass=as.numeric(newtest$Pclass)
dtest=cbind(sex, newtest[, c(3,4)] )
dtest=cbind(dtest, pclass)



install.packages("e1071")
library("e1071")
svm<- svm(Survived~., data = newtrain)
pred=predict(svm, newdata=newtest, type=c("class"))
myids=c("PassengerId")
id_col=testset[myids]
newpred=cbind(id_col, pred)
colnames(newpred)=c("Passengerid", "Survived")



install.packages("randomForest")
library(randomForest)
rfm <- randomForest(Survived~., data=newtrain, ntree=5)
print(rfm)
predRF <- predict(rfm, newtest, type=c("class"))
myids=c("PassengerId")
id_col=testset[myids]
newpred=cbind(id_col, pred)
colnames(newpred)=c("Passengerid", "Survived")


rfm1 <- randomForest(Survived~., data=newtrain, ntree=10)
print(rfm1)
predRF1 <- predict(rfm1, newtest, type=c("class"))
myids=c("PassengerId")
id_col=testset[myids]
newpred1=cbind(id_col, pred)
colnames(newpred1)=c("Passengerid", "Survived")


rfm2 <- randomForest(Survived~., data=newtrain, ntree=25)
print(rfm2)
predRF2 <- predict(rfm2, newtest, type=c("class"))
myids=c("PassengerId")
id_col=testset[myids]
newpred2=cbind(id_col, pred)
colnames(newpred2)=c("Passengerid", "Survived")


rfm3 <- randomForest(Survived~., data=newtrain, ntree=50)
print(rfm3)
predRF3 <- predict(rfm3, newtest, type=c("class"))
myids=c("PassengerId")
id_col=testset[myids]
newpred3=cbind(id_col, pred)
colnames(newpred3)=c("Passengerid", "Survived")


rfm4 <- randomForest(Survived~., data=newtrain, ntree=100)
print(rfm4)
predRF4 <- predict(rfm4, newtest, type=c("class"))
myids=c("PassengerId")
id_col=testset[myids]
newpred4=cbind(id_col, pred4)
colnames(newpred4)=c("Passengerid", "Survived")

