library(pastecs)
library(rpart)
library(e1071)
library(adabag)
library(randomForest)
library(ROCR)
library(data.table)
library(mltools)

rm(list = ls())
WAUS = read.csv("WAUS2020.csv")
L = as.data.frame(c(1:49))
set.seed(30241510) # Your Student ID is the random seed
L = L[sample(nrow(L), 10, replace = FALSE),] # sample 10 locations
WAUS = WAUS[(WAUS$Location %in% L),]
WAUS = WAUS[sample(nrow(WAUS), 2000, replace = FALSE),] # sample 2000 rows

##########First Step##########
#Cleaning the data from NA value
WAUS = na.omit(WAUS)

#Set RainTomorrow column as factor
WAUS$RainTomorrow = as.factor(WAUS$RainTomorrow)
str(WAUS)

##########Question 1##########
# a. What is the proportion of rainy days to fine days
# analysing number of yes and no under RainToday
raindays_proportion = as.data.frame(table(WAUS$RainTomorrow)) # No = 560; Yes = 136
fine_days = round((raindays_proportion[1,2]/sum(raindays_proportion[,2]))*100, digits = 2)
rainy_days = round((raindays_proportion[2,2]/sum(raindays_proportion[,2]))*100, digits = 2)
cat("Proportion of fine days: ",fine_days) #80.46
cat("Proportion of rainy days: ",rainy_days) #19.54

# b. descriptions of the predictor (independent) variables
data_summary = as.data.frame(round(stat.desc(WAUS[,-c(1,2,3,4,10,12,13,24,25)]), 2))
data_summary = t(data_summary[c(3,4,5,8,9,12,13),])
##########Question 2##########
#Question 2 has been done before doing question 1. I remove the NA value because
#it is unecessary to be included. I have also change RainTomorrow as factor

##########Question 3##########
#Divide data into 70% training and 30% test
set.seed(30241510)
train.row = sample(1:nrow(WAUS), 0.7*nrow(WAUS))
train.data = WAUS[train.row,]
test.data = WAUS[-train.row,]

##########Question 4##########
#Implement a classification model using each of the following techniques
#Decision Tree
tree.waus = rpart(RainTomorrow~., data = train.data, method = "class")
tree.pred = predict(tree.waus, test.data, type = "class")

#Naïve Bayes
nb.waus = naiveBayes(RainTomorrow~., data = train.data)
nb.pred = predict(nb.waus, test.data)

#Bagging
ba.waus = bagging(RainTomorrow~., data = train.data, mfinal = 10)
ba.pred = predict.bagging(ba.waus, newdata = test.data)

#Boosting
bo.waus = boosting(RainTomorrow~., data = train.data, mfinal = 10)
bo.pred = predict.boosting(bo.waus, newdata = test.data)

#Random Forest
rf.waus = randomForest(RainTomorrow~., data = train.data)
rf.pred = predict(rf.waus, test.data)

##########Question 5##########
#Create confusion matrix and calculate accuracy
#Decision Tree - should I do cv?
tree.cm = table(actual = test.data$RainTomorrow, predicted = tree.pred)
tree.cm.accuracy = round(mean(tree.pred == test.data$RainTomorrow)*100, digits = 2)
cat("Accuracy of Decision Tree:",tree.cm.accuracy,"%")

#Naïve Bayes
nb.cm = table(actual = test.data$RainTomorrow, predicted = nb.pred)
nb.cm.accuracy = round(mean(nb.pred == test.data$RainTomorrow)*100, digits = 2)
cat("Accuracy of Naïve Bayes is:", nb.cm.accuracy,"%")
  
#Bagging
ba.cm = ba.pred$confusion
ba.cm.accuracy = round(mean(ba.pred$class == test.data$RainTomorrow)*100, digits = 2)
cat("Accuracy of Bagging is:", ba.cm.accuracy,"%")

#Boosting
bo.cm = bo.pred$confusion
bo.cm.accuracy = round(mean(bo.pred$class == test.data$RainTomorrow)*100, digits = 2)
cat("Accuracy of Boosting is:", bo.cm.accuracy,"%")

#Random Forest
rf.cm = table(actual = test.data$RainTomorrow, predicted = rf.pred)
rf.cm.accuracy = round(mean(rf.pred == test.data$RainTomorrow)*100, digits = 2)
cat("Accuracy of Random Forest is:", rf.cm.accuracy,"%")

##########Question 6##########
#Calculate confidence, construct ROC curve

#Decision tree
tree.waus.vector = predict(tree.waus, test.data, type = "prob")
tree.waus.pred = prediction(tree.waus.vector[,2], test.data$RainTomorrow)
tree.waus.perf = performance(tree.waus.pred,"tpr","fpr")
plot(tree.waus.perf, main = "ROC Curve of Classifiers")
abline(0,1)

#Naive Bayes
nb.waus.raw = predict(nb.waus, test.data, type = 'raw')
nb.waus.pred = prediction(nb.waus.raw[,2], test.data$RainTomorrow)
nb.waus.perf = performance(nb.waus.pred, "tpr", "fpr")
plot(nb.waus.perf, add = TRUE, col = "blueviolet")

#Bagging
ba.waus.pred = prediction(ba.pred$prob[,2], test.data$RainTomorrow)
ba.waus.perf = performance(ba.waus.pred, "tpr","fpr")
plot(ba.waus.perf, add = TRUE, col = "green")

#Boosting
bo.waus.pred = prediction(bo.pred$prob[,2], test.data$RainTomorrow)
bo.waus.perf = performance(bo.waus.pred,"tpr","fpr")
plot(bo.waus.perf, add = TRUE, col = "cyan")

#Random Forest
rf.waus.prob = predict(rf.waus, test.data, type = "prob")
rf.waus.pred = prediction(rf.waus.prob[,2], test.data$RainTomorrow)
rf.waus.perf = performance(rf.waus.pred, "tpr","fpr")
plot(rf.waus.perf, add = TRUE, col = "red")

#add legend to plot
legend("topleft", legend= c("Decision Tree","Naïve Bayes","Bagging","Boosting","Random Forest"),
       col = c("black","blueviolet","green","cyan","red"), lty=1, cex=0.8)


#Calculate AUC
#Decision Tree
tree.waus.auc = performance(tree.waus.pred,"auc")
tree.waus.auc = round(tree.waus.auc@y.values[[1]],2)
cat("Area Under the Curve (AUC) of Decision Tree is:", tree.waus.auc)

#Naive Baiyes
nb.waus.auc = performance(nb.waus.pred,"auc")
nb.waus.auc = round(nb.waus.auc@y.values[[1]],2)
cat("Area Under the Curve (AUC) of Naïve Bayes is:", nb.waus.auc)

#Bagging
ba.waus.auc = performance(ba.waus.pred, "auc")
ba.waus.auc = round(ba.waus.auc@y.values[[1]],2)
cat("Area Under the Curve (AUC) of Bagging is:", ba.waus.auc)

#Boosting
bo.waus.auc = performance(bo.waus.pred, "auc")
bo.waus.auc = round(bo.waus.auc@y.values[[1]],2)
cat("Area Under the Curve (AUC) of Boosting is:", bo.waus.auc)

#Random Forest
rf.waus.auc = performance(rf.waus.pred, "auc")
rf.waus.auc = round(rf.waus.auc@y.values[[1]],2)
cat("Area Under the Curve (AUC) of Random Forest is:", rf.waus.auc)

##########Question 7##########
#Calculate RMSE as another indicator
#Decision Tree
tree.rmse = round(sqrt(mean((as.numeric(tree.pred) - as.numeric(test.data$RainTomorrow))^2)),2)

#Naive Bayes
nb.rmse = round(sqrt(mean((as.numeric(nb.pred) - as.numeric(test.data$RainTomorrow))^2)),2)

#Bagging
ba.rmse = round(sqrt(mean((ifelse(ba.pred$class == "Yes",2,1) - as.numeric(test.data$RainTomorrow))^2)),2)

#Boosting
bo.rmse = round(sqrt(mean((ifelse(bo.pred$class == "Yes",2,1) - as.numeric(test.data$RainTomorrow))^2)),2)

#Random Forest
rf.rmse = round(sqrt(mean((as.numeric(rf.pred) - as.numeric(test.data$RainTomorrow))^2)),2)

classifiers.data  = data.frame(Classifiers = c("Decision Tree","Naïve Bayes","Bagging","Boosting","Random Forest"),
                   Accuracy = c(tree.cm.accuracy,nb.cm.accuracy,ba.cm.accuracy,bo.cm.accuracy,rf.cm.accuracy),
                   AUC = c(tree.waus.auc, nb.waus.auc, ba.waus.auc,  bo.waus.auc, rf.waus.auc),
                   RMSE = c(tree.rmse, nb.rmse, ba.rmse, bo.rmse, rf.rmse))

classifiers.data.accuracy = classifiers.data [order(-classifiers.data$Accuracy),]
classifiers.data.auc = classifiers.data [order(-classifiers.data$AUC),]
classifiers.data.rmse = classifiers.data [order(classifiers.data$RMSE),]

##########Question 8##########
#DecisionTree
tree.waus$variable.importance

#Naive Bayes - cannot tell

#Bagging
print(ba.waus$importance) #16 is used out of 24

#Boosting
print(bo.waus$importance) #20 is used out of 24

#RandomForest
print(rf.waus$importance)

#try removing RainToday from train and test data
train.data.rt = train.data[,-24]
test.data.rt = test.data[,-24]

#fit new data to all classifiers except Naive Bayes and calculate their accuracy
#Decision Tree
tree.waus.rt = rpart(RainTomorrow~., data = train.data.rt, method = "class")
tree.pred.rt = predict(tree.waus.rt, test.data.rt, type = "class")
tree.acc.rt = round(mean(tree.pred.rt == test.data.rt$RainTomorrow)*100, digits = 2)

#Bagging
ba.waus.rt = bagging(RainTomorrow~., data = train.data.rt, mfinal = 10)
ba.pred.rt = predict.bagging(ba.waus.rt, newdata = test.data.rt)
ba.acc.rt = round(mean(ba.pred.rt$class == test.data.rt$RainTomorrow)*100, digits = 2)

#Boosting
bo.waus.rt = boosting(RainTomorrow~., data = train.data.rt, mfinal = 10)
bo.pred.rt = predict.boosting(bo.waus.rt, newdata = test.data.rt)
bo.acc.rt = round(mean(bo.pred.rt$class == test.data.rt$RainTomorrow)*100, digits = 2)

#Random Forest
rf.waus.rt = randomForest(RainTomorrow~., data = train.data.rt)
rf.pred.rt = predict(rf.waus.rt, test.data.rt)
rf.acc.rt = round(mean(rf.pred.rt == test.data.rt$RainTomorrow)*100, digits = 2)

#Create comparison accuracy table
comparison_rt = data.frame (Classifiers = c("Previous Accuracy", "Current Accuracy"),
                            Decision.Tree = c(tree.cm.accuracy, tree.acc.rt),
                            Bagging = c(ba.cm.accuracy, ba.acc.rt),
                            Boosting = c(bo.cm.accuracy, bo.acc.rt),
                            Random.Forest = c(rf.cm.accuracy, rf.acc.rt))

##########Question 9##########
#Since Random Forest has the highest accuracy and AUC, and the lowest RMSE, we take random forest
#classifiers as our tree-based classifier. We will be trying to improve the model by
#1 - setting importance parameter to TRUE
#2 - increasing ntree of random forest
#3 - searching the best number of split

#Looking for better ntree
num.tree = c(550, 600, 650, 700, 750, 800, 850, 900, 950, 1000)
best.rf.acc = rf.cm.accuracy
best.rf = rf.waus
best.rf.pred = rf.pred

for (i in 501:1000){
  set.seed(30241510)
  rf.waus.new = randomForest(RainTomorrow~., data = train.data, importance = TRUE, ntree = i)
  rf.pred.new = predict(rf.waus.new, test.data)
  acc = round(mean(rf.pred.new == test.data$RainTomorrow)*100, digits = 2)
  if (acc > best.rf.acc){
    best.rf.acc = acc
    best.rf = rf.waus.new
    best.rf.pred = rf.pred.new }
}
cat("best number of tree is", best.rf$ntree, "with accuracy of", best.rf.acc,"%")

#looking for better mtry starting from minimum split + 1 to the number of predictors

for (i in (best.rf$mtry+1):24){
  set.seed(30241510)
  rf.mtry = randomForest(RainTomorrow~., data = train.data, importance = TRUE, ntree = best.rf$ntree, mtry = i)
  rf.mtry.pred = predict(rf.mtry, test.data)
  acc = round(mean(rf.mtry.pred == test.data$RainTomorrow)*100, digits = 2)
  if (acc > best.rf.acc){
    best.rf.acc = acc
    best.rf = rf.mtry
    best.rf.pred = rf.mtry.pred
  }
  
}
cat("best number of split is", best.rf$mtry, "with accuracy of", best.rf.acc,"%")

#Calculating RMSE of better random forest
best.rf.rmse = round(sqrt(mean((as.numeric(best.rf.pred) - as.numeric(test.data$RainTomorrow))^2)),2)

#Creating data frame to compare previous and current model
rf.comparison = data.frame(Classifier = c("Random.Forest.1", "Random.Forest.2"),
                           ntree = c(rf.waus$ntree, best.rf$ntree),
                           mtry = c(rf.waus$mtry, best.rf$mtry),
                           Accuracy = c(rf.cm.accuracy, best.rf.acc),
                           RMSE = c(rf.rmse, rf.rmse.rt))

##########Question 10##########
#implements ANN - we will include all attributes and
#we will first do one hot encoding for non-numerical data and change
#the value of "Yes" to 1 and "No" to 0 for RainToday and RainTomorrow

library(neuralnet)
train.data[,24:25] = ifelse(train.data[,24:25] == "Yes", 1, 0)
test.data[,24:25] = ifelse(test.data[,24:25] == "Yes", 1, 0)

train.data = one_hot(as.data.table(train.data))
test.data = one_hot(as.data.table(test.data))

#Normalised the data with min max normalisation
normalise = function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

train.data = as.data.frame(lapply(train.data, FUN = normalise))
test.data = as.data.frame(lapply(test.data, normalise))

predictors_name = colnames(train.data)
f = as.formula(paste("RainTomorrow ~",paste(predictors_name[!predictors_name %in% "RainTomorrow"], collapse = " + ")))
waus.nn = neuralnet(f, train.data, hidden = c(10,4,2), linear.output = FALSE)
waus.nn.pred = compute(waus.nn, test.data[,1:69])
to_compare = round(waus.nn.pred$net.result, 0)
round(mean(to_compare == test.data$RainTomorrow)*100, digits = 2)

#Create a for loop to know the best number of hidden layer(s)
nn.acc = 0
hidden.layer = 0
best.nn = 0
for (i in 1:23){
  set.seed(30241510)
  waus.nn = neuralnet(f, train.data, hidden = i, linear.output = FALSE)
  waus.nn.pred = compute(waus.nn, test.data[,1:69])
  waus.nn.pred = as.data.frame(round(waus.nn.pred$net.result,0))
  current_acc = round(mean(waus.nn.pred$V1 == test.data$RainTomorrow)*100, digits = 2)
  if (current_acc > nn.acc){
    nn.acc = current_acc
    hidden.layer = i
    best.nn = waus.nn
  }
}
cat("Accuracy of ANN: ", nn.acc, "%\n")
cat("Number of hidden layer(s): ", hidden.layer)