library(e1071)
library(rpart)
library(rpart.plot)
library(class)
library(caTools)
library(randomForest)
library(C50)
library(neuralnet)
library(kknn)
library(ElemStatLearn)
library(caret)
library(ROSE)
library(pROC)
library(ROCR)
library(DT)



#Churnset load
Churn <- read.csv("C:/Users/sindh/Desktop/ADM-Project/Churn.csv", header = TRUE)
summary(Churn)
#Data clean
Churn = Churn[,-1]
Churn$TotalCharges[is.na(Churn$TotalCharges)] <- round(mean(Churn$TotalCharges, na.rm = TRUE))
backup <- Churn

# Outliers on Tenure
x <- Churn$tenure
A <- quantile(x, probs=c(.25, .75))
B <- quantile(x, probs=c(.05, .95))
H <- 1.5 * IQR(x)
x[x < (A[1] - H)] <- B[1]
x[x > (A[2] + H)] <- B[2]
boxplot(Churn[,c('tenure')], horizontal=FALSE, axes= TRUE,main = "Tenure BoxPlot", col = "light yellow", notch = TRUE, border = 'red', las =2)


# Outliers on TotalCharges
x <- Churn$TotalCharges
A <- quantile(x, probs=c(.25, .75))
B <- quantile(x, probs=c(.05, .95))
H <- 1.5 * IQR(x)
x[x < (A[1] - H)] <- B[1]
x[x > (A[2] + H)] <- B[2]
boxplot(Churn[,c('TotalCharges')], horizontal=FALSE, axes= TRUE,main = "Total Charges BoxPlot", col = "light yellow", notch = TRUE, border = 'red', las =2)

#Converting variables into factors. 
Churn$gender = as.numeric(factor(Churn$gender, levels = c('Male','Female'), labels = c(0,1)))
Churn$Partner = as.numeric(factor(Churn$Partner, levels = c('Yes','No'), labels = c(0,1)))
Churn$Dependents= as.numeric(factor(Churn$Dependents, levels = c('Yes','No'), labels = c(0,1)))
Churn$PhoneService = as.numeric(factor(Churn$PhoneService, levels = c('Yes','No'), labels = c(0,1)))
Churn$MultipleLines = as.numeric(factor(Churn$MultipleLines, levels = c('Yes','No','No phone service'), labels = c(0,1,2)))
Churn$PaperlessBilling = as.numeric(factor(Churn$PaperlessBilling, levels = c('Yes','No'), labels = c(0,1)))
Churn$InternetService = as.numeric(factor(Churn$InternetService, levels = c('DSL','Fiber optic','No'), labels = c(0,1,2)))
Churn$OnlineSecurity = as.numeric(factor(Churn$OnlineSecurity, levels = c('Yes','No internet service','No'), labels = c(0,1,2)))
Churn$OnlineBackup = as.numeric(factor(Churn$OnlineBackup, levels = c('Yes','No internet service','No'), labels = c(0,1,2)))
Churn$DeviceProtection = as.numeric(factor(Churn$DeviceProtection, levels = c('Yes','No internet service','No'), labels = c(0,1,2)))
Churn$TechSupport = as.numeric(factor(Churn$TechSupport, levels = c('Yes','No internet service','No'), labels = c(0,1,2)))
Churn$StreamingTV = as.numeric(factor(Churn$StreamingTV, levels = c('Yes','No internet service','No'), labels = c(0,1,2)))
Churn$StreamingMovies = as.numeric(factor(Churn$StreamingMovies, levels = c('Yes','No internet service','No'), labels = c(0,1,2)))
Churn$Contract = as.numeric(factor(Churn$Contract, levels = c('Month-to-month','One year','Two year'), labels = c(0,1,2)))
Churn$PaymentMethod = as.numeric(factor(Churn$PaymentMethod, levels = c('Electronic check','Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'), labels = c(0,1,2,3)))
Churn$Churn = factor(Churn$Churn, levels = c('Yes','No'), labels = c(0,1))


#Spliting(75:25) the Churn into test and training  
split <- sample.split(Churn$Churn, SplitRatio = 0.75)
train <- subset(Churn, split == TRUE)
test <- subset(Churn, split == FALSE)
train[-20] = scale(train[-20])
test[-20] = scale(test[-20])

# Top 4 parameters impacting Churn using Random forest  
RF <- randomForest(Churn~., data= Churn, importance=TRUE, ntree=300)
Predict <- predict(RF, test,type ="class")
table(Real=test[,20],Predict)

# Accuracy rate - Random Forest
ACC <- (test[,20]== Predict) *100
RF_accuracy <- sum(ACC)/length(ACC)
RF_accuracy

# Error rate - Random Forest
error<- (test[,20]!=Predict )
errorRate<-sum(error)/length(error)
errorRate 

#Importance of the parameters
importance(RF)
order(importance(RF))
varImpPlot(RF, main = 'Parameters contributing the most to accuracy ofmodel')

#Precision, Recall & F1 Values
precision <- posPredValue(Predict, test[,20], positive="1")
precision
recall <- sensitivity(Predict, test[,20], positive="1")
recall
F1 <- (2 * precision * recall) / (precision + recall)
F1
df1 <- c('Random Forest',RF_accuracy,errorRate,precision,recall,F1)


# Classification models
# C5.0

C50 <- C5.0(Churn~tenure+TotalCharges+Contract+MonthlyCharges,data=train )
C50Predict <-predict(C50,test, type="class")
table(actual=test[,20],C50=C50Predict)

# Accuracy rate - C5.0
ACC <- (test[,20]== C50Predict) *100
c50_accuracy <- sum(ACC)/length(ACC)
c50_accuracy
plot(C50)

# Error rate - C5.0
error<- (test[,20]!=C50Predict)
errorRate<-sum(error)/length(error)
errorRate 

#Calculating Precision, Recall $ F1
precision <- posPredValue(C50Predict, test[,20], positive="1")
recall <- sensitivity(C50Predict, test[,20], positive="1")
F1 <- (2 * precision * recall) / (precision + recall)
precision
recall
F1

df2 <- c('C5.0',c50_accuracy,errorRate,precision,recall,F1)
df2

# Applying NaiveBayes 
NB <- naiveBayes(Churn~tenure+TotalCharges+ MonthlyCharges+Contract, data = train)
NBPredict<- predict(NB,test)
table(NB=NBPredict,Class=test$Churn)


# Accuracy rate - NaiveBayes
ACC <- (test[,20]== NBPredict) *100
NBaccuracy <- sum(ACC)/length(ACC)
NBaccuracy

# Error rate - NaiveBayes
error<- (test[,20]!=NBPredict)
errorRate<-sum(error)/length(error)
errorRate 

#Calculating Precision, Recall $ F1
precision <- posPredValue(NBPredict, test[,20], positive="1")
precision
recall <- sensitivity(NBPredict, test[,20], positive="1")
recall
F1 <- (2 * precision * recall) / (precision + recall)
F1

df3 <- c('NaiveBayes',NBaccuracy,errorRate,precision,recall,F1)
df3

# Applying Decision Tree 
DT <- rpart(Churn~tenure+TotalCharges+ MonthlyCharges+Contract,data =train)
DTPredict <-predict( DT,test, type="class")
table(actual=test[,20],DT=DTPredict)

# Accuracy rate - Decsion Tree
ACC <- (test[,20]== DTPredict) *100
accuracy <- sum(ACC)/length(ACC)
accuracy
prp(DT)
rpart.plot(DT, type = 1)

# Error rate - Desicion Tree
error<- (test[,20]!=DTPredict)
errorRate<-sum(error)/length(error)
errorRate 

#Calculating Precision, Recall $ F1
precision <- posPredValue(DTPredict, test[,20], positive="1")
precision
recall <- sensitivity(DTPredict, test[,20], positive="1")
recall
F1 <- (2 * precision * recall) / (precision + recall)
F1

df4 <- c('Decision Tree',accuracy,errorRate,precision,recall,F1)
df4

# SVM
SVM = svm(formula = Churn~tenure+TotalCharges+ MonthlyCharges + Contract, data =train, type = 'C-classification', kernel = 'linear')
Predict <- predict(SVM, test)
table(actual=test[,20],Predict)

# Accuracy rate - SVM
ACC <- (test[,20]== Predict) *100
accuracy <- sum(ACC)/length(ACC)
accuracy

# Error rate - SVM
error<- (test[,20]!=Predict )
errorRate<-sum(error)/length(error)
errorRate

#Calculating Precision, Recall $ F1
precision <- posPredValue(Predict, test[,20], positive="1")
precision
recall <- sensitivity(Predict, test[,20], positive="1")
recall
F1 <- (2 * precision * recall) / (precision + recall)
F1

df5 <- c('SVM',accuracy,errorRate,precision,recall,F1)
df5

#KNN
# When K = 1
KNN1 <- knn(train[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],test[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],train[,20],k=1)
confusionMatrix <- table(Prediction=KNN1,Actual=test[,20])
confusionMatrix
ACC <- sum(test[,20]==KNN1)/nrow(test)*100 
ACC
#Calculating Precision, Recall $ F1
precision <- posPredValue(KNN1, test[,20], positive="1")
precision
recall <- sensitivity(KNN1, test[,20], positive="1")
recall
F1 <- (2 * precision * recall) / (precision + recall)
F1
KNN11 <- c('KNN1',ACC,errorRate,precision,recall,F1)
KNN11
# When K = 5
KNN5 <- knn(train[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],test[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],train[,20],k=5)
confusionMatrix5 <- table(Prediction=KNN5,Actual=test[,20])
confusionMatrix5
ACC5 <- sum(test[,20]==KNN5)/nrow(test)*100
ACC5

#Calculating Precision, Recall $ F1
precision <- posPredValue(KNN5, test[,20], positive="1")
precision
recall <- sensitivity(KNN5, test[,20], positive="1")
recall
F1 <- (2 * precision * recall) / (precision + recall)
F1
KNN12 <- c('KNN5',ACC5,errorRate,precision,recall,F1)

# When K = 10
KNN10 <- knn(train[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],test[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],train[,20],k=10)
confusionMatrix10 <- table(Prediction=KNN10,Actual=test[,20])
confusionMatrix10
ACC10 <- sum(test[,20]==KNN10)/nrow(test)*100
ACC10
#Calculating Precision, Recall $ F1
precision <- posPredValue(KNN10, test[,20], positive="1")
precision
recall <- sensitivity(KNN10, test[,20], positive="1")
recall
F1 <- (2 * precision * recall) / (precision + recall)
F1
KNN13 <- c('KNN10',ACC10,errorRate,precision,recall,F1)

# When K = 30
KNN30 <- knn(train[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],test[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')],train[,20],k=30)
confusionMatrix30 <- table(Prediction=KNN30,Actual=test[,20])
confusionMatrix30
ACC30 <- sum(test[,20]==KNN30)/nrow(test)*100
ACC30
#Calculating Precision, Recall $ F1
precision <- posPredValue(KNN30, test[,20], positive="1")
precision
recall <- sensitivity(KNN30, test[,20], positive="1")
recall
F1 <- (2 * precision * recall) / (precision + recall)
F1
KNN30 <- c('KNN30',ACC30,errorRate,precision,recall,F1)
KNNTable <- rbind(KNN11,KNN12,KNN13,KNN30)
colnames(KNNTable) <- c('K Model Value','Accuracy','Error','Precision','Recall','F1')
datatable(KNNTable)

#Comparing ROC of models
# Plotting the ROC curve - RF
PredictProb_RF <- predict(RF, test, type = "prob")
auc <- auc(test$Churn,PredictProb_RF[,2])
plot(roc(test$Churn,PredictProb_RF[,2]),colorize = TRUE,col = "red")

# Plotting the ROC curve - DT
PredictProb_DT <- predict(DT, test, type = "prob") # Using probability
auc <- auc(test$Churn,PredictProb_DT[,2])
auc
plot(roc(test$Churn,PredictProb_DT[,2]), col = "yellow")

# Plotting the ROC curve - C5.0
PredictProb_c50 <- predict(C50, test, type = "prob") 
auc <- auc(test$Churn,PredictProb_c50[,1])
auc
plot(roc(test$Churn,PredictProb_c50[,1]))

# Plotting the ROC curve - Multiple thresholds TPR & FPR
PredictProb_NB <- predict(NB, test, type = "raw") # Using probability
auc <- auc(test$Churn,PredictProb_NB[,2])
plot(roc(test$Churn,PredictProb_nb[,2]),col = "green")

# List of predictions
plot(roc(test$Churn,PredictProb_RF[,2]),colorize = TRUE,col = "Gray", main = "Comparing RF, CART, C50 & NB")
plot(roc(test$Churn,PredictProb_DT[,1]), add = TRUE, colorize = TRUE, col = "Red")
plot(roc(test$Churn,PredictProb_c50[,1]), add = TRUE, colorize = TRUE, col = "Black")
plot(roc(test$Churn,PredictProb_NB[,2]), add = TRUE, colorize = TRUE,col = "blue")


#ANN
NN  <- neuralnet(Churn~ tenure + TotalCharges + MonthlyCharges + Contract,train, hidden=8, threshold=0.10, stepmax = 1e6)
NN$result.matrix
plot(NN)
#Prediction using neural network 
NNResults <-compute(nn, test[,c('tenure','TotalCharges', 'MonthlyCharges','Contract')])
NNResults
ANN=as.numeric(NNResults$net.result)
ANN
# Rounding the Generated values 
ANN1<-round(ANN)
# Confusion Matrix
ANN_cat<-ifelse(ANN<1.5,1,2)
table(Actual=test$Churn,ANN1)
# Finding the Accuracy rate
ACC<- (test$Churn!=ANN_cat)
error
errorRate<-sum(error)/length(error)
errorRate
accuracy <- 1 - errorRate
accuracy

# ROC curve 
detach(package:neuralnet,unload = T)
NN.pred = prediction(ANN, test$Churn)
pref <- performance(NN.pred, "tpr", "fpr")
plot(pref,col = "red", main = "Neural Network curve")

#Visualizing KNN table:
library(DT)
CC <- rbind(df1,df2,df3,df4,df5)
CC <- data.frame(CC)
colnames(CC) <- c('Model','Accuracy','Error-Rate','Precision','Recall','F1')
CC
datatable(CC)
