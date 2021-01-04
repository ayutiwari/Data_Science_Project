# Project Name : Diabetes Risk Prediction - Medical Lab Data Analysis
# Student : Ayu Tiwari
# Course : HarvardX-Data Science: Capstone
# Date of submission : 12/28/2020
############################################################

##############################
# installing required packages
##############################

if(!require(tidyverse))    install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret))        install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggthemes))     install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(ggcorrplot))   install.packages("ggcorrplot", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(magrittr))     install.packages("magrittr", repos = "http://cran.us.r-project.org")
if(!require(rpart))        install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot))   install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(neighbr))      install.packages("neighbr", repos = "https://cran.rstudio.com/bin/macosx/contrib/4.0/neighbr_1.0.3.tgz")

###################################
# DATA SET
###################################
install.packages("neighbr")
library(tidyverse)
library(caret)
library(ROCR)
library(class)
library(rpart)
library(rpart.plot)
library(randomForest)
library(neighbr)

#read data from csv file and replace spaces from column name
data <- read_csv("diabetes_data_upload.csv", col_types = "dffffffffffffffff")
colnames(data) <- make.names(colnames(data))

#rearrange column values
col <- which(data[1,]=="No")
for (i in col)
{
  data[,i] <- factor(data[[i]], levels = c("Yes","No"))
}

#Data set structure
str(data)

#Data Dimensions
dim(data)

#display first 6 records
head(data)

##########################################
# Data Exploration and Transformation
##########################################

#Call preprocess function to convert age column values
temp_df <- preProcess(data, method="range")
dia_df <- predict(temp_df, newdata = data)

#convert class values to Positive = 1 and Negative = 0
levels(dia_df$class) <-  c(1,0)

#convert Gender Male=1 and Female=0
dia_df$Gender <- ifelse(dia_df$Gender== "Male", 1, 0)

#Convert other columns Yes=1 and No=0
dia_df$Polyuria <- ifelse(dia_df$Polyuria == "Yes", 1, 0)
dia_df$Polydipsia <- ifelse(dia_df$Polydipsia == "Yes", 1, 0)
dia_df$sudden.weight.loss <- ifelse(dia_df$sudden.weight.loss == "Yes", 1, 0)
dia_df$weakness <- ifelse(dia_df$weakness == "Yes", 1, 0)
dia_df$Polyphagia <- ifelse(dia_df$Polyphagia == "Yes", 1, 0)
dia_df$Genital.thrush <- ifelse(dia_df$Genital.thrush == "Yes", 1, 0)
dia_df$visual.blurring <- ifelse(dia_df$visual.blurring == "Yes", 1, 0)
dia_df$Itching <- ifelse(dia_df$Itching == "Yes", 1, 0)
dia_df$Irritability <- ifelse(dia_df$Irritability == "Yes", 1, 0)
dia_df$delayed.healing <- ifelse(dia_df$delayed.healing == "Yes", 1, 0)
dia_df$partial.paresis <- ifelse(dia_df$partial.paresis == "Yes", 1, 0)
dia_df$muscle.stiffness <- ifelse(dia_df$muscle.stiffness == "Yes", 1, 0)
dia_df$Alopecia <- ifelse(dia_df$Alopecia == "Yes", 1, 0)
dia_df$Obesity <- ifelse(dia_df$Obesity == "Yes", 1, 0)

#Correlation of every pair of features
set.seed(1, sample.kind="Rounding")
#unfactor class variable to numeric
temp_df <- dia_df
temp_df$class<- as.numeric(as.character(temp_df$class))

#generate correlation of all features
corr_df<- temp_df %>% cor()
print(corr_df)

#Correlation plot
heatmap(as.matrix(corr_df), Colv = NA, Rowv = NA, scale="row")

#Count of available Positive and Negative results outcomes of diabetes
#find out how many people have diabetes and how many don’t
temp_df %>% group_by(class) %>% summarize(count=n())

#plot
temp_df %>% ggplot(aes(class)) + geom_bar(fill = c("darkgreen", "red")) + ggtitle("Diabetes Positive and Negative outcome")

#Check on Diabetes cases by average age group

#calculate average age for diabetic and non-diabetic cases
x<- temp_df %>% group_by(class) %>% summarise(avg_age=mean(Age))

#plot
x %>% ggplot(aes(x=class, y=avg_age)) + geom_bar(stat="identity", fill = c("red", "darkgreen")) + ggtitle("Average age group and Diabetes outcome")

#Scatterplots of this data set
pairs(temp_df, col=data$class)

#relationship of features Polyuria and Polydipsia
temp_df %>% ggplot(aes(Polyuria, Polydipsia, colour = class)) + geom_jitter(height = 0.3, width = 0.2) + ggtitle("Polydipsia and Polyuria Prevalence in diabetes")

#relationship of features Sudden weight Loss and Weakness
temp_df %>% ggplot(aes(sudden.weight.loss, weakness, colour = class)) + geom_jitter(height = 0.3, width = 0.2) + xlab("Sudden Weight Loss") + ylab("Weakness") + ggtitle("Sudden Weight Loss and Weakness Prevalence of Diabetes")

#Age and Obesity in diabetes relation
temp_df %>% ggplot(aes(Obesity, Age, colour = class)) + geom_jitter(height = 0.3, width = 0.2) + xlab("Obesity") + ylab("Age") + ggtitle("Age and Obesity Prevalence of Diabetes")

#Genital Thrush and Visual Blurring symptoms relationship with diabetes outcome
temp_df %>%
  ggplot(aes(Genital.thrush, visual.blurring, colour = class)) +
  geom_jitter(height = 0.3, width = 0.2) +
  xlab("Genital Thrush") +
  ylab("Visual Blurring") +
  ggtitle("Genital Thrush and Visual Blurring Prevalence of Diabetes")

############################################################
# Preparing Data Set for Training, Test and Validation
############################################################

#create a validation data set
set.seed(88, sample.kind="Rounding")

validation_index <- createDataPartition(data$class, times = 1, p = 0.20, list = FALSE)
validation <- data %>% slice(validation_index)
diabetes <- data %>% slice(-validation_index)

#create train and test sets from diabetes data set
set.seed(16, sample.kind="Rounding")

test_index <- createDataPartition(diabetes$class, times = 1, p = 0.20, list = FALSE)
train <- diabetes %>% slice(-test_index)
test <- diabetes %>% slice(test_index)

###############################
# MACHINE LEARNING MODEL DESIGN AND SIMULATION
###############################

# 1)Logistic regression

#range of tuning parameters
tune_p <- seq(0.2, 0.5, by = 0.01)

#repeat experiment 5 times
folds <- 5

#Matrix to store the mean of accuracy
lr_acc_p <- matrix(nrow = folds, ncol = length(tune_p))
#create data partitions
part_data <- createFolds(1:nrow(train), k = folds)
for (i in 1:folds)
{
  #create train and test sets
  temp_train <- train %>% slice(-part_data[[i]])
  temp_test <- train %>% slice(part_data[[i]])
  #generate a matrix with mean of accuracy and sensitivity
  lr_acc_p[i,] <- sapply(tune_p, function(p){
    #apply logistic regression model
    train_lr_model <- glm(as.numeric(class=="Positive")~., family = "binomial", data = temp_train)
    #obtain the predictions (these are probabilities)
    predict_res <- predict(train_lr_model, temp_test, type = "response")
    #if prediction > p then classify diabetes as Positive outcome otherwise negative outcome.
    t_lr_cm <- confusionMatrix(ifelse(predict_res > p, "Positive","Negative") %>% factor(levels = c("Positive","Negative")), temp_test$class)
    #return the mean of the accuracy and sensitivity
    return(mean(c(t_lr_cm$overall["Accuracy"],
                  t_lr_cm$byClass["Sensitivity"])))
  })
}
#Calculate Mean
x<-colMeans(lr_acc_p)
max(x)

#find optimal value of P
p <- tune_p[which.max(x)]
print(p)

#plot
hist(x, main="Linear Regression average accuracies")

#Logistic Regression model Overall Accuracy
LR_model  <- glm(as.numeric(class=="Positive")~., family = "binomial", data = train)
summary(LR_model)

#Prediction from Linear regression model
predict_test <- predict(LR_model, test, type = 'response')

#ROC curve calculation
roc_predict_train <- predict(LR_model, type = 'response')
roc_prediction <- prediction(roc_predict_train, train$class)
roc_performance <- performance(roc_prediction, 'tpr','fpr')
plot(roc_performance, colorize = TRUE, text.adj = c(-0.2,1.7))

#consider p = 0.3 as the measuring probability threshold and if prediction result is greater than p will be considered patient is diabetic, if prediction is less than 0.3 will be considered non-diabetic.
x <- ifelse(predict_test > p, "Positive","Negative") %>% factor(levels = c("Positive","Negative"))

#confusion matrix
cm_lr <- confusionMatrix(x, test$class)

#save accuracy and sensitivity
lr_acc <- cm_lr$overall["Accuracy"]
lr_sen <- cm_lr$byClass["Sensitivity"]

print(lr_acc)
print(lr_sen)

# 2)	K-Nearest Neighbors
#KNN model will use optimal value of from k=3, 5, 7 or 9, number of neighbors will be considered in the KNN model for comparison.
#Data Transformation for KNN - Remove Age, Gender and class columns and convert other column values to boolean (TRUE or FALSE)
knn_mdl_train <- {train[-c(1, 2, 17)]=="Yes"} %>% as_tibble
#Add Gender column with boolean values
knn_mdl_train <- cbind(train[2] == "Male", knn_mdl_train)
#add class column back
knn_mdl_train <- cbind(knn_mdl_train, train[17])

#transform test data set
knn_mdl_test <- {test[-c(1, 2, 17)]=="Yes"} %>% as_tibble
knn_mdl_test <- cbind(test[2] == "Male", knn_mdl_test)

#define k = 3,5,7 and 9
tune_k <- seq(3, 9, by = 2)
folds <- 5
knn_acc_k <- matrix(nrow = folds, ncol = length(tune_k))
set.seed(4, sample.kind="Rounding")
part_data <- createFolds(1:nrow(train), k = folds)

for (i in 1:folds)
{

  #create train and test sets and remove the prediction column class
  temp_train <- knn_mdl_train %>% slice(-part_data[[i]])
  temp_test <- knn_mdl_train %>% slice(part_data[[i]]) %>% select(-c("class"))

  #apply KNN algo to estimate the cutoff value of k
  knn_acc_k[i,] <- sapply(tune_k, function(k){
    #create the knn model using jaccard distance metric
    temp_knn <- knn(train_set = temp_train,
                    test_set = temp_test,
                    k = k,
                    categorical_target = "class",
                    comparison_measure="jaccard")

    #predictions
    temp_preds_knn <- temp_knn$test_set_scores$categorical_target %>% factor(levels = c("Positive","Negative"))

    #create the confusion matrix
    temp_knn_cm <- confusionMatrix(temp_preds_knn, knn_mdl_train %>% slice(part_data[[i]]) %>% .$class)

    #return the mean of the accuracy and sensitivity
    return(mean(c(temp_knn_cm$overall["Accuracy"], temp_knn_cm$byClass["Sensitivity"])))
  })
}

#maximum value of KNN 5 folds accuracy
x<-colMeans(knn_acc_k)
max(x)

#find the optimal value of k
opt_k <- tune_k[which.max(colMeans(knn_acc_k))]
print(opt_k)

#plot
hist(x, main="KNN average accuracies for k = 3,5,7 and 9")

#generate a model with optimal value k=3
knn_model <- knn(train_set = knn_mdl_train,
                 test_set = knn_mdl_test,
                 k = opt_k,
                 categorical_target = "class",
                 comparison_measure="jaccard")
#Prediction using test data set
predict_knn <- knn_model$test_set_scores$categorical_target %>% factor(levels = c("Positive","Negative"))

#Generate confusion matrix
knn_cm <- confusionMatrix(predict_knn, test$class)

#Accuracy and Sensitivity
knn_acc <- knn_cm$overall["Accuracy"]
knn_sen <- knn_cm$byClass["Sensitivity"]

print(knn_acc)
print(knn_sen)

# 3) Decision Tree
#create decision tree model
set.seed(64, sample.kind="Rounding")

#generate the range of cutoff values to decide the optimal cutoff value
dec_tree_model <- train(class~.,
                        method = "rpart",
                        tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)),
                        data = diabetes)

#plot from samples results from decision tree. Optimal cp cutoff value
ggplot(dec_tree_model, highlight = TRUE)

#cp error estimation projection plot
dec_tree_model$results %>%
  ggplot(aes(x = cp, y = Accuracy)) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(x = cp,
                    ymin = Accuracy - AccuracySD,
                    ymax = Accuracy + AccuracySD))

#optimal cp value
cp_cuttoff <- dec_tree_model$bestTune

print(cp_cuttoff)

#generate a new model using train data with optimal cp value
dtree <- rpart(class~., cp = cp_cuttoff, data = train)

#plot tree structure
rpart.plot(dtree, type = 5)
title("Decision Tree")

#Measure the importance of feature variables
importance <- t(dtree$variable.importance)
print(importance)

#run experiment 5 folds to find the optimal value of tuning parameter
#tune cutoff parameter
tune_p <- seq(0.1, 0.8, by = 0.025)
folds <- 5
temp_dtree_acc <- matrix(nrow = folds, ncol = length(tune_p))
set.seed(4, sample.kind="Rounding")
part_data <- createFolds(1:nrow(train), k = folds)

#The cross validation returns the mean of the accuracy and sensitivity
for(i in 1:folds) {

  #prepare train and test sets
  temp_train <- train %>% slice(-part_data[[i]])
  temp_test <- train %>% slice(part_data[[i]])

  #Accuracy results dtree model with cutoff p
  temp_dtree_acc[i,] <- sapply(tune_p, function(p){

    #decision tree model
    temp_dtree_mod <- rpart(class~., cp = cp_cuttoff, data = temp_train)

    #Prediction calls
    temp_dtree_predict <- predict(temp_dtree_mod, temp_test) %>%
      as_tibble %$%
      ifelse(Positive > p, "Positive", "Negative") %>%
      factor(levels = c("Positive", "Negative"))

    #Generate Confusion Matrix
    temp_dtree_cm <- confusionMatrix(temp_dtree_predict, temp_test$class)

    # Mean of Accuracy and Sensitivity
    return(mean(c(temp_dtree_cm$overall["Accuracy"], temp_dtree_cm$byClass["Sensitivity"])))

  })
}

#Median is used to define the tune parameter cutoff, this will reduce the error because of many repetitions of values which are matching the same opt_p
opt_p <- median(tune_p[min_rank(desc(colMeans(temp_dtree_acc)))==1])

print(opt_p)

#apply model on validation dataset
dtree_predict <- predict(dtree, validation) %>%
  as_tibble %$%
  ifelse(Positive > opt_p, "Positive", "Negative") %>%
  factor(levels = c("Positive", "Negative"))

#confusion matrix
dtree_cofmat <- confusionMatrix(dtree_predict, validation$class)

#Accuracy and Sensitivity
dtree_acc <- dtree_cofmat $overall["Accuracy"]
dtree_sen <- dtree_cofmat $byClass["Sensitivity"]

print(dtree_acc)
print(dtree_sen)

# 4) Random Forest
set.seed(11, sample.kind="Rounding")

#create train model
rf<- randomForest(class~., data = train, ntree=100)

#importance of variables
print(rf$importance)

#Random Forest tree structure all terminal nodes will be shown as -1
getTree(rf, 1, labelVar=TRUE)

#do the prediction based on 100 trees random forest
rf_prdt <- predict(rf, test)

#generate confusion matrix
rf_cm <- confusionMatrix(rf_prdt, test$class)

#Accuracy and Sensitivity
rf_100_acc <- rf_cm$overall["Accuracy"]
rf_100_sen <- rf_cm$byClass["Sensitivity"]

print(rf_100_acc)
print(rf_100_sen)


############################################################
# Final Results from best suited model (Random Forest)
############################################################

#Random Forest-Final Result Validation

set.seed(1, sample.kind="Rounding")

#create train model using diabetes data set
rf_final <- randomForest(class~., data = diabetes, ntree=100)

#do the prediction based on 100 trees random forest
rf_predict_final <- predict(rf_final, validation)

#generate confusion matrix
rf_cm_final <- confusionMatrix(rf_predict_final, validation$class)

#Accuracy and Sensitivity
rf_acc_final <- rf_cm_final$overall["Accuracy"]
rf_sen_final <- rf_cm_final$byClass["Sensitivity"]

print(rf_acc_final) 
print(rf_sen_final)

#print confusion Matrix
print(rf_cm_final)
