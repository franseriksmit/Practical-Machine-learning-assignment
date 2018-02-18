---
output: 
  html_document: 
    keep_md: yes
---
##Predicting the manner in which Weight Lifting Exercises were done.
author: "F.Smit"
date: "18 februari 2018"


## Executive Summary
This project was carried out as the final assignment of the Coursera course "Practical Machine Learning". The purpose is to predict the manner in which weight lifting exercises were done.

##Background and Introduction
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, we will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participant They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The five ways are exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Only Class A corresponds to correct performance. The goal of this project is to predict the manner in which they did the exercise, i.e., Class A to E. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

##Loading the data
We will first load the required packages and then download the training and testing sets from the given URL's



The trainingset contains 19622 observations of 160 variables, the testing set contains 20 observations of the same amount of variabes. We will try to make a decent prediction of the outcome of the "classe" variable in the training set.

##Data cleaning
We will delete all columns in the training and the testing set that contain any missing values. Also, the first seven columns will be deleted since they are meta-variables containing information about the set and do not have any predicting value about the training cq test set.

```r
training<-training[, colSums(is.na(training))==0]
testing<-testing[,colSums(is.na(testing))==0]
training<-training[,-c(1:7)]
testing<-testing[,-c(1:7)]
```
The cleaned datasets training contains 19622 observations of 53 variables and the test set has 20 observations of the same 53 variables.

##Data splitting
We will now split the data into a training and a validation set.

```r
set.seed(13)
inTrain<-createDataPartition(y=training$classe, p=0.7, list=0)
training<-training[inTrain,]
valid<-training[-inTrain,]
```
##Prediction models 
Looking at the data, probably the best models to be used are either a decision tree or using random forests. 
First we will built the decision tree using the rpart method.
In practice, k=5 or k=10 are used when doing k-fold cross validation. Here we consider 5-fold (k=5) cross validation (default setting in trainControl function is 10) during implementation of the algorithm to save computing time. Data transformation is not necessary.


```r
control <- trainControl(method = "cv", number = 5)
modfit<-train(classe~., data=training, method="rpart",trControl=control)
print(modfit, digits = 4)
```

```
## CART 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10989, 10989, 10989, 10990, 10991 
## Resampling results across tuning parameters:
## 
##   cp       Accuracy  Kappa  
##   0.04028  0.5001    0.34556
##   0.04974  0.4617    0.28641
##   0.11250  0.3465    0.09471
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was cp = 0.04028.
```

```r
fancyRpartPlot(modfit$finalModel)
```

![](pml_files/figure-html/unnamed-chunk-4-1.png)<!-- -->
Now we will predict the outcome of the rpart method,using the validation set. This is required to get the out-of-smaple errors.

```r
predict1 <- predict(modfit, valid)
(conf1t <- confusionMatrix(valid$classe, predict1))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1077   13   88    0    2
##          B  343  195  249    0    0
##          C  330   13  378    0    0
##          D  313  113  242    0    0
##          E  120   40  281    0  331
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4799          
##                  95% CI : (0.4646, 0.4953)
##     No Information Rate : 0.5288          
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.3193          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4934  0.52139  0.30533       NA  0.99399
## Specificity            0.9470  0.84230  0.88131   0.8382  0.88379
## Pos Pred Value         0.9127  0.24778  0.52427       NA  0.42876
## Neg Pred Value         0.6248  0.94642  0.74758       NA  0.99940
## Prevalence             0.5288  0.09060  0.29990   0.0000  0.08067
## Detection Rate         0.2609  0.04724  0.09157   0.0000  0.08018
## Detection Prevalence   0.2859  0.19065  0.17466   0.1618  0.18702
## Balanced Accuracy      0.7202  0.68185  0.59332       NA  0.93889
```
Taking the accuray from the numbers being shown (0.4799), the out of sample error will be 0.5201. We can discart this model as this apparently fails in predicting a correct outcome.
Next up: random forests

```r
modfit2<-train(classe~., data=training, method="rf", trControl=control)
print(modfit2, digits = 4)
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10989, 10990, 10990, 10990, 10989 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa 
##    2    0.9919    0.9898
##   27    0.9920    0.9899
##   52    0.9816    0.9767
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```
Now we will predict the outcome of the random forest method, using the validation set for getting the out-of-smaple errors.

```r
predict2 <- predict(modfit2, valid)
(conf2t <- confusionMatrix(valid$classe, predict2))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1180    0    0    0    0
##          B    0  787    0    0    0
##          C    0    0  721    0    0
##          D    0    0    0  668    0
##          E    0    0    0    0  772
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9991, 1)
##     No Information Rate : 0.2859     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000    1.000
## Specificity            1.0000   1.0000   1.0000   1.0000    1.000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000    1.000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000    1.000
## Prevalence             0.2859   0.1906   0.1747   0.1618    0.187
## Detection Rate         0.2859   0.1906   0.1747   0.1618    0.187
## Detection Prevalence   0.2859   0.1906   0.1747   0.1618    0.187
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000    1.000
```
For this dataset, random forests seem to be perfect as the accuracy is 1. The out of sample error is (1-1)=0. This may be because of the fact that many of the variables are highly correlated. 

##Prediction on test set. 
Using random forests, we will predict the outcome variable classe from the testing set. 

```r
  (predict(modfit2, testing))
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
