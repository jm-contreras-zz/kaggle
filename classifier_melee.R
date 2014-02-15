#############
# LOAD DATA #
#############

# Load training data from Titanic competition
data = read.csv('train.csv')

##################
# PRE-PROCESSING #
##################

# Convert target variable to factor
data$Survived = as.factor(data$Survived)

# Declare missing age variables
titles = c('Mr', 'Mrs', 'Master', 'Miss', 'Dr', 'Rev')
n.title = length(titles)
miss.age = is.na(data$Age)
n.ex = nrow(data)
title.vec = vector(mode = 'numeric', length = n.ex)

# Loop through titles
for (i.title in 1:n.title) {
    
    # Impute missing ages
    this.title = paste(titles[i.title], '. ', sep = '')
    title.ind = data$Name %in% grep(this.title, data$Name, value = T)
    title.mean.age = mean(data$Age[title.ind], na.rm = T)
    data$Age[title.ind & miss.age] = title.mean.age
    
    # Fill title vector
    title.vec[title.ind] = i.title
    
}

# Incorporate title vector as feature (group Drs. and Revs. with Messrs.)
title.vec[title.vec > 4] = 1
data$Title = as.factor(title.vec)

# Remove non-features from data set
col.to.rm = names(data)[c(1, 4, 9, 11)]
data = data[, -which(names(data) %in% col.to.rm)]

# Create dummy variables
library(caret)
dv = dummyVars(~ Sex + Embarked + Title, data)
dv = predict(dv, data)

# Create data set with dummy variables
data.dv = subset(data, select= -c(Sex, Embarked, Title))
data.dv = cbind(data.dv, dv)

###############
# REPETITIONS #
###############

# Initialize repetitions data frame
n.rep = 25
emp.vec = rep(0, n.rep)
rep.df = data.frame(DT  = emp.vec, # Decision trees
                    RF  = emp.vec, # Random forests
                    NB  = emp.vec, # Naive Bayes
                    LR  = emp.vec, # Logistic regression
                    NET = emp.vec, # Neural networks
                    LDA = emp.vec, # Linear discriminant analysis
                    ADA = emp.vec, # Adaptive boosting
                    KNN = emp.vec, # K-nearest neighbors
                    SVM = emp.vec, # Support vector machines
                    ENS = emp.vec) # Ensemble

for (i.rep in 1:n.rep) {

    ####################
    # CROSS-VALIDATION #
    ####################

    # Load package
    library(cvTools)
    
    # Create cross-validation folds
    n.fold = 10
    cv = cvFolds(n.ex, n.fold)
    folds = cv$which
    examples = cv$subsets
    
    # Initialize cross-validation data frame
    emp.vec = rep(0, n.fold)
    cv.df = data.frame(DT  = emp.vec, # Decision trees
                       RF  = emp.vec, # Random forests
                       NB  = emp.vec, # Naive Bayes
                       LR  = emp.vec, # Logistic regression
                       NET = emp.vec, # Neural networks
                       LDA = emp.vec, # Linear discriminant analysis
                       ADA = emp.vec, # Adaptive boosting
                       KNN = emp.vec, # K-nearest neighbors
                       SVM = emp.vec, # Support vector machines
                       ENS = emp.vec) # Ensemble
    
    ##################
    # CLASSIFICATION #
    ##################
    
    # Loop through folds
    for (i.fold in 1:n.fold) {
        
        # Communicate progress
        print(paste0('Working on fold ', i.fold, ' in repetition ', i.rep, '.'),
              quote = F)
        
        # Declare indices for train and test examples
        train.ind = examples[folds != i.fold]
        test.ind = examples[folds == i.fold]
        
        # Create testing data
        test.data = data[test.ind, ]
        test.data.dv = data.dv[test.ind, ]
        
        # Actual test classes
        real = test.data$Survived
        
        ##################
        # DECISION TREES #
        ##################
        
        # Load package
        library(rpart)
        
        # Train, test, and evaluate
        tree = rpart(Survived ~ ., data, subset = train.ind)
        pred = predict(tree, test.data)
        cv.df[i.fold, 'DT'] = mean(round(pred[, 2]) == real)
        
        ##################
        # RANDOM FORESTS #
        ##################
        
        # Load package
        library(randomForest)
        
        # Train, test, and evaluate
        rf = randomForest(Survived ~ ., data, subset = train.ind)
        pred = predict(rf, test.data)
        cv.df[i.fold, 'RF'] = mean(pred == real)
    
        ###############
        # NAIVE BAYES #
        ###############
        
        # Load package
        library(e1071)
        
        # Train, test, and evaluate
        nb = naiveBayes(Survived ~ ., data, subset = train.ind)
        pred = predict(nb, test.data)
        cv.df[i.fold, 'NB'] = mean(pred == real)
    
        #######################
        # LOGISTIC REGRESSION #
        #######################
        
        # Train, test, and evaluate
        lr = glm(Survived ~ ., data,
                 family = binomial(link = 'logit'),
                 subset = train.ind)
        pred = predict(lr, test.data, type = 'response')
        cv.df[i.fold, 'LR'] = mean(round(pred) == real)
        
        ###################
        # NEURAL NETWORKS #
        ###################
        
        # Load package
        library(nnet)
        
        # Train, test, and evaluate
        nnet = nnet(Survived ~ ., data, subset = train.ind, size = 8)
        pred = predict(nnet, test.data)
        cv.df[i.fold, 'NET'] = mean(round(pred) == real)
        
        ################################
        # LINEAR DISCRIMINANT ANALYSIS #    
        ################################
        
        # Load package
        library(MASS)
        
        # Train, test, and evaluate
        lda = lda(Survived ~ ., data, subset = train.ind)
        pred = predict(lda, test.data)
        cv.df[i.fold, 'LDA'] = mean(pred$class == real)
        
        #####################
        # ADAPTIVE BOOSTING #    
        #####################
        
        # Load package
        library(ada)
        
        # Train, test, and evaluate
        ada = ada(Survived ~ ., data, subset = train.ind)
        pred = predict(ada, test.data)
        cv.df[i.fold, 'ADA'] = mean(pred == real)
        
        #######################
        # K-NEAREST NEIGHBORS #
        #######################
        
        # Load package
        library(kknn)
        
        # Train, test, and evaluate
        knn = kknn(Survived ~ ., data.dv[train.ind, ], test.data.dv)
        cv.df[i.fold, 'KNN'] = mean(knn$fitted.values == real)
        
        ###########################
        # SUPPORT VECTOR MACHINES #
        ###########################
        
        # Load package
        library(e1071)
        
        # Train, test, and evaluate
        svm = svm(Survived ~ ., data.dv, subset = train.ind)
        pred = predict(svm, test.data.dv)
        cv.df[i.fold, 'SVM'] = mean(pred == real)
        
        ############
        # EMSEMBLE #
        ############
        
        # Compute mean accuracy
        cv.df[i.fold, 'ENS'] = rowMeans(cv.df[i.fold, 1:9])
    
    }
    
    # Compute repetition means
    rep.df[i.rep, ] = colMeans(cv.df)
    
}
