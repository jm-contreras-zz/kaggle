# Source external functions
source('selectFeatures.R')

# Load data
train = read.csv('train.csv')
test = read.csv('test.csv')
target.train = train$Survived

# Declare missing age variables
titles = c('Mr', 'Mrs', 'Master', 'Miss', 'Dr', 'Rev')
n.title = length(titles)
train.missing.age = is.na(train$Age)
test.missing.age = is.na(test$Age)
n.train.ex = nrow(train)
train.title = vector(mode = 'numeric', length = n.train.ex)
n.test.ex = nrow(test)
test.title = vector(mode = 'numeric', length = n.test.ex)

# Loop through titles
for (i.title in 1:n.title) {
    
    # Declare current title
    this.title = paste(titles[i.title], '. ', sep = '')
    
    # Impute missing ages in train data
    train.title.index = train$Name %in% grep(this.title, train$Name, value = TRUE)
    title.mean.age = mean(train$Age[train.title.index], na.rm = TRUE)
    train$Age[train.title.index & train.missing.age] = title.mean.age
    
    # Impute missing ages in test data
    test.title.index = test$Name %in% grep(this.title, test$Name, value = TRUE)
    test$Age[test.title.index & test.missing.age] = title.mean.age
    
    # Fill title vector
    train.title[train.title.index] = i.title
    test.title[test.title.index] = i.title
    
}

# Incorporate title vector as feature
train.title[train.title > 4] = 0
train$Title = as.factor(train.title)

# Incorporate title vector as feature
test.title[test.title > 4] = 0
test$Title = as.factor(test.title)

# Declare features
features = c('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title')
    
# Select features
best.features = selectFeatures(train[, features], target.train, 1, 0.8);
    
# Make model formula
names.best.features = names(best.features)
formula.features = paste(names.best.features, collapse = ' + ')
model.formula = as.formula(paste('target.train ~', formula.features))

# Train model
model = glm(model.formula, data = train[, names.best.features], family = binomial(link = 'logit'))

# Test model
predictions = predict(model, test, type = 'response')
submission = round(predictions)

# Fix two cases with missing values
submission[89] = 1
submission[153] = 0

# Write submission file
write.table(submission, file = 'Submission.csv', row.names = FALSE, col.names = FALSE)
