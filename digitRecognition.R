# SETUP

# Load packages
library(caret)
library(randomForest)

# Source external functions
source('selectFeatures.R')

# Load training data
train = read.csv('train.csv')
test = read.csv('test.csv')

# Separate classes from training data
class = as.factor(train$label)
train = train[, -1]

# Declarations
n.pix.row = 28  # Number of pixels per pixel matrix row
n.test = nrow(test)

# CREATE NEW FEATURES

create.features = function(data, n.pix.row) {
    
    # COUNT DARK PIXELS
    dark.pix = rowSums(data > 0)
    
    # RECONSTRUCT IMAGES TO COMPUTE SYMMETRY
    
    # Create pixel matrix
    pix.matrix = matrix(1:(n.pix.row ^ 2), nrow = n.pix.row, byrow = T)
    
    # Identify top and bottom pixels
    top.pix = pix.matrix[1:(n.pix.row / 2), ]
    bot.pix = pix.matrix[n.pix.row:(n.pix.row / 2 + 1), ]
    
    # Compute horizontal symmetry
    hor.sym = rowSums(abs(data[, top.pix] - data[, bot.pix]))
    
    # Identify left and right pixels
    left.pix = pix.matrix[, 1:(n.pix.row / 2)]
    right.pix = pix.matrix[, n.pix.row:(n.pix.row / 2 + 1)]
    
    # Compute vertical symmetry
    ver.sym = rowSums(abs(data[, right.pix] - data[, left.pix]))
    
    # Concatenate new features
    new.features = cbind(dark.pix, hor.sym, ver.sym)
    
    # Return them
    return(new.features)
    
}

# PRE-PROCESS

pre.process = function(data, new.features) {
    
    # Remove features with zero and near-zero variance
    zero.var.ind = nearZeroVar(data)
    data = data[, -zero.var.ind]
    
    # Remove correlated features
    cor.ind = findCorrelation(cor(data), cutoff = 0.75)
    data = data[, -cor.ind]
    
    # Add new features to data
    data = cbind(data, new.features)
    
    # Return data
    return(data)
    
}

# CREATE NEW FEATURES AND PRE-PROCESS DATA

# Training data
new.features = create.features(train, n.pix.row)
train = pre.process(train, new.features)

#Testing data 
new.features = create.features(test, n.pix.row)
test = cbind(test, new.features)
test = test[, names(train)]

# TRAIN AND TEST

# Train random forest
rf = randomForest(train, class)

# Test random forest
predictions = predict(rf, test)
submission = as.data.frame(cbind(1:n.test, as.numeric(predictions) - 1))
names(submission) = c('ImageId', 'Label')

# Write submission file
write.table(submission, file = 'Submission.csv', row.names = F, sep = ',')
