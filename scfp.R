# SETUP

# Load package
library(cluster)

# PRE-PROCESS

pre.process = function(data, train.or.test) {

    # Declare variables
    n.example = nrow(data)
    
    # Identify cities
    kmeans.out = kmeans(data[, c('latitude', 'longitude')], 4)
    data$city = as.factor(kmeans.out$cluster)
    
    # Replace rare tags with 'other' tag
    tag.count = sort(table(data$tag_type))
    rare.tags = names(tag.count[tag.count < 2])
    rare.tags.ind = which(data$tag_type %in% rare.tags)
    data$tag_type = as.character(data$tag_type)
    data$tag_type[rare.tags.ind] = 'other'
    
    # Identify common tags
    tag.table = sort(table(data$tag_type), decreasing = T)
    
    # Assign keywords for text mining
    if (train.or.test = 'train') {
    
        key.words = c('trash', 'tree', 'hole', 'graffiti', 'light', 'hydrant', 'sign',
                  'overgrowth', 'sidewalk', 'blighted', 'traffic', 'snow', 'drain',
                  'road', 'bridge', 'bike', 'homeless', 'flood', 'abandon',
                  'crosswalk', 'drug', 'robbe', 'meter', 'animal', 'bench', 'smell',
                  'loud', 'test', 'idling', 'signal', 'rat', 'hot', 'prosti',
                  'roadkill', 'reckless driv', 'pedestrian light', 'zoning', 'lost',
                  'public art', 'public concern')
                  
    } else if (train.or.test = 'test') {
    
        key.words = c('trash', 'graffiti', 'hole', 'overgrowth', 'tree', 'light',
                      'blighted', 'sidewalk', 'sign', 'drain', 'bike', 'road',
                      'traffic', 'abandon', 'street sign', 'drug', 'crosswalk',
                      'bridge', 'noise', 'idling', 'bench', 'animal', 'rat', 'hydrant',
                      'loud', 'test', 'idling', 'signal', 'rat', 'hot', 'prosti',
                      'smell', 'homeless', 'meter', 'pedestrian light', 'robbe',
                      'snow', 'hot', 'reckless driv', 'test', 'zoning', 'roadkill',
                      'flood', 'prosti')
                      
    }
    
    n.keys = length(key.words)
    
    # Identify examples with missing tags
    no.tag = is.na(data$tag_type)
    
    # Impute missing tags
    for (i.key in 1:n.keys) {
        
        this.tag = names(tag.table[i.key])
        this.key.word = key.words[i.key]
        this.key.ind = grepl(this.key.word, data$description, ignore.case = T)
        this.key.no.tag = this.key.ind & no.tag
        n.replace = sum(this.key.no.tag)
        
        if (n.replace > 0) {
            data$tag_type[this.key.no.tag] = this.tag
            no.tag = is.na(data$tag_type)
            print(paste('Tagged', n.replace, this.tag))
        } else {
            print(paste('Skipped', this.tag))
            next
        }
        
    }
    
    # Replace remaining missing values with 'other'
    data$tag_type[no.tag] = 'other'
    
    # Convert 'tag_type' column from character to factor class
    data$tag_type = as.factor(data$tag_type)
    
    # Remove factors from 'source' column
    data = transform(data, source = as.character(source))
    
    # Identify examples with missing tags
    no.tag = is.na(data$source)
    
    # Replace missing values with 'other'
    data$source[no.tag] = 'other'
    
    # Convert 'source' column from character to factor class
    data$source = as.factor(data$source)
    
    # Unpack date and time
    all.date.time = strsplit(as.character(data$created_time), split = ' ')
    
    # Initialize time and date matrix
    when.matrix = matrix(nrow = n.example, ncol = 6)
    
    # Reformat time and date values to populate time and date matrix
    for (i.ex in 1:n.example) {
        
        # Reformat time and date values
        this.date = as.Date(all.date.time[[i.ex]][1])
        this.time = as.numeric(strsplit(all.date.time[[i.ex]][2], split = ':')[[1]])
        
        # Populate time and date matrix
        when.matrix[i.ex, 1] = as.numeric(format(this.date, "%Y"))  # Year
        when.matrix[i.ex, 2] = as.numeric(format(this.date, "%m"))  # Month
        when.matrix[i.ex, 3] = as.numeric(format(this.date, "%d"))  # Day
        when.matrix[i.ex, 4] = as.POSIXlt(this.date)$wday           # Weekday
        when.matrix[i.ex, 5] = this.time[1]                         # Hour
        when.matrix[i.ex, 6] = this.time[2]                         # Minute
        
    }
    
    # Convert matrix to data frame
    when.df = as.data.frame(when.matrix)
    names(when.df) = c('year', 'month', 'day', 'weekday', 'hour', 'minute')
    data = cbind(data, when.df)

}

# Load and pre-process training and testing data
train = pre.process(read.csv('train.csv'), 'train')
test = pre.process(read.csv('test.csv'), 'test')

# TRAIN AND TEST

# Declare target variables
targets = c('num_views', 'num_votes', 'num_comments')
n.target = length(targets)

# Initialize submission matrix
submission = matrix(nrow = nrow(test), ncol = n.target + 1)
submission[, 1] = test$id

# Loop through targets
for (i.target in 1:n.target) {
    
    # Define targets
    this.target = paste(targets[i.target], sep = '')
    
    # Train model
    this.formula = paste(this.target, '~ city + tag_type + source + year +',
                         'month + day + weekday + hour + minute')
    
    # Train model
    model = glm(this.formula, data = train)
    
    # Test model
    predictions = predict(model, test)
    
    # Transform and save predictions
    submission[, i.target + 1] = predictions
    
}

# Remove negative predictions
submission[submission < 0] = 0

# Write submission file
write.table(submission, file = 'submission.csv', col.names = c('id', targets),
            sep = ',', row.names = F)
