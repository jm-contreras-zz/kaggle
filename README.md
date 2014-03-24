kaggle
======

**Repository of R scripts that I wrote to take part in three predictive modeling competitions hosted by [Kaggle](http://www.kaggle.com/).** Each script requires that the training and testing data from its respective competition are located in the current working directory in the form of CSV files. As output, each script produces a CSV file called *submission*, which can be submitted to its competition.

- **scfp.R**, written for the [See Click Fix Predict competition](http://www.kaggle.com/c/see-click-predict-fix), uses linear regression to predict the number of user comments, views, and votes on non-emergency issues from 4 US cities. Its operations include the following.
  - Clustering of issue coordinates with *k*-means to identify the city of provenance as a feature
  - Imputation of missing issue tags (e.g., 'graffiti') with keyword matching from known tags
  - Extraction of time features (e.g., hour) from date-time strings

- **titanic.R**, written for the [Titanic competition](http://www.kaggle.com/c/titanic-gettingStarted), uses logistic regression to predict which passengers survived the sinking of the eponymous ship. Its operations include the following.
  - Imputation of missing age data
  - Extraction of passengers' titles from their names for use as a feature
  - Feature selection with guided random forests (via wrapper **selectFeatures.R**)

- **digits.R**, written for the [Digit Recognizer competition](http://www.kaggle.com/c/digit-recognizer), uses random forests to analyze pixel data from black-and-white digit images to predict the number that each image represents. Its operations include the following.
  - Creation of new features (number of dark pixels in each image and metrics for horizontal and vertical symmetry)
  - Removal of redundant and low-variance features

- **classifierMelee.R** uses multiple rounds of *k*-fold cross-validation with the training data from the Titanic competition to assess the classification accuracy of 9 different classifiers. The results are stored in a data frame for further analysis or visualization.
