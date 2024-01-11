Page 1
install.packages("tidyverse")
install.packages("caret")
install.packages("randomForest")
install.packages("keras")
install.packages("e1071")
install.packages("rpart")
# Load the necessary libraries
library(tidyverse) # For data manipulation and visualization
library(caret) # For data splitting
library(randomForest) # For random forest algorithm
library(keras) # For neural network models
library(e1071) # For SVM
library(rpart) # For decision trees
# Step 2: Load the dataset
data <- read.csv('youtube adview.csv')
# Check the dataset structure and dimensions
print(dim(data))
print(str(data))
# Step 3: Visualize the dataset
# Ensure data has numeric values for correlation calculation
numeric_data <- data %>% select_if(is.numeric)
if (ncol(numeric_data) > 0) {
 corr <- cor(numeric_data, use = "complete.obs") # Handle NA values
 heatmap(corr, symm = TRUE)
}
# Convert category to numeric values
category <- c('A'=1,'B'=2,'C'=3,'D'=4,'E'=5,'F'=6,'G'=7,'H'=8)
data$category <- as.numeric(factor(data$category, levels = names(category), labels = category))
# Step 4: Clean the dataset
# Print the number of NA values in each column
print(colSums(is.na(data)))
# Remove rows with NA values and duplicate rows
data <- na.omit(data)
data <- unique(data)
# Ensure columns are numeric
numeric_columns <- c('views', 'likes', 'dislikes', 'comment', 'adview')
data[numeric_columns] <- lapply(data[numeric_columns], function(x) as.numeric(as.character(x)))
# Handle any conversion errors by removing rows with NAs that arise from conversion
data <- na.omit(data)
# Step 5: Transform attributes and preprocess data
# Factorize non-numeric attributes and then convert to numeric
factor_columns <- c('duration', 'vidid', 'published')
data[factor_columns] <- lapply(data[factor_columns], function(x) as.numeric(as.factor(x)))
# Visualize the distribution of the category and adview
hist(data$category, main = "Category distribution", xlab = "Category")
plot(data$adview, main = "Adview Distribution", xlab = "Index", ylab = "Adviews")
data <- data[data$adview < 2000000, ]
plot(data$adview, main = "Filtered Adview Distribution", xlab = "Index", ylab = "Adviews")
# Step 6: Normalize data and split into sets
# Define features and target variable
X <- subset(data, select = -c(adview))
y <- data$adview
# Normalize features
X_scaled <- as.data.frame(scale(X))
# Split the data into training and testing sets
set.seed(42)
Page 2
split <- createDataPartition(y, p = 0.8, list = FALSE)
train_data <- data[split, ]
test_data <- data[-split, ]
# Step 7: Train models
linear_reg <- lm(adview ~ ., data = train_data)
svr_reg <- svm(adview ~ ., data = train_data)
dt_reg <- rpart(adview ~ ., data = train_data)
rf_reg <- randomForest(adview ~ ., data = train_data, ntree = 200, mtry = 3)
# Step 8: Train an Artificial Neural Network
# Ensure that keras is properly configured and available
if (!keras::is_keras_available()) {
 stop("Keras is not available")
}
ann <- keras_model_sequential()
ann %>%
 layer_dense(units = 128, activation = 'relu', input_shape = ncol(X_scaled)) %>%
 layer_dense(units = 64, activation = 'relu') %>%
 layer_dense(units = 1, activation = 'linear')
compile(ann, optimizer = 'adam', loss = 'mean_squared_error')
fit(ann, as.matrix(train_data[, -which(names(train_data) == "adview")]), train_data$adview, epochs =# Step 9: Evaluate models
# Predictions
predictions <- list(
 linear_pred = predict(linear_reg, newdata = test_data),
 svr_pred = predict(svr_reg, newdata = test_data),
 dt_pred = predict(dt_reg, newdata = test_data),
 rf_pred = predict(rf_reg, newdata = test_data),
 ann_pred = predict(ann, as.matrix(test_data[, -which(names(test_data) == "adview")]))
)
# Compute Mean Squared Errors
errors <- sapply(predictions, function(pred) mean((test_data$adview - pred)^2))
print(errors)
# Step 10: Save models
saveRDS(rf_reg, file = 'best_youtube_adview_predictor.rds')
save_model_hdf5(ann, filepath = 'ann_youtube_adview.h5')