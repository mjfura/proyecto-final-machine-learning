# Install and load the necessary libraries
# install.packages("tm")
# install.packages("SnowballC")
# install.packages("e1071")
# install.packages("caret")
# install.packages("data.table")

# Set working directory (adjust to your own path)
setwd("C:/Users/nicol/OneDrive/MIA/Cursos/Aprendizaje Sup/Proyecto de Aplicacion/REPO GITHUB/proyecto-final-machine-learning/aprendizaje-supervizado")

library(tm)
library(SnowballC)
library(e1071)
library(caret)
library(data.table)
library(Matrix)  # For sparse matrix support

# Increase the protection stack size
#options(expressions = 9000)

# # Create a larger sample dataset
# set.seed(123)  # For reproducibility
# sample_texts <- c(
#   "I love this product, it is amazing!", "This is the worst purchase I have ever made.",
#   "The product is okay, not great but not terrible either.", "Fantastic! I'm very happy with this purchase.",
#   "Terrible experience, will not buy again.", "It's decent, could be better.",
#   "Absolutely wonderful, highly recommend!", "Not what I expected, quite disappointing.",
#   "Neutral feelings, it's just okay.", "Amazing quality, very satisfied."
# )
# 
# sample_sentiments <- c("positive", "negative", "neutral")
# 
# # Generate 100 samples
# sample_data <- data.table(
#   text = sample(sample_texts, 100, replace = TRUE),
#   sentiment = sample(sample_sentiments, 100, replace = TRUE)
# )
# 
# # Save the dataset to a CSV file
# write.csv(sample_data, "sample_dataset.csv", row.names = FALSE)



# Cargar los datos
dataset_path <- "dataset/train.csv"
df_total <- fread(dataset_path, header = TRUE, sep = ",")

# Keep only the columns "text" and "sentiment"
df_total <- df_total[, .(text, sentiment)]


# Check the loaded data
#print(df)

# Assuming df is your data frame
df <- df_total[1:4000, ]


# Preprocess text function
preprocess_text <- function(text) {
  text <- tolower(text)
  text <- removePunctuation(text)
  text <- removeNumbers(text)
  text <- removeWords(text, stopwords("en"))
  text <- wordStem(text, language = "en")  # Stemming using SnowballC
  text <- stripWhitespace(text)
  return(text)
}

# Apply preprocessing
df[, text := sapply(text, preprocess_text)]

# Create a corpus and DTM
corpus <- Corpus(VectorSource(df$text))
dtm <- DocumentTermMatrix(corpus)

# Inspect the DTM to understand its structure and content
#inspect(dtm[1:10, 1:10])

# Apply TF-IDF weighting to the DTM
dtm_tfidf <- weightTfIdf(dtm)
dtm_matrix <- as.matrix(dtm_tfidf)

# View the dimensions of the matrix
#print(dim(dtm_matrix))

# View the first few rows and columns
#print(head(dtm_matrix[, 1:10]))  # First 10 columns of the first few rows

# View a summary of the matrix
#print(summary(dtm_matrix))

# Convert back to data table
dtm_df <- as.data.table(dtm_matrix)
dtm_df[, sentiment := factor(df$sentiment)]  # Convert sentiment to factor

# Split data into training and test sets
set.seed(123)
trainIndex <- createDataPartition(dtm_df$sentiment, p = 0.8, list = FALSE)
trainData <- dtm_df[trainIndex]
testData <- dtm_df[-trainIndex]

# Check data splitting
#print(table(trainData$sentiment))
#print(table(testData$sentiment))

# Train SVM model with TF-IDF features



cost0=c()
for(j in seq(0.01,2, by=0.05)) {
  cost0=cbind(cost0, svm(sentiment ~ ., data = trainData, kernel = "linear", cost=j, cross=5)$tot.accuracy)
}

plot(seq(0.01,2, by=0.05),cost0, type="o", pch=20, ylab="Accuracy", xlab= "C" )
abline(h =cost0[which.max(cost0)], v=seq(0.01,2, by=0.05)[which.max(cost0)], lty=2, col=2)


which.max(cost0)

#Accuracy
cost0[which.max(cost0)]

#Valor de C
seq(0.01,2, by=0.05)[which.max(cost0)]






#svm_model1 <- svm(sentiment ~ ., data = trainData, kernel = "linear", cost = 1, scale = FALSE)
#  
# 
# 
# 
# # svm_model2 <- svm(sentiment ~ ., data = trainData, kernel = "linear", cost = 0.1, scale = FALSE)
# # 
# # svm_model3 <- svm(sentiment ~ ., data = trainData, kernel = "linear", cost = 0.1, scale = FALSE)
# # 
# # svm_model4 <- svm(sentiment ~ ., data = trainData, kernel = "linear", cost = 0.1, scale = FALSE)
# 
# 
# 
# 
# 
# 
# Predict and evaluate

# print("modelo1")
# predictions <- predict(svm_model1, newdata = testData)
# print(table(predictions, testData$sentiment))
# confusionMatrix(predictions, testData$sentiment)

# # print("modelo2")
# # predictions <- predict(svm_model2, newdata = testData)
# # print(table(predictions, testData$sentiment))
# # confusionMatrix(predictions, testData$sentiment)
# 
# print("modelo3")
# predictions <- predict(svm_model3, newdata = testData)
# print(table(predictions, testData$sentiment))
# confusionMatrix(predictions, testData$sentiment)
# 
# print("modelo4")
# predictions <- predict(svm_model4, newdata = testData)
# print(table(predictions, testData$sentiment))
# confusionMatrix(predictions, testData$sentiment)