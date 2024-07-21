# Install and load the necessary libraries
install.packages("tm")
install.packages("SnowballC")
install.packages("e1071")
install.packages("caret")
install.packages("data.table")

# Set working directory (adjust to your own path)
setwd("C:/Users/nicol/OneDrive/MIA/Cursos/Aprendizaje Sup/Proyecto de Aplicacion/REPO GITHUB/proyecto-final-machine-learning/aprendizaje-supervizado")

library(tm)
library(SnowballC)
library(e1071)
library(caret)
library(data.table)
library(Matrix)  # For sparse matrix support

# Increase the protection stack size
options(expressions = 5000)

# Create a small sample dataset
sample_data <- data.table(
  text = c("I love this product, it is amazing!",
           "This is the worst purchase I have ever made.",
           "The product is okay, not great but not terrible either.",
           "Fantastic! I'm very happy with this purchase.",
           "Terrible experience, will not buy again.",
           "It's decent, could be better.",
           "Absolutely wonderful, highly recommend!",
           "Not what I expected, quite disappointing.",
           "Neutral feelings, it's just okay.",
           "Amazing quality, very satisfied."),
  sentiment = c("positive", "negative", "neutral", "positive", "negative",
                "neutral", "positive", "negative", "neutral", "positive")
)

# Save the dataset to a CSV file
write.csv(sample_data, "sample_dataset.csv", row.names = FALSE)

# Load and preprocess the sample dataset
dataset_path <- "sample_dataset.csv"
df <- fread(dataset_path, header = TRUE, sep = ",")

# Check the loaded data
print(df)

# Keep only the columns "text" and "sentiment"
df <- df[, .(text, sentiment)]

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
inspect(dtm[1:10, 1:10])

# Apply TF-IDF weighting to the DTM
dtm_tfidf <- weightTfIdf(dtm)
dtm_matrix <- as.matrix(dtm_tfidf)

# View the dimensions of the matrix
print(dim(dtm_matrix))

# View the first few rows and columns
print(head(dtm_matrix[, 1:10]))  # First 10 columns of the first few rows

# View a summary of the matrix
print(summary(dtm_matrix))

# Perform PCA to reduce dimensions
pca_result <- prcomp(dtm_matrix, scale. = TRUE, center = TRUE)

# Number of principal components to retain (e.g., 5 components for this small dataset)
num_components <- 5
dtm_pca <- pca_result$x[, 1:num_components]

# Convert back to data table
dtm_df <- as.data.table(dtm_pca)
dtm_df[, sentiment := factor(df$sentiment)]  # Convert sentiment to factor

# Split data into training and test sets
set.seed(123)
trainIndex <- createDataPartition(dtm_df$sentiment, p = 0.8, list = FALSE)
trainData <- dtm_df[trainIndex]
testData <- dtm_df[-trainIndex]

# Train SVM model with reduced dimensions
svm_model <- svm(sentiment ~ ., data = trainData, kernel = "linear", cost = 1, scale = FALSE)

# Predict and evaluate
predictions <- predict(svm_model, newdata = testData)
confusionMatrix(predictions, testData$sentiment)