getwd()


# Instalar y cargar las bibliotecas necesarias

install.packages("tm")
install.packages("SnowballC")
install.packages("e1071")
install.packages("caret")
install.packages("data.table")

setwd("C:/Users/nicol/OneDrive/MIA/Cursos/Aprendizaje Sup/Proyecto de Aplicacion/REPO GITHUB/proyecto-final-machine-learning/aprendizaje-supervizado")


library(tm)
library(SnowballC)
library(e1071)
library(caret)
library(data.table)

# Increase the protection stack size
#options(expressions = 5000)

# Cargar los datos

#df <- read.csv("tweets.csv", stringsAsFactors = FALSE)

dataset_path <- "dataset/train.csv"
#df <- read.csv(dataset_path, header = TRUE, sep = ",", fileEncoding = "ISO-8859-1")

df <- fread(dataset_path, header = TRUE, sep = ",")

# Keep only the columns "text" and "sentiment"
df <- df[, .(text, sentiment)]
https://www.csie.ntu.edu.tw/~cjlin/papers/quadworkset.pdf
head(df)


# Preprocess text function
preprocess_text <- function(text) {
  text <- tolower(text)
  text <- removePunctuation(text)
  text <- removeNumbers(text)
  text <- removeWords(text, stopwords("en"))
  text <- wordStem(text)
  text <- stripWhitespace(text)
  return(text)
}

# Apply preprocessing
df[, text := sapply(text, preprocess_text)]

# Create a corpus and DTM
corpus <- Corpus(VectorSource(df$text))
dtm <- DocumentTermMatrix(corpus)
dtm_tfidf <- weightTfIdf(dtm)
dtm_matrix <- as.matrix(dtm_tfidf)

# Prepare data for modeling
dtm_df <- as.data.table(dtm_matrix)
dtm_df[, sentiment := df$sentiment]

# Split data into training and test sets


?createDataPartition
set.seed(123)
trainIndex <- createDataPartition(dtm_df$sentiment, p = 0.1, list = FALSE)
trainData <- dtm_df[trainIndex]
testData <- dtm_df[-trainIndex]

head(trainData)
summary(trainData)


# Train SVM model

# Start memory profiling
memory.profile()
svm_model <- svm(sentiment ~ ., data = trainData)
memory.profile()

# Predict and evaluate
predictions <- predict(svm_model, newdata = testData)
confusionMatrix(predictions, testData$sentiment)