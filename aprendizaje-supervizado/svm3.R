
# Cargar las bibliotecas necesarias
library(e1071)
library(tm)
library(SnowballC)



setwd("C:/Users/nicol/OneDrive/MIA/Cursos/Aprendizaje Sup/Proyecto de Aplicacion/REPO GITHUB/proyecto-final-machine-learning/aprendizaje-supervizado")

# Cargar los datos

dataset_path <- "dataset/train.csv"
datos <- read.csv(dataset_path, header = TRUE, sep = ",", fileEncoding = "ISO-8859-1")

# Preprocesar los datos
corpus <- Corpus(VectorSource(datos$text))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removeWords, stopwords("english"))

# Crear la matriz de términos del documento con la ponderación TF-IDF
dtm <- DocumentTermMatrix(corpus)
dtm <- weightTfIdf(dtm)

# Dividir los datos en conjuntos de entrenamiento y prueba
set.seed(1234)
train_indices <- sample(1:nrow(dtm), nrow(dtm)*0.7)
train_dtm <- dtm[train_indices, ]
test_dtm <- dtm[-train_indices, ]
train_sentiments <- datos$sentiment[train_indices]
test_sentiments <- datos$sentiment[-train_indices]

# Convertir la variable dependiente en un factor
train_sentiments <- as.factor(train_sentiments)
test_sentiments <- as.factor(test_sentiments)

# Entrenar el modelo SVM y hacer predicciones
svm_model <- svm(train_dtm, train_sentiments, kernel = "linear")
predictions <- predict(svm_model, newdata = test_dtm)



