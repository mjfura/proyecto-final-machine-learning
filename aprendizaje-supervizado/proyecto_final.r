# INSTALAR LIBRERIAS
install.packages("tm")
install.packages("doParallel")
install.packages('ranger')
install.packages("textstem")
# !INSTALAR LIBRERIAS
# CARGAR LIBRERIAS
library(ggplot2)
library(tm)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(doParallel)
library(ranger)
library(glmnet)
library(textstem)

# !CARGAR LIBRERIAS

# CARGAR DATASET
dataset_path <- "dataset/train.csv"
dataset_tweets <- read.csv(dataset_path, header = TRUE, sep = ",", fileEncoding = "ISO-8859-1")
# !CARGAR DATASET

# PREPROCESAMIENTO DE DATOS
head(dataset_tweets)
colnames(dataset_tweets)
modificar_dataset <- function(dataset) {
    colnames(dataset)[colnames(dataset) == "Time.of.Tweet"] <- "time_of_tweet"
    colnames(dataset)[colnames(dataset) == "Age.of.User"] <- "age_user"
    colnames(dataset)[colnames(dataset) == "Population..2020"] <- "population_2020"
    colnames(dataset)[colnames(dataset) == "Land.Area..Km.."] <- "land_area_km"
    colnames(dataset)[colnames(dataset) == "Density..P.Km.."] <- "density_p_km"
    dataset$sentiment <- as.factor(dataset$sentiment)
    dataset$time_of_tweet <- as.factor(dataset$time_of_tweet)
    dataset$age_user <- as.factor(dataset$age_user)
    dataset$Country <- as.factor(dataset$Country)
    dataset <- dataset[, c("sentiment", "text", "selected_text", "age_user", "population_2020", "land_area_km", "density_p_km", "time_of_tweet", "Country")]
    return(dataset)
}
dataset_tweets <- modificar_dataset(dataset_tweets)
colnames(dataset_tweets)
dim(dataset_tweets)
dataset_tweets$Country
dataset_tweets <- subset(dataset_tweets, Country %in% c("Australia", "Canada", "Ireland", "New Zealand", "South Africa", "United Kingdom", "United States of America")) 
# !PREPROCESAMIENTO DE DATOS
dim(dataset_tweets)
table(dataset_tweets$Country)
head(dataset_tweets)
# ANÁLISIS EXPLORATORIO DE DATOS
nrow(dataset_tweets)
summary(dataset_tweets)
dim(dataset_tweets)
str(dataset_tweets)

table(dataset_tweets$sentiment)
table(dataset_tweets$time_of_tweet)
table(dataset_tweets$age_user)
table(dataset_tweets$Country)

barplot(table(dataset_tweets$sentiment), main = "Frecuencia de Sentimientos", xlab = "Sentimiento", ylab = "Frecuencia")
mostrar_distribucion_sentimientos <- function(dataset, columna,label) {
    ggplot(dataset, aes(x = columna, fill = sentiment)) +
        geom_bar(position = "fill") +
        scale_y_continuous(labels = scales::percent_format()) +
        labs(
            title = "Distribución de Sentimientos",
            x = label,
            y = "Porcentaje de Sentimientos",
            fill = "Sentimiento"
        ) +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
}
mostrar_distribucion_sentimientos(dataset_tweets, dataset_tweets$time_of_tweet, "Momento del día")
mostrar_distribucion_sentimientos(dataset_tweets, dataset_tweets$Country, "Países")
mostrar_distribucion_sentimientos(dataset_tweets, dataset_tweets$age_user, "Edad del usuario")
# !ANÁLISIS EXPLORATORIO DE DATOS

# VECTORIZACION

get_corpus <- function(dataset) {
    removeURL <- function(x) gsub("http\\S+|www\\.\\S+", "", x)
    # removeShortWords <- function(x) gsub("\\b\\w{1}\\b", "", x)
    # removeSpecialCharacters <- function(x) gsub("[^a-zA-Z\\s]", "", x)
    # dataset$text <- iconv(dataset$text, to = "UTF-8")
    corpus <- VCorpus(VectorSource(dataset$text))
    corpus <- tm_map(corpus, content_transformer(removeURL))
    corpus <- tm_map(corpus, content_transformer(tolower))
    corpus <- tm_map(corpus, removePunctuation)
    # corpus <- tm_map(corpus, content_transformer(removeSpecialCharacters))
    corpus <- tm_map(corpus, removeNumbers)
    # corpus <- tm_map(corpus, content_transformer(removeShortWords))
    corpus <- tm_map(corpus, removeWords, stopwords("en"))
    lemmatize_corpus <- function(corpus) {
        return(tm_map(corpus, content_transformer(lemmatize_strings)))
    }
    corpus <- lemmatize_corpus(corpus)
    return(corpus)

}
get_df_vectorizado <- function(dataset, vocabulary = NULL) {
    corpus <- get_corpus(dataset)
    control <- list(dictionary = vocabulary)
    if (missing(vocabulary)) {
        control <- list(bounds = list(global = c(5, Inf)))
    }
    dtm <- DocumentTermMatrix(corpus, control = control)
    vocabulario <- findFreqTerms(dtm)

    dtm_tfidf <- weightTfIdf(dtm)
    dtm_matrix <- as.matrix(dtm_tfidf)
    df_dtm_matrix <- as.data.frame(dtm_matrix)
    return(list(df_vectorizado=df_dtm_matrix, vocabulario=vocabulario))
}


calcular_pca <- function(df_dataset){
    pca_resultado <- prcomp(df_dataset, center = TRUE, scale. = TRUE)
    valores_propios <- pca_resultado$sdev^2
    varianza_explicada <- valores_propios / sum(valores_propios)
    varianza_acumulada <- cumsum(varianza_explicada)

    # Imprimir la varianza explicada y acumulada
    print(varianza_explicada)
    print(varianza_acumulada)
    return(pca_resultado)
}
# !VECTORIZACION

# SPLIT DATASET
split_dataset <- function(dataset) {
    set.seed(123)
    ind <- splitTools::partition(dataset$sentiment, p = c(0.9, 0.1))
    train <- dataset[ind$'1', ]
    valid <- dataset[ind$'2', ]
    train_x <- train[, c("text", "age_user", "time_of_tweet","Country")]
    train_y <- train$sentiment
    valid_x <- valid[, c("text", "age_user", "time_of_tweet", "Country")]
    valid_y <- valid$sentiment
    return(list(train_x=train_x, train_y=train_y, valid_x=valid_x, valid_y=valid_y))
}

result <- split_dataset(dataset_tweets)
train_x <- result$train_x
train_y <- result$train_y
valid_x <- result$valid_x
valid_y <- result$valid_y

# VISUALIZACION DE DATOS DESPUES DE LA PARTICIION
data_train <- cbind(train_x, train_y)
colnames(data_train)
colnames(data_train)[colnames(data_train) == "train_y"] <- "sentiment"
mostrar_distribucion_sentimientos(data_train, data_train$time_of_tweet, "Momento del día")
mostrar_distribucion_sentimientos(dataset_tweets, dataset_tweets$Country, "Países")
mostrar_distribucion_sentimientos(data_train, data_train$age_user, "Edad del usuario")
# !VISUALIZACION DE DATOS DESPUES DE LA PARTICIION

# !SPLIT DATASET
# VECTORIZACION TRAIN
result <- get_df_vectorizado(train_x)
train_x_vectorizado <- result$df_vectorizado
train_vocabulario <- result$vocabulario
dim(train_x_vectorizado)
train_vocabulario
train_x_vectorizado
head(train_x_vectorizado)
# resultado_pca <- calcular_pca(train_x_vectorizado)

# !VECTORIZACION TRAIN
# VECTORIZACION VALID
result <- get_df_vectorizado(valid_x, vocabulary = train_vocabulario)
valid_x_vectorizado <- result$df_vectorizado
dim(valid_x_vectorizado)
# !VECTORIZACION VALID



# MODELO ARBOL DE DECISION
#TODO: Toma mucho tiempo su ejecución, por el momento ignorar
modelo_arbol <- rpart(formula = as.factor(train_y) ~ ., data = as.data.frame(train_x), method = "class") 
rpart.plot(modelo_arbol)
predictions <- predict(modelo_arbol, as.data.frame(valid_x), type = "class")
matrix_confusion <- confusionMatrix(predictions, valid_y,mode="everything")
matrix_confusion
# !MODELO ARBOL DE DECISION
# K-FOLD
#TODO: Toma mucho tiempo su ejecución, por el momento ignorar
trC <- trainControl(method = "cv", number = 10)
arbol_kfold <- train(y=train_y,x=train_x, method = "rpart", trControl = trC)
arbol_kfold$results
arbol_kfold$resample
arbol_kfold$bestTune
predictions <- predict(arbol_kfold, valid_x)

matrix_confusion <- confusionMatrix(predictions, valid_y, mode = "everything")
matrix_confusion
# !K-FOLD
# MODELO RANDOM FOREST
modelo_ranger <- ranger(
    y = train_y,
    x = train_x_vectorizado,
    num.trees = 200,
    verbose = TRUE,
    mtry = sqrt(ncol(train_x_vectorizado)),
    min.node.size = 0.05,
    num.threads = detectCores() - 1 # Utiliza todos los núcleos disponibles menos uno
)

all(names(train_x) %in% names(valid_x))
predictions <- predict(modelo_ranger, data = valid_x_vectorizado, type = "response")
predictions$predictions
matrix_confusion <- confusionMatrix(predictions$predictions, valid_y, mode = "everything")
matrix_confusion
# !MODELO RANDOM FOREST

# K-FOLD
#TODO: Toma mucho tiempo su ejecución, por el momento ignorar
resultado <- get_df_vectorizado(dataset_tweets)
data_train_vectorizado <- resultado$df_vectorizado
data_train_vocabulario <- resultado$vocabulario
dim(data_train_vectorizado)
control <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

# Configura los argumentos específicos del método, incluyendo num.threads
methodArgs <- list(
    num.threads = detectCores() - 1, # Ajusta el número de hilos aquí
    verbose = TRUE,
    num.trees = 50
)
size = length(data_train_vectorizado)
sqr_size=sqrt(size)
tuneGrid <- expand.grid(
    .mtry = c(sqr_size), # Ejemplos de valores para mtry
    .min.node.size = c(1), # Ejemplos de valores para min.node.size
    .splitrule = "gini" # Ajusta según sea clasificación o regresión
)
# Entrena el modelo usando caret con ranger, pasando num.threads
modelo <- train(
    y = dataset_tweets$sentiment,
    x = data_train_vectorizado, # Asegúrate de que df es tu dataframe
    method = "ranger",
    trControl = control
    #tuneGrid = tuneGrid,
    #methodArgs = methodArgs # Pasa los argumentos específicos del método aquí
)

# Ver los resultados del modelo
modelo$results
modelo$resample
modelo$bestTune

# !K-FOLD

# TESTING
dataset_path_test <- "dataset/test.csv"
dataset_tweets_test <- read.csv(dataset_path_test, header = TRUE, sep = ",", fileEncoding = "ISO-8859-1")
dim(dataset_tweets_test)
colnames(dataset_tweets_test)
modificar_dataset_test <- function(dataset) {
    colnames(dataset)[colnames(dataset) == "Time.of.Tweet"] <- "time_of_tweet"
    colnames(dataset)[colnames(dataset) == "Age.of.User"] <- "age_user"
    colnames(dataset)[colnames(dataset) == "Population..2020"] <- "population_2020"
    colnames(dataset)[colnames(dataset) == "Land.Area..Km.."] <- "land_area_km"
    colnames(dataset)[colnames(dataset) == "Density..P.Km.."] <- "density_p_km"
    dataset$sentiment <- as.factor(dataset$sentiment)
    dataset$time_of_tweet <- as.factor(dataset$time_of_tweet)
    dataset$age_user <- as.factor(dataset$age_user)
    dataset$Country <- as.factor(dataset$Country)
    dataset <- dataset[, c("sentiment", "text", "age_user", "population_2020", "land_area_km", "density_p_km", "time_of_tweet", "Country")]
    return(dataset)
}
dataset_tweets_test <- modificar_dataset_test(dataset_tweets_test)
resultado <- get_df_vectorizado(dataset_tweets_test, vocabulary = train_vocabulario)
data_test_vectorizado <- resultado$df_vectorizado
dim(data_test_vectorizado)
length(dataset_tweets_test$sentiment)
predictions <- predict(modelo_ranger, data = data_test_vectorizado, type = "response")
matrix_confusion <- confusionMatrix(predictions$predictions, dataset_tweets_test$sentiment, mode = "everything")
matrix_confusion
# !TESTING