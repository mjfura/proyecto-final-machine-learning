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
# !PREPROCESAMIENTO DE DATOS

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
    corpus <- VCorpus(VectorSource(dataset$selected_text))
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
        control <- list(bounds = list(global = c(3, Inf)))
    }
    dtm <- DocumentTermMatrix(corpus, control = control)
    vocabulario <- findFreqTerms(dtm)

    dtm_tfidf <- weightTfIdf(dtm)
    dtm_matrix <- as.matrix(dtm_tfidf)
    df_dtm_matrix <- as.data.frame(dtm_matrix)
    # df_dtm_matrix$time_of_tweet <- dataset$time_of_tweet
    # df_dtm_matrix$age_user <- dataset$age_user
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


# SELECCION DE CARACTERISTICAS
# cv_model <- cv.glmnet(df_dtm_matrix, dataset_tweets$sentiment, alpha = 1, family = "multinomial") # Usar "gaussian" para regresión

# !SELECCION DE CARACTERISTICAS

# !VECTORIZACION

# SPLIT DATASET
split_dataset <- function(dataset) {
    set.seed(123)
    ind <- splitTools::partition(dataset$sentiment, p = c(0.8, 0.2))
    train <- dataset[ind$'1', ]
    valid <- dataset[ind$'2', ]
    train_x <- train[, c("selected_text", "age_user", "time_of_tweet","Country")]
    train_y <- train$sentiment
    valid_x <- valid[, c("selected_text", "age_user", "time_of_tweet", "Country")]
    valid_y <- valid$sentiment
    return(list(train_x=train_x, train_y=train_y, valid_x=valid_x, valid_y=valid_y))
}
# ind <- splitTools::partition(dataset_tweets$sentiment, p = c(0.8, 0.2))
# train_x <- df_dtm_matrix[ind$`1`, ]
# train_y <- dataset_tweets[ind$`1`, ]$sentiment
# valid_x <- df_dtm_matrix[ind$`2`, ]
# valid_y <- dataset_tweets[ind$`2`, ]$sentiment
# length(train_y)
# dim(train_x)
result <- split_dataset(dataset_tweets)
train_x <- result$train_x
train_y <- result$train_y
valid_x <- result$valid_x
valid_y <- result$valid_y
data_train <- cbind(train_x, train_y)
colnames(data_train)
colnames(data_train)[colnames(data_train) == "train_y"] <- "sentiment"

mostrar_distribucion_sentimientos(data_train, data_train$time_of_tweet, "Momento del día")
mostrar_distribucion_sentimientos(dataset_tweets, dataset_tweets$Country, "Países")
mostrar_distribucion_sentimientos(data_train, data_train$age_user, "Edad del usuario")
# !SPLIT DATASET
# VECTORIZACION TRAIN
result <- get_df_vectorizado(train_x)
train_x_vectorizado <- result$df_vectorizado
train_vocabulario <- result$vocabulario
dim(train_x_vectorizado)
train_vocabulario
train_x_vectorizado
head(train_x_vectorizado)
resultado_pca <- calcular_pca(train_x_vectorizado)
# VECTORIZACION VALID
result <- get_df_vectorizado(valid_x, vocabulary = train_vocabulario)
valid_x_vectorizado <- result$df_vectorizado
dim(valid_x_vectorizado)
# !VECTORIZACION TRAIN



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
    num.trees = 100,
    verbose = TRUE,
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
control <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

# Configura los argumentos específicos del método, incluyendo num.threads
methodArgs <- list(
    num.threads = detectCores() - 1, # Ajusta el número de hilos aquí
    verbose = TRUE
    #num.trees = 50
)
tuneGrid <- expand.grid(
    .mtry = c(sqrt(4023) / 2, sqrt(4023), sqrt(4023) * 1.5), # Ejemplos de valores para mtry
    .min.node.size = c(5, 10), # Ejemplos de valores para min.node.size
    .splitrule = "gini" # Ajusta según sea clasificación o regresión
)
# Entrena el modelo usando caret con ranger, pasando num.threads
modelo <- train(
    y=train_y,
    x = df_dtm_matrix_train, # Asegúrate de que df es tu dataframe
    method = "ranger",
    trControl = control,
    #tuneGrid = tuneGrid,
    methodArgs = methodArgs # Pasa los argumentos específicos del método aquí
)

# Ver los resultados del modelo
print(modelo)
# !K-FOLD

# TESTING
dataset_path_test <- "dataset/test.csv"
dataset_tweets_test <- read.csv(dataset_path_test, header = TRUE, sep = ",", fileEncoding = "ISO-8859-1")
dataset_tweets_test <- modificar_dataset(dataset_tweets_test)
# !TESTING