# INSTALAR LIBRERIAS
install.packages("tm")
install.packages("doParallel")
install.packages('ranger')
install.packages("textstem")
install.packages("Rtsne")
install.packages("e1071")
install.packages("LiblineaR")
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
library(Rtsne)
library(dplyr)
library(text)
library(textstem)
library(e1071)
library(LiblineaR)

# !CARGAR LIBRERIAS

# CARGAR DATASET
dataset_path <- "dataset/train.csv"
dataset_tweets <- read.csv(dataset_path, header = TRUE, sep = ",", fileEncoding = "ISO-8859-1")
# !CARGAR DATASET
# EMBEDDING
dim(dataset_tweets)
load_embeddings <- function(filepath) {
    embeddings_env <- new.env(hash = TRUE) # Usar un entorno para almacenar los embeddings

    # Leer todo el contenido del archivo
    content <- readLines(filepath, encoding = "UTF-8")

    # Procesar cada línea
    for (line in content) {
        elements <- strsplit(line, " ")[[1]] # Dividir la línea en elementos
        if (length(elements) > 1) { # Asegurar que la línea tiene elementos suficientes
            token <- elements[1] # El primer elemento es el término
            vector <- as.numeric(elements[-1]) # Convertir los elementos restantes a numéricos
            embeddings_env[[token]] <- vector # Asignar el vector al término en el entorno
        }
    }

    return(embeddings_env) # Retornar el entorno con los embeddings
}


embeddings_path <- "/Users/marcofura/Downloads/glove/glove.twitter.27B.200d.txt"
embeddings <- load_embeddings(embeddings_path)
embeddings[["work"]]
# !EMBEDDING
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
    lemmatize_corpus <- function(corpus) {
        return(tm_map(corpus, content_transformer(lemmatize_strings)))
    }
    corpus <- lemmatize_corpus(corpus)
    corpus <- tm_map(corpus, removePunctuation)
    # corpus <- tm_map(corpus, content_transformer(removeSpecialCharacters))
    corpus <- tm_map(corpus, removeNumbers)
    # corpus <- tm_map(corpus, content_transformer(removeShortWords))
    corpus <- tm_map(corpus, removeWords, stopwords("en"))
    return(corpus)

}
texto_1 <- "I am working in the park very good"
texto_2 <- "He was doing in the park"
texto_3 <- "They have worked in the park better"
texto_4 <- "I hate work in the park"
df <- data.frame(text = c(texto_1, texto_2, texto_3,texto_4))
corpus <- get_corpus(df)
corpus
embeddings[["park"]]
inspect(corpus)
inspect(corpus[[4]])
result <- get_df_vectorizado_embeddings(df,embeddings)
result
result$df_vectorizado
result$vocabulario
get_df_vectorizado <- function(dataset, vocabulary = NULL) {
    corpus <- get_corpus(dataset)
    # Incorporando bigramas
    # BigramTokenizer <- function(x) {
    #     unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)
    # }

    # control <- list(tokenize = BigramTokenizer)
    # if (!is.null(vocabulary)) {
    #     control$dictionary <- vocabulary
    # } else {
    #     control$bounds <- list(global = c(1, Inf))
    # }

    control <- list(dictionary = vocabulary)
    if (missing(vocabulary)) {
        control <- list(bounds = list(global = c(2, Inf)))
    }
    dtm <- DocumentTermMatrix(corpus, control = control)
    vocabulario <- findFreqTerms(dtm)

    dtm_tfidf <- weightTfIdf(dtm)
    dtm_matrix <- as.matrix(dtm_tfidf)
    df_dtm_matrix <- as.data.frame(dtm_matrix)
    # df_dtm_matrix$Country <- dataset$Country
    # df_dtm_matrix$age_user <- dataset$age_user
    # df_dtm_matrix$time_of_tweet <- dataset$time_of_tweet
    # df_dtm_matrix$population_2020 <- dataset$population_2020
    # df_dtm_matrix$land_area_km <- dataset$land_area_km
    # df_dtm_matrix$density_p_km <- dataset$density_p_km

    
    return(list(df_vectorizado=df_dtm_matrix, vocabulario=vocabulario))
}
vectorize_document <- function(document, embeddings, vector_size) {
    words <- strsplit(document, "\\s+")[[1]]
    vectors <- list()

    for (word in words) {
        if (!is.null(embeddings[[word]])) {
            vectors[[length(vectors) + 1]] <- embeddings[[word]]
        }
    }

    if (length(vectors) == 0) {
        return(rep(0, vector_size)) # Retorna un vector de ceros si no hay vectores
    }

    document_vector <- Reduce("+", vectors) / length(vectors)
    return(document_vector)
}
get_df_vectorizado_embeddings <- function(dataset, embeddings) {
    # Preparar el corpus
    corpus <- get_corpus(dataset)

    # Convertir el corpus a texto plano para procesamiento
    textos <- sapply(corpus, as.character)

    # Vectorizar cada documento
    vectores_documentos <- lapply(textos, vectorize_document, embeddings = embeddings)

    # Filtrar documentos que no pudieron ser vectorizados (NULLs)
    vectores_documentos <- Filter(Negate(is.null), vectores_documentos)

    # Convertir la lista de vectores en un data.frame
    df_vectores <- do.call(rbind, vectores_documentos)
    df_vectores <- as.data.frame(df_vectores)

    return(df_vectores)
}
resultado <- get_df_vectorizado_embeddings(df,embeddings)
names(embeddings)[10]
dim(resultado)
head(resultado)
colnames(resultado)
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
# VECTORIZACION EMBEDDINGS
dim(train_x)
train_x_vectorizado <- get_df_vectorizado_embeddings(train_x,embeddings)
valid_x_vectorizado <- get_df_vectorizado_embeddings(valid_x,embeddings)
dim(train_x_vectorizado)
dim(valid_x_vectorizado)
head(train_x_vectorizado)
# !VECTORIZACION EMBEDDINGS


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
train_vocabulario
length(train_y)
dim(train_x_vectorizado)
set.seed(123)
modelo_ranger <- ranger(
    y = train_y,
    x = train_x_vectorizado,
    num.trees = 500,
    verbose = TRUE,
    mtry = sqrt(ncol(train_x_vectorizado)),
    importance = "impurity",
    #min.node.size = 0.05,
    num.threads = detectCores() - 1 # Utiliza todos los núcleos disponibles menos uno
)
importancia_vars <- modelo_ranger$variable.importance
importancia_vars_ordenadas <- sort(importancia_vars, decreasing = TRUE)
par(mar = c(7, 4, 4, 2) + 0.1)

# Crear el gráfico de barras con etiquetas más grandes y en un ángulo de 45 grados para mejor legibilidad
barplot(importancia_vars_ordenadas[1:10], 
        main = "Importancia de las Variables en el Modelo Ranger", 
        las = 2, 
        cex.names = 1, # Ajustar el tamaño de las etiquetas
        names.arg = names(importancia_vars_ordenadas[1:10]), 
        las = 2) # Ángulo de las etiquetas

# Restaurar los márgenes a los valores predeterminados (opcional)
importancia_vars_ordenadas[1:50]
par(mar = c(5, 4, 4, 2) + 0.1)
all(names(train_x) %in% names(valid_x))
predictions <- predict(modelo_ranger, data = valid_x_vectorizado, type = "response")
predictions$predictions
matrix_confusion <- confusionMatrix(predictions$predictions, valid_y, mode = "everything")
matrix_confusion
# limitar frecuencia a 3
# !MODELO RANDOM FOREST
# Modelo SVM
# modelo_svm <- svm(train_x_vectorizado, train_y, kernel = "linear", cost = 1, scale = FALSE)
modelo_svm_liblinear <- LiblineaR(
    data = train_x_vectorizado,
    target = train_y,
    type = 7, # L2-regularized L2-loss support vector classification
    cost = 0.1, # El parámetro de coste, similar al C en otras implementaciones SVM
    verbose = TRUE
) # Muestra información del proceso de entrenamiento
modelo_svm_liblinear
predicciones <- predict(modelo_svm_liblinear, valid_x_vectorizado)
matrix_confusion <- confusionMatrix(predicciones$predictions, valid_y, mode = "everything")
matrix_confusion
# !Modelo SVM

# K-FOLD
#TODO: Toma mucho tiempo su ejecución, por el momento ignorar
corpus <- get_corpus(dataset_tweets)
num_documentos_vacios <- sum(sapply(corpus, nchar) == 0)
num_documentos_vacios
inspect(corpus[[4]])
dataset_tweets$text[4]
resultado <- get_df_vectorizado(dataset_tweets)
data_train_vectorizado <- resultado$df_vectorizado
data_train_vocabulario <- resultado$vocabulario
dim(data_train_vectorizado)
control <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

# Configura los argumentos específicos del método, incluyendo num.threads
methodArgs <- list(
    num.threads = detectCores() - 1, # Ajusta el número de hilos aquí
    verbose = TRUE,
    #num.trees = 50
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
    y = train_y,
    x = train_x_vectorizado, # Asegúrate de que df es tu dataframe
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
# MODELO FINAL

resultado <- get_df_vectorizado(dataset_tweets)
data_vectorizado <- resultado$df_vectorizado
data_vocabulario <- resultado$vocabulario
data_vectorizado$sentiment <- dataset_tweets$sentiment
ind <- splitTools::partition(data_vectorizado$sentiment, p = c(0.9, 0.1))
train <- data_vectorizado[ind$"1", ]
valid <- data_vectorizado[ind$"2", ]
train_x <- select(train, -sentiment)
train_y <- train$sentiment
valid_x <- select(valid, -sentiment)
valid_y <- valid$sentiment
dim(data_vectorizado)
set.seed(123)
modelo_ranger <- ranger(
    y = dataset_tweets$sentiment,
    x = data_vectorizado,
    num.trees = 500,
    verbose = TRUE,
    mtry = sqrt(ncol(data_vectorizado)),
    #min.node.size = 0.05,
    num.threads = detectCores() - 1 # Utiliza todos los núcleos disponibles menos uno
)
predicciones <- predict(modelo_ranger, data = valid_x, type = "response")
matrix_confusion <- confusionMatrix(predicciones$predictions, valid_y, mode = "everything")
matrix_confusion
# !MODELO FINAL
# TESTING
dataset_path_test <- "dataset/test.csv"
dataset_tweets_test <- read.csv(dataset_path_test, header = TRUE, sep = ",", fileEncoding = "ISO-8859-1")
dim(dataset_tweets_test)
colnames(dataset_tweets_test)
modificar_dataset_test <- function(dataset) {
    dataset_limpio <- na.omit(dataset)

    # Eliminar filas con valores vacíos
    dataset <- dataset_limpio[sapply(dataset_limpio, function(x) !any(x == "")), ]
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
resultado <- get_df_vectorizado(dataset_tweets_test, vocabulary = data_vocabulario)
data_test_vectorizado <- resultado$df_vectorizado
dim(data_test_vectorizado)
length(dataset_tweets_test$sentiment)
table(dataset_tweets_test$sentiment)
dataset_tweets_test$sentiment

predictions <- predict(modelo_ranger, data = data_test_vectorizado, type = "response")
matrix_confusion <- confusionMatrix(predictions$predictions, dataset_tweets_test$sentiment, mode = "everything")
matrix_confusion
# !TESTING
#VISUALIZACION
dim(valid_x)
nrow(unique(valid_x))
indices_unicos <- !duplicated(data_test_vectorizado)
valid_x_unico <- data_test_vectorizado[indices_unicos, ]
etiquetas_unicas <- dataset_tweets_test$sentiment[indices_unicos]
tsne_resultados <- Rtsne(valid_x_unico, dims = 2, perplexity = 30, verbose = TRUE)
tsne_resultados
tsne_df <- as.data.frame(tsne_resultados$Y)

tsne_df$etiqueta <- etiquetas_unicas
ggplot(tsne_df, aes(x = V1, y = V2, color = etiqueta)) +
    geom_point() +
    theme_minimal() +
    labs(
        title = "t-SNE: Componentes con Colores por Etiqueta Real",
        x = "Componente 1",
        y = "Componente 2",
        color = "Etiqueta"
    ) +
    scale_color_discrete(name = "Etiqueta")
ggplot(tsne_df, aes(x = V1, y = V2, color = predictions$predictions[indices_unicos])) +
    geom_point() +
    theme_minimal() +
    labs(
        title = "t-SNE: Componentes con Colores por Etiqueta Predecidas",
        x = "Componente 1",
        y = "Componente 2",
        color = "Etiqueta"
    ) +
    scale_color_discrete(name = "Etiqueta")
#!VISUALIZACION