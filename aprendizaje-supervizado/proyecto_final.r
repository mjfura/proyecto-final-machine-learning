# INSTALAR LIBRERIAS
install.packages("tm")
# !INSTALAR LIBRERIAS
# CARGAR LIBRERIAS
library(ggplot2)
library(tm)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)

# !CARGAR LIBRERIAS

# CARGAR DATASET
dataset_path <- "dataset/train.csv"
dataset_tweets <- read.csv(dataset_path, header = TRUE, sep = ",", fileEncoding = "ISO-8859-1")
# !CARGAR DATASET

# PREPROCESAMIENTO DE DATOS
head(dataset_tweets)
colnames(dataset_tweets)
colnames(dataset_tweets)[colnames(dataset_tweets) == "Time.of.Tweet"] <- "time_of_tweet"
colnames(dataset_tweets)[colnames(dataset_tweets) == "Age.of.User"] <- "age_user"
colnames(dataset_tweets)[colnames(dataset_tweets) == "Population..2020"] <- "population_2020"
colnames(dataset_tweets)[colnames(dataset_tweets) == "Land.Area..Km.."] <- "land_area_km"
colnames(dataset_tweets)[colnames(dataset_tweets) == "Density..P.Km.."] <- "density_p_km"
colnames(dataset_tweets)

dataset_tweets$sentiment <- as.factor(dataset_tweets$sentiment)
dataset_tweets$time_of_tweet <- as.factor(dataset_tweets$time_of_tweet)
dataset_tweets$age_user <- as.factor(dataset_tweets$age_user)
dataset_tweets$Country <- as.factor(dataset_tweets$Country)
dataset_tweets <- dataset_tweets[, c("sentiment","text","age_user","population_2020","land_area_km","density_p_km", "time_of_tweet", "Country")]
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


removeURL <- function(x) gsub("http\\S+|www\\.\\S+", "", x)
get_corpus <- function(dataset) {
    corpus <- VCorpus(VectorSource(dataset$text))
    corpus <- tm_map(corpus, content_transformer(removeURL))
    corpus <- tm_map(corpus, content_transformer(tolower))
    corpus <- tm_map(corpus, removePunctuation)
    corpus <- tm_map(corpus, removeNumbers)
    corpus <- tm_map(corpus, removeWords, stopwords("en"))
    return(corpus)
}
dataset_tweets$text <- iconv(dataset_tweets$text, to = "UTF-8")
dataset_tweets$text
corpus <- get_corpus(dataset_tweets)
inspect(corpus[[6]])
dtm <- DocumentTermMatrix(corpus, control = list(bounds = list(global = c(5, Inf))))
dtm_tfidf <- weightTfIdf(dtm)

terminos_entrenamiento <- Terms(dtm_tfidf)
dtm_matrix <- as.matrix(dtm_tfidf)
dim(dtm_matrix)
dim(dataset_tweets)
set.seed(123)
ind <- splitTools::partition(dataset_tweets$sentiment, p = c(0.8, 0.2))
train_x <- dtm_matrix[ind$`1`, ]
train_y <- dataset_tweets[ind$`1`, ]$sentiment
valid_x <- dtm_matrix[ind$`2`, ]
valid_y <- dataset_tweets[ind$`2`, ]$sentiment
length(train_y)
dim(train_x)
# !VECTORIZACION

# MODELO ARBOL DE DECISION
modelo_arbol <- rpart(formula = as.factor(train_y) ~ ., data = as.data.frame(train_x), method = "class") 
rpart.plot(modelo_arbol)
predictions <- predict(modelo_arbol, as.data.frame(valid_x), type = "class")
matrix_confusion <- confusionMatrix(predictions, valid_y,mode="everything")
matrix_confusion
# !MODELO ARBOL DE DECISION

# MODELO RANDOM FOREST
df_train_x=as.data.frame(train_x)
ncol(df_train_x)
colnames(train_x) <- make.names(colnames(train_x), unique = TRUE)
modelo_random_forest <- randomForest(as.factor(train_y) ~ ., data = as.data.frame(train_x), ntree = 500)
predictions <- predict(modelo_random_forest, as.data.frame(valid_x))
matrix_confusion <- confusionMatrix(predictions, valid_y,mode="everything")
matrix_confusion
# !MODELO RANDOM FOREST
