# CARGAR LIBRERIAS
library(ggplot2)

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