# CARGAR DATASET
dataset_path <- "dataset/train.csv"
dataset_tweets <- read.csv(dataset_path, header = TRUE, sep = ",", fileEncoding = "ISO-8859-1")
# !CARGAR DATASET

# PREPROCESAMIENTO DE DATOS
head(dataset_tweets)
colnames(dataset_tweets)
dataset_tweets$sentiment <- as.factor(dataset_tweets$sentiment)


# !PREPROCESAMIENTO DE DATOS

# ANÁLISIS EXPLORATORIO DE DATOS
nrow(dataset_tweets)
summary(dataset_tweets)
dim(dataset_tweets)
str(dataset_tweets)
# !ANÁLISIS EXPLORATORIO DE DATOS