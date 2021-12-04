# Detecção de Fraudes no Tráfego de Cliques em Propagandas de Aplicações Mobile

## Etapa 1 - Definição do Problema de Negócio

"
Este é o 1º projeto final do curso Big Data Analytics com R e Microsoft Azure Machine Learning fornecido pela Data Science Academy.

De fato, chegar até aqui foi um grande desafio e ao mesmo tempo muito gratificante pelas experiências adquiridas ao longo deste curso. Neste projeto temos por 
objetivo criar um modelo de Machine Learning capaz de prever fraudes de cliques em propagandas Mobile.

Esse desafio está disponível no Kaggle pela TalkingData, que é uma plataforma de Big Data independente da China e que lida diariamente para detectar as possíveis fraudes entre
os bilhões de cliques contabilizados todos os dias.

Dessa forma, criarei um modelo simples e que seja capaz de generalizar de forma satisfatória para novos dados.

"

## Etapa 2 - Obtenção dos dados e carregamento

"
Os dados estão disponíveis na competição do kaggle (https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection), fiz o download e estou a trabalhar no Rstudio localmente.
Como o repositório não disponibiliza de tal volume de espaço, os arquivos não serão disponibilizados aqui.
"
### Definindo Diretório de Trabalho
setwd("~/Documents/Estudos/DSA/BigDataRAzure/Cap-22/AdtrackingFraudDetection")
getwd()

### Carregando os dados de train e test

"
Como os conjuntos de dados são enormes, vou carregar samples e depois decidir o volume que o trabalharei (de acordo com aquilo que tenho de recursos).
"
set.seed(123)
sample.train <- read.csv("train.csv", nrows = 1000)
sample.test <- read.csv("test.csv", nrows = 1000 )

head(sample.train)
str(sample.train)

head(sample.test)
str(sample.test)

### nomes das colunas que temos nos datasets
train_cols = c("ip", "app", "device", "os","channel", "click_time", "attributed_time", "is_attributed")

"
Dessa forma já conseguimos ver os tipos de dados de cada feature e as colunas que usaremos para criar um subseting dos dados para trabalharmos.

Obs. Por uma questão de viabilidade de memória eu utilizarei as últimas 10.000.000 de samples dos dados de train, assim facilitará minha manipulação inicial. 
Uma alternativa seria utilizar um serviço em cloud ou até mesmo um BD com sql para manipulação inicial.
"
#install.packages("R.utils")
library(readr)

nL <- R.utils::countLines("train.csv")
lt <- 10000000
df_train <- read_csv("train.csv", col_names = train_cols, skip = nL - lt, show_col_types = TRUE)
head(df_train)


## Etapa 3 - EDA

### Verificando os dados de treino

### Como podemos observar nã há a coluna "attributed_time" nos dados de teste, logo, vamos excluí-la dos dados de treino e também há uma grande qtd de NAs
table(is.na(df_train$attributed_time)) 
df_train$attributed_time <- NULL

### Verificando a variável is_attributed
table(df_train$is_attributed)

"
Como podemos observar acima há muito mais dados de não download dos app o que gera um desbalanceamento muito grande.
Temos duas opções :
1 - utilizar balanceamento com o SMOTE, mas, devido o tamanho tornaria quase impossível trabalhar localmente;
2 - Separar as classes e fazer um subsample já que possuímos muitos dados. (optamos por esta)
"
library(tidyr)
library(dplyr)

df_train_maj <- as.data.frame(filter(df_train, is_attributed == 0))
df_train_min <- as.data.frame(filter(df_train, is_attributed == 1))

"
Agora que já temos os dados da classe marjoritária, vou selecionar 1.000.000 de amostras e fazer o join com a classe minoritária. 
Dessa forma, além de diminuir a necessidade de recursos, melhoramos o balanceamento do dataset.
É óbvio que no caso de aplicação real seria o ideal usar quanto mais dados possível. Porém devido a memória disponível apenas consigo o volume acima.
"
### Selecionando as rows
indexes <- sample(1:nrow(df_train_maj), 1000000, replace = F)
df_train_sample <- df_train_maj[indexes,]

### Fazendo o Join dos datasets
join_cols <- c("ip", "app", "device", "os","channel", "click_time", "is_attributed")
df_train2 <-  merge.data.frame(x=df_train_sample, y=df_train_min, by.x = join_cols, by.y = join_cols, all.x = T, all.y = T)

head(df_train2)

table(df_train2$is_attributed)

"
Agora podemos iniciar com algumas visualizações gráficas
"
library(ggplot2)

### Visualizando a quantidade de valores diferentes nas variáveis categóricas
df_train2 %>%
  select(ip, app, device, os, channel) %>%
  summarise(distinct_ip = length(unique(ip)), 
            distinct_app = length(unique(app)),
            distinct_device = length(unique(device)),
            distinct_os = length(unique(os)),
            distinct_channel = length(unique(channel)))

### vizualizando os dados da variável is_attributed
paste("Os dados de downloads representam apenas",round(table(df_train2$is_attributed)[2] / nrow(df_train2),2), "% de todo o conjunto de dados")

### Vizualizando a correlação entre as variáveis
library(lattice)
cols <- c("ip", "app", "os", "channel", "device", "is_attributed")
metodos <- c("pearson", "spearman")
cors <- lapply(metodos, function(method) 
  (cor(df_train2[, cols], method = method)))

plot.cors <- function(x, labs){
  diag(x) <- 0.0 
  plot( levelplot(x, 
                  main = paste("Plot de Correlação usando Método", labs),
                  scales = list(x = list(rot = 90), cex = 1.0)) )
}

Map(plot.cors, cors, metodos)

"
Podemos notar uma certa correlação entre a variável target, Ip e App, como a variável ip é apenas um identificador não poderá fazer parte do nosso modelo.
No entanto, podemos criar posteriormente uma variável com a quantidade de clicks por ip e sim ainda teremos a informação de certos usuário.

Podemos depois seguir com uma análise temporal após realizarmos algumas transformações nas datas.
"

## Etapa 4 - Data Munging

"
O nosso conjunto de dados armazena cada clik por linha, à qual possui o registro exato do tempo. Podemos agora extrair esses dados e separá-los em features específicas, pois,
se o número de clicks forem muitos próximos para o mesmo ip, há uma grande chance de utilização de fraudes.
Outro Ponto será que não poderemos utilizar o ip pois assim não fará sentido o modelo aprender essa informação, logo, poderemos tratá-la realizando o count de todos os clicks
por ip.
"

### Contando os clicks por ip e adicionando ao dataframe
df <- df_train2 %>% 
  group_by(ip) %>%
  summarise(clicks_by_ip = length(channel))
"
Esse ponto acima foi tomado por decisão pelo fato que o ip em si não trás valor preditivo, mas, ao trabalharmos com a quantidade de clicks por ip
conseguimos manter e atribuir uma característica daquele determinado ip usuário.
"

### Join dos datasets resultantes
df_train3 <- merge.data.frame(x=df_train2, y=df, by="ip")

head(df_train3)

### Agora podemos extrair as informações de datatime
extract.features <- function(df){
  
  df$day_of_year <- strftime(df$click_time, format = "%j")
  df$day_of_week <- strftime(df$click_time, format = "%u")
  df$hour <- strftime(df$click_time, format = "%H")
  df$min <- strftime(df$click_time, format = "%M")
  df$sec <- strftime(df$click_time, format = "%S")
  df$click_time <- NULL
  return(df)
}

### Extraindo as features
df_train3 <- extract.features(df_train3)

### Convertendo o tipo de variáveis para int para podermos prever valores com dados que não seriam possíveis em factores definidos
variables_int <- c("app", "device", "os","channel", "clicks_by_ip", "day_of_year", "day_of_week", "hour", "min", "sec")
to.int <- function(df, variables){
  for (variable in variables) {
    df[[variable]] <- as.integer(df[[variable]])
  }
  return(df)
}

df_train3 <- to.int(df=df_train3, variables = variables_int)

### variável target as factor
df_train3$is_attributed <- as.factor(df_train3$is_attributed)
str(df_train3)

### Excluindo o ip
df_train3$ip <- NULL

## Etapa 5 - Criação do Modelo
"
Devo mencionar que esse processo foi extremamente demorado, embora eu tenha uma máquina com 20G de memória, tive alguns erros no Rstudio e no processo de treino parei de contar
as vezes em que tive que reiniciar tudo devido desligamentos por limit size memory.

Apenas tenho abaixo dois dos modelos à qual obtive melhores resultados, também devo informar que testei e alterei os hiperparâmetros em todo o tempo que obtive disponível. mas,
devido o recurso de tempo e memória, esse é o melhor que posso entregar.

Acredito que a DSA tem sempre alertado para esse processo, tentar simplificar e entregar aquilo que é possível com o recursos que temos.
"

### split de dados em treino e teste
ind <- sample(1:nrow(df_train3), nrow(df_train3) * 0.7)
train <- df_train3[ind,]
test <- df_train3[-ind,]


### Carregando algumas bibliotecas
library(caret)
library(lightgbm)

### Criando o modelo com o lightgbm

### Criando os dados no formato para o lgbm e definindo os parâmetros
lgb.train <- lgb.Dataset(data = data.matrix(train[,-5]), label = data.matrix(train[,5]))
lgb.test <- data.matrix(test[,-5])

params <- list(objective = "binary",
               metric = "auc",
               min_sum_hessian_in_leaf = 1,
               feature_fraction = 0.7,
               bagging_fraction = 0.7,
               bagging_freq = 5,
               min_data = 100,
               max_bin = 50,
               lambda_l1 = 1,
               lambda_l2 = 1.3,
               min_data_in_bin=100,
               min_gain_to_split = 10,
               min_data_in_leaf = 30,
               is_unbalance = TRUE)

### encontrando a melhor iteração do modelo com cv
model_lgbm_cv <- lgb.cv(params = params, 
                     data = lgb.train,  
                     learning_rate = 0.02, 
                     num_leaves = 35,
                     num_threads = 2 ,
                     nrounds = 5000, 
                     early_stopping_rounds = 50,
                     eval_freq = 20, 
                     nfold = 10, stratified = TRUE)

best_iter <- model_lgbm_cv$best_iter

### treinando o modelo
model_lgbm <- lgb.train(data = lgb.train, 
                       params = params, 
                       nrounds = best_iter, 
                       eval_freq = 20, 
                       num_leaves= 35,
                       learning_rate=0.02,
                       num_threads=2)

### Realizando as previsões
pred_lgbm <- predict(model_lgbm, lgb.test)
pred_lgbm

### Criando a confusionMatrix e avaliando o modelo
confM_lgbm <- confusionMatrix(table(round(pred_lgbm), test$is_attributed), positive = "1")

"
Podemos observar o melhor valor à qual consegui depois de testar alguns parâmentros com o tempo disponível.
Também pode-se observar que o nosso modelo possui uma Sensitivity de 0.9, resultado esse ocasionado pelo erros dos FN (639) que considero um número elevado de erros.
Tendo em vista que esse valor compreende a mais de 10% dos acertos em previsão de Downloads"

### Gerando a curva ROC
library(ROCR)
df_ROC <- data.frame(real = test$is_attributed,
                        predict = pred_lgbm)

pred_ROC_ada <- prediction(as.numeric(df_ROC$predict), as.numeric(df_ROC$real))
perf_ROC_ada <- performance(pred_ROC_ada, "tpr","fpr") 
plot(perf_ROC_ada, col = rainbow(10))

### Vizualizando as features importantes
lgb.plot.importance(lgb.importance(model_lgbm), measure = 'Gain')

"
podemos notar que as features app, clicks_by_ip, channel, min, os, sec e device são as mais importantes de acordo com o modelo.
"

### Criando um modelo com XGBoost

#install.packages("xgboost")
library(xgboost)

### vamos criar o data.matrix para trabalharmos com o xgb
x_train <- data.matrix(train[,-5])
y_train <- data.matrix(train[,5])

xgb_train <- xgb.DMatrix(data=x_train, label=y_train)

x_test <- data.matrix(test[,-5])
y_test <- data.matrix(test[,5])

xgb_test <- xgb.DMatrix(data = x_test, label=y_test)

### treinando o modelo
model_xgb <- xgboost(data = xgb_train,
                     eta = 0.3,
                     max_depth = 10, 
                     nround=100, 
                     subsample = 0.5,
                     colsample_bytree = 0.5,
                     eval_metric = "auc",
                     objective = "binary:logistic",
                     nthread = 2,
                     verbose = 1,
                     )

### Realizando as previsões
pred_xgb <- predict(model_xgb, xgb_test)

pred_xgb_fac <- as.factor(round(pred_xgb))

### Avaliando o modelo
confM_xgb <- confusionMatrix(table(pred_xgb_fac, test$is_attributed), positive = "1")

### Gerando a curva ROC
df_ROC <- data.frame(real = test$is_attributed,
                     predict = pred_xgb_fac)

pred_ROC_ada <- prediction(as.numeric(df_ROC$predict), as.numeric(df_ROC$real))
perf_ROC_ada <- performance(pred_ROC_ada, "tpr","fpr") 
plot(perf_ROC_ada, col = rainbow(10))

### verificando as features mais importantes
importance_matrix <- xgb.importance(colnames(x_train), model = model_xgb)
xgb.plot.importance(importance_matrix = importance_matrix, rel_to_first = TRUE, xlab = "Relative Importance")

"
Como podemos observar embora o xgb obteve melhor acc e a métrica sensivity foi muito inferior ao lgbm. O primeiro modelo obteve melhores resultados do ponto de vista de balanceamento.

Também é possível notar que as features importance são todas bem próximas e neste caso também é possível testar outras técnicas de feature engineering.

"
## Criando o arquivo de submissão para o kaggle
df_test <- read_csv("test.csv")
head(df_test)

### Realizando as transformações
df_test <- extract.features(df_test)
head(df_test)

df_temp <- df_test %>% 
  group_by(ip) %>%
  summarise(clicks_by_ip = length(channel))

df_test <- merge.data.frame(x=df_test, y=df_temp, by="ip")

df_test <- to.int(df=df_test, variables = variables_int)
head(df_test)
str(df_test)

### Criando o arquivo para submissão
submission_df <- as.data.frame(df_test[,"click_id"])

colnames(submission_df) <- "click_id"

submission_df$click_id <- as.integer(submission_df$click_id)
head(submission_df)
str(submission_df)

cols_null <- c("click_id", "ip")
df_test[,cols_null] <- NULL

df_test <- data.matrix(df_test)

### Criando as predições com os dois modelos

sub_lgbm <- predict(model_lgbm, data = df_test)
submission_df["is_attributed"] <- as.integer(round(sub_lgbm))
head(submission_df)
write.csv(submission_df, "submission_df_lgbm.csv", row.names = FALSE)

sub_xgb <- predict(model_xgb, newdata = df_test)
submission_df["is_attributed"] <- as.integer(round(sub_xgb))
write.csv(submission_df, "submission_df_xgb.csv", row.names = FALSE)

## Comentários Finais

xgb <-  data.frame("public score" = 0.70615,
              "private score"= 0.74566)
lgbm <-  data.frame("public score" = 0.89343,
                    "private score"= 0.90133)

"
Acima temos os scores obtidos pelos modelos, de fato o lgbm obteve uma capacidade de generalização infinitamente superior. Não podemos deixar de comentar que usei apenas
1M dos mais de 180M de samples para treinar o modelo, com certeza se neste momento dispusesse de mais capacidade de processamento nosso modelo teria algum implemento.

No entanto considero satisfeito com aquilo que alcancei, espero que possa continuar neste processo.

Esse foi sem dúvidas o meu maior desafio até agora, trabalhar com um volume de dados razoável trouxe a tona os problemas reais em gerenciar os recursos. Em segundo lugar
veio o fator tomada de decisão à qual tive que definir onde deveria parar com os recuros que tinha para fazer.
De fato, há muito que poderia e deve ser feito, outras técnicas de feature engineering e outros modelos.

Quanto a decisão do modelo final deu-se devido a complexidade dos dados, e depois de testar alguns, esses foram os que melhor resultaram.


Cumprimentos

Jair Oliveira
"