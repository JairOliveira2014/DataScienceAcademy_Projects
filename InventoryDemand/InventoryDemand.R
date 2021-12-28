# Prevendo Demanda de Estoque com Base em Vendas

## 1 - Definição do Problema
"
Este é o projeto final número 2 do curso Big Data Analytics com R e Microsoft Azure Machine Learning que faz parte da formação Cientista de Dados fornecida pela Data Science Academy (www.datascienceacademy.com.br).

O problema de negócio à qual iremos trabalhar a seguir é algo comum às empresas que realizam vendas de produtos, prever a demanda de estoques de forma efetiva evita a necessidade de volumes financeiros
parados em estoque que poderiam ser melhor alocados em áreas mais estratégicas, além de diminuir perdas por retorno de produtos não vendidos.

Sendo assim, utilizaremos os dados disponíveis no kaggle (https://www.kaggle.com/c/grupo-bimbo-inventory-demand), fornecidos pelo grupo Bimbo Mexicano que atende com mais de 100 produtos frescos de panificação
nas suas 45.000 lojas. Como esse produto tem um pequeno tempo de validade, uma demanda efetiva reduz custos significantes de devolução e perda de produtos.

Obs. Muitas das decisões que serão tomadas neste projeto teve influências em pesquisas nas discussões do desáfio no site do kaggle.
"

## 2 - Obtenção dos Dados
"
Os dados foram obtidos do link compartilhado acima à qual realizei o download e trabalharei localmente com o Rstudio.
"
### Definindo diretório de trabalho
setwd("~/Documents/Estudos/DSA/BigDataRAzure/Cap-22/InventoryDemand")
getwd()

### Carregando algumas bibliotecas e os dados com data.table
library(data.table)
library(bigreadr)
library(fasttime)
library(dplyr)
library(lubridate)
library(caret)

client <- fread("cliente_tabla.csv")
product <- fread("producto_tabla.csv")
town <- fread("town_state.csv")
train <- fread("train.csv")
test <- fread("test.csv")

## 3 - Análise Exploratória dos dados
"
Já podemos observar que nossos dados estão normalizados em algumas tables.
"
head(client)
head(product)
head(town)
head(train)
head(test)

str(client)
str(product)
str(town)
str(train)
str(test)
"
Logo de início podemos perceber algumas informações:
- A tabela cliente possui 930,500 samples, mas, de pronto já vemos que ID 4 aparece duplicado, porém com o nome diferente. 
  Este ponto já sido alertado no enunciado da competição.
- Na tabela produto há 2,592 ID's e descrições, porém consegue-se perceber que o ID 0 é para produto não identificado, provavelmente utiliza-se esse ID enquanto o produto
  não esteja com o cadastro completo.
- A tabela town possui 790 samples com Id da agência, identificação da cidade e estado.
- Nos dados de treino possuimos 74,180,464 samples com as informações que usaremos para criar nossos modelos.
- Já nos dados de testes temos 6,999,251 samples, porém, não há as features Venta_uni_hoy e Venta_hoy já que trata-se de dados do futuro e não há ainda essa informação, 
  outro ponto que teremos que nos atentar é o facto de que provavelmente há dados novos de produtos ou até lojas nos dados de teste e logo teremos que tratá-los, neste ponto,
  usar agg de médias pode ser uma ótima solução para esse tipo de tratativa.
- Também há de perceber que a maioria das features possuem o tipo de dado errado, logo, teremos de tratar.
"

### Verificando se há dados NA's e duplicados

anyNA(client)
table(duplicated(client$Cliente_ID)) 
"Não há valores NA's, porém, há 4862 Registros duplicados."

anyNA(product)
table(duplicated(product$Producto_ID))
" Não há valores NA's e nem registros duplicados na tabela product."

anyNA(town)
table(duplicated(town$Agencia_ID))
" Não há valores NA's e nem registros duplicados na tabela town."

anyNA(train)
table(duplicated(train))
" Não há valores NA's e nem registros duplicados na tabela train."

anyNA(test)
table(duplicated(test))
" Não há valores NA's e nem registros duplicados na tabela test."

### Estatísticas básicas dos dados de treino
summary(train)
"
Três informações importantes:
- feature semana: vai de 3 à 9
- features relacionadas à vendas, contém valores baixos à grandes números, que podem configurar-se com outliers. Provavelmente teremos que realizar alguma transformação como exp ou log.
"
### uniques datas
train %>%
  select(Agencia_ID, Canal_ID, Ruta_SAK, Cliente_ID, Producto_ID) %>%
  summarise(Agencia_ID = length(unique(Agencia_ID)), 
            Canal_ID = length(unique(Canal_ID)),
            Ruta_SAK = length(unique(Ruta_SAK)),
            Cliente_ID = length(unique(Cliente_ID)),
            Producto_ID = length(unique(Producto_ID)))
"
Podemos perceber que temos exemplos de vendas em 552 agências, por 9 canais diferentes em 3,603 rotas diferentes com 1799 produtos que foram vendidos para 880,604 clientes.

A seguir criaremos algumas visualizações para extrair melhores insights a respeito de como os dados distribuem-se.
"
### Agora Podemos criar algumas visualizações para fornecer mais clareza a respeito dos dados de treino
library(ggplot2)

train %>%
  mutate_at(c(var = "Semana"), as.factor) %>%
  group_by(Semana) %>%
  summarise(frequency = n()) %>%
  ggplot(aes(x= Semana, y= frequency)) + 
  geom_bar(stat = 'identity', alpha = 0.75, fill = "blue") +
  labs(title = "Frequência de Vendas por Semana") +
  theme_bw()
" Podemos observar acima que a frequência de vendas semanais tem aproximadamente a mesma distribuição"

train %>%
  group_by(Producto_ID) %>%
  summarise(frequency = n()) %>%
  ggplot(aes(x = frequency)) + 
  geom_boxplot(fill = "red", alpha = 0.75) +
  labs(title = "Box plot dos produtos")
"Fica claro em observar acima que há alguns produtos com frequência de vendas maiores que outros, o que é já esperado."

train %>%
  group_by(Cliente_ID) %>%
  summarise(frequency = n()) %>%
  ggplot(aes(x= frequency)) +
  geom_boxplot(fill = "green", alpha = 0.75) +
  labs(title = "BoxPlot da feature Cliente") +
  theme_bw()
"Ao observar o gráfico acima vemos que há um cliente que realiza uma frenquência de compras muito superior aos demais, abaixo podemos ver qual cliente é este."

train %>%
  group_by(Cliente_ID) %>%
  summarise(frequency = n()) %>%
  arrange(desc(frequency)) %>%
  inner_join(client, by= "Cliente_ID")
" Podemos observar que o cliente PUEBLA REMISSION tem uma frequência muito superior ao segundo, isso provavemente pode gerar alguma distorção nos nossos modelos.
  Poderemos lidar com este tópico aplicando alguma tranformações que penalise valores extremos."

train %>%
  mutate_at(c(var= "Canal_ID"), as.factor) %>%
  group_by(Canal_ID) %>%
  summarise(frequency = n()) %>%
  ggplot(aes(x= Canal_ID, y= frequency)) +
  geom_bar(stat = 'identity', fill = "blue", alpha =0.75) +
  theme_bw() +
  labs(title = "Análise canais de vendas")
"Pode-se perceber que o canal de vendas 1 realiza mais de 90% de todas as vendas"

train %>% 
  ggplot(aes(x = Demanda_uni_equil)) +
  geom_boxplot(fill = "red", alpha = 0.75) +
  theme_bw() +
  labs(title = "Análise da Previsão de Demandas")
 " Podemos observar como a variável target está distribuída, de fato há valores que apresentam uma característica aos extremos o que reforça a transformação que precisamos."
 
## 4 - Data Transformation

 " Agora que já conhecemos um pouco melhor nossos dados podemos realizar algumas transformações.
  
  De início sabemos que temos nos dados de treino temos 7 semanas e teremos que realizar a previsão nos dados de 2 semanas. como não temos dados do futuro, o que nos resta é
  fazer o Join com os dados de treino e criar algumas features com base nos produtos e clientes. Também usaremos a última semana e apenas os valores médios das 3 semanas anteriores para
  construirmos nosso dataset (essa última operação deu-se devido a disponibilidade de recursos computacionais)."
 
### De início vamos criar um conjunto de dados com a semana 9 e selecionar algumas varíavesi que não estão nos dados de test
train_1 <- train %>% 
              filter(Semana == 9) %>%
              select(Semana,
                     Agencia_ID,
                     Canal_ID,
                     Ruta_SAK,
                     Cliente_ID,
                     Producto_ID,
                     Demanda_uni_equil)
head(train_1)
head(test)

### Agora vamos fazer o join com os dados testes para criar as features
"Antes de realizarmos os join teremos que criar uma feature ID nos dados de treino e uma feature de controle em ambos para identificação"
train_1$id <- NA
train_1$indicador <- 0

" Podemos avançar agora nos dados de test"
test$indicador <- 1
test$Demanda_uni_equil <- 0

df <- rbind(train_1, test)
table(df$Semana)

### Agora vamos realizar a transformação na nossa variável target
" Uma ótima opção é aplicar uma transformação exponencial ou logarítima, optámos por um log, mas, antes teremos de verificar se há valores 0, pois, neste caso teríamos de aplicar
  log1"
length(train[Demanda_uni_equil==0])
train$Demanda_uni_equil <- log1p(train$Demanda_uni_equil)
df$Demanda_uni_equil <- log1p(df$Demanda_uni_equil)

" Agora podemos verificar como ficou a distribuição da variável target"
df %>%
  ggplot(aes(x=Demanda_uni_equil)) +
  geom_boxplot(fill= "red", alpha= 0.75) +
  theme_bw() +
  labs(title = "Boxplots variável Demanda_uni_equil")

df %>% 
  ggplot(aes(x=Demanda_uni_equil)) +
  geom_density(fill ="red", alpha= 0.75) +
  theme_bw() +
  labs(title = "Distribuição da variável target")
" Podemos perceber claramente que a distribuição de daos melhorou de forma expressiva"  


### Agora vamos criar algumas features com base nos dados passados e assim resolver o problema com os dados de test
df <- train %>%
  filter(Semana > 5 & Semana < 9) %>%
  select(Producto_ID, Demanda_uni_equil) %>%
  group_by(Producto_ID) %>%
  summarise(Med_dem_by_prod = mean(Demanda_uni_equil), 
            Freq_by_prod = n()) %>%
  right_join(df, by= "Producto_ID")

head(df,20)
tail(df)

df <- train %>%
  filter(Semana > 5 & Semana < 9) %>%
  select(Cliente_ID, Demanda_uni_equil) %>%
  group_by(Cliente_ID) %>%
  summarise(Med_dem_by_client = mean(Demanda_uni_equil),
            Freq_by_client = n()) %>%
  right_join(df, by= "Cliente_ID")

head(df,20)
tail(df)
  
df <- train %>% 
  filter(Semana > 5 & Semana < 9) %>%
  select(Cliente_ID, Producto_ID, Demanda_uni_equil) %>%
  group_by(Cliente_ID, Producto_ID) %>%
  summarise(Med_dem_by_ClientProd = mean(Demanda_uni_equil),
            freq_by_ClientProd = n()) %>%
  right_join(df, by=c("Cliente_ID", "Producto_ID"))
      
head(df)
tail(df)  

" Como podemos observar acima criamos news features para resolver o problema de falta de variáveis dos dados de teste.
como a distribuição de vendas semanais é semelhante, apenas criamos as features utilizando como referência as semanas
de 6 à 8.

Sendo assim podemos avançar e criar o modelo preditivo. Caso o nosso modelo não obtenha as melhores métricas podemos fazer feature engineering
com algumas das features que descartamos anteriormente"

### Tratandos os missing values gerado pelo feature engineering
" Neste momento trataremos os Missing Values gerados pelo processo de feature engineering. 
  Poderíamos tomar duas descisões:
  - Preencher com um valor 0;
  - Preencher com a média;
  Neste primeiro caso optámos pela segunda opção e depois testaremos outra opção."

df <- data.frame(sapply(df, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x)))
head(df)

### Padronizando de algumas features para diminuir a necessidade por recurso computacional e trabalharmos na mesma escala
" Uma questão aqui é o porquê de padronizarmos algumas features que são IDs, o motivo é que precisamos destas features como preditoras e como os valores
  estão em uma escala muito grande o modelo pode penalizar aquelas em que possui valores maiores.
"
cols <-  c("Cliente_ID", 
           "Producto_ID",
           "Med_dem_by_ClientProd", 
           "freq_by_ClientProd",
           "Med_dem_by_client", 
           "Freq_by_client", 
           "Med_dem_by_prod", 
           "Freq_by_prod",
           "Semana",
           "Agencia_ID",
           "Canal_ID",
           "Ruta_SAK")

preproc <- preProcess(df[, cols], method="range")
df <- predict(preproc, df)
head(df)

### Dividindo os dados em treino e test novamente
train <- df %>% filter(indicador == 0)
train$id <- NULL
train$indicador <- NULL
head(train)

test <- df %>% filter(indicador == 1)
test$indicador <- NULL
test$Demanda_uni_equil <- NULL
head(test)

## 5 - Criando Modelo Preditivo

set.seed(123)

### Criando o modelo com o lightgbm
library(lightgbm)

### Criando os dados no formato para o lgbm e definindo os parâmetros
lgb.train <- lgb.Dataset(data = data.matrix(train[,-13]), label = data.matrix(train[,13]))
lgb.test <- data.matrix(test[,-13])

params <- list(objective = "regression",
               metric = "L2",
               lambda_l1 = 1.1,
               lambda_l2 = 1.1,
               min_sum_hessian = 8,
               learning_rate = 0.03, 
               num_leaves = 30L,
               max_depth = 10L,
               num_threads = 6)
" metric = rmse, l2 = 1.3"

### encontrando a melhor iteração do modelo com cv
model_lgbm_cv <- lgb.cv(params = params, 
                        data = lgb.train,  
                        nrounds = 2000L, 
                        early_stopping_rounds = 50,
                        eval_freq = 20, 
                        nfold = 10, 
                        verbose = 1)

best_iter <- model_lgbm_cv$best_iter

### treinando o modelo
model_lgbm <- lgb.train(data = lgb.train, 
                        params = params, 
                        nrounds = 2000L)

### Realizando as previsões
pred_lgbm <- predict(model_lgbm, lgb.test)
pred_lgbm

### Salvando em um dataframe com as informações
submission_lgbm <- as.data.frame(test$id)
colnames(submission_lgbm) <- "id"
submission_lgbm$id <- as.integer(submission_lgbm$id)

### Transformando os valores previstos para a escala normal
submission_lgbm["Demanda_uni_equil"] <- as.integer(expm1(pred_lgbm))
  
head(submission_lgbm)   
summary(submission_lgbm)

" Caso haja alguns valores negativos da demanda, logo, teremos que substituir esses valores por 0 "
submission_lgbm$Demanda_uni_equil <- sapply(submission_lgbm$Demanda_uni_equil, function(x) ifelse(x < 0, 0, x))

summary(submission_lgbm$Demanda_uni_equil)
str(submission_lgbm)

### Criando o arquivo de submission
write.csv(submission_lgbm, "submission_lgbm.csv", row.names = FALSE)

"
Private Score <- 0.51018
Public Score <- 0.49596

Podemos observar o melhor valor à qual consegui depois de testar alguns parâmentros com o tempo disponível.
"

### Criando um modelo com xgboost

library(xgboost)

### vamos criar o data.matrix para trabalharmos com o xgb
x_train <- data.matrix(train[,-13])
y_train <- data.matrix(train[,13])

xgb_train <- xgb.DMatrix(data=x_train, label=y_train)

x_test <- data.matrix(test[,-13])

### Criando o modelo
model_xgb <- xgboost(xgb_train, 
               objective = "reg:squarederror",
               booster = "gbtree",
               eta = 0.2,
               max_depth = 15, 
               nround= 100, 
               subsample = 0.7,
               colsample_bytree = 0.85,
               eval_metric = "rmsle",
               nthread = 6, 
               maximize = FALSE,
               verbose = 1)

### Realizando as previsões
pred_xgb <- predict(model_xgb, x_test)

### Salvando em um dataframe com as informações
submission_xgb <- as.data.frame(test$id)
colnames(submission_xgb) <- "id"
submission_xgb$id <- as.integer(submission_xgb$id)

### Transformando os valores previstos para a escala normal
submission_xgb["Demanda_uni_equil"] <- as.integer(expm1(pred_xgb))

head(submission_xgb)   
summary(submission_xgb)

" Vamos substituir os valores menores que 0, por 0, caso existam! "
submission_xgb$Demanda_uni_equil <- sapply(submission_xgb$Demanda_uni_equil, function(x) ifelse(x < 0, 0, x))

summary(submission_xgb$Demanda_uni_equil)
str(submission_xgb)

### Criando o arquivo de submission
write.csv(submission_xgb, "submission_xgb.csv", row.names = FALSE)

" 
Private Score <- 0.51539
Public Score <- 0.49707

Acima podemos observar o melhor resultado alcançado.
" 
## Comentários Finais

" Escolhermos dois modelos que são relativamentes rápidos durante o treino e que possuem booster com base em árvores ou gradientes descendentes.

O Lgbm obteve melhores métricas, embora que muito aquem daquilo que seria o top da competição. Sem dúvidas,  a escolha dos hiperparâmentros é uma das questões mais difíceis.

Contudo, foi gratificante trabalhar com um modelo de previsão de demandas, e sem dúvidas há outras possibilidades que poderiam ser exploradas e algumas transformações poderiam ser melhores trabalhadas. 
Porém, devido estar trabalhando localmente e com memória limitada, algumas decisões foram de encontro aos recursos disponíveis, o que acredito que será uma realidade quando estiver
trabalhando na área.

Como estou cursando a formação Cientista de Dados os cursos seguintes abordaram modelos diferentes, ajustes e até processamento em nuvem que ajudarão na construção de melhores modelos.

Caso tenha alguma sugestão agradeceria grandemente.

Jair Oliveira
"

