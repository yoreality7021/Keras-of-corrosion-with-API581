#AI model-1：預測PH & H2O-----
#先行切割15%當作驗證使用-----
pph=read.csv(file.choose())
pph=pph[,-1]
#將aspen模擬出的PH值以0.001取代
pph$Y2=replace(pph$Y2,pph$Y2<0,0.001)
#不要科學記號
options(scipen = 999)
smp.size = floor(0.85*nrow(pph)) #85%，nrow()是用來擷取資料筆數
train.ind = sample(seq_len(nrow(pph)), smp.size) #85%訓練資料
train1 = pph[train.ind,] #85%訓練
test1 = pph[-train.ind,] #15%驗證
nrow(train1)
nrow(test1)
write.csv(train1,file="C:/Users/bigje/Desktop/(R)train85%.csv")
write.csv(test1,file="C:/Users/bigje/Desktop/(R)var15%.csv")

#在從85%中切割80/20-----
smp.size = floor(0.80*nrow(train1)) #80%，nrow()是用來擷取資料筆數
train.ind = sample(seq_len(nrow(train1)), smp.size) #20%測試資料
train2 = train1[train.ind,] 
test2 = train1[-train.ind,] 
nrow(train2)
nrow(test2)
write.csv(train2,file="C:/Users/bigje/Desktop/(R)train80%.csv")
write.csv(test2,file="C:/Users/bigje/Desktop/(R)test20%.csv")

#直接讀取抽樣資料-----
train2=read.csv(file.choose())
test2=read.csv(file.choose())

#正規化(maxmin)-----
maxmin=function (x){ (x - min(x)) / (max(x)-min(x)) }
{
  pphdf=as.data.frame(lapply(train2, maxmin))
  pphtrain.x=as.matrix(pphdf[,-6:-7])
  pphtrain.y=as.matrix(pphdf[,6:7])
  pphtestdf=as.data.frame(lapply(test2, maxmin))
  pphtest.x=as.matrix(pphtestdf[,-6:-7])
  pphtest.y=as.matrix(pphtestdf[,6:7])
}

#[Keras NN]模型建立-----
library(devtools)
library(R6)
library(curl)
#從github下載安裝keras套件
# devtools::install_github("rstudio/keras")
#啟用keras套件
library(keras)
#安裝keras集TensorFlow(CPU版&GPU版，二擇一)
#install_keras()                    #CPU版，第一次安裝使用
#install_keras(tensorflow = "gpu")  #GPU版
{
  model0 = keras_model_sequential()
  model0 %>%
    layer_dense(units = 128, activation = 'relu', 
                kernel_initializer = 'he_normal', input_shape = c(5)) %>%
    #layer_dropout(rate = 0.1) %>%
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dense(units = 2, activation = 'sigmoid' )
  summary(model0)
  
  # # loss function(損失函數)
  # model0 %>% compile(
  #   loss = 'mean_squared_logarithmic_error', optimizer = 'adam', metrics = c('accuracy'))
  # #loss function(L2)
  model0 %>% compile(
    loss = 'mean_squared_error', optimizer = 'adam', metrics = c('accuracy') )
  # #loss function(L1)
  # model0 %>% compile(
  #   loss = 'mean_absolute_error', optimizer = 'adam', metrics = c('accuracy')) )
  
  #keras training
  model00 = model0 %>% fit( 
    pphtrain.x, pphtrain.y, epochs = 500, batch_size=16)
  #keras training model save in workspace
  #setwd("C:/Users/bigje/Desktop/keras")
  model0 %>% save_model_hdf5("keras-model.h5", "C:/Users/bigje/Desktop/keras",
                             overwrite = FALSE)
  # #若要讀取為
  # new_model = load_model_hdf5("keras-model.h5")
  # newmodel = load_model_hdf5("C/Users/bigje/Desktop/keras/keras.h5")
  # #評估模型loss & accuracy
  # model0 %>% evaluate(pphtest.x, pphtest.y)
  # #plot(model0)
  # #預測PH & H2O
  # pred0=data.frame(predict(model0, pphtest.x))
  # pred0=cbind(pred0,pphtest.y)
  # colnames(pred0)=c("Y1","Y2","T1","T2")
  
}

# #test predict data 預測值還原
# remaxminY1 = function (x){ x*(max(test2$Y1)-min(test2$Y1))+min(test2$Y1) }
# remaxminY2 = function (x){ x*(max(test2$Y2)-min(test2$Y2))+min(test2$Y2) }
# b=data.frame(cbind(remaxminY1(pred0$Y1),remaxminY1(pred0$T1),
#                    remaxminY2(pred0$Y2),remaxminY2(pred0$T2)) )
# colnames(b)=c("predY1","TrueY1","predY2","TrueY2")
# #test predict data 模行估計指標
# a=data.frame(cbind( R_squared(b$TrueY1,b$predY1),R_squared(b$TrueY2,b$predY2),
#                     RMSE(b$predY1,b$TrueY1),RMSE(b$predY2,b$TrueY2),
#                     mape(b$TrueY1,b$predY1),mape(b$TrueY2,b$predY2)))
# colnames(a)=c("R2_H2O","R2_PH","RMSE_H2O","RMSE_PH","MAPE_H2O","MAPE_PH")


# #驗證15%的數據-----
# v15=read.csv(file.choose())
# v15=v15[,-1]
# #正規化(0~1之間)(maxmin)
# maxmin=function (x){ (x - min(x)) / (max(x)-min(x)) }
# {
#   v15testdf=as.data.frame(lapply(v15, maxmin))
#   v15test.x=as.matrix(v15testdf[,-6:-7])
#   v15test.y=as.matrix(v15testdf[,6:7])
# }
# #Keras NN model
# {
#   #預測PH & H2O
#   v15pred0=data.frame(predict(model0, v15test.x))
#   v15pred0=cbind(v15pred0,v15test.y)
#   colnames(v15pred0)=c("Y1","Y2","T1","T2")
# }
# 
# #var15%預測值還原
# remaxminv15Y1 = function (x){ x*(max(v15$Y1)-min(v15$Y1))+min(v15$Y1) }
# remaxminv15Y2 = function (x){ x*(max(v15$Y2)-min(v15$Y2))+min(v15$Y2) }
# v15b=data.frame(cbind(remaxminv15Y1(v15pred0$Y1),remaxminv15Y1(v15pred0$T1),
#                       remaxminv15Y2(v15pred0$Y2),remaxminv15Y2(v15pred0$T2)) )
# colnames(v15b)=c("predY1","TrueY1","predY2","TrueY2")
# #var15%模型估計指標
# v15a=data.frame(cbind( R_squared(v15b$TrueY1,v15b$predY1),
#                        R_squared(v15b$TrueY2,v15b$predY2),
#                        RMSE(v15b$predY1,v15b$TrueY1),RMSE(v15b$predY2,v15b$TrueY2),
#                        mape(v15b$TrueY1,v15b$predY1),mape(v15b$TrueY2,v15b$predY2)))
# colnames(v15a)=c("R2_H2O","R2_PH","RMSE_H2O","RMSE_PH","MAPE_H2O","MAPE_PH")
# 
# View(rbind(a,v15a))

#AI model2：API 581-----
# #KNN
# #class包
# library(class)#載入其中的knn
# ktrain=read.csv(file.choose()) #mpy1(1)
# ktrain=ktrain[,-1]
# ktrain$Temp=as.numeric(ktrain$Temp)
# kx_train=as.matrix( ktrain[,1:2] )
# ky_train=as.matrix(ktrain[,-1:-2])
# colnames(ky_train)=c("HCI")
# # kx_test=kx_train
# # ky_test=ky_train
# #knn(分群=2)
# kpre = knn(train = kx_train, test = kx_test, cl = ky_train, k = 2)
# # kt=data.frame(cbind(cl,as.character(kpre) ))#輸出
# # kt$X1=as.numeric(kt$X1)
# # kt$X2=as.numeric(kt$X2)

#xgboost
library(xgboost)
library(Matrix)
df=read.csv(file.choose())
df=df[,-1]
#先轉成稀疏矩陣
dtrain = xgb.DMatrix(data = as.matrix(df[,1:2]), label = df$HCI)
dtest = xgb.DMatrix(data = as.matrix(df[,1:2]), label = df$HCI)
#模型建立
xgb.model =  xgboost(data = dtrain,
                     max_depth=10, eta=0.3, min_child_weight=1,  
                     objective='reg:linear', 
                     nround=100)
#預測
xgbpred = predict(xgb.model, dtest)
xgbpred1 = data.frame(cbind(round(xgbpred,2),df$HCI))
names(xgbpred1)=c("pred","true")
#存檔
save(xgb.model, file = "D:/AI-work/FCFC-ARO3/R-code/Rda/xgbmodel.rda")
save(xgbpred1, file = "D:/AI-work/FCFC-ARO3/R-code/Rda/xgbpred.rda")
#新數據進入keras NN預測-----
yu=data.frame(280,80,1,0.005,0.001)
colnames(yu)=c("x1","x2","x3","x4","x5")
#正規化新數據(使用訓練資料格式)
yux1maxmin=function (x){ (x - min(train2$x1)) / (max(train2$x1)-min(train2$x1)) }
yux2maxmin=function (x){ (x - min(train2$x2)) / (max(train2$x2)-min(train2$x2)) }
yux3maxmin=function (x){ (x - min(train2$x3)) / (max(train2$x3)-min(train2$x3)) }
yux4maxmin=function (x){ (x - min(train2$x4)) / (max(train2$x4)-min(train2$x4)) }
yux5maxmin=function (x){ (x - min(train2$x5)) / (max(train2$x5)-min(train2$x5)) }
#儲存
save(yux1maxmin, file = "D:/AI-work/FCFC-ARO3/R-code/Rda/maxmin-nex1.rda")
save(yux2maxmin, file = "D:/AI-work/FCFC-ARO3/R-code/Rda/maxmin-nex2.rda")
save(yux3maxmin, file = "D:/AI-work/FCFC-ARO3/R-code/Rda/maxmin-nex3.rda")
save(yux4maxmin, file = "D:/AI-work/FCFC-ARO3/R-code/Rda/maxmin-nex4.rda")
save(yux5maxmin, file = "D:/AI-work/FCFC-ARO3/R-code/Rda/maxmin-nex5.rda")
#將新數據正規化
yum=as.matrix(cbind(yux1maxmin(yu$x1),yux2maxmin(yu$x2),yux3maxmin(yu$x3),
                    yux4maxmin(yu$x4),yux5maxmin(yu$x5)))
#還原預測值(使用訓練資料格式)
yupred=data.frame(predict(model0,yum))
reyu1=function(x) { x*(max(train2$Y1)-min(train2$Y1))+min(train2$Y1)}
reyu2=function(x) { x*(max(train2$Y2)-min(train2$Y2))+min(train2$Y2)}
reyu=data.frame(cbind(reyu1(yupred$X1),reyu2(yupred$X2)))
reyu
colnames(reyu)=c("H2O","PH")

# #PH預測值若<0，以0.001取代
# {
#   if(reyu$PH<0){
#   reyu$PH = 0.001
#   }
#   reyu
# }
##輸入操作溫度(管段1 or 2 or 3/4)
# nempy = matrix(c(68.2,reyu$PH),ncol=2)
# colnames(nempy)=c("Temp","PH")
netemp = matrix(c(68.2,55,44.3,44.3))
nempy = cbind(netemp,reyu$PH)
colnames(nempy)=c("Temp","PH")
#管段1~4的矩陣型態
nempy1=matrix(nempy[1,1:2],ncol=2)
nempy2=matrix(nempy[2,1:2],ncol=2)
nempy3=matrix(nempy[3,1:2],ncol=2)
nempy4=matrix(nempy[4,1:2],ncol=2)

# #再進入AI model-2進行KNN，預測出該管段的腐蝕率(beta)
# beta1=knn(train = kx_train, test = nempy, cl = ky_train, k = 2)
# beta1=as.character(beta1)
# beta1=as.data.frame(as.numeric(beta1))
# colnames(beta1)=c("HCI")
# beta1
#再進入AI model-2進行xgboost，預測出該管段的腐蝕率(beta)
beta1=data.frame(round(predict(xgb.model, nempy1),2))
names(beta1)=c("HCl")
beta2=data.frame(round(predict(xgb.model, nempy2),2))
names(beta2)=c("HCl")
beta3=data.frame(round(predict(xgb.model, nempy3),2))
names(beta3)=c("HCl")
beta4=data.frame(round(predict(xgb.model, nempy4),2))
names(beta4)=c("HCl")

##修正係數(xlsx)
library(openxlsx)
alphatable1=read.xlsx("D:/AI-work/FCFC-ARO3/修正係數/新修正係數.xlsx", sheet = "Corrosion") #sheet = 1，第一張工作表
alphatable2=read.xlsx("D:/AI-work/FCFC-ARO3/修正係數/新修正係數.xlsx", sheet = "Interaction")

suw = as.numeric(alphatable1[1,6])
rate1 = data.frame(as.numeric(alphatable2[9:12,8]))
names(rate1)="rate"
su=data.frame(as.numeric(alphatable1[c(3:6),5]))
su=cbind(data.frame(as.numeric(alphatable1[c(3:6),2])),su)
names(su)=c("x2","x4")
su_x4_sum = sum(as.numeric(alphatable1[3:6,5]))
#各管段的修正係數公式(輸入H2O預測值，x1)
{
s1 = function (x1){
  ((((x1/su_x4_sum*su[1,2])*rate1[1,1])*1000/suw/10000)/su[1,1])*2
}
s2 = function (x1){
  ((((x1/su_x4_sum*su[2,2])*rate1[2,1])*1000/suw/10000)/su[2,1])*2
}
s3 = function (x1){
  ((((x1/su_x4_sum*su[3,2])*rate1[3,1])*1000/suw/10000)/su[3,1])*2
}
s4 = function (x1){
  ((((x1/su_x4_sum*su[4,2])*rate1[4,1])*1000/suw/10000)/su[4,1])*2
}
}
#save the alpha calculation and function
save(su, file = "D:/AI-work/FCFC-ARO3/R-code/Rda/alpha-1.rda")
save(su_x4_sum, file = "D:/AI-work/FCFC-ARO3/R-code/Rda/alpha-2.rda")
save(suw, file = "D:/AI-work/FCFC-ARO3/R-code/Rda/alpha-3.rda")
save(rate1, file = "D:/AI-work/FCFC-ARO3/R-code/Rda/alpha-4.rda")
save(s1, file = "D:/AI-work/FCFC-ARO3/R-code/Rda/alpha-function1.rda")
save(s2, file = "D:/AI-work/FCFC-ARO3/R-code/Rda/alpha-function2.rda")
save(s3, file = "D:/AI-work/FCFC-ARO3/R-code/Rda/alpha-function3.rda")
save(s4, file = "D:/AI-work/FCFC-ARO3/R-code/Rda/alpha-function4.rda")
#各管段修正係數(alpha)
{
s1alpha = round(s1(reyu[1,1]),5)
s2alpha = round(s2(reyu[1,1]),5)
s3alpha = round(s3(reyu[1,1]),5)
s4alpha = round(s4(reyu[1,1]),5)
}
#修正後腐蝕率
{
fab1 = s1alpha * beta1
fab2 = s2alpha * beta2
fab3 = s3alpha * beta3
fab4 = s4alpha * beta4
}
#輸出
{
  cat(c("Corrosion Prediction Result\n","Corrosion in PC1 = ", as.numeric(fab1),"\n",
        "Corrosion in PC2 = ", as.numeric(fab2),"\n",
        "Corrosion in PC3 = ", as.numeric(fab3),"\n",
        "Corrosion in PC4 = ", as.numeric(fab4)))
  
}
#模型評估指標(套件跟自訂)-----
{
  # #模型評估指標(Metrics套件)
  # #平均絕對誤差 MAE
  # MAE = mae(actual,predicted)
  # #平均絕對百分誤差 MAPE
  # MAPE = mape(actual,predicted)
  # #均方誤差 MSE
  # MSE = mse(actual, predicted)
  # #均方根誤差 RMSE
  # RMSE = rmse(actual, predicted)
  # #平均均方對數誤差 MSLE
  # MSLE = msle(actual, predicted)
  # 
  # #模型評估指標(自行建立)
  # #絕對誤差 E
  # E = function(actual,predicted){ actual-predicted }
  # #取絕對值後的E
  # aE  = function(actual,predicted){ abs(actual-predicted) }
  # #相對誤差 e
  # e = function(actual,predicted){ (actual-predicted)/actual }
  # #平均絕對誤差 MAE
  # MAE = function(actual,predicted){ 1/length(actual)*sum(abs(actual-predicted)) }
  # #平均絕對百分誤差 MAPE
  # MAPE = function(actual,predicted){ 1/length(actual)*sum(abs((actual-predicted)/actual))}
  # #均方誤差 MSE
  # MSE = function(actual,predicted) {1/length(actual)*sum((actual-predicted)^2)}
  # #正規化均方誤差 NMSE
  # NMSE = function(actual,predicted){ (sum((actual-predicted)^2)/sum((actual-mean(actual))^2)}
  # #均方根誤差 RMSE
  # RMSE = function(actual,predicted){ sqrt(1/length(actual)*sum((actual-predicted)^2)) }
  # #平均均方對數誤差 MSLE
  # MSLE = function(actual,predicted){ 1/length(actual)*sum((log(1+actual,base=exp(1))-log(1+predicted,base=exp(1)))^2) }
  # 
  # #R Squared
  # R_squared = function(actual,predicted){ 1-sum((actual-predicted)^2)/sum((actual-mean(actual))^2)}
  # 
}