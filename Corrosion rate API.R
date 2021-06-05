library(keras)
library(dplyr)
library(xgboost)
library(openxlsx)
#載入rda檔
{
# setwd("D:\\AI-work\\FCFC-ARO3\\R-code\\Rda")
path = getwd()
filenames = list.files(path=paste(path, sep =""), pattern="*.rda")

for (i in filenames){
  load(file=paste(path,"/", i, sep =""))
}
}
#載入keras的NN預測模型
kmodel = load_model_hdf5("keras-model.h5")

##新數據進入keras NN預測-----
###整體預測與輸出結果###
{
# {  
# #可自行輸入數值
# i1=as.numeric(readline("請輸入入料量："))
# i2=as.numeric(readline("請輸入含水量："))
# i3=as.numeric(readline("請輸入含氯量："))
# i4=as.numeric(readline("請輸入C5+雜質："))
# i5=as.numeric(readline("請輸入C4雜質："))
# yu=data.frame(cbind(i1,i2,i3,i4,i5))
# colnames(yu)=c("x1","x2","x3","x4","x5")
# }
{
#給定既定數值
yu=data.frame(280,80,1,0.005,0.001)
colnames(yu)=c("x1","x2","x3","x4","x5")
}
#將新數據正規化
yum=as.matrix(cbind(yux1maxmin(yu$x1),yux2maxmin(yu$x2),yux3maxmin(yu$x3),
                    yux4maxmin(yu$x4),yux5maxmin(yu$x5)))
#進行預測(含水量與pH值)-----
yupred=data.frame(predict(kmodel,yum))
#還原預測值(使用訓練資料格式)
reyu=data.frame(cbind(reyu1(yupred$X1),reyu2(yupred$X2)))
colnames(reyu)=c("H2O","PH")

##輸入操作溫度(管段1 or 2 or 3/4)-----
netemp = matrix(c(68.2,55,44.3,44.3))
nempy = cbind(netemp,reyu$PH)
colnames(nempy)=c("Temp","PH")
#管段1~4的矩陣型態
nempy1=matrix(nempy[1,1:2],ncol=2)
nempy2=matrix(nempy[2,1:2],ncol=2)
nempy3=matrix(nempy[3,1:2],ncol=2)
nempy4=matrix(nempy[4,1:2],ncol=2)
#再進入AI model-2進行xgboost，預測出該管段的腐蝕率(beta)-----
beta1=data.frame(round(predict(xgb.model, nempy1),2))
names(beta1)=c("HCl")
beta2=data.frame(round(predict(xgb.model, nempy2),2))
names(beta2)=c("HCl")
beta3=data.frame(round(predict(xgb.model, nempy3),2))
names(beta3)=c("HCl")
beta4=data.frame(round(predict(xgb.model, nempy4),2))
names(beta4)=c("HCl")

#各管段修正係數(alpha)-----
{
  s1alpha = round(s1(reyu[1,1]),5)
  s2alpha = round(s2(reyu[1,1]),5)
  s3alpha = round(s3(reyu[1,1]),5)
  s4alpha = round(s4(reyu[1,1]),5)
}
#修正後腐蝕率-----
{
  fab1 = s1alpha * beta1
  fab2 = s2alpha * beta2
  fab3 = s3alpha * beta3
  fab4 = s4alpha * beta4
}
#輸出結果-----
{
  cat(c("##Corrosion Prediction Result##\n",
        "Corrosion in PC1 = ", as.numeric(fab1),"\n",
        "Corrosion in PC2 = ", as.numeric(fab2),"\n",
        "Corrosion in PC3 = ", as.numeric(fab3),"\n",
        "Corrosion in PC4 = ", as.numeric(fab4),"\n"))
}

# fab5=data.frame(rbind(fab1,fab2,fab3,fab4))
# fab5=cbind(NA,fab5)
# colnames(fab5)=c("names","value")
# fab5[1:4,1]=c("PC1","PC2","PC3","PC4")
# 
# print(fab5)

}

###整體預測與輸出結果###
