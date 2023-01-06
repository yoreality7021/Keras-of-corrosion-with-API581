# Corrosion rate predict(python)

## Summary
主要是利用深度類神經先行預測含水量與pH值，再使用knn model來建立API 581的數值型態，先行計算出初步的腐蝕率(beta)；

再根據廠方提供的修正係數公式，計算出修正係數(alpha)；

最後，再把alpha與beta相乘，得出修正後的腐蝕率。

### AI-model 1：Keras model(DNN)

安裝keras於虛擬環境中，還有須安裝tensorflow(2.2以上(含))，是連動的
```
pip install keras
```

數據匯入後，先把數據分成85/15，15為驗證集，再把85拆成80/20

訓練集與測試集的數據型態都需轉成矩陣，並使用正規化(min-max)，讓數據都介於0~1之間

#訓練與測試的x&y正規化需分開設定，Ex：
```
trainx = preprocessing.MinMaxScaler()
```

#fit_transform()為正規化(0~1之間)
```
trainx_minmax = trainx.fit_transform(npx_train).reshape(npx_train.shape[0],npx_train.shape[1])
```

若要還原到正常數值得化就使用原本的訓練集數據逆推。
```
retrain_x = trainx.inverse_transform(trainx_minmax)
```

接著開始進入keras模型進行預測，輸入層1層、隱藏層1層、輸出層1層、輸入元5個、輸出元2個

經過迭代搜尋法比較後，神經元數為128個較佳、batch_size為16較佳、使用500次迭代

活化函數使用ReLU，kernel使用he_normal，輸出層時再使用sigmoid收斂

使用模型評估指標，訓練集與測試集的R^2都在0.99多，RMSE與MAPE都很小，誤差極小

再將經過深度類神經模型(keras model)預測出來的含水量與pH值另外存檔

pH值要跟各管段操作溫度一起進入AI-model 2預測出初步的腐蝕率(beta)

### AI-model 2：knn model for API 581

因為API 581屬於非線性的資料庫，因此嘗試使用多種機器學習方法來建立，問題都會使預測值產生負值，最後使用KNN來建立，沒有負值產生
```
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train, y_train)
```

用混淆矩陣呈現預測結果
```
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(ytest,pred))
```

### AI-model 1 and 2 to calculate corrosion rate

輸入各管段的操作溫度與經AI-model 1預測出來的pH值，進AI-model 2得出初步的腐蝕率(beta)

經AI-model 1預測出來的含水量預測值，透過廠方提供的修正係數公式表，得出各管段的修正係數(alpha)

最後再將各管段修正係數(alpha)與各管段初步的腐蝕率(beta)相乘，得出修正後的腐蝕率，即為各管段的腐蝕率
