import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
df=pd.read_csv("Housing.csv")
print("first five row of the data:",df.head())
print("null values in the data:",df.isnull().sum())
print("duplicated values in the data:",df.duplicated().sum())
objcol=[x for x in df.columns if df[x].dtype=="object"]
print("object columns:",objcol)
le=LabelEncoder()
for i in objcol:
  df[i]=le.fit_transform(df[i])
print("data after encoding:",df.head())
sns.heatmap(df.corr(),annot=True)
plt.title("Correlation Matrix")
plt.show()
plt.plot(df["price"],df["area"],marker="o")
plt.title("Price vs Area")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()
sns.pairplot(df)
sns.regplot(x="area",y="price",data=df)
plt.title("Price vs Area")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()
x=df.drop("price",axis=1)
y=df["price"]
xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(xtr,ytr)
pred=lr.predict(xte)
mae=mean_absolute_error(yte,pred)
mse=mean_squared_error(yte,pred)
rmse=np.sqrt(mse)
r2=r2_score(yte,pred)
print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)
print("R2:",r2)
plt.plot(xte,pred,marker="o",color="r")
plt.plot(xte,yte,marker="o",color="g")
plt.show()