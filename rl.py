from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns
df = fetch_california_housing(as_frame=True).frame  # alvo: MedHouseVal

print(df[["MedHouseVal", "AveRooms", "AveOccup", "HouseAge", "AveBedrms", "Population", "Latitude", "Longitude"]].describe())
print(df.head(10)) 
# print(df.info())
# df.hist(bins=50)
# plt.show()

corr = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    annot=True,  
    fmt=".2f",  
    cmap="coolwarm",  
    vmin=-1,
    vmax=1,  
    linewidths=0.5, 
)

plt.title("Matriz de Correlação - California Housing")
plt.show()