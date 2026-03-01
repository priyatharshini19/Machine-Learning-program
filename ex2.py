import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import accuracy_score,precision_score
data = {
    'Quiz 1' : [7,8,6,9,5,7,8,10,9,6],
    'Quiz 2' : [8,7,5,9,6,8,9,10,8,7],
    'Internal 1' : [40,42,35,45,30,38,41,48,44,36],
    'Internal 2' : [42,40,34,46,32,39,43,47,45,37],
    'Assignment_Seminar' : [4,5,3,5,3,4,4,5,5,3],
    'Final_Score' : [78,82,70,90,65,76,84,95,88,72]
}
df = pd.DataFrame(data)
x = df[['Quiz 1','Quiz 2','Internal 1','Internal 2','Assignment_Seminar']]
y_reg = df ['Final_Score']
y_class = (df['Final_Score'] >= 75).astype(int)
reg_model = LinearRegression()
reg_model.fit(x,y_reg)
clf = LogisticRegression()
clf.fit(x,y_class)
y_pred_class = clf.predict(x)
accuracy = accuracy_score(y_class,y_pred_class) * 100
precision = precision_score(y_class,y_pred_class) * 100
print (f"\n Accuracy Score :{accuracy : 2f} % ")
print(f"Precision Score : {precision : 2f} % ")
print("\n---Predict Your Final marks---")
q1 = float(input("Enter Quiz 1 marks(out of 10): "))
q2 = float(input("Enter Quiz 2 marks(out of 10):"))
i1 = float(input("Enter Internal 1 marks(out of 50):"))
i2 = float(input("Enter Internal 2 marks(out of 50):"))
a_s = float(input("Enter Assignment_Seminar marks(out of 5):"))
Predicted_Score = reg_model.predict([[q1,q2,i1,i2,a_s]]) [0]
Components = ['Quiz 1','Quiz 2','Internal 1','Internal 2','Assignment_Seminar','Predicted Final']
Scores = [q1,q2,i1,i2,a_s,Predicted_Score]
plt.bar(Components,Scores,color = ['skyblue','skyblue','orange','orange','green','red'])
plt.title('Student Performance Components vs Predicted Final Score')
plt.ylabel('marks')
plt.ylim(0,100)
plt.show()