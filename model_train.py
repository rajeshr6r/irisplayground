from  sklearn import  datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

iris=datasets.load_iris()
x=iris.data
y=iris.target


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5)


classifier=tree.DecisionTreeClassifier()
classifier.fit(x_train,y_train)

predictions=classifier.predict(x_test)
print(accuracy_score(y_test,predictions))

# save the model to disk
import pickle
filename = './model/iris_model.sav'
pickle.dump(classifier, open(filename, 'wb'))
 
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict([x_test[0]])
print(result)


# A well-defined prediction

input_data = {"sepal_length":3.5,"sepal_width":4.5,"petal_length":6.7,"petal:width":7.8}
prediction_mapper={0:'Iris-Setosa',1:'Iris-Versicolour',2:'Iris-Virginica'}
prediction=loaded_model.predict([list(input_data.values())])
output=prediction_mapper.get(prediction[0])
