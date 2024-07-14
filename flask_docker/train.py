#train svm classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#train data 10 is mapped to prediction 1
#train data 100 is mappen to prediction 0 
train_data = [[10], [100]]
train_target = [1,0]
clf_lda = LinearDiscriminantAnalysis()
clf_lda.fit(train_data, train_target)

#saved trained classifier as joblib file for server to use
import joblib
joblib.dump(clf_lda, "LDA_clf.joblib")