import streamlit as st
import io
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Function that will generate the output to be shown
def generate_output(user_input):
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    some_python_code(user_input)  # Your python code goes here
    output = new_stdout.getvalue()
    sys.stdout = old_stdout
    return output

# Python code to be executed on button press
def some_python_code(user_input):
    # print(f"This is the user input: {user_input}")
    
    def read_and_encode():
      url = user_input
    
      # Read the data into a pandas DataFrame
      data = pd.read_csv(url, na_values=['None'])
      data = data.drop('date', axis=1)
    
      # Remove rows with null or None values
      data = data.dropna()
    
      # Create a LabelEncoder object
      le = LabelEncoder()
    
      # Apply the LabelEncoder to each column
      data_encoded = data.apply(le.fit_transform)
      return data_encoded, le
    
    
    
    
    
    def train_model(data_encoded, le, feature_columns, target_column):
    
      # Separate the data into features (X) and the target variable (y)
      X = data_encoded[feature_columns]
      y = data_encoded[target_column]
    
      # Apply SMOTE
      oversample = SMOTE(k_neighbors=2) # aumentar o quitar cuando tenga mas data, por defecto seria 6
      X, y = oversample.fit_resample(X, y)
    
      # Define base models
      base_models = [
          ('gnb', GaussianNB()),
          ('dt', DecisionTreeClassifier(random_state=42)),
          ('knn', KNeighborsClassifier()),
          # ('rf', RandomForestClassifier(random_state=42)),
          ('svc', SVC(probability=True))
      ]
    
      # Define meta-model
      meta_model = LogisticRegression()
    
      # Create a StackingClassifier
      clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=LeaveOneOut()) # LeaveOneOut pasarlo a 5 cuando tenga mas data para que sean 5 folds
    
      # Create a LeaveOneOut object
      loo = LeaveOneOut()
    
      # Perform leave-one-out cross-validation
      accuracy = []
      tn = 0
      fn = 0
      fp = 0
      tp = 0
      for train_index, test_index in loo.split(X):
          X_train, X_test = X.iloc[train_index], X.iloc[test_index]
          y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
          # Train the model
          clf.fit(X_train, y_train)
    
          # Predict the target variable for the testing set
          y_pred = clf.predict(X_test)
    
          # Compute the accuracy of the model
          accuracy.append(accuracy_score(y_test, y_pred))
          if (y_test.values[0] == 0) and (y_pred[0] == 0):
            tn = tn+1
          elif (y_test.values[0] == 0) and (y_pred[0] == 1):
            fn = fn+1
          elif (y_test.values[0] == 1) and (y_pred[0] == 0):
            fp = fp+1
          elif (y_test.values[0] == 1) and (y_pred[0] == 1):
            tp = tp+1
          
    
      acc = sum(accuracy)/len(accuracy)
      valores_confusion = np.array([[tn,fn],[fp,tp]])
      print(f"{target_column} Accuracy in leave-one-out cross-validation: {acc}")
      return clf, acc, valores_confusion
    
    
    
    
    df, le = read_and_encode()
    
    model_4_am_to_8_am, acc_model_4_am_to_8_am, valores_confusion_model_4_am_to_8_am = train_model(df, le, ["4_pm_tp_4_am_last_night"], "4_am_to_8_am")
    model_8_am_to_9_30_am, acc_model_8_am_to_9_30_am, valores_confusion_model_8_am_to_9_30_am = train_model(df, le, ["4_pm_tp_4_am_last_night", "4_am_to_8_am"], "8_am_to_9_30_am")
    model_9_30_am_to_10_am, acc_model_9_30_am_to_10_am, valores_confusion_model_9_30_am_to_10_am = train_model(df, le, ["4_pm_tp_4_am_last_night", "4_am_to_8_am", "8_am_to_9_30_am"], "9_30_am_to_10_am")
    model_10_am_to_11_am, acc_model_10_am_to_11_am, valores_confusion_model_10_am_to_11_am = train_model(df, le, ["4_pm_tp_4_am_last_night", "4_am_to_8_am", "8_am_to_9_30_am", "9_30_am_to_10_am"], "10_am_to_11_am")
    model_11_am_to_12_30_m, acc_model_11_am_to_12_30_m, valores_confusion_model_11_am_to_12_30_m = train_model(df, le, ["4_pm_tp_4_am_last_night", "4_am_to_8_am", "8_am_to_9_30_am", "9_30_am_to_10_am", "10_am_to_11_am"], "11_am_to_12_30_m")
    model_12_30_m_to_2_pm, acc_model_12_30_m_to_2_pm, valores_confusion_model_12_30_m_to_2_pm = train_model(df, le, ["4_pm_tp_4_am_last_night", "4_am_to_8_am", "8_am_to_9_30_am", "9_30_am_to_10_am", "10_am_to_11_am", "11_am_to_12_30_m"], "12_30_m_to_2_pm")
    model_2_pm_to_4_pm, acc_model_2_pm_to_4_pm, valores_confusion_model_2_pm_to_4_pm = train_model(df, le, ["4_pm_tp_4_am_last_night", "4_am_to_8_am", "8_am_to_9_30_am", "9_30_am_to_10_am", "10_am_to_11_am", "11_am_to_12_30_m", "12_30_m_to_2_pm"], "2_pm_to_4_pm")
    model_4_pm_tp_4_am_these_night, acc_model_4_pm_tp_4_am_these_night, valores_confusion_model_4_pm_tp_4_am_these_night = train_model(df, le, ["4_pm_tp_4_am_last_night", "4_am_to_8_am", "8_am_to_9_30_am", "9_30_am_to_10_am", "10_am_to_11_am", "11_am_to_12_30_m", "12_30_m_to_2_pm", "2_pm_to_4_pm"], "4_pm_tp_4_am_these_night")
    
    
    
    
    
    
    
    
    
    
    
    
    url = user_input
    
    # Read the data into a pandas DataFrame
    data = pd.read_csv(url, na_values=['None'])
    data = data.drop('date', axis=1)
    last_row_with_data = data.dropna(how='all').iloc[-1]
    
    # Preserve only the columns with no null values
    last_row_with_data_no_null = last_row_with_data.dropna()
    
    print(last_row_with_data_no_null)
    
    
    def decision_theory_bayes_minimum_risk(predictions, confussion):
      
      tn = confussion[0,0]
      fn = confussion[0,1]
      fp = confussion[1,0]
      tp = confussion[1,1]
    
      #Matriz de predicciones P (Prefiero llamarla matriz de precisión, Es similar a la matriz de confusión pero respecto a los valores reales y no a la predicción, filas son valores reales coluimnas son predccion y de ahi Sale la sensitividad y especificidad)
      confusion=np.array([[tn/(tn+fn),fn/(tn+fn)],[fp/(tp+fp),tp/(tp+fp)]]) #Esta seria la matriz de confusion normal, el porcentaje es calculado respecto a la prediccion, de los que predije 0 a cuantos les atine y a cuantos no
      # print("Matriz de confusion")
      # print(confusion)
      P=np.array([[tn/(tn+fp),fp/(tn+fp)],[fn/(fn+tp),tp/(fn+tp)]]) #Esta si es la matriz de presición, el porcentaje es calculado respecto al valor rteal, de los que eran 0 a cuantos les atine y a cuantos no
      # print("Matriz P de precisión")
      # print(P) #De todos los que son 0 a cuantos detecte y a cuantos no, de todos los que son 1 a cuantos detecte y a cuantos no
    
      #Matriz de frecuencua Q (Prefiero llamarla matriz de predicciones), probabilidad preducha de ser 0 y probabilidad predicha de ser 1
      Q = [[predictions[0,0],0],[0,predictions[0,1]]]
      # print("Matriz Q de predicciones")
      # print(Q)
    
      #print(np.matmul((confusion.transpose()),Q))
      #print(np.matmul((P.transpose()),Q))
    
      #print((P.transpose())*Q)#Mal multiplicadas
    
      #Matriz R
      R=np.matmul((P.transpose()),Q)
      # print("Matriz R")
      # print(R)
    
      #Función de frecuencia de la predicción
      prob0=R[0][0]+R[0][1]
      prob1=R[1][0]+R[1][1]
      # print("Función de frecuencia de la predicción 0")
      # print(prob0)
      # print("Función de frecuencia de la predicción 1")
      # print(prob1)
    
      #Matriz estocastica P*
      PAsterisco=np.array([[R[0][0]/prob0,R[0][1]/prob0],
                          [R[1][0]/prob1,R[1][1]/prob1]])
      # print("Matriz estocastica P*")
      # print(PAsterisco)
    
      if prob0 >= prob1:
        return [prob0,prob1], 0
      else:
        return [prob0,prob1], 1
    
    
    
    
    
    #--------------------------------------------------------------------------------------------------------------------#
    
    if len(last_row_with_data_no_null) == 1:
      # Define the new data point
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_4_am_to_8_am.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_4_am_to_8_am.predict_proba(new_data_encoded)
      
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_4_am_to_8_am)
    
      print(f"\n\n Prediction for: 4_am_to_8_am")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_4_am_to_8_am}")
      print(f"Confussion: {valores_confusion_model_4_am_to_8_am}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_1 = "decreasing"
      else:
        value_fw_1 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [value_fw_1]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_8_am_to_9_30_am.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_8_am_to_9_30_am.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_8_am_to_9_30_am)
    
      print(f"\n\n Prediction for: 8_am_to_9_30_am")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_8_am_to_9_30_am}")
      print(f"Confussion: {valores_confusion_model_8_am_to_9_30_am}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_2 = "decreasing"
      else:
        value_fw_2 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [value_fw_1],
          "8_am_to_9_30_am": [value_fw_2]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_9_30_am_to_10_am.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_9_30_am_to_10_am.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_9_30_am_to_10_am)
    
      print(f"\n\n Prediction for: 9_30_am_to_10_am")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_9_30_am_to_10_am}")
      print(f"Confussion: {valores_confusion_model_9_30_am_to_10_am}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_3 = "decreasing"
      else:
        value_fw_3 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [value_fw_1],
          "8_am_to_9_30_am": [value_fw_2],
          "9_30_am_to_10_am": [value_fw_3]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_10_am_to_11_am.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_10_am_to_11_am.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_10_am_to_11_am)
    
      print(f"\n\n Prediction for: 10_am_to_11_am")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_10_am_to_11_am}")
      print(f"Confussion: {valores_confusion_model_10_am_to_11_am}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_4 = "decreasing"
      else:
        value_fw_4 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [value_fw_1],
          "8_am_to_9_30_am": [value_fw_2],
          "9_30_am_to_10_am": [value_fw_3],
          "10_am_to_11_am": [value_fw_4]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_11_am_to_12_30_m.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_11_am_to_12_30_m.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_11_am_to_12_30_m)
    
      print(f"\n\n Prediction for: 11_am_to_12_30_m")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_11_am_to_12_30_m}")
      print(f"Confussion: {valores_confusion_model_11_am_to_12_30_m}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_5 = "decreasing"
      else:
        value_fw_5 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [value_fw_1],
          "8_am_to_9_30_am": [value_fw_2],
          "9_30_am_to_10_am": [value_fw_3],
          "10_am_to_11_am": [value_fw_4],
          "11_am_to_12_30_m": [value_fw_5]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_12_30_m_to_2_pm.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_12_30_m_to_2_pm.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_12_30_m_to_2_pm)
    
      print(f"\n\n Prediction for: 12_30_m_to_2_pm")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_12_30_m_to_2_pm}")
      print(f"Confussion: {valores_confusion_model_12_30_m_to_2_pm}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
    
    ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_6 = "decreasing"
      else:
        value_fw_6 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [value_fw_1],
          "8_am_to_9_30_am": [value_fw_2],
          "9_30_am_to_10_am": [value_fw_3],
          "10_am_to_11_am": [value_fw_4],
          "11_am_to_12_30_m": [value_fw_5],
          "12_30_m_to_2_pm": [value_fw_6]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_2_pm_to_4_pm.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_2_pm_to_4_pm.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_2_pm_to_4_pm)
    
      print(f"\n\n Prediction for: 2_pm_to_4_pm")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_2_pm_to_4_pm}")
      print(f"Confussion: {valores_confusion_model_2_pm_to_4_pm}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
    
    ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_7 = "decreasing"
      else:
        value_fw_7 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [value_fw_1],
          "8_am_to_9_30_am": [value_fw_2],
          "9_30_am_to_10_am": [value_fw_3],
          "10_am_to_11_am": [value_fw_4],
          "11_am_to_12_30_m": [value_fw_5],
          "12_30_m_to_2_pm": [value_fw_6],
          "2_pm_to_4_pm": [value_fw_7]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_4_pm_tp_4_am_these_night.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_4_pm_tp_4_am_these_night.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_4_pm_tp_4_am_these_night)
    
      print(f"\n\n Prediction for: 4_pm_tp_4_am_these_night")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_4_pm_tp_4_am_these_night}")
      print(f"Confussion: {valores_confusion_model_4_pm_tp_4_am_these_night}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    #--------------------------------------------------------------------------------------------------------------------#
    
    elif len(last_row_with_data_no_null) == 2:
      # Define the new data point
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_8_am_to_9_30_am.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_8_am_to_9_30_am.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_8_am_to_9_30_am)
    
      print(f"\n\n Prediction for: 8_am_to_9_30_am")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_8_am_to_9_30_am}")
      print(f"Confussion: {valores_confusion_model_8_am_to_9_30_am}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_1 = "decreasing"
      else:
        value_fw_1 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [value_fw_1]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_9_30_am_to_10_am.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_9_30_am_to_10_am.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_9_30_am_to_10_am)
    
      print(f"\n\n Prediction for: 9_30_am_to_10_am")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_9_30_am_to_10_am}")
      print(f"Confussion: {valores_confusion_model_9_30_am_to_10_am}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_2 = "decreasing"
      else:
        value_fw_2 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [value_fw_1],
          "9_30_am_to_10_am": [value_fw_2]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_10_am_to_11_am.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_10_am_to_11_am.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_10_am_to_11_am)
    
      print(f"\n\n Prediction for: 10_am_to_11_am")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_10_am_to_11_am}")
      print(f"Confussion: {valores_confusion_model_10_am_to_11_am}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
    ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_3 = "decreasing"
      else:
        value_fw_3 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [value_fw_1],
          "9_30_am_to_10_am": [value_fw_2],
          "10_am_to_11_am": [value_fw_3]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_11_am_to_12_30_m.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_11_am_to_12_30_m.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_11_am_to_12_30_m)
    
      print(f"\n\n Prediction for: 11_am_to_12_30_m")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_11_am_to_12_30_m}")
      print(f"Confussion: {valores_confusion_model_11_am_to_12_30_m}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
    ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_4 = "decreasing"
      else:
        value_fw_4 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [value_fw_1],
          "9_30_am_to_10_am": [value_fw_2],
          "10_am_to_11_am": [value_fw_3],
          "11_am_to_12_30_m": [value_fw_4]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_12_30_m_to_2_pm.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_12_30_m_to_2_pm.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_12_30_m_to_2_pm)
    
      print(f"\n\n Prediction for: 12_30_m_to_2_pm")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_12_30_m_to_2_pm}")
      print(f"Confussion: {valores_confusion_model_12_30_m_to_2_pm}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
    ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_5 = "decreasing"
      else:
        value_fw_5 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [value_fw_1],
          "9_30_am_to_10_am": [value_fw_2],
          "10_am_to_11_am": [value_fw_3],
          "11_am_to_12_30_m": [value_fw_4],
          "12_30_m_to_2_pm": [value_fw_5]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_2_pm_to_4_pm.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_2_pm_to_4_pm.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_2_pm_to_4_pm)
    
      print(f"\n\n Prediction for: 2_pm_to_4_pm")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_2_pm_to_4_pm}")
      print(f"Confussion: {valores_confusion_model_2_pm_to_4_pm}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
    ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_6 = "decreasing"
      else:
        value_fw_6 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [value_fw_1],
          "9_30_am_to_10_am": [value_fw_2],
          "10_am_to_11_am": [value_fw_3],
          "11_am_to_12_30_m": [value_fw_4],
          "12_30_m_to_2_pm": [value_fw_5],
          "2_pm_to_4_pm": [value_fw_6]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_4_pm_tp_4_am_these_night.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_4_pm_tp_4_am_these_night.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_4_pm_tp_4_am_these_night)
    
      print(f"\n\n Prediction for: 2_pm_t4_pm_tp_4_am_these_nighto_4_pm")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_4_pm_tp_4_am_these_night}")
      print(f"Confussion: {valores_confusion_model_4_pm_tp_4_am_these_night}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
    #--------------------------------------------------------------------------------------------------------------------#
    
    elif len(last_row_with_data_no_null) == 3:
      # Define the new data point
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_9_30_am_to_10_am.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_9_30_am_to_10_am.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_9_30_am_to_10_am)
    
      print(f"\n\n Prediction for: 9_30_am_to_10_am")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_9_30_am_to_10_am}")
      print(f"Confussion: {valores_confusion_model_9_30_am_to_10_am}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_1 = "decreasing"
      else:
        value_fw_1 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [value_fw_1]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_10_am_to_11_am.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_10_am_to_11_am.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_10_am_to_11_am)
    
      print(f"\n\n Prediction for: 10_am_to_11_am")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_10_am_to_11_am}")
      print(f"Confussion: {valores_confusion_model_10_am_to_11_am}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_2 = "decreasing"
      else:
        value_fw_2 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [value_fw_1],
          "10_am_to_11_am": [value_fw_2]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_11_am_to_12_30_m.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_11_am_to_12_30_m.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_11_am_to_12_30_m)
    
      print(f"\n\n Prediction for: 11_am_to_12_30_m")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_11_am_to_12_30_m}")
      print(f"Confussion: {valores_confusion_model_11_am_to_12_30_m}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_3 = "decreasing"
      else:
        value_fw_3 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [value_fw_1],
          "10_am_to_11_am": [value_fw_2],
          "11_am_to_12_30_m": [value_fw_3]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_12_30_m_to_2_pm.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_12_30_m_to_2_pm.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_12_30_m_to_2_pm)
    
      print(f"\n\n Prediction for: 12_30_m_to_2_pm")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_12_30_m_to_2_pm}")
      print(f"Confussion: {valores_confusion_model_12_30_m_to_2_pm}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_4 = "decreasing"
      else:
        value_fw_4 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [value_fw_1],
          "10_am_to_11_am": [value_fw_2],
          "11_am_to_12_30_m": [value_fw_3],
          "12_30_m_to_2_pm": [value_fw_4]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_2_pm_to_4_pm.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_2_pm_to_4_pm.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_2_pm_to_4_pm)
    
      print(f"\n\n Prediction for: 2_pm_to_4_pm")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_2_pm_to_4_pm}")
      print(f"Confussion: {valores_confusion_model_2_pm_to_4_pm}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_5 = "decreasing"
      else:
        value_fw_5 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [value_fw_1],
          "10_am_to_11_am": [value_fw_2],
          "11_am_to_12_30_m": [value_fw_3],
          "12_30_m_to_2_pm": [value_fw_4],
          "2_pm_to_4_pm": [value_fw_5]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_4_pm_tp_4_am_these_night.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_4_pm_tp_4_am_these_night.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_4_pm_tp_4_am_these_night)
    
      print(f"\n\n Prediction for: 4_pm_tp_4_am_these_night")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_4_pm_tp_4_am_these_night}")
      print(f"Confussion: {valores_confusion_model_4_pm_tp_4_am_these_night}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
    
    #--------------------------------------------------------------------------------------------------------------------#
    elif len(last_row_with_data_no_null) == 4:
      # Define the new data point
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [last_row_with_data_no_null["9_30_am_to_10_am"]]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_10_am_to_11_am.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_10_am_to_11_am.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_10_am_to_11_am)
    
      print(f"\n\n Prediction for: 10_am_to_11_am")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_10_am_to_11_am}")
      print(f"Confussion: {valores_confusion_model_10_am_to_11_am}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_1 = "decreasing"
      else:
        value_fw_1 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [last_row_with_data_no_null["9_30_am_to_10_am"]],
          "10_am_to_11_am": [value_fw_1]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_11_am_to_12_30_m.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_11_am_to_12_30_m.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_11_am_to_12_30_m)
    
      print(f"\n\n Prediction for: 11_am_to_12_30_m")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_11_am_to_12_30_m}")
      print(f"Confussion: {valores_confusion_model_11_am_to_12_30_m}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_2 = "decreasing"
      else:
        value_fw_2 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [last_row_with_data_no_null["9_30_am_to_10_am"]],
          "10_am_to_11_am": [value_fw_1],
          "11_am_to_12_30_m": [value_fw_2]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_12_30_m_to_2_pm.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_12_30_m_to_2_pm.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_12_30_m_to_2_pm)
    
      print(f"\n\n Prediction for: 12_30_m_to_2_pm")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_12_30_m_to_2_pm}")
      print(f"Confussion: {valores_confusion_model_12_30_m_to_2_pm}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
    
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_3 = "decreasing"
      else:
        value_fw_3 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [last_row_with_data_no_null["9_30_am_to_10_am"]],
          "10_am_to_11_am": [value_fw_1],
          "11_am_to_12_30_m": [value_fw_2],
          "12_30_m_to_2_pm": [value_fw_3]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_2_pm_to_4_pm.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_2_pm_to_4_pm.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_2_pm_to_4_pm)
    
      print(f"\n\n Prediction for: 2_pm_to_4_pm")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_2_pm_to_4_pm}")
      print(f"Confussion: {valores_confusion_model_2_pm_to_4_pm}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_4 = "decreasing"
      else:
        value_fw_4 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [last_row_with_data_no_null["9_30_am_to_10_am"]],
          "10_am_to_11_am": [value_fw_1],
          "11_am_to_12_30_m": [value_fw_2],
          "12_30_m_to_2_pm": [value_fw_3],
          "2_pm_to_4_pm": [value_fw_4]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_4_pm_tp_4_am_these_night.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_4_pm_tp_4_am_these_night.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_4_pm_tp_4_am_these_night)
    
      print(f"\n\n Prediction for: 4_pm_tp_4_am_these_night")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_4_pm_tp_4_am_these_night}")
      print(f"Confussion: {valores_confusion_model_4_pm_tp_4_am_these_night}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
    #--------------------------------------------------------------------------------------------------------------------#
    elif len(last_row_with_data_no_null) == 5:
      # Define the new data point
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [last_row_with_data_no_null["9_30_am_to_10_am"]],
          "10_am_to_11_am": [last_row_with_data_no_null["10_am_to_11_am"]]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_11_am_to_12_30_m.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_11_am_to_12_30_m.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_11_am_to_12_30_m)
    
      print(f"\n\n Prediction for: 11_am_to_12_30_m")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_11_am_to_12_30_m}")
      print(f"Confussion: {valores_confusion_model_11_am_to_12_30_m}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_1 = "decreasing"
      else:
        value_fw_1 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [last_row_with_data_no_null["9_30_am_to_10_am"]],
          "10_am_to_11_am": [last_row_with_data_no_null["10_am_to_11_am"]],
          "11_am_to_12_30_m": [value_fw_1]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_12_30_m_to_2_pm.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_12_30_m_to_2_pm.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_12_30_m_to_2_pm)
    
      print(f"\n\n Prediction for: 12_30_m_to_2_pm")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_12_30_m_to_2_pm}")
      print(f"Confussion: {valores_confusion_model_12_30_m_to_2_pm}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_2 = "decreasing"
      else:
        value_fw_2 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [last_row_with_data_no_null["9_30_am_to_10_am"]],
          "10_am_to_11_am": [last_row_with_data_no_null["10_am_to_11_am"]],
          "11_am_to_12_30_m": [value_fw_1],
          "12_30_m_to_2_pm": [value_fw_2]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_2_pm_to_4_pm.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_2_pm_to_4_pm.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_2_pm_to_4_pm)
    
      print(f"\n\n Prediction for: 2_pm_to_4_pm")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_2_pm_to_4_pm}")
      print(f"Confussion: {valores_confusion_model_2_pm_to_4_pm}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_3 = "decreasing"
      else:
        value_fw_3 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [last_row_with_data_no_null["9_30_am_to_10_am"]],
          "10_am_to_11_am": [last_row_with_data_no_null["10_am_to_11_am"]],
          "11_am_to_12_30_m": [value_fw_1],
          "12_30_m_to_2_pm": [value_fw_2],
          "2_pm_to_4_pm": [value_fw_3]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_4_pm_tp_4_am_these_night.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_4_pm_tp_4_am_these_night.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_4_pm_tp_4_am_these_night)
    
      print(f"\n\n Prediction for: 4_pm_tp_4_am_these_night")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_4_pm_tp_4_am_these_night}")
      print(f"Confussion: {valores_confusion_model_4_pm_tp_4_am_these_night}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
    #--------------------------------------------------------------------------------------------------------------------#
    elif len(last_row_with_data_no_null) == 6:
      # Define the new data point
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [last_row_with_data_no_null["9_30_am_to_10_am"]],
          "10_am_to_11_am": [last_row_with_data_no_null["10_am_to_11_am"]],
          "11_am_to_12_30_m": [last_row_with_data_no_null["11_am_to_12_30_m"]]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_12_30_m_to_2_pm.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_12_30_m_to_2_pm.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_12_30_m_to_2_pm)
    
      print(f"\n\n Prediction for: 12_30_m_to_2_pm")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_12_30_m_to_2_pm}")
      print(f"Confussion: {valores_confusion_model_12_30_m_to_2_pm}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_1 = "decreasing"
      else:
        value_fw_1 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [last_row_with_data_no_null["9_30_am_to_10_am"]],
          "10_am_to_11_am": [last_row_with_data_no_null["10_am_to_11_am"]],
          "11_am_to_12_30_m": [last_row_with_data_no_null["11_am_to_12_30_m"]],
          "12_30_m_to_2_pm": [value_fw_1]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_2_pm_to_4_pm.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_2_pm_to_4_pm.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_2_pm_to_4_pm)
    
      print(f"\n\n Prediction for: 2_pm_to_4_pm")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_2_pm_to_4_pm}")
      print(f"Confussion: {valores_confusion_model_2_pm_to_4_pm}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_2 = "decreasing"
      else:
        value_fw_2 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [last_row_with_data_no_null["9_30_am_to_10_am"]],
          "10_am_to_11_am": [last_row_with_data_no_null["10_am_to_11_am"]],
          "11_am_to_12_30_m": [last_row_with_data_no_null["11_am_to_12_30_m"]],
          "12_30_m_to_2_pm": [value_fw_1],
          "2_pm_to_4_pm": [value_fw_2]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_4_pm_tp_4_am_these_night.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_4_pm_tp_4_am_these_night.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_4_pm_tp_4_am_these_night)
    
      print(f"\n\n Prediction for: 4_pm_tp_4_am_these_night")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_4_pm_tp_4_am_these_night}")
      print(f"Confussion: {valores_confusion_model_4_pm_tp_4_am_these_night}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
    #--------------------------------------------------------------------------------------------------------------------#
    elif len(last_row_with_data_no_null) == 7:
      # Define the new data point
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [last_row_with_data_no_null["9_30_am_to_10_am"]],
          "10_am_to_11_am": [last_row_with_data_no_null["10_am_to_11_am"]],
          "11_am_to_12_30_m": [last_row_with_data_no_null["11_am_to_12_30_m"]],
          "12_30_m_to_2_pm": [last_row_with_data_no_null["12_30_m_to_2_pm"]]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_2_pm_to_4_pm.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_2_pm_to_4_pm.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_2_pm_to_4_pm)
    
      print(f"\n\n Prediction for: 2_pm_to_4_pm")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_2_pm_to_4_pm}")
      print(f"Confussion: {valores_confusion_model_2_pm_to_4_pm}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
      ####--predicting next hour----##########################################
    
      if final_prediction == 0:
        value_fw_1 = "decreasing"
      else:
        value_fw_1 = "increasing"
    
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [last_row_with_data_no_null["9_30_am_to_10_am"]],
          "10_am_to_11_am": [last_row_with_data_no_null["10_am_to_11_am"]],
          "11_am_to_12_30_m": [last_row_with_data_no_null["11_am_to_12_30_m"]],
          "12_30_m_to_2_pm": [last_row_with_data_no_null["12_30_m_to_2_pm"]],
          "2_pm_to_4_pm": [value_fw_1]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_4_pm_tp_4_am_these_night.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_4_pm_tp_4_am_these_night.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_4_pm_tp_4_am_these_night)
    
      print(f"\n\n Prediction for: 4_pm_tp_4_am_these_night")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_4_pm_tp_4_am_these_night}")
      print(f"Confussion: {valores_confusion_model_4_pm_tp_4_am_these_night}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")
    
    
    
    
    
    #--------------------------------------------------------------------------------------------------------------------#
    elif len(last_row_with_data_no_null) == 8:
      # Define the new data point
      new_data = pd.DataFrame({
          "4_pm_tp_4_am_last_night": [last_row_with_data_no_null["4_pm_tp_4_am_last_night"]],
          "4_am_to_8_am": [last_row_with_data_no_null["4_am_to_8_am"]],
          "8_am_to_9_30_am": [last_row_with_data_no_null["8_am_to_9_30_am"]],
          "9_30_am_to_10_am": [last_row_with_data_no_null["9_30_am_to_10_am"]],
          "10_am_to_11_am": [last_row_with_data_no_null["10_am_to_11_am"]],
          "11_am_to_12_30_m": [last_row_with_data_no_null["11_am_to_12_30_m"]],
          "12_30_m_to_2_pm": [last_row_with_data_no_null["12_30_m_to_2_pm"]],
          "2_pm_to_4_pm": [last_row_with_data_no_null["2_pm_to_4_pm"]]
      })
    
      # Encode the new data point using the same encoding scheme used for training the model
      new_data_encoded = new_data.apply(le.transform)
    
      # Generate a prediction for the new data point
      prediction = model_4_pm_tp_4_am_these_night.predict(new_data_encoded)
    
      # Generate prediction probabilities for the new data point
      prediction_proba = model_4_pm_tp_4_am_these_night.predict_proba(new_data_encoded)
    
      probs, final_prediction = decision_theory_bayes_minimum_risk(prediction_proba, valores_confusion_model_4_pm_tp_4_am_these_night)
    
      print(f"\n\n Prediction for: 4_pm_tp_4_am_these_night")
      print(f"Prediction: {prediction}")
      print(f"Prediction Probabilities: {prediction_proba}")
      print(f"Accuracy: {acc_model_4_pm_tp_4_am_these_night}")
      print(f"Confussion: {valores_confusion_model_4_pm_tp_4_am_these_night}")
      print(f"Final prediction: {final_prediction}")
      print(f"Corrected probs: {probs}")

# Streamlit app interface
st.title('Streamlit Predict Button')

# Text box for user input
user_input = st.text_input('Execute between 10 am and 11 am, if predicion for 11:00 to 12_30 is increase with >65% confidence, then invest (Never invest in a different schedule):')

if st.button('Predict'):
    if user_input:  # Execute only if user has provided some input
        output = generate_output(user_input)
        st.text(output)
    else:
        st.warning('Please input a string.')
