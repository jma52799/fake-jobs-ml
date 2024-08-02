import matplotlib.pyplot as plt
import pandas as pd
import os
from os.path import exists
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def plot_training_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def model_evaluation(model, X_test, y_test):
    return model.evaluate(X_test, y_test)


def performance_report(model, X_test, y_test, model_name, type, report_dir='../output/reports/'):
    os.makedirs('../output/models/', exist_ok=True)

    #time = date.today()
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # predict probabilities for test set
    yhat_probs = model.predict(X_test, verbose=0)

    # predict crisp classes for test set
    #yhat_classes = model.predict_classes(X_test, verbose=0)
    yhat_classes = (model.predict(X_test) > 0.5).astype("int32")

    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes)
    print('F1 score: %f' % f1)

    if exists(report_dir + 'report.csv'):
        df = pd.read_csv(report_dir + 'report.csv')
    else:
        df = pd.DataFrame(columns=['time','model_name', 'type', 'accuracy', 'precision', 'recall', 'f1'])

    new_row = pd.DataFrame({
      'time': [time],
      'model_name': [model_name],
      'type': [type],
      'accuracy': [accuracy],
      'precision': [precision],
      'recall': [recall],
      'f1': [f1]
    })

    df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(report_dir + 'report.csv', index=False)
    return df