from keras.models import load_model
import numpy as np

MODEL_PATH = 'MultiLayerPerceptron_CIFAR10.h5'
X_TEST = 'x_test.npz'
Y_TEST = 'y_test.npz'
CLASSES = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model  = load_model(MODEL_PATH)
x_test = np.load(X_TEST)
y_test = np.load(Y_TEST)

#################  Making Predictions #######################
x_test=x_test['arr_0']
y_test=y_test['arr_0']

# first 10 predictions
predictions = model.predict_classes(x_test[:10])
tru_val = y_test[:10].reshape(10,)
for i in range(10):
    print('Prediction:',CLASSES[predictions[i]],'\tActual:',CLASSES[tru_val[i]])
#############################################################