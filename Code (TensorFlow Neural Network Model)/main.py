# O3-enabled BLE Weather Station Predicting Air Quality w/ TensorFlow
#
# Windows, Linux, or Ubuntu
#
# By Kutluhan Aktar
#
# Via Nano 33 BLE, collate local weather data, build and train a TensorFlow neural network model, and run the model to predict air quality.
# 
#
# For more information:
# https://www.theamplituhedron.com/projects/O3_enabled_BLE_Weather_Station_Predicting_Air_Quality_w_TensorFlow/

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tflite_to_c_array import hex_to_c_array
from labels import labels

# Create a class to build a neural network model after visualizing and scaling (normalizing) the local weather data collected by the Arduino Nano 33 BLE.
class Air_Quality_Level:
    def __init__(self, csv_path):
        self.inputs = []
        self.labels = []
        self.model_name = "air_quality_level"
        # Read the collated local weather data set:
        self.df = pd.read_csv(csv_path)
    # Create graphics for each requested column.
    def graphics(self, column_1, column_2, x_label, y_label):
        # Show the requested data column from the data set:
        plt.style.use("dark_background")
        plt.gcf().canvas.set_window_title('O3-enabled BLE Weather Station Predicting Air Quality')
        plt.hist2d(self.df[column_1], self.df[column_2], cmap="summer_r")
        plt.colorbar()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(x_label)
        plt.show()
    # Visualize data before creating and training the neural network model.
    def data_visualization(self):
        # Scrutinize data columns to build a model with appropriately formatted data:
        self.graphics('Temperature', 'Ozone_Concentration', 'Temperature', 'Ozone Concentration')
        self.graphics('Altitude', 'Ozone_Concentration', 'Altitude', 'Ozone Concentration')
        self.graphics('Pressure', 'Ozone_Concentration', 'Pressure', 'Ozone Concentration')
        self.graphics('Wind Speed', 'Ozone_Concentration', 'Wind Speed', 'Ozone Concentration')
    # Scale (normalize) data to define appropriately formatted inputs.
    def scale_data_and_define_inputs(self):
        self.df["scaled_Temperature"] = self.df["Temperature"] / 100
        self.df["scaled_Altitude"] = self.df["Altitude"] / 100
        self.df["scaled_Ozone"] = self.df["Ozone_Concentration"] / 1000
        self.df["scaled_Pressure"] = self.df["Pressure"] / 100000
        self.df["scaled_Wind_Speed"] = self.df["Wind Speed"] / 30
        # Create the inputs array by utilizing the scaled variables:
        for i in range(len(self.df)):
            self.inputs.append(np.array([self.df["scaled_Temperature"][i], self.df["scaled_Altitude"][i], self.df["scaled_Ozone"][i], self.df["scaled_Pressure"][i], self.df["scaled_Wind_Speed"][i]]))
        self.inputs = np.asarray(self.inputs)
    # Assign labels for each weather data input according to the Air Quality Index (AQI) on the given date.
    def define_and_assign_labels(self):
        self.labels = labels
    # Split inputs and labels into training and test sets.
    def split_data(self):
        l = len(self.df)
        # (95%, 5%) - (training, test)
        self.train_inputs = self.inputs[0:int(l*0.95)]
        self.test_inputs = self.inputs[int(l*0.95):]
        self.train_labels = self.labels[0:int(l*0.95)]
        self.test_labels = self.labels[int(l*0.95):]
    # Build and train an artificial neural network (ANN) model to make predictions on air quality levels (classes) based on the local weather data.
    def build_and_train_model(self):
        # Build the neural network:
        self.model = keras.Sequential([
            keras.Input(shape=(5,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])
        # Compile:
        self.model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        # Train:
        self.model.fit(self.train_inputs, self.train_labels, epochs=100)
        # Test the model accuracy:
        print("\n\nModel Evaluation:")
        test_loss, test_acc = self.model.evaluate(self.test_inputs, self.test_labels) 
        print("Evaluated Accuracy: ", test_acc)
    # Save the model for further usage:
    def save_model(self):
        self.model.save("model/{}.h5".format(self.model_name))        
    # Convert the TensorFlow Keras H5 model (.h5) to a TensorFlow Lite model (.tflite).
    def convert_TF_model(self, path):
        #model = tf.keras.models.load_model(path + ".h5")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        #converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        # Save the recently converted TensorFlow Lite model.
        with open(path + '.tflite', 'wb') as f:
            f.write(tflite_model)
        print("\r\nTensorFlow Keras H5 model converted to a TensorFlow Lite model!\r\n")
        # Convert the recently created TensorFlow Lite model to hex bytes (C array) to generate a .h file string.
        with open("model/{}.h".format(self.model_name), 'w') as file:
            file.write(hex_to_c_array(tflite_model, self.model_name))
        print("\r\nTensorFlow Lite model converted to a C header (.h) file!\r\n")
    # Run Artificial Neural Network (ANN):
    def Neural_Network(self, save):
        self.scale_data_and_define_inputs()
        self.define_and_assign_labels()
        self.split_data()
        self.build_and_train_model()
        if save:
            self.save_model()
    
# Define a new class object named 'air_quality_level':
air_quality_level = Air_Quality_Level("data/air_quality_data_set.csv")

# Visualize data columns:
air_quality_level.data_visualization()

# Artificial Neural Network (ANN):        
air_quality_level.Neural_Network(True)        
        
# Convert the TensorFlow Keras H5 model to a TensorFlow Lite model:
air_quality_level.convert_TF_model("model/{}".format(air_quality_level.model_name))
     