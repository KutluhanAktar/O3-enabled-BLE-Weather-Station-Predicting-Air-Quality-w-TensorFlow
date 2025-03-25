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

from bluepy import btle
from struct import unpack
from csv import writer
from time import sleep
import datetime

class air_quality:
    def __init__(self):
        # Define the Arduino Nano 33 BLE's address information:
        self.MAC_Address = "4c:f9:a9:9a:b2:da"
        self.Service_UUID_Address = "19B10000-E8F2-537E-4F6C-D104768A1214"
        self.Characteristic_UUID_Address = "19B10001-E8F2-537E-4F6C-D104768A1214"
        # Define the peripheral device:
        self.device = btle.Peripheral(self.MAC_Address)
        # Define the characteristics:
        self.characteristics = self.device.getCharacteristics()
    # Display the given service's information if required.
    def print_service(self):
        service = self.device.getServiceByUUID(btle.UUID(self.Service_UUID_Address))
        print(service.getCharacteristics())
    # Obtain the local weather data from the Arduino Nano 33 BLE. 
    def obtain_characteristics(self):
        self.air_data = []
        # Create the air quality data array:
        for data in self.characteristics:
            if(data.uuid == self.Characteristic_UUID_Address):
                #print(data.read())
                self.air_data.extend(unpack('ffiii', data.read()))
                # Add the date to the air quality data array:
                _date = datetime.datetime.now().strftime("%m-%d-%y_%H:%M:%S")
                self.air_data.append(_date)
                # Print the air quality data array:
                print(self.air_data)
        sleep(1)
    # Insert the recently generated air quality data array into the given CSV file.
    def insert_data_to_CSV(self, file_name):
        with open(file_name, "a", newline="") as f:
            # Add a new row:
            writer(f).writerow(self.air_data)
            f.close()

air_quality_data = air_quality()
#air_quality_data.print_service()

try:
    while True:
        # Get the updated characteristics and insert them into the given CSV file every 30 seconds:
        air_quality_data.obtain_characteristics()
        air_quality_data.insert_data_to_CSV("air_quality_data_set.csv")
        sleep(30)
except KeyboardInterrupt:
        # Disconnect BLE:
        air_quality_data.device.disconnect()
        print("\r\nPeripheral Disconnected!")
