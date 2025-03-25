         /////////////////////////////////////////////  
        //     O3-enabled BLE Weather Station      //
       //   Predicting Air Quality w/ TensorFlow  //
      //             ---------------             //
     //          (Arduino Nano 33 BLE)          //           
    //             by Kutluhan Aktar           // 
   //                                         //
  /////////////////////////////////////////////

//
// Via Nano 33 BLE, collate local weather data, build and train a TensorFlow neural network model, and run the model to predict air quality.
//
// For more information:
// https://www.theamplituhedron.com/projects/O3_enabled_BLE_Weather_Station_Predicting_Air_Quality_w_TensorFlow/
//
//
// Connections
// Arduino Nano 33 BLE :  
//                                DFRobot IIC Ozone Sensor
// A4  --------------------------- SDA
// A5  --------------------------- SCL
//                                BMP180 Barometric Pressure/Temperature/Altitude Sensor
// A4  --------------------------- SDA
// A5  --------------------------- SCL
//                                SSD1306 OLED Display (128x64)
// A4  --------------------------- SDA
// A5  --------------------------- SCL
//                                DFRobot Anemometer Kit
// A0  --------------------------- S (Yellow)
//                                5mm Green LED
// D2  --------------------------- +


// Include the required libraries.
#include <ArduinoBLE.h>
#include "DFRobot_OzoneSensor.h"
#include <Adafruit_BMP085.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// Create the BLE service:
BLEService air_quality_service("19B10000-E8F2-537E-4F6C-D104768A1214");

// Create the data characteristic and allow the remote device (central) to read and write:
BLECharacteristic airDataCharacteristic("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite, 20);

// Define the collect number. The collection range is 1-100: 
#define COLLECT_NUMBER   20 

// To modify the I2C address, configure the hardware IIC address by the dial switch - A0, A1 (ADDRESS_0 for [0 0]), (ADDRESS_1 for [1 0]), (ADDRESS_2 for [0 1]), (ADDRESS_3 for [1 1]).              
/*  
    The default IIC device address is ADDRESS_3: 
       ADDRESS_0               0x70  
       ADDRESS_1               0x71
       ADDRESS_2               0x72
       ADDRESS_3               0x73
*/
#define Ozone_IICAddress ADDRESS_3

// Define the IIC Ozone Sensor.
DFRobot_OzoneSensor Ozone;

// Define the timer for the IIC Ozone Sensor.
unsigned long ozone_timer = 0;
unsigned long timer = 0;

// Define the BMP180 Barometric Pressure/Temperature/Altitude Sensor.
Adafruit_BMP085 bmp;

// Define the anemometer kit's voltage signal pin (yellow).
#define  anemometer_signal A0

// Define the notification LED.
#define notification 2

// Define the SSD1306 screen settings:
#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 64 // OLED display height, in pixels
#define OLED_RESET    -1 // Reset pin # (or -1 if sharing Arduino reset pin)

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Define monochrome graphics:
static const unsigned char PROGMEM _error [] = {
0x00, 0x00, 0x00, 0x00, 0x00, 0x3F, 0xFC, 0x00, 0x00, 0xE0, 0x07, 0x00, 0x01, 0x80, 0x01, 0x80,
0x06, 0x00, 0x00, 0x60, 0x0C, 0x00, 0x00, 0x30, 0x08, 0x01, 0x80, 0x10, 0x10, 0x03, 0xC0, 0x08,
0x30, 0x02, 0x40, 0x0C, 0x20, 0x02, 0x40, 0x04, 0x60, 0x02, 0x40, 0x06, 0x40, 0x02, 0x40, 0x02,
0x40, 0x02, 0x40, 0x02, 0x40, 0x02, 0x40, 0x02, 0x40, 0x02, 0x40, 0x02, 0x40, 0x02, 0x40, 0x02,
0x40, 0x02, 0x40, 0x02, 0x40, 0x02, 0x40, 0x02, 0x40, 0x03, 0xC0, 0x02, 0x40, 0x01, 0x80, 0x02,
0x40, 0x00, 0x00, 0x02, 0x60, 0x00, 0x00, 0x06, 0x20, 0x01, 0x80, 0x04, 0x30, 0x03, 0xC0, 0x0C,
0x10, 0x03, 0xC0, 0x08, 0x08, 0x01, 0x80, 0x10, 0x0C, 0x00, 0x00, 0x30, 0x06, 0x00, 0x00, 0x60,
0x01, 0x80, 0x01, 0x80, 0x00, 0xE0, 0x07, 0x00, 0x00, 0x3F, 0xFC, 0x00, 0x00, 0x00, 0x00, 0x00
};
static const unsigned char PROGMEM _weather [] = {
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x02, 0x10, 0x00, 0x00,
0x02, 0x11, 0x00, 0x00, 0x01, 0x10, 0x00, 0x00, 0x01, 0x12, 0x00, 0x00, 0x10, 0xFC, 0x10, 0x00,
0x09, 0xFE, 0x60, 0x00, 0x07, 0x87, 0x80, 0x00, 0x03, 0x02, 0x00, 0x00, 0x03, 0x00, 0x7E, 0x00,
0x02, 0x00, 0xFF, 0x80, 0x02, 0x01, 0xC1, 0xC0, 0x03, 0x01, 0x80, 0xC0, 0x02, 0x1F, 0x00, 0x60,
0x04, 0x3B, 0x00, 0x60, 0x18, 0x60, 0x00, 0x60, 0x10, 0x40, 0x00, 0x70, 0x00, 0xC0, 0x00, 0x18,
0x07, 0xC0, 0x00, 0x0C, 0x0F, 0xC0, 0x00, 0x04, 0x0C, 0x00, 0x00, 0x04, 0x08, 0x00, 0x00, 0x04,
0x18, 0x00, 0x00, 0x04, 0x08, 0x00, 0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0E, 0x00, 0x00, 0x38,
0x07, 0xFF, 0xFF, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

// Create a struct (data) including all air quality data parameters:
struct data {
  float _temperature;
  float _altitude;
  int ozoneConcentration;
  int _pressure;
  int wind_speed;
};

// Define the data holders: 
struct data air_Quality_Data;
float _temperature, _altitude, _real_altitude;
int ozoneConcentration, _pressure, _sea_level_pressure, wind_speed;

void setup(){
  Serial.begin(9600);
  // Wait for the serial monitor to be initialized so as to display this peripheral device's address information successfully:
  //while(!Serial);

  pinMode(notification, OUTPUT);

  // Start the timer:
  ozone_timer = millis();

  // Initialize the SSD1306 screen:
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.display();
  delay(1000);
  
  // Check the IIC Ozone Sensor connection status:
  while(!Ozone.begin(Ozone_IICAddress)){
    Serial.println("IIC Ozone Sensor is not found!");
    err_msg();
    delay(1000);
  }
  Serial.println("\nIIC Ozone Sensor is connected successfully!\n");
  
  /*   
     Set IIC Ozone Sensor mode:
       MEASURE_MODE_AUTOMATIC    // active  mode
       MEASURE_MODE_PASSIVE      // passive mode
  */
  Ozone.SetModes(MEASURE_MODE_PASSIVE);

  // Check the BMP180 Barometric Pressure/Temperature/Altitude Sensor connection status: 
  while(!bmp.begin()){
    Serial.println("BMP180 Barometric Pressure/Temperature/Altitude Sensor is not found!");
    err_msg();
    delay(1000);
  }
  Serial.println("\nBMP180 Barometric Pressure/Temperature/Altitude Sensor is connected successfully!\n");

  // Check the BLE initialization status:
  while(!BLE.begin()){
    Serial.println("BLE initialization is failed!");
    err_msg();
  }
  Serial.println("\nBLE initialization is successful!\n");
  // Print this peripheral device's address information:
  Serial.print("MAC Address: "); Serial.println(BLE.address());
  Serial.print("Service UUID Address: "); Serial.println(air_quality_service.uuid());
  Serial.print("Characteristic UUID Address: ");Serial.println(airDataCharacteristic.uuid());
  Serial.println();

  // Set the local name this peripheral advertises: 
  BLE.setLocalName("AirQuality");
  // Set the UUID for the service this peripheral advertises:
  BLE.setAdvertisedService(air_quality_service);

  // Add the given characteristic to the service:
  air_quality_service.addCharacteristic(airDataCharacteristic);

  // Add the service to the device:
  BLE.addService(air_quality_service);

  // Assign event handlers for connected, disconnected devices to this peripheral:
  BLE.setEventHandler(BLEConnected, blePeripheralConnectHandler);
  BLE.setEventHandler(BLEDisconnected, blePeripheralDisconnectHandler);

  // Set the initial value for the given characteristic: 
  airDataCharacteristic.writeValue((byte)0);

  // Start advertising:
  BLE.advertise();
  Serial.println(("Bluetooth device active, waiting for connections..."));

  display.clearDisplay();   
  display.setTextSize(2); 
  display.setTextColor(SSD1306_BLACK, SSD1306_WHITE);
  display.setCursor(0,0);
  display.println("Heating...");
  display.display();
}


void loop(){
  // Wait until the IIC Ozone Sensor heats for 3 minutes.
  while (millis() - ozone_timer < 3*60*1000){ if (millis() - timer > 1000){ timer = millis(); } }
  // Transmit the collected weather (air quality) data to Raspberry Pi over BLE every 20 seconds.
  if (millis() - timer > 20000){
    update_characteristics();
    // After updating characteristics, notify the user.
    display.clearDisplay();   
    display.drawBitmap(48, 0, _weather, 32, 32, SSD1306_WHITE);
    display.setTextSize(1); 
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0,40);
    display.println("Given BLE characteristics are updated successfully!");
    display.display();
    digitalWrite(notification, HIGH); delay(1500); digitalWrite(notification, LOW);  
    timer = millis();
  }
  
  collect_ozone_concentration();
  delay(1000);
  collect_BMP180_data();
  delay(500);
  collect_anemometer_data();
  delay(250);
  
  // Show the collected weather (air quality) data on the screen.
  show_weather_data();

  // Poll for BLE events:
  BLE.poll();
}

void collect_ozone_concentration(){
  ozoneConcentration = Ozone.ReadOzoneData(COLLECT_NUMBER);
  Serial.print("\n\nOzone Concentration => "); Serial.print(ozoneConcentration); Serial.println(" PPB");
}

void collect_BMP180_data(){
  _temperature = bmp.readTemperature();
  _pressure = bmp.readPressure();
  // Calculate altitude assuming 'standard' barometric pressure of 1013.25 millibars (101325 Pascal).
  _altitude = bmp.readAltitude();
  _sea_level_pressure = bmp.readSealevelPressure();
  // To get a more precise altitude measurement, use the current sea level pressure, which will vary with the weather conditions. 
  _real_altitude = bmp.readAltitude(101500);
  // Print the data generated by the BMP180 Barometric Pressure/Temperature/Altitude Sensor.
  Serial.print("Temperature => "); Serial.print(_temperature); Serial.println(" *C");
  Serial.print("Pressure => "); Serial.print(_pressure); Serial.println(" Pa");
  Serial.print("Altitude => "); Serial.print(_altitude); Serial.println(" meters");
  Serial.print("Pressure at sea level (calculated) => "); Serial.print(_sea_level_pressure); Serial.println(" Pa");
  Serial.print("Real Altitude => "); Serial.print(_real_altitude); Serial.println(" meters");
}

void collect_anemometer_data(){
  float outvoltage = (analogRead(A0) * (3.3 / 1023.0)) + 0.1;
  // Calculate the wind speed (level) [1 - 30] according to the output voltage: 
  wind_speed = 6 * outvoltage;
  // Print the data generated by the Anemometer Kit.
  Serial.print("Wind Speed (Level) => "); Serial.print(wind_speed);
}

void show_weather_data(){
  display.clearDisplay();   
  display.setTextSize(1); 
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0,8); 
  display.println("Ozone Con. => " + String(ozoneConcentration) + " PPB");
  display.println("Wind Speed => " + String(wind_speed));
  display.println("Temp. => " + String(_temperature) + " *C");
  display.println("Pressure => " + String(_pressure) + " Pa");
  display.println("Altitude => " + String(_altitude) + " m");
  display.display();  
}

void err_msg(){
  // Show the error message on the SSD1306 screen.
  display.clearDisplay();   
  display.drawBitmap(48, 0, _error, 32, 32, SSD1306_WHITE);
  display.setTextSize(1); 
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0,40); 
  display.println("Check the serial monitor to see the error!");
  display.display();  
}

void blePeripheralConnectHandler(BLEDevice central) {
  // Central connected event handler:
  Serial.print("Connected event, central: ");
  Serial.println(central.address());
}

void blePeripheralDisconnectHandler(BLEDevice central) {
  // Central disconnected event handler:
  Serial.print("Disconnected event, central: ");
  Serial.println(central.address());
}

void update_characteristics(){
  // Update the data characteristic with the data struct:
  air_Quality_Data._temperature = _temperature;
  air_Quality_Data._altitude = _altitude;
  air_Quality_Data.ozoneConcentration = ozoneConcentration;
  air_Quality_Data._pressure = _pressure;
  air_Quality_Data.wind_speed = wind_speed;
  airDataCharacteristic.writeValue((byte *) &air_Quality_Data, 20);
}
