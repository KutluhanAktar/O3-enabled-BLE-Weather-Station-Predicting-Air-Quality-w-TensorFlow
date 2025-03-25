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
//                                Button (6x6)
// D3  --------------------------- +


// Include the required libraries.
#include "DFRobot_OzoneSensor.h"
#include <Adafruit_BMP085.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// Import the required TensorFlow modules.
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/version.h"

// Import the converted TensorFlow Lite model.
#include "air_quality_level.h"

// TFLite globals, used for compatibility with Arduino-style sketches:
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  // Create an area of memory to use for input, output, and other TensorFlow arrays.
  constexpr int kTensorArenaSize = 15 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace

// Define the threshold value for the model outputs (results).
float threshold = 0.75;

// Define the air quality level (class) names:
String classes[] = {"Good", "Moderate", "Unhealthy"};

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

// Define the model initialization button.
#define run_model 3

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
static const unsigned char PROGMEM _good [] = {
0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x7F, 0x80, 0x00, 0x00, 0x00, 0xFF, 0xC0, 0x00, 0x00,
0x01, 0xFF, 0xC0, 0x00, 0x00, 0x03, 0xFF, 0xE0, 0x00, 0x00, 0x03, 0xFF, 0xFC, 0x00, 0x00, 0x3F,
0xFF, 0xFF, 0x00, 0x00, 0x7F, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0x80, 0x00, 0xFF, 0xFF,
0xFF, 0x80, 0x01, 0xFF, 0xFF, 0xFF, 0xC0, 0x03, 0xFF, 0xFF, 0xFF, 0xC0, 0x07, 0xFF, 0xFF, 0xFF,
0xC0, 0x0F, 0xFF, 0xFF, 0xFF, 0xC0, 0x0F, 0xFF, 0xFF, 0xFF, 0xE0, 0x0F, 0xFF, 0xFF, 0xFF, 0xF0,
0x0F, 0xFF, 0xFF, 0xFF, 0xF0, 0x0F, 0xFF, 0xFC, 0xFF, 0xF0, 0x07, 0xFF, 0xF8, 0x7F, 0xF0, 0x03,
0xF8, 0x1C, 0x7F, 0xE0, 0x01, 0xF1, 0x1C, 0xFE, 0x00, 0x00, 0x00, 0xF8, 0xBC, 0x00, 0x00, 0x00,
0x79, 0x00, 0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x3E, 0x00, 0x00, 0x00, 0x00, 0x3C,
0x00, 0x00, 0x00, 0x00, 0x3C, 0x00, 0x00, 0x00, 0x00, 0x3C, 0x00, 0x00, 0x00, 0x00, 0x3C, 0x00,
0x00, 0x00, 0x00, 0x7C, 0x00, 0x00, 0x00, 0x00, 0x7C, 0x00, 0x00, 0x00, 0x00, 0x7E, 0x00, 0x00,
0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0x03, 0xFF, 0xC0, 0x00, 0x00,
0x1F, 0x7F, 0x70, 0x00, 0x00, 0x40, 0x67, 0x0C, 0x00, 0x00, 0x00, 0xC1, 0x00, 0x00, 0x00, 0x01,
0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};
static const unsigned char PROGMEM _moderate [] = {
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0F, 0xC0, 0x00, 0x00, 0x00, 0x30, 0x30, 0x00, 0x00,
0x00, 0x40, 0x18, 0x00, 0x00, 0x00, 0xC0, 0x08, 0x00, 0x00, 0x00, 0x80, 0x04, 0x00, 0x00, 0x00,
0x80, 0x04, 0x00, 0x00, 0x70, 0x00, 0x04, 0x00, 0x03, 0x86, 0x00, 0x04, 0x00, 0x04, 0x01, 0x00,
0x04, 0x00, 0x08, 0x00, 0x80, 0x08, 0x00, 0x10, 0x00, 0x40, 0x08, 0x00, 0x10, 0x00, 0x78, 0x10,
0x00, 0x10, 0x00, 0x46, 0x00, 0x00, 0x20, 0x00, 0x41, 0x80, 0x00, 0x20, 0x00, 0x00, 0x80, 0x00,
0xE0, 0x00, 0x00, 0x40, 0x01, 0x00, 0x60, 0x00, 0x40, 0x02, 0x00, 0x18, 0x00, 0x20, 0x02, 0x00,
0x08, 0x00, 0x20, 0x00, 0x00, 0x08, 0x00, 0x20, 0x00, 0x00, 0x08, 0x00, 0x40, 0x3F, 0xFF, 0xF0,
0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x01, 0x80, 0x00, 0x00, 0x00, 0x06,
0x00, 0x00, 0x00, 0x7F, 0xF8, 0x00, 0x07, 0xFC, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
0x3E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
};
static const unsigned char PROGMEM _unhealthy [] = {
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x0E, 0x78, 0x00, 0x00, 0x00, 0x7F, 0x1C,
0x00, 0x00, 0x00, 0xFF, 0xEE, 0x00, 0x00, 0x00, 0x7F, 0xE7, 0x80, 0x00, 0x00, 0x07, 0xF3, 0x80,
0x00, 0x00, 0x00, 0x78, 0x60, 0x00, 0x00, 0x00, 0x3F, 0x74, 0x00, 0x00, 0x00, 0x26, 0x7C, 0x00,
0x00, 0x00, 0x07, 0x04, 0x00, 0x00, 0x00, 0x01, 0x86, 0x00, 0x00, 0x00, 0x01, 0x86, 0x00, 0x00,
0x00, 0x01, 0x86, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xC7, 0x80, 0x00, 0x00,
0x01, 0xC7, 0x80, 0x00, 0x00, 0x01, 0xC7, 0x80, 0x00, 0x02, 0x03, 0xC7, 0x80, 0x00, 0x02, 0x03,
0xC7, 0x80, 0x00, 0x02, 0x0B, 0xE7, 0x80, 0x00, 0x02, 0x53, 0xE7, 0x80, 0x00, 0x07, 0xFF, 0xEF,
0x80, 0x00, 0x0F, 0xFF, 0xFF, 0xE0, 0x00, 0x0F, 0xFF, 0xFF, 0xE0, 0x00, 0x08, 0x08, 0x3F, 0xE0,
0x00, 0x08, 0x08, 0x3F, 0xE0, 0x00, 0x0F, 0xFF, 0xFF, 0xE0, 0x00, 0x08, 0x08, 0x3F, 0xE0, 0x00,
0x08, 0x08, 0x3F, 0xFF, 0xF0, 0x0F, 0xFF, 0xFF, 0xE0, 0x08, 0x0F, 0xFF, 0xFF, 0xE0, 0x08, 0x1F,
0xFF, 0xFF, 0xE0, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

// Define the data holders: 
float _temperature, _altitude, _real_altitude;
int ozoneConcentration, _pressure, _sea_level_pressure, wind_speed;

void setup(){
  Serial.begin(9600);

  pinMode(notification, OUTPUT);
  pinMode(run_model, INPUT_PULLUP);

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

  // TensorFlow Lite Model settings:
  
  // Set up logging (will report to Serial, even within TFLite functions).
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure.
  model = tflite::GetModel(air_quality_level);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while(1);
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model.
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize,
    error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while(1);
  }

  // Assign model input and output buffers (tensors) to pointers.
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  delay(1000);
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

  collect_ozone_concentration();
  delay(1000);
  collect_BMP180_data();
  delay(500);
  collect_anemometer_data();
  delay(250);
  
  // Show the collected weather (air quality) data on the screen.
  show_weather_data();

 // Execute the TensorFlow Lite model to make predictions on the air quality levels (classes). 
 if(digitalRead(run_model) == LOW) run_inference_to_make_predictions();
}

void run_inference_to_make_predictions(){    
    // Scale (normalize) values (local weather data) depending on the model and copy them to the input buffer (tensor):
    model_input->data.f[0] = _temperature / 100;
    model_input->data.f[1] = _altitude / 100;
    model_input->data.f[2] = ozoneConcentration / 1000;
    model_input->data.f[3] = _pressure / 100000;
    model_input->data.f[4] = wind_speed / 30;
    
    // Run inference:
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on the given input.");
    }

    // Read predicted y values (air quality classes) from the output buffer (tensor): 
    for(int i = 0; i<3; i++){
      if(model_output->data.f[i] >= threshold){
        // Display the detection result (class).
        display.clearDisplay(); 
        if(i == 0) display.drawBitmap(44, 0, _good, 40, 40, SSD1306_WHITE);
        if(i == 1) display.drawBitmap(44, 0, _moderate, 40, 40, SSD1306_WHITE);
        if(i == 2) display.drawBitmap(44, 0, _unhealthy, 40, 40, SSD1306_WHITE);
        // Print:
        int str_x = classes[i].length() * 11;
        display.setTextSize(2); 
        display.setTextColor(SSD1306_WHITE);
        display.setCursor((SCREEN_WIDTH - str_x) / 2, 48);
        display.println(classes[i]);
        display.display();
        digitalWrite(notification, HIGH); delay(1500); digitalWrite(notification, LOW); 
      }
    }
    // Exit and clear.
    delay(3000);

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
