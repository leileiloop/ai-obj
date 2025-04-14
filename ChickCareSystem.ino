#include <DHT.h>
#include <HX711.h>
#include <Wire.h>
#include <RTClib.h> // DS3231 RTC library
#include <Servo.h>   
#include <EEPROM.h>

// Pin definitions for DHT11 sensor
#define DHTPIN 12
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// Load cell pins for food, water, and additional weight sensors
#define LOADCELL_DOUT_FOOD  2
#define LOADCELL_SCK_FOOD  3
#define LOADCELL_DOUT_WATER 4
#define LOADCELL_SCK_WATER 5
#define LOADCELL_DOUT  6
#define LOADCELL_SCK   7

const int TRIG_PIN_WATER = 36;
const int ECHO_PIN_WATER = 38;
const int TRIG_PIN_FOOD = 40;
const int ECHO_PIN_FOOD = 42;

HX711 scaleFood, scaleWater, scale; // Load cells for food, water, and additional weight

// Relay pins
const int relay1 = 44, relay2 = 46, relay3 = 58, relay4 = 50;
const int waterPumpRelay = 9, exhaustFanRelay = 11;
const int relayUVPin = 10, relaySprinklePin = 8;

// Servo pin for dispensing food
Servo foodServo;
const int servoPin = 13;

// Stepper motor control pins
const int stepPin = 30, dirPin = 32;
const int enablePin = 34;   // Connect to ENABLE on DRV885

// RTC setup
RTC_DS3231 rtc;
DateTime t;

float calibration_factor_food = -88525;
float calibration_factor_water = -88525;
float calibration_factor = -88525;
float waterWeightThreshold = 0.00;
float foodWeightThreshold = 0.00;
int temperatureThreshold = 38;
int humidityThreshold = 70;

unsigned long lastUpdateTime = 0;
const unsigned long updateInterval = 2000;

// Variables for time management
unsigned long previousMillis = 0; 
const unsigned long intervalDay1To7 = 60 * 60 * 1000; // 1 hour in milliseconds
const unsigned long intervalDay8To28 = 10 * 60 * 1000; // 10 minutes in milliseconds
const unsigned long intervalDay29Onward = 15 * 60 * 1000; // 15 minutes in milliseconds
unsigned long duration = 0;

// Variables to track the current day and cycle
int current2Day = 1;   // Tracks the current day
int cyclesPerDay = 6; // Default cycles per day (Day 1 to 7)
DateTime lastCycleTime; // Tracks the time of the last cycle

bool lightsState;
bool fanState;

int waterMaxHeight = 25, foodMaxHeight = 35;
long durationWater, distanceWater, durationFood, distanceFood;
int waterPercentage, foodPercentage;
int currentDay = 0, lastActivatedDay = 0;
unsigned long startMillis = 0;
const unsigned long runDuration = 30000;
bool devicesRunning = false;

unsigned long startTime = 0;
bool dispensing = false;
bool foodDispensed = false;

unsigned long durationStartTime = 0; // Stores the start time of the UV light ON period
bool uvLightOn = false; // Tracks if the UV light is currently on

const unsigned long pumpDuration = 30000; // Pump duration in milliseconds (1 minute)
const int intervalHours = 8;             // Interval in hours to turn on the pump


void setup() {
  Serial.begin(9600);
  dht.begin();

  // Initialize Load cells
  setupLoadCells();
  
  // Initialize Ultrasonic sensors
  setupUltrasonicSensors();

  // Setup Servo motor for food dispensing
  foodServo.attach(servoPin);
  foodServo.write(120);

  // Setup Relay pins
  setupRelays();

  if (!rtc.begin()) {
    Serial.println("Couldn't find RTC");
    while (1);
  }
  
  // Uncomment if the RTC is not set
   //rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
  

  lastCycleTime = rtc.now(); // Initialize lastCycleTime

  
   
  
  // Stepper motor setup
  setupStepperMotor();



}

void loop() {
  // Check if a serial command is received
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();  // Remove spaces and newline characters
    processCommand(command);
  }

  // Run scheduled tasks every updateInterval milliseconds
  if (millis() - lastUpdateTime >= updateInterval) {
    lastUpdateTime = millis();
    readTemperatureHumidity();
    controlLightsAndFan();
    controlWaterAndFoodLevels();
    checkAndRunScheduledDevices();
    measureWaterAndFoodLevels();
    checkAndRunScheduledUVLight();
    scheduleAndDispenseFood();
    scheduleWaterPump();
  }
}


void setupLoadCells() {
  scaleFood.begin(LOADCELL_DOUT_FOOD, LOADCELL_SCK_FOOD);
  scaleFood.set_scale(calibration_factor_food);
  scaleFood.tare();

  scaleWater.begin(LOADCELL_DOUT_WATER, LOADCELL_SCK_WATER);
  scaleWater.set_scale(calibration_factor_water);
  scaleWater.tare();

  scale.begin(LOADCELL_DOUT, LOADCELL_SCK);
  scale.set_scale(calibration_factor);
  scale.tare();
}

void setupUltrasonicSensors() {
  pinMode(TRIG_PIN_WATER, OUTPUT);
  pinMode(ECHO_PIN_WATER, INPUT);
  pinMode(TRIG_PIN_FOOD, OUTPUT);
  pinMode(ECHO_PIN_FOOD, INPUT);
}

void setupRelays() {
  pinMode(relay1, OUTPUT);
  pinMode(relay2, OUTPUT);
  pinMode(relay3, OUTPUT);
  pinMode(relay4, OUTPUT);
  pinMode(waterPumpRelay, OUTPUT);
  pinMode(exhaustFanRelay, OUTPUT);
  pinMode(relayUVPin, OUTPUT);
  pinMode(relaySprinklePin, OUTPUT);
  pinMode(enablePin, OUTPUT);


  // Turn off all relays initially
  digitalWrite(relay1, LOW);
  digitalWrite(relay2, LOW);
  digitalWrite(relay3, LOW);
  digitalWrite(relay4, LOW);

  digitalWrite(waterPumpRelay, HIGH);
  digitalWrite(exhaustFanRelay, HIGH);
  digitalWrite(relayUVPin, HIGH);
  digitalWrite(relaySprinklePin, HIGH);
}

void setupStepperMotor() {
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);

  digitalWrite(dirPin, HIGH);  // Set motor direction
  // Enable the driver (assuming active low ENABLE pin)
  digitalWrite(enablePin, HIGH);

}

// Function to process serial commands
void processCommand(String command) {
  if (command == "STOP_CONVEYOR") {
    digitalWrite(enablePin, HIGH);  // Stop stepper motor
    Serial.println("{\"Conveyor\":\"OFF\"}");
  } 
  else if (command == "STOP_UV_LIGHT") {
    digitalWrite(relayUVPin, HIGH);  // Turn off UV light
    Serial.println("{\"UVLight\":\"OFF\"}");
  } 
  else if (command == "STOP_SPRINKLE") {
    digitalWrite(relaySprinklePin, LOW);  // Turn off Sprinkle system
    Serial.println("{\"Sprinkle\":\"OFF\"}");
  }
}

void readTemperatureHumidity() {
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  
  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("DHT sensor error!");
    return;
  }

  Serial.print("{\"Hum\":");
  Serial.print(humidity);
  Serial.print(",\"Temp\":");
  Serial.print(temperature);
  Serial.println("}");
}

void controlLightsAndFan() {

  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  // Check temperature and control lights
  if (temperature <= temperatureThreshold) {
    turnOnLights();
    lightsState = true;  // Lights are ON
  } else {
    turnOffLights();
    lightsState = false; // Lights are OFF
  }

  // Check humidity and control the exhaust fan
  if (humidity > humidityThreshold) {
    digitalWrite(exhaustFanRelay, LOW); // Turn on exhaust fan
    fanState = true; // Exhaust fan is ON
  } else {
    digitalWrite(exhaustFanRelay, HIGH); // Turn off exhaust fan
    fanState = false; // Exhaust fan is OFF
  }

  // Log the state of the lights and fan
  Serial.print("{\"Lights\":");
  Serial.print(lightsState ? "\"ON\"" : "\"OFF\"");
  Serial.print(",\"ExhaustFan\":");
  Serial.print(fanState ? "\"ON\"" : "\"OFF\"");
  Serial.println("}");
}

void controlWaterAndFoodLevels() {
  // Water level monitoring
  scaleWater.set_scale(calibration_factor_water);
  float waterWeight = scaleWater.get_units(5); // Measure water weight

  // Validate water weight
  if (waterWeight <= 0) {
    Serial.println("{\"Error\":\"Invalid Water Weight\"}");
    waterWeight = 0; // Ensure it does not go below 0
  }

  Serial.print("{\"Water_Weight\":");
  Serial.print(waterWeight);
  Serial.println("}");

  // Control water pump based on water weight
  if (waterWeight < waterWeightThreshold) {
    Serial.println("{\"message\":\"Water is Low\"}");
  } else {
    Serial.println("{\"message\":\"Sufficient Water\"}");
  }

  // Food level monitoring
  scaleFood.set_scale(calibration_factor_food);
  float foodWeight = scaleFood.get_units(5); // Measure food weight

  // Validate food weight
  if (foodWeight < 0) {
    Serial.println("{\"Error\":\"Invalid Food Weight\"}");
    foodWeight = 0; // Ensure it does not go below 0
  }

  Serial.print("{\"Food_Weight\":");
  Serial.print(foodWeight);
  Serial.println("}");

  // Independent food scheduling continues, but check weight for logging
  if (foodWeight < foodWeightThreshold) {
    Serial.println("{\"message\":\"Food is Low\"}");
  } else {
    Serial.println("{\"message\":\"Sufficient Food\"}");
  }

  // Chicken weight monitoring
  scale.set_scale(calibration_factor);
  float weight = scale.get_units(5); // Measure chicken weight

  // Validate chicken weight
  if (weight < 0) {
    Serial.println("{\"Error\":\"Invalid Weight\"}");
    weight = 0; // Ensure it does not go below 0
  }

  Serial.print("{\"Weight\":");
  Serial.print(weight);
  Serial.println("}");
}

void checkAndRunScheduledDevices() {
  DateTime now = rtc.now();
  int currentDay = now.day();
  int currentHour = now.hour();
  int currentMinute = now.minute();
  
   // Check the schedule
  if ((currentDay >= lastActivatedDay + 1 || 
       (currentDay < lastActivatedDay && (31 - lastActivatedDay + currentDay) >= 1)) &&
      currentHour == 10 && currentMinute == 0) {
      
      // Perform the action
      Serial.println("Scheduled action performed!");
      lastActivatedDay = currentDay;

    // Activate devices
    startMillis = millis();
    devicesRunning = true;
    Serial.println("Devices activated.");
  }

  // Check if devices should still be running
  if (devicesRunning) {
    if (millis() - startMillis < runDuration) {

      digitalWrite(enablePin, LOW);
      
      digitalWrite(stepPin, HIGH);
      delayMicroseconds(4000);
      digitalWrite(stepPin, LOW);
      delayMicroseconds(4000);
      digitalWrite(relaySprinklePin, HIGH);
      Serial.println("{\"Conveyor\":\"ON\"}");
      Serial.println("{\"Sprinkle\":\"ON\"}");
    } else {
      // Turn off devices after the duration
      digitalWrite(enablePin, HIGH);
      digitalWrite(relaySprinklePin, LOW);
      devicesRunning = false;
      Serial.println("{\"Conveyor\":\"OFF\"}");
      Serial.println("{\"Sprinkle\":\"OFF\"}");
    }
  }
}


void measureWaterAndFoodLevels() {
  // Measure water level
  digitalWrite(TRIG_PIN_WATER, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN_WATER, LOW);
  durationWater = pulseIn(ECHO_PIN_WATER, HIGH);
  distanceWater = durationWater * 0.034 / 2;

  waterPercentage = map(distanceWater, 0, waterMaxHeight, 100, 0);
  waterPercentage = constrain(waterPercentage, 0, 100);
  Serial.print("{\"Water_Level\":");
  Serial.print(waterPercentage);
  Serial.println("}");

  // Measure food level
  digitalWrite(TRIG_PIN_FOOD, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN_FOOD, LOW);
  durationFood = pulseIn(ECHO_PIN_FOOD, HIGH);
  distanceFood = durationFood * 0.034 / 2;

  foodPercentage = map(distanceFood, 0, foodMaxHeight, 100, 0);
  foodPercentage = constrain(foodPercentage, 0, 100);
  Serial.print("{\"Food_Level\":");
  Serial.print(foodPercentage);
  Serial.println("}");
}


void checkAndRunScheduledUVLight() {
  DateTime now = rtc.now();

  // Ensure `current2Day` starts from the correct value (e.g., based on RTC date)
  static bool initialized = false;
  if (!initialized) {
    current2Day = 1; // Initialize based on your system's start day
    initialized = true;
  }

  // Update exposure duration and cycles
  if (current2Day <= 7) {
    duration = intervalDay1To7;
    cyclesPerDay = 6;
  } else if (current2Day >= 8 && current2Day <= 28) {
    duration = intervalDay8To28;
    cyclesPerDay = 12;
  } else if (current2Day >= 29) {
    duration = intervalDay29Onward;
    cyclesPerDay = 12;
  }

  unsigned long cycleIntervalSeconds = (24 * 60 * 60) / cyclesPerDay;

  // Ensure `lastCycleTime` is initialized correctly
  static DateTime lastCycleTime = now;

  // Check if the cycle interval has elapsed
  if (!uvLightOn && (now.unixtime() - lastCycleTime.unixtime()) >= cycleIntervalSeconds) {
    digitalWrite(relayUVPin, LOW); // Activate UV light relay
    Serial.println("{\"UVLight\":\"ON\"}");
    uvLightOn = true;
    durationStartTime = millis(); // Record the start time
    lastCycleTime = now;          // Update the last cycle time
  }

  // Check if the UV light has been on for the set duration
  if (uvLightOn && (millis() - durationStartTime >= (unsigned long)duration * 1000)) {
    digitalWrite(relayUVPin, HIGH); // Deactivate UV light relay
    Serial.println("{\"UVLight\":\"OFF\"}");
    uvLightOn = false;             // Reset the state
  }

  // Update the day counter if the day has changed
  static int lastDay = now.day();
  if (now.day() != lastDay) {
    current2Day++;
    lastDay = now.day();
  }
}


void turnOnLights() {
  digitalWrite(relay1, LOW);
  digitalWrite(relay2, LOW);
  digitalWrite(relay3, LOW);
  digitalWrite(relay4, LOW);
}

void turnOffLights() {
  digitalWrite(relay1, HIGH);
  digitalWrite(relay2, HIGH);
  digitalWrite(relay3, HIGH);
  digitalWrite(relay4, HIGH);
}



void scheduleWaterPump() {
  static int lastActivationHour = -1;    // Tracks the last activation hour
  static bool pumpRunning = false;       // Flag to track if the pump is currently running
  static unsigned long pumpStartTime = 0; // Time when the pump was turned on

  DateTime now = rtc.now();              // Get the current time from the RTC
  unsigned long currentMillis = millis(); // Current time in milliseconds

  // Check if the pump needs to be turned on
  if ((now.hour() % intervalHours == 0) && now.minute() == 0 && now.second() == 0 && lastActivationHour != now.hour()) {
    digitalWrite(waterPumpRelay, LOW);   // Turn on the pump
    Serial.println("{\"Water\":\"ON\"}");
    pumpRunning = true;                  // Mark the pump as running
    pumpStartTime = currentMillis;       // Record the start time
    lastActivationHour = now.hour();     // Update the last activation hour
  }

  // Turn off the pump after the specified duration
  if (pumpRunning && (currentMillis - pumpStartTime >= pumpDuration)) {
    digitalWrite(waterPumpRelay, HIGH);  // Turn off the pump
    Serial.println("{\"Water\":\"OFF\"}");
    pumpRunning = false;                 // Mark the pump as not running
  }
}





void scheduleAndDispenseFood() {
  // Define the dispensing schedule (e.g., 3 times per day at specific times)
  const int numDispensesPerDay = 3; // Number of times to dispense food daily
  const int dispenseHours[numDispensesPerDay] = {8, 14, 20}; // Hours to dispense food (24-hour format)
  static bool foodDispensedToday[numDispensesPerDay] = {false, false, false}; // Flags for each dispense time

  DateTime now = rtc.now(); // Get the current time from the RTC

  // Loop through the scheduled times
  for (int i = 0; i < numDispensesPerDay; i++) {
    // Check if it's time to dispense food and it hasn't been dispensed yet today
    if (now.hour() == dispenseHours[i] && now.minute() == 0 && !foodDispensedToday[i]) {
      dispenseFood(); // Call the dispensing function
      foodDispensedToday[i] = true; // Mark this slot as dispensed

      // Log the dispense event
      Serial.print("{\"ScheduledFoodDispense\":\"Hour ");
      Serial.print(dispenseHours[i]);
      Serial.println("\"}");
    }

    // Reset the flag for the next day at midnight
    if (now.hour() == 0 && now.minute() == 0) {
      foodDispensedToday[i] = false;
    }
  }
}


void dispenseFood() {
  static unsigned long startTime = 0; // Track the start time
  static bool dispensing = false;    // Dispensing state
  static bool foodDispensed = false; // Food dispensed state

  if (!dispensing) {
    // Start the dispensing process
    Serial.println("{\"Food\":\"ON\"}");
    foodServo.write(75); // Rotate the servo to dispense position
    startTime = millis(); // Record the start time
    dispensing = true;
    foodDispensed = false; // Reset the food dispensed flag
  }

  // Check if the dispensing time (10 seconds) has elapsed
  if (dispensing && !foodDispensed) {
    if (millis() - startTime >= 5000) { // 10 seconds
      foodServo.write(113); // Move the servo back to the initial position
      Serial.println("{\"Food\":\"OFF\"}");
      dispensing = false;
      foodDispensed = true; // Set the flag to indicate food has been dispensed
    }
  }
}

long microsecondsToCentimeters(long microseconds) {
  return microseconds / 29 / 2; // Speed of sound: 29 microseconds per cm round trip
}
