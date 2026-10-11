/*
 * GARUDA: ESP32 MPU6050 & Helmet Visor Controller
 * Dual-stream edge telemetry for GARUDA Rider Intent Prediction
 */

#include <Wire.h>

const uint8_t MPU6050_ADDRESS = 0x68;
const uint8_t LED_GREEN = 25;
const uint8_t LED_YELLOW = 26;
const uint8_t LED_RED = 27;
const float ACCEL_SCALE = 16384.0f;
const float GYRO_SCALE = 131.0f;

void setLeds(bool green, bool yellow, bool red) {
  digitalWrite(LED_GREEN, green ? HIGH : LOW);
  digitalWrite(LED_YELLOW, yellow ? HIGH : LOW);
  digitalWrite(LED_RED, red ? HIGH : LOW);
}

int16_t readWord(uint8_t reg) {
  Wire.beginTransmission(MPU6050_ADDRESS);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDRESS, (uint8_t)2);
  return (int16_t)((Wire.read() << 8) | Wire.read());
}

void setup() {
  Serial.begin(115200);
  Serial.println("GARUDA_HARDWARE_READY");
  Wire.begin();
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  Wire.beginTransmission(MPU6050_ADDRESS);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission(true);
  setLeds(true, false, false);
}

void loop() {
  if (Serial.available()) {
    char command = Serial.read();
    if (command == 'C') setLeds(true, false, false);
    if (command == 'W') setLeds(false, true, false);
    if (command == 'E') setLeds(false, false, true);
  }
  int16_t ax = readWord(0x3B), ay = readWord(0x3D), az = readWord(0x3F);
  int16_t gx = readWord(0x43), gy = readWord(0x45), gz = readWord(0x47);
  Serial.printf("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", (float)gx / GYRO_SCALE, (float)gy / GYRO_SCALE, (float)gz / GYRO_SCALE, (float)ax / ACCEL_SCALE * 9.81f, (float)ay / ACCEL_SCALE * 9.81f, (float)az / ACCEL_SCALE * 9.81f);
  delay(10);
}
