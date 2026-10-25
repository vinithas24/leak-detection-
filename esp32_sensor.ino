// OPTIONAL hardware: ESP32 + 2 flow sensors (YF-S201) + pressure sensor (analog).
// Prints CSV lines: flow_in,flow_out,pressure,water_usage  (read via USB serial)
volatile int pulsesIn = 0, pulsesOut = 0;
void IRAM_ATTR countIn()  { pulsesIn++; }
void IRAM_ATTR countOut() { pulsesOut++; }

void setup() {
  Serial.begin(115200);
  pinMode(25, INPUT_PULLUP); pinMode(26, INPUT_PULLUP);
  attachInterrupt(25, countIn, FALLING);
  attachInterrupt(26, countOut, FALLING);
  Serial.println("flow_in,flow_out,pressure,water_usage");
}
void loop() {
  pulsesIn = pulsesOut = 0; delay(1000);
  float flowIn  = pulsesIn  / 7.5;           // YF-S201: Hz / 7.5 = L/min
  float flowOut = pulsesOut / 7.5;
  float volts = analogRead(34) * 3.3 / 4095.0;
  float pressure = (volts - 0.5) * 30.0;     // 0.5-4.5V sensor, 0-120 psi (adjust!)
  Serial.printf("%.2f,%.2f,%.2f,%.2f\n", flowIn, flowOut, pressure, flowOut);
}
