void setup() {
  // put your setup code here, to run once:
  pinMode(7, INPUT);
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  bool val = digitalRead(7);
  if(val==0) Serial.println("OBST");
  delay(50);
}
