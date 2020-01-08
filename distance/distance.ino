void setup() {
  // put your setup code here, to run once:
  pinMode(7, INPUT);
    pinMode(8, OUTPUT);
  Serial.begin(9600);

}

void loop() {
  // put your main code here, to run repeatedly:
  int val = digitalRead(7);
  if(val==0) Serial.println("OBST");
  else Serial.println("Liber colegu");
  //delay(500);
  if(val==0) digitalWrite(8, HIGH);
  else digitalWrite(8, LOW);
  //Serial.println(!val);
  delay(500);
}
