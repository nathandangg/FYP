//Final Ver wo Bluetooth
#include <math.h>

const int rotdirPin = 6; //LOW for moving in the positive direction
const int rotstepPin = 5; //HIGH for moving in the positive direction
const int homeswitchPin = 7;
const int rotswitchPin = 9;
const int sliderdirPin = 2; 
const int sliderstepPin = 3; 
const int rotateDelay = 3000; //rotate speed control
const int sliderDelay = 2000; //slider speed control
const int maxdist = 5001;
const int stepsPerTranslation = 100; //2cm
const long BAUDRATE = 9600;    // This is the default communication baud rate of the HC-05 module

float L = 500 * 5;
float L3 = 0;
float H = 360 * 5;
float calibratedangle = 0;

float currentdist = 0;
float currentangle = 0;

void setup()
{
	pinMode(sliderstepPin, OUTPUT);
	pinMode(sliderdirPin, OUTPUT);
	pinMode(rotdirPin, OUTPUT);
	pinMode(rotstepPin, OUTPUT);
	pinMode(homeswitchPin, INPUT);
	pinMode(rotswitchPin, INPUT);
	digitalWrite(sliderstepPin,LOW);
 	digitalWrite(rotstepPin,LOW);
  Serial.begin(9600);          // Init hardware serial
  Homing();
  mainfunction();
}

void loop()
{
	//mainfunction();
}

void resetLength()
{
  digitalWrite(sliderdirPin, HIGH);
  while(digitalRead(homeswitchPin) == HIGH)
  {
    digitalWrite(sliderstepPin, HIGH);
    delayMicroseconds(sliderDelay);
    digitalWrite(sliderstepPin, LOW);
    delayMicroseconds(sliderDelay);
  }

  delay(1000);
  
	digitalWrite(sliderdirPin, LOW);
	while(digitalRead(homeswitchPin) == LOW)
	{
		digitalWrite(sliderstepPin, HIGH);
		delayMicroseconds(sliderDelay);
		digitalWrite(sliderstepPin, LOW);
		delayMicroseconds(sliderDelay);
	}
  delay(1000);
  
  digitalWrite(sliderdirPin, HIGH);
  while(digitalRead(homeswitchPin) == HIGH)
  {
    digitalWrite(sliderstepPin, HIGH);
    delayMicroseconds(sliderDelay);
    digitalWrite(sliderstepPin, LOW);
    delayMicroseconds(sliderDelay);
  }

	delay(1000);
  
	currentdist = 0;
}

void resetAngle()
{ 
	digitalWrite(rotdirPin, HIGH); 
	while(digitalRead(rotswitchPin) == HIGH)
	{
  	digitalWrite(rotstepPin,HIGH);
  	delayMicroseconds(2500); 
  	digitalWrite(rotstepPin,LOW);
  	delayMicroseconds(1000);
	}
 	digitalWrite(rotstepPin,LOW);
	delay(1000);
  
	currentangle = 0;

  // Move to 90 to avoid collision
	digitalWrite(rotdirPin, LOW); 
  for(int x = currentangle; x < 423 ;x++)
  {
    digitalWrite(rotstepPin, HIGH);
    delayMicroseconds(rotateDelay);
    digitalWrite(rotstepPin, LOW);
    delayMicroseconds(rotateDelay);

    currentangle++;
  }
}
void Homing()
{
//  Serial.println("Homing");
  resetAngle();
  resetLength();
}

void mainfunction()
{
	float newangle;
  float newdist;

	while(currentdist < maxdist)
	{
  	if(Serial.available())
  	{
      String input = Serial.readStringUntil('\n'); // Read the incoming data until a newline character
      input.trim(); // Remove leading and trailing whitespaces

      // Start next Ascan signal
      if(input.startsWith("a")){
        
        // calculate target dist
        if(round(currentdist) == 0 && round(currentangle) == 423)
          newdist = 0;
        else
          newdist = currentdist + stepsPerTranslation;

  			// move to target dist
        int directionL = (newdist > currentdist) ? HIGH : LOW;
        digitalWrite(sliderdirPin, directionL);
      
        while (round(currentdist) != round(newdist)) {
          digitalWrite(sliderstepPin, HIGH);
          delayMicroseconds(sliderDelay); 
          digitalWrite(sliderstepPin, LOW);
          delayMicroseconds(sliderDelay);
      
          if (directionL == HIGH) {
            currentdist++;
          } else {
            currentdist--;
          }
        }

        // Find target Angle
        if (currentdist < L) {
            L3 = L - currentdist;
            newangle = atan(H / L3) * 847 / M_PI;
        } else if (currentdist > L) {
            L3 = currentdist - L;
            newangle = 847 - (atan(H/ L3)) * 840 / M_PI;
        } else {
            newangle = 423;
        }
//        Serial.print("Calculated Angle: ");
//        Serial.println(newangle);

        // rotate to new angle
        int directionR = (newangle > currentangle) ? LOW : HIGH;
        digitalWrite(rotdirPin, directionR);
      
        while (round(currentangle) != round(newangle)) {
          digitalWrite(rotstepPin, HIGH);
          delayMicroseconds(rotateDelay); 
          digitalWrite(rotstepPin, LOW);
          delayMicroseconds(rotateDelay);
      
          if (directionR == LOW) {
            currentangle++;
          } else {
            currentangle--;
          }
        }
//        Serial.print("Real Angle: ");
//        Serial.println(currentangle);
        
        //transmit back to module
        Serial.write('b'); 
  		}
      if(input.startsWith("c")){
        Homing();
      }

      // Check if the received data starts with "P " and contains two numbers eg: P 20 90
      if (input.startsWith("P ")) {
        int spaceIndex = input.indexOf(' ');
        if (spaceIndex != -1) {
          String number1Str = input.substring(spaceIndex + 1); // Get the part after the space
          int targetPosition = number1Str.toInt(); // Convert it to an integer
          
          int directionL = (targetPosition > currentdist) ? HIGH : LOW;
          digitalWrite(sliderdirPin, directionL);
        
          while (round(currentdist) != targetPosition) {
            digitalWrite(sliderstepPin, HIGH);
            delayMicroseconds(sliderDelay); 
            digitalWrite(sliderstepPin, LOW);
            delayMicroseconds(sliderDelay);
        
            if (directionL == HIGH) {
              currentdist++;
            } else {
              currentdist--;
            }
          }
          int spaceIndex2 = number1Str.indexOf(' ');
          if (spaceIndex2 != -1) {
            String number2Str = number1Str.substring(spaceIndex2 + 1); // Get the part after the second space
            int targetAngle = number2Str.toInt(); // Convert it to an integer

            int directionR = (targetAngle > currentangle) ? LOW : HIGH;
            digitalWrite(rotdirPin, directionR);
          
            while (round(currentangle) != targetAngle) {
              digitalWrite(rotstepPin, HIGH);
              delayMicroseconds(rotateDelay); 
              digitalWrite(rotstepPin, LOW);
              delayMicroseconds(rotateDelay);
          
              if (directionR == LOW) {
                currentangle++;
              } else {
                currentangle--;
              }
            }
          }
        }
        Serial.write('b');
//      Serial.print("Current distance: ");
//      Serial.println(currentdist);
//      Serial.print("Current angle: ");
//      Serial.println(currentangle);
      }
  	}
  }
}
