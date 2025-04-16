#include <Servo.h>

// Define 5 servos for the prosthetic hand
Servo thumbServo;
Servo indexServo;
Servo middleServo;
Servo ringServo;
Servo pinkyServo;

void setup() {
  Serial.begin(9600);  // Start serial communication

  // Attach servos to pins
  thumbServo.attach(3);
  indexServo.attach(5);
  middleServo.attach(6);
  ringServo.attach(9);
  pinkyServo.attach(10);

  // Initialize hand to open position
  openHand();
}

void loop() {
  if (Serial.available()) {
    int gesture = Serial.parseInt();  // Receive gesture ID from Python

    if (gesture > 0) {  // Validate input
      performGesture(gesture);
    }
  }
}

// === Open all fingers ===
void openHand() {
  thumbServo.write(0);
  indexServo.write(0);
  middleServo.write(0);
  ringServo.write(0);
  pinkyServo.write(0);
}

// === Close all fingers ===
void closeHand() {
  thumbServo.write(90);
  indexServo.write(90);
  middleServo.write(90);
  ringServo.write(90);
  pinkyServo.write(90);
}

// === Execute motion based on the predicted gesture ===
void performGesture(int gesture) {
  // Always start from an open position for safety
  openHand();
  delay(200);  // Small pause for smoother transitions

  switch (gesture) {
    case 1:  // Fist
      closeHand();
      break;

    case 2:  // Point
      indexServo.write(90);
      break;

    case 3:  // Peace Sign
      indexServo.write(90);
      middleServo.write(90);
      break;

    case 4:  // Thumbs Up
      thumbServo.write(90);
      break;

    case 5:  // Pinky Out
      pinkyServo.write(90);
      break;

    case 11:  // Relax/Open Hand
      openHand();
      break;

    case 12:  // Strong Grip
      closeHand();
      break;

    default:
      // Unknown gesture, do nothing (or stay open for safety)
      openHand();
      break;
  }
}
