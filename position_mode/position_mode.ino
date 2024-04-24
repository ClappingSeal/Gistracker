#include <DynamixelShield.h>

#if defined(ARDUINO_AVR_UNO) || defined(ARDUINO_AVR_MEGA2560)
  #include <SoftwareSerial.h>
  SoftwareSerial soft_serial(7, 8); // DYNAMIXELShield UART RX/TX
  #define DEBUG_SERIAL soft_serial
#elif defined(ARDUINO_SAM_DUE) || defined(ARDUINO_SAM_ZERO)
  #define DEBUG_SERIAL SerialUSB    
#else
  #define DEBUG_SERIAL Serial
#endif

const uint8_t DXL_ID1 = 1;
const uint8_t DXL_ID2 = 2;
const float DXL_PROTOCOL_VERSION = 1.0;

DynamixelShield dxl;

void setup() {
  DEBUG_SERIAL.begin(115200);
  dxl.begin(1000000);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  dxl.ping(DXL_ID1);
  dxl.torqueOff(DXL_ID1);
  dxl.setOperatingMode(DXL_ID1, OP_POSITION);
  dxl.torqueOn(DXL_ID1);

  dxl.ping(DXL_ID2);
  dxl.torqueOff(DXL_ID2);
  dxl.setOperatingMode(DXL_ID2, OP_POSITION);
  dxl.torqueOn(DXL_ID2);
}

void loop() {
  if (Serial.available() > 0) {
    String inputString = Serial.readStringUntil('\n');
    inputString.trim();

    // 입력이 '1'이면 모터의 현재 위치 출력
    if (inputString == "1") {
      int currentPosition1 = dxl.getPresentPosition(DXL_ID1, UNIT_DEGREE);
      int currentPosition2 = dxl.getPresentPosition(DXL_ID2, UNIT_DEGREE);
      DEBUG_SERIAL.print(currentPosition1);
      DEBUG_SERIAL.print(" ");
      DEBUG_SERIAL.println(currentPosition2);
    } else {
      int firstSpaceIndex = inputString.indexOf(' ');
      int secondSpaceIndex = inputString.indexOf(' ', firstSpaceIndex + 1);
      int thirdSpaceIndex = inputString.indexOf(' ', secondSpaceIndex + 1);

      if (firstSpaceIndex != -1 && secondSpaceIndex != -1 && thirdSpaceIndex != -1) {
        int goalPosition1 = inputString.substring(0, firstSpaceIndex).toInt();
        int goalPosition2 = inputString.substring(firstSpaceIndex + 1, secondSpaceIndex).toInt();
        int velocity1 = inputString.substring(secondSpaceIndex + 1, thirdSpaceIndex).toInt();
        int velocity2 = inputString.substring(thirdSpaceIndex + 1).toInt();

        // 첫 번째 모터 제어
        dxl.setGoalVelocity(DXL_ID1, velocity1); // 모터 1의 속도 설정
        dxl.setGoalPosition(DXL_ID1, goalPosition1, UNIT_DEGREE); // 모터 1의 위치 설정

        // 두 번째 모터 제어
        dxl.setGoalVelocity(DXL_ID2, velocity2); // 모터 2의 속도 설정
        dxl.setGoalPosition(DXL_ID2, goalPosition2, UNIT_DEGREE); // 모터 2의 위치 설정
      }
    }
  }
}

