/*
 * eval_radio.h
 */

#ifndef INC_EVAL_RADIO_H_
#define INC_EVAL_RADIO_H_

// Radio evaluation parameters
#define MIN_PA_LEVEL -30 // initial Tx transmit power, in dBm
#define MAX_PA_LEVEL 15 // final Tx transmit power, in dBm
#define N_PACKETS 1000 // number of packets transmitted for each Tx power level
#define PAYLOAD_LEN 100 // payload length of the transmitted packets
#define PACKET_DELAY 1 // delay between two packets, in seconds
#define USE_BUTTON 0   // Use the button to trigger the transmission of packets

void eval_radio(void);

#endif /* INC_EVAL_RADIO_H_ */
