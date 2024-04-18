/*
 * eval_radio.c
 */

#include <adc_dblbuf.h>
#include "eval_radio.h"
#include "config.h"
#include "main.h"
#include "s2lp.h"

// Adding include to use UART
#include "usart.h"


void eval_radio(void)
{
	uint8_t uart_buf[1];

	DEBUG_PRINT("[DBG] Radio evaluation mode\r\n");

	uint8_t buf[PAYLOAD_LEN];
	for (uint16_t i=0; i < PAYLOAD_LEN; i++) {
		buf[i] = (uint8_t) (i & 0xFF);
	}

	for (int32_t lvl = MIN_PA_LEVEL; (lvl <= MAX_PA_LEVEL | !USE_BUTTON); lvl++) {
		// Adding the option to send the packets automatically
		if(USE_BUTTON) {
			btn_press = 0;
			DEBUG_PRINT("=== Press button B1 to start evaluation at %ld dBm\r\n", lvl);
			while (!btn_press) {
				__WFI();
			}
			S2LP_SetPALeveldBm(lvl);
		}

		// We send the power lvl directly through UART :-)
		else {
			HAL_UART_Receive(&hlpuart1, uart_buf, 1, 0xFFFFFFFF);
			S2LP_SetPALeveldBm((int32_t) *uart_buf - 128);
		}
		DEBUG_PRINT("=== Configured PA level to %ld dBm, sending %d packets at this level\r\n", lvl, N_PACKETS);

		for (uint16_t i=0; i < N_PACKETS; i++) {
			HAL_StatusTypeDef err = S2LP_Send(buf, PAYLOAD_LEN);
			if (err) {
				Error_Handler();
			}

			for(uint16_t j=0; j < PACKET_DELAY; j++) {
				HAL_GPIO_WritePin(GPIOB, LD2_Pin, GPIO_PIN_SET);
				HAL_Delay(5000);
				HAL_GPIO_WritePin(GPIOB, LD2_Pin, GPIO_PIN_RESET);
				HAL_Delay(5000);
			}
		}

		printf("Packets sent\n");
	}

	DEBUG_PRINT("=== Finished evaluation, reset the board to run again\r\n");
	while (1);
}
