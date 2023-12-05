/*
 * packet.c
 */

#include "aes_ref.h"
#include "config.h"
#include "packet.h"
#include "main.h"
#include "utils.h"

// uniformly randomly chosen key
const uint8_t AES_Key[16]  = {
                            0x34,0xFC,0xE5,0xA5,
							0x86,0x5A,0x98,0x25,
							0xB8,0x30,0x2F,0x0B,
							0x6E,0x06,0x90,0x2D};

void tag_cbc_mac(uint8_t *tag, const uint8_t *msg, size_t msg_len) {
	// Allocate a buffer of the key size to store the input and result of AES
	// uint32_t[4] is 4*(32/8)= 16 bytes long
	uint32_t statew[4] = {0};
	// state is a pointer to the start of the buffer
	uint8_t *state = (uint8_t*) statew;
    size_t i;

    uint8_t block[16] = {0};
    // TO DO : Complete the CBC-MAC_AES
    for (i = 0; i < ((int) (msg_len/16)); i++){
		for(int j = 0; j < 16; j++) {
			block[j] = state[j] ^ msg[i*16+j];
		}

    	AES128_encrypt(block, AES_Key);

		for(int j = 0; j < 16; j++) {
			state[j] = block[j];
		}
    }

    int rest = msg_len % 16;

    if(rest) {
    	for(i = 0; i < rest; i++) {
    		block[i] = state[i] ^ msg[(int) (msg_len/16) * 16 + i];
    	}

		for(i = rest; i < 16; i++) {
			block[i] = state[i];
		}

    	AES128_encrypt(block, AES_Key);

		for(int j = 0; j < 16; j++) {
			state[j] = block[j];
		}
    }

    // Copy the result of CBC-MAC-AES to the tag.
    for (int j=0; j<16; j++) {
        tag[j] = state[j];
    }
}

// Assumes payload is already in place in the packet
int make_packet(uint8_t *packet, size_t payload_len, uint8_t sender_id, uint32_t serial) {
    size_t packet_len = payload_len + PACKET_HEADER_LENGTH + PACKET_TAG_LENGTH;
//    // Initially, the whole packet header is set to 0s
//    memset(packet, 0, PACKET_HEADER_LENGTH);
//    // So is the tag
//	memset(packet + payload_len + PACKET_HEADER_LENGTH, 0, PACKET_TAG_LENGTH);

    packet[0] = 0x00;
    packet[1] = sender_id;
    packet[2] = (uint8_t) ((payload_len >> 8) & 0xFF);
    packet[3] = (uint8_t) (payload_len & 0xFF);
    packet[4] = (uint8_t) ((serial >> 24) & 0xFF);
    packet[5] = (uint8_t) ((serial >> 16) & 0xFF);
    packet[6] = (uint8_t) ((serial >> 8) & 0xFF);
    packet[7] = (uint8_t) (serial & 0xFF);

	// TO DO :  replace the two previous command by properly
	//			setting the packet header with the following structure :
	/***************************************************************************
	 *    Field       	Length (bytes)      Encoding        Description
	 ***************************************************************************
	 *  r 					1 								Reserved, set to 0.
	 * 	emitter_id 			1 					BE 			Unique id of the sensor node.
	 *	payload_length 		2 					BE 			Length of app_data (in bytes).
	 *	packet_serial 		4 					BE 			Unique and incrementing id of the packet.
	 *	app_data 			any 							The feature vectors.
	 *	tag 				16 								Message authentication code (MAC).
	 *
	 *	Note : BE refers to Big endian
	 *		 	Use the structure 	packet[x] = y; 	to set a byte of the packet buffer
	 *		 	To perform bit masking of the specific bytes you want to set, you can use
	 *		 		- bitshift operator (>>),
	 *		 		- and operator (&) with hex value, e.g.to perform 0xFF
	 *		 	This will be helpful when setting fields that are on multiple bytes.
	*/

	// For the tag field, you have to calculate the tag. The function call below is correct but
	// tag_cbc_mac function, calculating the tag, is not implemented.
    tag_cbc_mac(packet + payload_len + PACKET_HEADER_LENGTH, packet, payload_len + PACKET_HEADER_LENGTH);

    return packet_len;
}
