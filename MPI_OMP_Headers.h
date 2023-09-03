
#ifndef MYPROTO_H_
#define MYPROTO_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Mutant.h"

#define INPUT_FILE_NAME "input.dat"
#define OUTPUT_FILE_NAME "output.dat"

#define CUDA_PRECENT 0.8

#define ARGS_IN_INFO 4
#define W_ARR_SIZE 4
#define SEQ1_MAX_SIZE 10001
#define SEQ2_MAX_SIZE 5001

#define SIGNS_TABLE_SIZE 27 /* 26 letters A-Z plus '-' sign */

#define STAR '*'
#define COLON ':'
#define POINT '.'
#define SPACE ' '

#define NOT_FOUND '%'

#define RED   "\x1B[31m"
#define RESET "\x1B[0m"

#define TAG_SEND 0 /* MPI SEND */

typedef struct file_info
{
	int goal; /* 0 - maximum , 1 - minimum */
    double w[W_ARR_SIZE];
    char seq1[SEQ1_MAX_SIZE];
    char seq2[SEQ2_MAX_SIZE];

}FILE_INFO;


typedef enum {ERROR_ARGS=1, ERROR_FILE, ERROR_READ, ERROR_MALLOC, ERROR_CUDA} errorsTypes;


void mutant_type_initiate();
void info_type_initiate();

void initSignsTableOMP();
int getMaxOffset(int sizeOfSeq1, int sizeOfSeq2);
int startProccessJob(int rank, FILE_INFO* info, Mutant* bestMutant, int firstOffset, int lastOffset);

/* --------------------- File Functions --------------------- */
void readSequence(FILE* f, char* seq);
void readFromFile(FILE_INFO* info);
void writeResultsToFile(Mutant* theBest, char* seq2);
/* ---------------------------------------------------------- */

void printSeq(char* seq, int length, int offset);
void printOffset(int offset);



#endif /* MYPROTO_H_ */
