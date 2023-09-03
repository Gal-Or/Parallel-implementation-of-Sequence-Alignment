#ifndef MUTANT_H_
#define MUTANT_H_
#define _CRT_SECURE_NO_WARNINGS
#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARGS_IN_MUTANT 6

typedef struct _mutant
{
	int offset;
	int indexToChange;
	double changeInScore;
	double newScore;
	double originalScore;
	char newLetter;

}Mutant;


void printMutant(Mutant* mutant, char* seq2);
Mutant* compareMutantByScore(Mutant* m1, Mutant* m2, int goal);


#endif /* MUTANT_H_*/