
#include "Mutant.h"
#include "MPI_OMP_Headers.h"


 void printMutant(Mutant* mutant, char* seq2)
{
	printOffset(mutant->offset);
	int size = strlen(seq2) - 1;

	for (int i = 0; i < size; i++)
	{
		if (i == mutant->indexToChange)
			printf(RED"%c", mutant->newLetter);
		else
			printf(RESET"%c", seq2[i]);
	}
	printf("\n");
}

 Mutant* compareMutantByScore(Mutant* m1, Mutant* m2, int goal)
{
	switch (goal)
		{
		case 0: //maximum

			if (m1->newScore > m2->newScore)
				return m1;
			else
				return m2;

		case 1: //minimum
			if (m1->newScore > m2->newScore)
				return m2;
			else
				return m1;

		default:
			return NULL;
		}
}