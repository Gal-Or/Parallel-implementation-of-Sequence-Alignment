/*
 ============================================================================
 Name        : Final_Project.c
 Author      : Gal Or , ID -> 316083690  
 ============================================================================
 */
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include "MPI_OMP_Headers.h"
#include "Mutant.h"


extern MPI_Datatype mpi_mutant_type; /* mutant data type */
extern MPI_Datatype mpi_info_type;   /* file info data type */

int main(int argc, char* argv[]){
	int  my_rank;			 /* rank of process */
	int  p;      			 /* number of processes */
	
	MPI_Status status ;   	/* return status for receive */
	FILE_INFO info; 		/* information from input file */
	int numOfOffsets;  		/* number of optional offsets */
	int startOffset; 		/* the first offset in process's offsets */
	int endOffset; 			/* the last offset in process's offsets - not include in pricess job */
	int numOfIteration;		/* number of offsets for specific process */
	Mutant bestMutant;		/* best mutant of each procces found */
	int success =0;
	
	
	omp_set_num_threads(4);

	/* start up MPI */
	MPI_Init(&argc, &argv);
	
	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
	
	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p); 
		
	/* data type definitions */
	mutant_type_initiate();
	info_type_initiate();
	initSignsTableOMP(); 	// init the signs table - each process needs it to do his job

	/* Divide the tasks between both processes */
	if (my_rank == 0 ) // process p0
	{

		readFromFile(&info); // p0 read data from input file 
		MPI_Send(&info, 1, mpi_info_type, 1, TAG_SEND, MPI_COMM_WORLD); // p0 send file info to p1
		
		numOfOffsets = getMaxOffset(strlen(info.seq1)-1, strlen(info.seq2)-1);
		MPI_Send(&numOfOffsets, 1, MPI_INT, 1, TAG_SEND, MPI_COMM_WORLD); // p0 send number of offsets to p1

		numOfIteration = numOfOffsets/2; // p0 do job for helf of the offsets

	}
	else // process p1
	{
		MPI_Recv(&info, 1 , mpi_info_type, 0, TAG_SEND , MPI_COMM_WORLD, &status); // p0 recive file info from p1
		MPI_Recv(&numOfOffsets, 1 , MPI_INT, 0, TAG_SEND , MPI_COMM_WORLD, &status); // p0 recive file info from p1

		numOfIteration = numOfOffsets -numOfOffsets/2; // p1 do job for the rest(numOfOffsets - helf) of the offsets

	}
	/* each process calculate his start offset and last offset according to his rank */
	startOffset = my_rank * (numOfOffsets/2);
	endOffset = startOffset + numOfIteration;
	success = startProccessJob(my_rank, &info, &bestMutant,startOffset, endOffset); // each process do his job

	if(my_rank ==0) // pocess p0
	{
		Mutant* theBest;
		Mutant theBestFromP1;
		MPI_Recv(&theBestFromP1, 1 , mpi_mutant_type, 1, TAG_SEND , MPI_COMM_WORLD, &status); // p0 recive the best mutant that p1 found
		if(success == 1)
			theBest = compareMutantByScore(&bestMutant, &theBestFromP1, info.goal); // p0 compar between his best mutant and p0 best mutant
		else
			theBest = &theBestFromP1;


		writeResultsToFile(theBest, info.seq2); // p0 writes the best mutant to output file
	}
	else // process p1
	{
		MPI_Send(&bestMutant, 1, mpi_mutant_type, 0, TAG_SEND, MPI_COMM_WORLD); // p1 send best mutant that he found to p0
	}


	/* free all data types */
	MPI_Type_free(&mpi_info_type);
    MPI_Type_free(&mpi_mutant_type);

	/* shut down MPI */
	MPI_Finalize(); 
	return 0;
}
