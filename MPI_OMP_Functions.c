#define _CRT_SECURE_NO_WARNINGS
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MPI_OMP_Headers.h"
#include "MutualFunctions.h"


char signsTable[SIGNS_TABLE_SIZE][SIGNS_TABLE_SIZE];

MPI_Datatype mpi_mutant_type;
MPI_Datatype mpi_info_type;
 

void mutant_type_initiate()
{
	/*
		This function initialize mutant data type for MPI send recive.
	*/
	int mBlockLength[ARGS_IN_MUTANT]={1,1,1,1,1,1};/*number of blocks for each parameter*/

	MPI_Aint mDisplacements[ARGS_IN_MUTANT]={ offsetof(struct _mutant, offset),
											  offsetof(struct _mutant, indexToChange),
											  offsetof(struct _mutant, changeInScore),
											  offsetof(struct _mutant, newScore),
											  offsetof(struct _mutant, originalScore),
											  offsetof(struct _mutant, newLetter)};

	MPI_Datatype mTypes[ARGS_IN_MUTANT]={ MPI_INT,MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_CHAR };

	/* creating the new data type struct mutant */
	MPI_Type_create_struct(ARGS_IN_MUTANT, mBlockLength, mDisplacements,mTypes, &mpi_mutant_type);
	MPI_Type_commit(&mpi_mutant_type);
}

void info_type_initiate()
{
	/*
		This function initialize info data type for MPI send recive.
	*/
	int iBlockLength[ARGS_IN_INFO]={1, W_ARR_SIZE, SEQ1_MAX_SIZE, SEQ2_MAX_SIZE};//number of blocks for each parameter
	
	MPI_Aint iDisplacements[ARGS_IN_INFO]={ offsetof(struct file_info, goal),
											offsetof(struct file_info, w),
											offsetof(struct file_info, seq1),
											offsetof(struct file_info, seq2)};
	
	MPI_Datatype iTypes[ARGS_IN_INFO]={ MPI_INT, MPI_DOUBLE, MPI_CHAR, MPI_CHAR};
	/* creating the new data type struct info */
	MPI_Type_create_struct(ARGS_IN_INFO, iBlockLength, iDisplacements, iTypes, &mpi_info_type);
	MPI_Type_commit(&mpi_info_type);
}


int startProccessJob(int rank,FILE_INFO* info, Mutant* bestMutant, int firstOffset, int lastOffset)
{
	/*
		This function calculats range of offsets for CUDA job and for OMP job.
		One of OMP threads starts CUDA, and the rest of the threads do job for OMP offsets.
		At the end in bestMutant there is the best mutant of CUDA & OMP offsets.
	*/
	Mutant cudaBestMutant;
	int numOfOffsets =  lastOffset - firstOffset;
	int cudaIterations,OMPIterations; 
	
	if(numOfOffsets == 0) /* if this process has no job -> return */
		return 0;

	if(numOfOffsets == 1 && CUDA_PRECENT > 0) /* if this process has job for only one offset && cuda play-> cuda do the job */
	{
		cudaIterations = 1;	
		OMPIterations = 0;
	}
	else /* if this process has more than one offset the job will be divid by CUDA precent for job */
	{
		cudaIterations = numOfOffsets * CUDA_PRECENT;
		OMPIterations = numOfOffsets - cudaIterations;
	}
	/* calculate range of offsets to CUDA and MPI according to numOfIteration */
	int cudaFirst = firstOffset;
	int cudaLast = cudaFirst + cudaIterations;
	int ompFirst = cudaLast;
	int ompLast = ompFirst + OMPIterations;

	/* init bast mutant for this process */
	initMutant(bestMutant, firstOffset, calcOrginalScore(firstOffset, info->seq1, info->seq2, info->w));

	
	#pragma omp parallel /* starts OMP job */
	{	

		int num_of_threads = omp_get_num_threads();
		int tid = omp_get_thread_num();

		if( tid == 0 && numOfOffsets > 0 && cudaIterations > 0) /* if there if job for cuda thread number 0 in OMP in charge of starts cuda and check cuda results */
		{
			Mutant* h_allMutants= computeOnGPU(cudaIterations , info, cudaFirst ); /* start cuda */
			if (h_allMutants == NULL)
			{
				printf("Failed compute on GPU...");
				MPI_Abort(MPI_COMM_WORLD,ERROR_CUDA);
			}
		
			copyMutant(&cudaBestMutant,&h_allMutants[0]);
	
			for(int i = 1; i<cudaIterations ; i++) /* chack cuda results */
				changeMutantByScore(&cudaBestMutant, &h_allMutants[i], info->goal);
				
			free(h_allMutants); /* free cpu memory */
			
		}
		else if( OMPIterations > 0  ) /* If there if job for OMP each thread (1,2,3) calculate his range from OMP offsets. 
										 If cuda has no job thread number 0 will do this too  */
		{
			if(cudaIterations > 0) /* checks if thread number 0 do cuda job */
			{
				tid--;
				num_of_threads =3;
			}

			Mutant bestThreadMutant;
			/* each thread calculate his range of offsets */
			int offsetsForThread = OMPIterations/num_of_threads;
			int threadFirst = ompFirst + offsetsForThread * tid;
			int threadLast = threadFirst + offsetsForThread;

			if(OMPIterations % num_of_threads != 0 && tid == num_of_threads-1) /* if the total OMP number of thread non divied by number of threads the last thread get the rest of the job */
				threadLast += OMPIterations % num_of_threads;
			
			for(int i =threadFirst ; i<threadLast ; i++ ) /* each thread do his job */
			{
				findBestMutant(info, &bestThreadMutant, i);
				#pragma omp critical
				{
					changeMutantByScore(bestMutant, &bestThreadMutant, info->goal); /* each thread check if his result is better then the current result- one thread at the time */
				}
			}
		}
	}
	if(cudaIterations > 0) /* if cuda had job -> check cuda finel result compare to omp result */
		changeMutantByScore(bestMutant, &cudaBestMutant, info->goal);	
	
	return 1;	
}


void readSequence(FILE* f, char* seq) //create sequence from file
{
	/* 
		This function reads single sequence from file. 
	*/

	if (fscanf(f, "%s", seq) != 1)
	{
		printf("Failed reading sequence from file..\n");
		fclose(f);
		MPI_Abort(MPI_COMM_WORLD,ERROR_READ);
	}
}

void readFromFile(FILE_INFO* info)
{
	/* 
		This function reads info from file. 
	*/

	FILE* f = fopen(INPUT_FILE_NAME, "r");
	char strGoal[8] = "";

	if (!f)
	{
		printf("Failed opening input file..\n");
		fclose(f);
		MPI_Abort(MPI_COMM_WORLD,ERROR_FILE);
	}

	if (fscanf(f, "%lf %lf %lf %lf", &info->w[0], &info->w[1], &info->w[2], &info->w[3]) != W_ARR_SIZE)
	{
		printf("Failed reading weights from file..\n");
		fclose(f);
		MPI_Abort(MPI_COMM_WORLD,ERROR_READ);
	}
	readSequence(f, info->seq1);
	readSequence(f, info->seq2);

	if (fscanf(f, "%s", strGoal) != 1)
	{
		printf("Failed reading goal from file ..\n");
		fclose(f);
		MPI_Abort(MPI_COMM_WORLD,ERROR_READ);
	}
	if (strcmp(strGoal, "maximum") == 0)
		info->goal = 0;
	else
		info->goal = 1;

	fclose(f);
}

void initSignsTableOMP()
{
	/*
		This function init CPU signs table according to the rules.
		This function fill the table with OMP/
	*/
	#pragma omp parallel for 
	for(int x = 0 ; x<SIGNS_TABLE_SIZE ; x++)
	{
		for(int y = 0; y<SIGNS_TABLE_SIZE ; y++ )
		{
			if(x == SIGNS_TABLE_SIZE - 1)
				if(y == SIGNS_TABLE_SIZE - 1)
					signsTable[x][y] = STAR;
				else
					signsTable[x][y] = SPACE;
			else 
				if(y == SIGNS_TABLE_SIZE - 1)
					signsTable[x][y] = SPACE;
				else
					signsTable[x][y] = copmare(x + ASCII, y + ASCII);
		}
	}
}

int getMaxOffset(int sizeOfSeq1, int sizeOfSeq2)
{
	/* 
		This function calculate the maximum optional offset for seq1 and seq2 
	*/
	return sizeOfSeq1 - sizeOfSeq2 +1;
}

void writeResultsToFile(Mutant* theBest, char* seq2)
{
	/*
		This function writes the best mutant, the offset and the score to output file.
	*/

	FILE* f = fopen(OUTPUT_FILE_NAME, "w");
	if (!f)
	{
		printf("Failed opening output file..\n");
		fclose(f);
		MPI_Abort(MPI_COMM_WORLD,ERROR_FILE);
	}

	seq2[theBest->indexToChange]= theBest->newLetter;
	if(fprintf(f, "%s\n", seq2) < 0)
		printf("Failed write mutant to output file..\n");

	if(fprintf(f, "Offset : %d , Alignment Score : %g", theBest->offset, theBest->newScore) < 0)
		printf("Failed write offset and score to output file..\n");
	


}


void printSeq(char* seq, int length, int offset)
{
	printOffset(offset);

	for (int i = 0; i < length; i++)
		printf("%c", seq[i]);

	printf("\n");
}

void printOffset(int offset)
{
	for (int i = 0; i < offset; i++)
		printf(" ");
}

