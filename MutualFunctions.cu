
#include <helper_cuda.h>
#include "MutualFunctions.h"



__host__ Mutant* computeOnGPU(int numOfOffsets, FILE_INFO* info, int firstOffset) {
	/*
		This function execute CUDA part - if execute threads on 1/4 from offsets.
	*/

    cudaError_t err = cudaSuccess; /* Error code to check return values for CUDA calls */

    /* execute CUDA for init signs table */
    dim3 threadsPerBlock(SIGNS_TABLE_SIZE, SIGNS_TABLE_SIZE);
    dim3 blocksPerGrid(1, 1);
    initCudaSignsTable<<<threadsPerBlock, blocksPerGrid>>>(info);
    cudaDeviceSynchronize();
    
	/* allocate d_info in GPU memory and copy values from CPU */
	FILE_INFO* d_info= create_d_info(info);
	/* allocate d_allMutants in GPU memory */
	size_t size = numOfOffsets * sizeof(Mutant);
	Mutant* d_allMutants= create_d_allMutants(size);

	/* execute CUDA for do job on cuda offsets */
    int threadsPerBlock2 = 256;
    int blocksPerGrid2 =(numOfOffsets + threadsPerBlock2 - 1) / threadsPerBlock2;
    doJobForOffset<<<blocksPerGrid2, threadsPerBlock2>>>(d_info, d_allMutants, firstOffset, numOfOffsets);
	cudaDeviceSynchronize();
	
	err = cudaGetLastError();
	checkCudaErr(err ,"Failed to launch kernel function 'doJobForOffset'");
    
	/* allocate h_allMutants in CPU memory and copy values from GPU memory */
	Mutant* h_allMutants = copy_d_allMutants_to_cpu(d_allMutants, size);
	freeAllGpuAllocations(d_info, d_allMutants );

    cudaDeviceSynchronize();
    return h_allMutants;
}

__host__ FILE_INFO* create_d_info(FILE_INFO* info)
{
	/*
		This function allocate memory on GPU for file info struct and copy the data
		from CPU to GPU.
	*/
	cudaError_t err =cudaSuccess;
	FILE_INFO* d_info; 
	err = cudaMalloc((void**)&d_info, sizeof(FILE_INFO));
	checkCudaErr(err, "Failed to allocate info device memory");

    double* w;
    char* seq1;
    char* seq2;
	err =cudaMalloc((void**)&w, W_ARR_SIZE);
	checkCudaErr(err , "Failed to allocate w device memory");
	err =cudaMalloc((void**)&seq1, SEQ1_MAX_SIZE);
	checkCudaErr(err , "Failed to allocate seq1 device memory");
	err =cudaMalloc((void**)&seq2, SEQ2_MAX_SIZE);
	checkCudaErr(err , "Failed to allocate seq2 device memory");

	err =cudaMemcpy(w, info->w, W_ARR_SIZE, cudaMemcpyHostToDevice);
	checkCudaErr(err , "Failed copy w memory from CPU to GPU");
	err =cudaMemcpy(seq1, info->seq1, SEQ1_MAX_SIZE, cudaMemcpyHostToDevice);
	checkCudaErr(err , "Failed copy seq1 memory from CPU to GPU");
	err =cudaMemcpy(seq2, info->seq2, SEQ2_MAX_SIZE, cudaMemcpyHostToDevice);	
	checkCudaErr(err , "Failed copy seq2 memory from CPU to GPU");
	
	err = cudaMemcpy(&(d_info->w), &w, sizeof(w), cudaMemcpyHostToDevice);
	checkCudaErr(err , "Failed copy w address");
	err =cudaMemcpy(&(d_info->seq1),&seq1 , sizeof(seq1), cudaMemcpyHostToDevice);
	checkCudaErr(err , "Failed copy seq1 address");
	err =cudaMemcpy(&(d_info->seq2),  &seq2, sizeof(seq2), cudaMemcpyHostToDevice);
	checkCudaErr(err , "Failed copy seq2 address");
	err =cudaMemcpy(d_info, info, sizeof(FILE_INFO), cudaMemcpyHostToDevice);

	return d_info;
}

__host__ Mutant* create_d_allMutants(int size)
{
	/*
		This function allocate memory on GPU for mutants struct array.
	*/
	
	cudaError_t err =cudaSuccess;
	Mutant* d_allMutants;
    err = cudaMalloc((void**)&d_allMutants, size);
	checkCudaErr(err , "Failed to allocate device memory");
	return d_allMutants;
}

__host__ Mutant* copy_d_allMutants_to_cpu(Mutant* d_allMutants, int size)
{
	/*
		This function allocate memory on CPU for file d_allMutants array and copy the data
		from GPU to CPU.
	*/
	cudaError_t err =cudaSuccess;
	Mutant* h_allMutants = (Mutant*)malloc(size);
	err = cudaMemcpy(h_allMutants, d_allMutants, size, cudaMemcpyDeviceToHost);
	checkCudaErr(err , "Failed to copy result array from device to host");

	return h_allMutants;
}

__host__ void freeAllGpuAllocations(FILE_INFO* d_info, Mutant* d_allMutants)
{
	/*
		This function free all allocated memory on GPU
	*/
	cudaError_t err =cudaSuccess;

	err = cudaFree(d_info);
	if (err != cudaSuccess) 
    {
        fprintf(stderr,"Failed to free device d_info - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err =cudaFree(d_allMutants);
	if (err != cudaSuccess) 
    {
        fprintf(stderr,"Failed to free device d_allMutants - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
}

__host__ void checkCudaErr(cudaError_t err , const char* msg)
{
	if (err != cudaSuccess) 
    {
        fprintf(stderr, "%s - %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}




__global__  void initCudaSignsTable(FILE_INFO* info) 
{
	/*
		This function init gpu signs table according to the rules.
		Each thread fill his cell according his tid in the table.
	*/
   
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    
	if(x == SIGNS_TABLE_SIZE - 1)
		if(y == SIGNS_TABLE_SIZE - 1)
			cudaSignsTable[x][y] = STAR;
		else
			cudaSignsTable[x][y] = SPACE;
	else 
		if(y == SIGNS_TABLE_SIZE - 1)
			cudaSignsTable[x][y] = SPACE;
		else
			cudaSignsTable[x][y] = copmare(x + ASCII, y + ASCII);

  
}

__global__  void doJobForOffset(FILE_INFO* info, Mutant* d_allMutants, int firstOffset, int numOfOffsets) 
{
	/*
		This function each thread finds best mutant for specific offset according to his tid. 
		each thread save his best mutant in d_allMutants  array in the index that metch his tid
	*/
	
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	// find best mutant for specific offset according to thread ID 
    if (i < numOfOffsets)
		findBestMutant(info, d_allMutants+i, i+firstOffset);	

}


__host__ __device__ void findBestMutant(FILE_INFO* info, Mutant* bestMutant, int offset)
{
	initMutant(bestMutant, offset, calcOrginalScore(offset, info->seq1, info->seq2, info->w));

	if (info->goal == 0) //maximum
			maxMutantForSpecificOffset(info, bestMutant);
	else //minimum
			minMutantForSpecificOffset(info, bestMutant);

}

__host__ __device__ char getSignFromTable(int i, int j)
{

	/*
		This function get sign from signs table according to index i and j.
		This finction get the sign each time from the From the appropriate table
		(OMP table-CPU or Cuda table-GPU) according to special '__CUDA_ARCH__' cuda define.
	*/

    #if(defined(__CUDA_ARCH__ )&& (__CUDA_ARCH__  > 0))
        return cudaSignsTable[i][j];
    #else
        return signsTable[i][j];
    #endif
}

__device__ __host__ int getLetterIndex(char letter)
{ 
	/*
		This function convert from letter to the relevant index in signs table.
	*/
	if(letter == HYPHEN)
		return SIGNS_TABLE_SIZE-1;
	else
		return letter - ASCII;
} 

__device__ __host__ int myStrlen(char* str)
{
	/* 
		This function gets the length of str.
	*/
    int count=0;
    while(str[count] != '\0')
        count++;
    
    return count;
}

__device__ __host__ char* myStrchr(char* str, char c)
{
	/*
		This function chacks if the char c is in str and returns his first address 
		that he founds it. If char c in not in str the finction returns NULL.
	*/
    int size = myStrlen(str);

    for(int i= 0; i<size ; i++)
    {
        if(str[i] == c)
            return &str[i];
    }
    return NULL;
}

__device__ __host__ char copmare(char c1, char c2) 
{
	/*
		This function compare between to chars and returns the relevant sign-
		check the difference between 2 chars.
	*/
	if (c1 == c2)
		return STAR;
	else if (checkIfConservative(c1, c2) == 1)
		return COLON;
	else if (checkIfSemiConservative(c1, c2) == 1)
		return POINT;
	else
		return SPACE;
}

__device__ __host__ int checkIfConservative(char c1, char c2)
{
	/*
		This function chacks if the two chars c1, c1 are in concervative group.
	*/
    char conservative[CON_SIZE][CON_LEN] = { "STA", "NEQK", "NDEQ","NHQK", "QHRK", "MILV", "MILF","HY", "FYW" };
	for (int i = 0; i < CON_SIZE; i++)
		if (myStrchr(conservative[i], c1) && myStrchr(conservative[i], c2))
			return 1;  // true - for :

	return 0; //false
}

__device__ __host__ int checkIfSemiConservative(char c1, char c2)
{
	/*
		This function chacks if the two chars c1, c1 are in semi-concervative group.
	*/
    char semiConservative[SEMI_SIZE][SEMI_LEN] = { "CSA", "ATV", "SAG", "STNK", "STPA","SGND", "SNDEQK", "NDEQHK", "NEQHRK", "FVLIM", "HFY" };
	for (int i = 0; i < SEMI_SIZE; i++)
		if (myStrchr(semiConservative[i], c1) && myStrchr(semiConservative[i], c2))
			return 1; // true - for .

	return 0; //false
}


__device__ __host__ double calcOrginalScore(int offset, char* seq1, char* seq2, double* w)
{
	/* 
		This function calculate the original score for seq1 and seq2.
	*/
	
	int numOfStars = 0;
	int numOfColons = 0;
	int numOfPoints = 0;
	int numOfSpaces = 0;
	int sizeOfSeq2 = myStrlen(seq2);
	char sign;

	for (int i = 0; i < sizeOfSeq2; i++)
	{
	
		sign = copmare(seq1[offset + i], seq2[i]);
		switch (sign)
		{
		case STAR:
			numOfStars++;
			break;

		case COLON:
			numOfColons++;
			break;

		case POINT:
			numOfPoints++;
			break;

		case SPACE:
			numOfSpaces++;
			break;

		default:
			break;
		}
	}
	
	return (w[0] * numOfStars) - (w[1] * numOfColons) - (w[2] * numOfPoints) - (w[3] * numOfSpaces);
}

__device__ __host__ double getWeightOfSign(char sign, FILE_INFO* info )
{
	switch (sign)
		{ 
		case STAR:
			return info->w[0];	
		case COLON:
			return -(info->w[1]);
		case POINT:
			return -(info->w[2]);	
		case SPACE:
			return -(info->w[3]);

		default:
			return 0;	
		}
}


__device__ __host__ void minMutantForSpecificOffset(FILE_INFO* info, Mutant* bestMutant)
{
	/* 
		This function finds the minimum mutant for specific offset and save it in bestMutant. 
	*/

	int sizeOfSeq2 = myStrlen(info->seq2);
	int SpecificOffset = bestMutant->offset;
	double changeToSign1 = 0, changeToSign2 = 0, changeToStar=0;
	char curSign, sign1, sign2 ;
	char tempLetter = NOT_FOUND;

	for (int i = 0; i < sizeOfSeq2; i++)
	{
		curSign = getSignFromTable(getLetterIndex(info->seq1[SpecificOffset + i]), getLetterIndex(info->seq2[i]));
		switch (curSign)
		{ // in all cases - calculate the change in score for all options to legal substitution
		case STAR:
			changeToSign1 = -info->w[0] - info->w[2]; //to point
			changeToSign2 = -info->w[0] - info->w[3]; //to space
			sign1 = POINT;
			sign2 = SPACE;
			break;
		case SPACE:
			changeToSign1 = info->w[3] - info->w[2]; //to point
			changeToSign2 = info->w[3] - info->w[1]; //to colon
			changeToStar = info->w[3] + info->w[0]; //to star
			sign1 = POINT;
			sign2 = COLON;
			break;
		case POINT:
			changeToSign1 = info->w[2] - info->w[1]; //to colon
			changeToSign2 = info->w[2] - info->w[3]; //to space
			changeToStar = info->w[2] + info->w[0]; //to star
			sign1 = COLON;
			sign2 = SPACE;
			break;
		case COLON:
			changeToSign1 = info->w[1] - info->w[2]; //to point
			changeToSign2 = info->w[1] - info->w[3]; //to space
			sign1 = POINT;
			sign2 = SPACE;
			break;
		}

		if (changeToSign1 < changeToSign2)
			tempLetter=findSubstitution(info, sign1, changeToSign1, sign2, changeToSign2, SpecificOffset, i, bestMutant);
		else
			tempLetter=findSubstitution(info, sign2, changeToSign2, sign1, changeToSign1, SpecificOffset, i, bestMutant);

		if ((curSign == POINT|| curSign == SPACE) && tempLetter == NOT_FOUND) 
		{ // if the substitution failed in case of POINT or SPACE it is always legal to switch to star
			tempLetter = getLetterFotSpesificSign(STAR, info->seq1[SpecificOffset + i], info->seq2[i]);
			changeMutantByScoreChange(bestMutant, changeToStar, tempLetter, i, info->goal);
		}
	}//for
}

__device__ __host__ void maxMutantForSpecificOffset(FILE_INFO* info, Mutant* bestMutant)
{
	/* 
		This function finds the maximum mutant for specific offset and save it in bestMutant 
	*/

	int sizeOfSeq2 = myStrlen(info->seq2);
	int SpecificOffset = bestMutant->offset;
	double tempChange; // new change in score
	char curSign, tempLetter; // new letter
	double changeToSpace = 0, changeToPoint = 0;

	for(int i = 0 ; i< sizeOfSeq2; i++ )
	{
		curSign = getSignFromTable(getLetterIndex(info->seq1[SpecificOffset + i]), getLetterIndex(info->seq2[i]) );
		tempChange = 0;
		tempLetter = NOT_FOUND;

		switch (curSign)
		{ // in all cases - calculate the change in score for all options to legal substitution
		case STAR:
			changeToPoint = -info->w[0] - info->w[2]; //to point
			changeToSpace = -info->w[0] - info->w[3]; //to space
			break;
		case SPACE:
			tempChange = info->w[0] + info->w[3]; //to star
			tempLetter = info->seq1[SpecificOffset + i];
			break;
		case POINT:
			tempChange = info->w[0] + info->w[2]; //to star
			tempLetter = info->seq1[SpecificOffset + i];
			break;
		case COLON:
			changeToSpace = info->w[1] - info->w[3]; //to Space
			changeToPoint = info->w[1] - info->w[2]; //to point
			break;
		}

		if(curSign == STAR || curSign == COLON)
			if (changeToSpace > changeToPoint)
				findSubstitution(info, SPACE, changeToSpace, POINT, changeToPoint, SpecificOffset, i, bestMutant);
			else
				findSubstitution(info, POINT, changeToPoint, SPACE, changeToSpace, SpecificOffset, i, bestMutant);

		else // curSign == SPACE || curSign == POINT
			changeMutantByScoreChange(bestMutant, tempChange, tempLetter, i, info->goal);
	}//for
	
}


__device__ __host__ char findSubstitution(FILE_INFO* info, char sign1, double changeToSign1, char sign2, double changeToSign2, int SpecificOffset, int i, Mutant* bestMutant)
{
	/* 
		This function try to find substitution to optional legal sign1, if it failed try to 
		find substitution to optional legal sign2. 
	*/
	double tempChange = 0;
	char tempLetter = NOT_FOUND;

	tempChange = changeToSign1;
	tempLetter = getLetterFotSpesificSign(sign1, info->seq1[SpecificOffset + i], info->seq2[i]);

	if (tempLetter == NOT_FOUND) 
	{/* checks if the replacement to sign1 failed && there is option for change 
		to sign2 -> try change to sign2*/
		tempChange = changeToSign2;
		tempLetter = getLetterFotSpesificSign(sign2, info->seq1[SpecificOffset + i], info->seq2[i]);
	}

	if(tempLetter != NOT_FOUND)
		changeMutantByScoreChange(bestMutant, tempChange, tempLetter, i, info->goal );
	return tempLetter;
}

__device__ __host__ char getLetterFotSpesificSign(char spesificSign ,char charToMatch, char charToChange)
{
	/* 
		This function search new letter that with the charToMatch according to roles give spesificSign.
	    The new letter needs to be switchable with charToChange according to substitution role. 
	*/

	int charIndex = charToMatch - ASCII;
	for (int i = 0; i < SIGNS_TABLE_SIZE; i++)
	{
		if (getSignFromTable(charIndex,i) == spesificSign) //looking for letter that will give the specific sign i want
		{
			if(!checkIfConservative(i + ASCII, charToChange)) //chack if it is legal to change with this letter according to substitution role
				return i + ASCII;
		}
	}
	return NOT_FOUND;
}

/*
 ============================================================================
 ----------------------------- Mutant Functions -----------------------------
 ============================================================================
*/

__device__ __host__ void initMutant(Mutant* mutant, int offset, double originalScore)
{
	/*
		This function init mutant values.
	*/
	mutant->changeInScore = 0;
	mutant->indexToChange = -1;
	mutant->newLetter = NOT_FOUND;
	mutant->newScore = -1;
	mutant->offset = offset;
	mutant->originalScore = originalScore;
}

__device__ __host__ void copyMutant(Mutant* bestMutantOfAllOffsets, Mutant* bestMutantSpecificOffset)
{
	/*
		This function copy values of bestMutantSpecificOffset to bestMutantOfAllOffsets
	*/
	bestMutantOfAllOffsets->changeInScore = bestMutantSpecificOffset->changeInScore;
	bestMutantOfAllOffsets->indexToChange = bestMutantSpecificOffset->indexToChange;
	bestMutantOfAllOffsets->newLetter = bestMutantSpecificOffset->newLetter;
	bestMutantOfAllOffsets->newScore = bestMutantSpecificOffset->newScore;
	bestMutantOfAllOffsets->offset = bestMutantSpecificOffset->offset;
	bestMutantOfAllOffsets->originalScore = bestMutantSpecificOffset->originalScore;
}

__device__ __host__ void changeMutantByScore(Mutant* bestMutantOfAllOffsets, Mutant* bestMutantSpecificOffset, int goal)
{
	/*
		This function compare mutants score according to the goal (min\max).
		If need do copy mutant.
	*/
	if(bestMutantSpecificOffset->newLetter ==  NOT_FOUND)
		return;
		
	switch (goal)
	{
	case 0: //maximum
		if (bestMutantSpecificOffset->newScore > bestMutantOfAllOffsets->newScore || bestMutantOfAllOffsets->newLetter == NOT_FOUND )
			copyMutant(bestMutantOfAllOffsets, bestMutantSpecificOffset);
		break;
	case 1: //minimum
		if (bestMutantSpecificOffset->newScore < bestMutantOfAllOffsets->newScore || bestMutantOfAllOffsets->newLetter == NOT_FOUND )
			copyMutant(bestMutantOfAllOffsets, bestMutantSpecificOffset);
		break;
	}
			    
}

__device__ __host__ int checkIfSwitch(Mutant* bestMutant, double tempChange, char tempLetter, int goal)
{
	/*
		This function checks if thr new mutant that founds is better than
		current best mutant.
		in this function tempLetter != NOT_FOUND.
	*/
	if (bestMutant->newLetter == NOT_FOUND)
			return 1; //yes
		else
			if ((tempChange > bestMutant->changeInScore && goal == 0) || (tempChange < bestMutant->changeInScore && goal == 1))
				return 1; //yes

		return 0; //no
}

__device__ __host__ void changeMutantByScoreChange(Mutant* bestMutant, double tempChange, char tempLetter, int i, int goal)
{
	/*
		This function checks if the new replacment is better the current best mutant.
		If yes, copy the values to best mutant.
	*/ 
	if (checkIfSwitch(bestMutant, tempChange, tempLetter, goal)) 
	{
		bestMutant->changeInScore = tempChange;
		bestMutant->newLetter = tempLetter;
		bestMutant->indexToChange = i;
		bestMutant->newScore = bestMutant->originalScore + tempChange;
	}
}


