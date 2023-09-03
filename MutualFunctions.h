#ifndef GLOBAL_DEVICE_H_
#define GLOBAL_DEVICE_H_
#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/device_launch_parameters.h>
#include "Mutant.h"
#include "MPI_OMP_Headers.h"

#define ASCII 65

#define CON_SIZE 9
#define CON_LEN 5
#define SEMI_SIZE 11
#define SEMI_LEN 7

#define HYPHEN '-'


extern char signsTable[SIGNS_TABLE_SIZE][SIGNS_TABLE_SIZE]; /* in CPU memory */
__device__ char cudaSignsTable[SIGNS_TABLE_SIZE][SIGNS_TABLE_SIZE]; /* in GPU memory */

/* ========================================================================================= 
   ----------------------------------- HOST FUNCTIONS --------------------------------------
   ---> Executed on the host
   ---> Only called frome the host
   ========================================================================================= */

__host__ Mutant* computeOnGPU(int numOfOffsets, FILE_INFO* info, int firstOffset);
__host__ FILE_INFO* create_d_info(FILE_INFO* info);
__host__ Mutant* create_d_allMutants(int size);
__host__ Mutant* copy_d_allMutants_to_cpu(Mutant* d_allMutants, int size);
__host__ void freeAllGpuAllocations(FILE_INFO* d_info, Mutant* d_allMutants);
__host__ void checkCudaErr(cudaError_t err , const char* msg);

/* ========================================================================================= 
   ---------------------------------- GLOBAL FUNCTIONS -------------------------------------
   ---> Executed on the device.
   ---> Only called frome the host.
   ---> Each thread do this function.
   ========================================================================================= */

__global__  void initCudaSignsTable(FILE_INFO* info);
__global__  void doJobForOffset(FILE_INFO* info, Mutant* d_allMutants, int firstOffset, int numOfOffsets);

/* ========================================================================================= 
   ------------------------------- DEVICE__HOST FUNCTIONS ----------------------------------
   ---> Executed on the device or on the host.
   ---> called from the device or from the host.
   ========================================================================================= */

__host__ __device__ void findBestMutant(FILE_INFO* info, Mutant* bestMutant, int offset);
__host__ __device__ char getSignFromTable(int i, int j);
__device__ __host__ int myStrlen(char* str);
__device__ __host__ char* myStrchr(char* str, char c);
__device__ __host__ int getLetterIndex(char letter);
__device__ __host__ char copmare(char c1, char c2);

__device__ __host__ int checkIfConservative(char c1, char c2);
__device__ __host__ int checkIfSemiConservative(char c1, char c2);

__device__ __host__ double calcOrginalScore(int offset, char* seq1, char* seq2, double* w);
__device__ __host__ double getWeightOfSign(char sign, FILE_INFO* info );

__device__ __host__ void minMutantForSpecificOffset(FILE_INFO* info, Mutant* bestMutant);
__device__ __host__ void maxMutantForSpecificOffset(FILE_INFO* info, Mutant* bestMutant);
__device__ __host__ char findSubstitution(FILE_INFO* info, char sign1, double changeToSign1, char sign2, double changeToSign2, int SpecificOffset, int i, Mutant* bestMutant);
__device__ __host__ char getLetterFotSpesificSign(char spesificSign ,char charToMatch, char charToChange);

/* ---------------------------------- MUTANT FUNCTIONS ------------------------------------- */

__device__ __host__ void initMutant(Mutant* mutant, int offset, double originalScore);
__device__ __host__ void copyMutant(Mutant* bestMutantOfAllOffsets, Mutant* bestMutantSpecificOffset);
__device__ __host__ void changeMutantByScore(Mutant* bestMutantOfAllOffsets, Mutant* bestMutantSpecificOffset, int goal);
__device__ __host__ int checkIfSwitch(Mutant* bestMutant, double tempChange, char tempLetter, int goal);
__device__ __host__ void changeMutantByScoreChange(Mutant* bestMutant, double tempChange, char tempLetter, int i, int goal);



#endif /* GLOBAL_DEVICE_H_ */