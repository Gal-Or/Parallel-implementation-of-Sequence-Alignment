build:
	mpicxx -fopenmp -c 	Final_Project.c -o Final_Project.o
	mpicxx -fopenmp -c MPI_OMP_Functions.c -o MPI_OMP_Functions.o
	mpicxx -fopenmp -c Mutant.c -o Mutant.o
	nvcc -I./inc -c MutualFunctions.cu -o MutualFunctions.o
	mpicxx -fopenmp -o mpiCudaOMP  Final_Project.o MPI_OMP_Functions.o Mutant.o MutualFunctions.o  /usr/local/cuda-9.1/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./mpiCudaOMP

run:
	mpiexec -np 2 ./mpiCudaOMP 

runOn2:
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./mpiCudaOMP
