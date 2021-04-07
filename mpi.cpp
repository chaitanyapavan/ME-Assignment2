#include <stdio.h>
#include <stdlib.h>
#include<time.h>
#include <string>
#include <iostream>
#include <random>
#include <cmath>
#include <bits/stdc++.h>
#include <mpi.h>

#define N 1000
#define NPES 1
#define ROOT 0
#define NR (N/NPES)

using namespace std;

int main(int argc, char *argv[]){
    int npes; // Number of PEs
    int mype; // My PE number
    int stat; // Error status
    MPI_Status status;

    int i,j,k;
    double start, end;
    
    uniform_real_distribution<double> distribution(0.0,1.0);
    default_random_engine generator;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes );
    MPI_Comm_rank(MPI_COMM_WORLD, &mype );
    start=MPI_Wtime(); 

    if (npes != NPES){
        if(mype == 0){
            fprintf(stdout, "The example is only for %d PEs\n", NPES);
        }
        MPI_Finalize();
        exit(1);
    }

    // Intializing A,B,C
    auto A = new float[N][N];
    auto B = new float[N][N];
    auto C = new float[N][N];
    for(i=0; i<N; i++){
        for(j=0; j<N;j++){
            A[i][j] =  distribution(generator);
            B[i][j] =  distribution(generator);
            C[i][j] = 0;
        }
    }

    MPI_Bcast(A, N*N, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(B, N*N, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Computing C=AB
    for(i=mype*NR; i<(mype+1)*NR; i++){
        for(j=0; j<N; j++){
            for(k=0; k<N; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(mype != ROOT){
        MPI_Send(&C[mype*NR][0], NR*N, MPI_FLOAT, ROOT, 10, MPI_COMM_WORLD);
    }

    if(mype == ROOT){
        for(i=1; i<NPES; ++i){
            MPI_Recv(&C[i*NR][0], NR*N, MPI_FLOAT, i, 10, MPI_COMM_WORLD, &status);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(C, N*N, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);


    // Transforming C into Upper trinagular matrix using doolittle algorithm
    auto lower = new float[N][N];
    auto upper = new float[N][N];
    
    for(i=0; i<N; ++i){
        for(j=0; j<N; ++j){
            lower[i][j] = 0;
            upper[i][j] = 0;
        }
    }

    for(i=0; i<N; ++i){
        // Upper triangular
        int temp = (N-i) / NPES;
        int start_index = i + (mype*temp);
        int end_index = i + ((mype+1)*temp);
        if(mype == (NPES-1)){
            end_index = N;
        }
        for(k=start_index; k<end_index; k++){
            float sum =0;
            for(j=0; j<i; j++){
                sum += (lower[i][j] * upper[j][k]);
            }
            upper[i][k] = C[i][k] - sum;
        }
        // Lower triangular
        for(k=start_index; k<end_index; ++k){
            if(i==k){
                lower[i][i] = 1;
            }
            else{
                float sum1 = 0;
                for (j = 0; j < i; j++){
                    sum1 += (lower[k][j] * upper[j][i]);
                }
                lower[k][i]= (C[k][i] - sum1) / upper[i][i];
            }
        }

        //cout << "hello" << endl;
        MPI_Barrier(MPI_COMM_WORLD);

        if(mype != ROOT){
            MPI_Send(&lower[start_index][i], end_index-start_index, MPI_FLOAT, ROOT, 5, MPI_COMM_WORLD);
            MPI_Send(&upper[i][start_index], end_index-start_index, MPI_FLOAT, ROOT, 5, MPI_COMM_WORLD);
        }

        if(mype == ROOT){
            for(j=1; j<NPES; ++j){
                int temp1 = (N-i) / NPES;
                int start_index1 = i + (j*temp1);
                int end_index1 = i + ((j+1)*temp1);
                if(j == (NPES-1)){
                    end_index1 = N;
                }
                MPI_Recv(&lower[start_index1][i], end_index1-start_index1, MPI_FLOAT, j, 5, MPI_COMM_WORLD, &status);
                MPI_Recv(&upper[i][start_index1], end_index1-start_index1, MPI_FLOAT, j, 5, MPI_COMM_WORLD, &status);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&lower[0][i], N, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
        MPI_Bcast(&upper[i][0], N, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);        
    }

    

    // checking
    // if(mype == ROOT){
    //     for(i=0; i<N; i++){
    //         for(j=0; j<N; j++){
    //             float temp = 0;
    //             for(k=0; k<N; k++){
    //                 temp += lower[i][k] * upper[k][j];
    //             }
    //             if(abs(temp - C[i][j]) > 0.05){
    //                 printf("%d %d %f %f\n", i, j, C[i][j], temp);
    //             }
    //         }
    //     }
    // }

    end=MPI_Wtime();
    printf("\n Time taken by process %d is %f \n",mype,(end - start));
    MPI_Finalize();
}