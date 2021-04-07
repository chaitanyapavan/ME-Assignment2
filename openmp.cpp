#include <stdio.h>
#include <stdlib.h>
#include<time.h>
#include <omp.h>
#include <string>
#include <iostream>
#include <random>
#include <cmath>
#include <bits/stdc++.h>

#define N 1000

using namespace std;

int main(int argc, char *argv[]){
    int i,j,k;
    
    uniform_real_distribution<double> distribution(0.0,1.0);
    default_random_engine generator;

    // Intializing A,B,C
    auto A = new float[N][N];
    auto B = new float[N][N];
    auto C = new float[N][N];
    // #pragma omp parallel for private(i, j) shared(A,B,C,distribution,generator)
    for(i=0; i<N; i++){
        for(j=0; j<N;j++){
            A[i][j] = distribution(generator) ;
            B[i][j] = distribution(generator) ;
            C[i][j] = 0;
        }
    }

    // Computing C=AB
    #pragma omp parallel for private(i, j, k) shared(A,B,C)
    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            for(k=0; k<N; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

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
        float sum =0;
        #pragma omp parallel for private(k,j) shared(lower, upper, C) firstprivate(sum)
        for(k=i; k<N; k++){
            for(j=0; j<i; j++){
                sum += (lower[i][j] * upper[j][k]);
            }
            upper[i][k] = C[i][k] - sum;
            sum = 0;
        }
        // Lower triangular
        float sum1 = 0;
        #pragma omp parallel for private(k,j) shared(lower, upper, C) firstprivate(sum1)
        for(k=i; k<N; ++k){
            if(i==k){
                lower[i][i] = 1;
            }
            else{
                for (j = 0; j < i; j++){
                    sum1 += (lower[k][j] * upper[j][i]);
                }
                lower[k][i]= (C[k][i] - sum1) / upper[i][i];
            }
            sum1=0;
        }
    }

    // checking
    // for(i=0; i<N; i++){
    //     for(j=0; j<N; j++){
    //         float temp = 0;
    //         for(k=0; k<N; k++){
    //             temp += lower[i][k] * upper[k][j];
    //         }
    //         if(abs(temp - C[i][j]) > 0.05){
    //             printf("%d %d %f %f\n", i, j, C[i][j], temp);
    //         }
    //     }
    // }
}