#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <ctime>
#include <time.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;

#define N 1024                      // define n*n matrix
#define S (N/1)                     // sparse matrix's nonzero elements per col (75% currently)
#define BLOCK_SIZE 32               // block size ( 256->16 / 1024->32 )
#define TEST_TIMES 100             // mul module caculate times (for easy record the running time)


// simple cuda mul kernel 
static void __global__ mul_kernel(int rowsize,int colsize,int colpitch,const float *d_a,const float *d_b,float *d_c)
{
    uint index= threadIdx.x + blockIdx.x * blockDim.x;
    if(rowsize <= index) return;
    float temp=0.0f;
    for(int j=0;j<TEST_TIMES;j++)
    {
        for(int i=0;i<rowsize;i++)
        {
            temp+=d_a[i*colpitch+index]*d_b[i+j*N];
        }
        d_c[index+j*N]=temp;
        
        temp = 0.0f;
        __syncthreads();
    }
}

// cuda mul kernel with shared memory
static void __global__ mul_kernel_shared(int rowsize,int colsize,int colpitch,const float *d_a,const float *d_b,float *d_c,const int sharedsize)
{
    __shared__ float s_b[N];
    float temp=0.0f;
    uint index= threadIdx.x + blockIdx.x * blockDim.x;
    for(int j=0;j<TEST_TIMES;j++)
    {
        for(int start=0;start<rowsize;start+=sharedsize)
        {
            // load shared memory (vec)
            __syncthreads();
            for(int i=threadIdx.x;i<sharedsize&&(i+start)<rowsize;i+=blockDim.x)
                s_b[i]=d_b[i+start+j*N];
            __syncthreads();
        
            if(rowsize <= index) continue;
            int end=start+sharedsize > rowsize ? rowsize : start+sharedsize;
            for(int i=start;i<end;i++)
            {
                temp+=d_a[i*colpitch+index]*s_b[i-start];
            }
        }
        if(index<colsize)
            d_c[index+j*N]=temp;
        temp = 0;
        __syncthreads();
    }
}


// cuda mul kernel with shared memory and csr
static void __global__ mul_kernel_shared_csr(int rowsize,int colsize,int colpitch,const int *d_row,const float *d_val,const float *d_b,float *d_c,const int sharedsize)
{
    __shared__ float s_b[N];
    float temp=0.0f;
    uint index= threadIdx.x + blockIdx.x * blockDim.x;

    for(int j=0;j<TEST_TIMES;j++)
    {
        // load shared memory (vec)
        for(int start=0;start<rowsize;start+=sharedsize)
        {
            for(int i=threadIdx.x;i<sharedsize&&(i+start)<rowsize;i+=blockDim.x)
            {
                s_b[i]=d_b[i+start+j*N];
            }
            __syncthreads();
        }
    

        for(int i=0;i<S;i++)
        {
            temp+=d_val[index+N*i]*s_b[d_row[index+i*N]];
        }
    
        if(index<colsize)
            d_c[index+j*N]=temp;
        temp = 0;
        __syncthreads();
    }
        
}


// use register cache row data
static void __global__ mul_kernel_shared_csr_reg(int rowsize,int colsize,int colpitch,const int *d_row,const float *d_val,const float *d_b,float *d_c,const int sharedsize)
{
    __shared__ float s_b[N];
    float temp=0.0f;
    float val[S];
    int row[S];

    uint index= threadIdx.x + blockIdx.x * blockDim.x;

    for(int i=0;i<S;i++)
    {
        val[i]=d_val[index+N*i];
        row[i]=d_row[index+i*N];
    }

    for(int j=0;j<TEST_TIMES;j++)
    {

    // load shared memory (vec)
        for(int start=0;start<rowsize;start+=sharedsize)
        {
            for(int i=threadIdx.x;i<sharedsize&&(i+start)<rowsize;i+=blockDim.x)
            {
                s_b[i]=d_b[i+start+j*N];
            }
            __syncthreads();
        }

    
    
        for(int i=0;i<S;i++)
        {
            temp+=val[i]*s_b[row[i]];
        }
        
        if(index<colsize)
            d_c[index+j*N]=temp;
        temp = 0;
        __syncthreads();
    }
        
}

// cpu matrix mul
void mul_cpu(float *a,float *b,float *c)
{
    for(int k=0;k<TEST_TIMES;k++)
        for(int i=0;i<N;i++)
        {
            c[i+k*N]=0;
            for(int j=0;j<N;j++)
                c[i+k*N]+=(*(a+i*N+j)**(b+j+k*N));
        }
}

// test cpu and gpu mul result
bool resultcompare(float *ref,float *test,float accu)
{
    for(int i=0;i<N*TEST_TIMES;i++)
    {
        if(fabs(*(ref+i)-*(test+i))>accu) return false;
    }
    return true;
}



int main()
{
    srand(time(0));

    // Memory Manage

    int *sma_a_col=new int[N*S];            // CSR row array
    float *sma_a_val=new float[N*S];        // CSR value array
  

    int *d_row;                             // CSR row array (transpose)
    float *d_val;                           // CSR value array (transpose)
    cudaMallocManaged(&d_row, sizeof(int)*N*S);
    cudaMallocManaged(&d_val, sizeof(float)*N*S);

    float *d_a,*d_b,*d_c;                   // Matrix A, Vector B, Result C
    cudaMallocManaged(&d_a, sizeof(float)*N*N);
    cudaMallocManaged(&d_b, sizeof(float)*N);
    cudaMallocManaged(&d_c, sizeof(float)*N);

    float *c1,*c2,*c3,*c4,*c5;               // Mul result C (on GPU : 1-4)
    cudaMallocManaged(&c1, sizeof(float)*N);
    cudaMallocManaged(&c2, sizeof(float)*N);
    cudaMallocManaged(&c3, sizeof(float)*N);
    cudaMallocManaged(&c4, sizeof(float)*N);
    cudaMallocManaged(&c5, sizeof(float)*N);
    

    // Init setting
    bool a_row[N];                          // nozero element flag
    int pos;
    float temp;

    // CPU timer
    clock_t begin,end;
    double timer;
    double total_csr=0,total_trans=0,total_cpu=0;  // Save total time of each work

    // GPU timer
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpu_timer;
    float total_gpu1=0,total_gpu2=0,total_gpu3=0,total_gpu4=0,total_gpu5=0;

    // GPU set threads & blocks
    uint threads=N;
    int sharedsize=threads;
    int blocknum=(N+threads-1)/threads;


    // * Test begin *

    for(int tt=0; tt<TEST_TIMES; tt++)
    {

    // Random fill in matrix & vector
    for (int i = 0; i < N; i++)
    {
        for(int j=0;j<N;j++)
            a_row[j]=false;

        for(int j=0;j<S;j++)
        {
            int temp_pos = rand() % N; 
            while(a_row[temp_pos])
            {
                temp_pos++;
                if(temp_pos==N) temp_pos=0;
            } 
            a_row[temp_pos]=true;
        }

        pos=S*i;
        for(int k=0;k<N;k++)
        {
            *(d_a+i*N+k)=0;
            if(a_row[k])
            {
                *(d_a+i*N+k)=rand()%10;
                //printf("row:%d val:%f \n",*(sma_a_col+pos),*(sma_a_val+pos));
                pos++;
            }
        }
        
    }

    for (int i = 0; i < N*TEST_TIMES; i++)
        *(d_b+i) = rand() % 10;

    // * Recording csr decoding time *
    begin=clock();
    for (int i = 0; i < N; i++)
    {
        pos=S*i;
        for(int k=0;k<N;k++)
        {
            if(*(d_a+i*N+k)!=0)
            {
                *(sma_a_col+pos)=k;
                *(sma_a_val+pos)=*(d_a+i*N+k);
                //printf("row:%d val:%f \n",*(sma_a_col+pos),*(sma_a_val+pos));
                pos++;
            }
        }
        
    }

    end=clock();
    timer=(double)(end-begin)/CLOCKS_PER_SEC;
    total_csr+=timer;
    //printf("The csr decoding time is %f ms.\n",timer*1000);
    

    // Cpu Mul reference
    begin=clock();

    for(int i=0;i<1;i++)
        mul_cpu(d_a,d_b,d_c);

    end=clock();
    timer=(double)(end-begin)/CLOCKS_PER_SEC;
    total_cpu+=timer;
    //printf("The total cpu run time is %f ms.\n",timer*1000);


    // Matrix tranpose (for memory coalesced)

    
    begin=clock();
    for (int i = 0; i < N; i++)
        for(int j = i+1; j < N; j++)
        {
            temp = *(d_a+j*N+i);
            *(d_a+j*N+i) = *(d_a+i*N+j);
            *(d_a+i*N+j) = temp;
        }

    for (int i = 0; i < N; i++)
        for(int j = 0; j < S; j++)
        {
            *(d_row+j*N+i)=*(sma_a_col+i*S+j);
            *(d_val+j*N+i)=*(sma_a_val+i*S+j);
        }
    end=clock();
    timer=(double)(end-begin)/CLOCKS_PER_SEC;
    total_trans+=timer;


    // * GPU Caculation Part *

    // 1.Normal Matrix mul kernel
    cudaEventRecord(start,0);

    mul_kernel<<<blocknum, threads>>>(N,N,N,d_a,d_b,c1);
    cudaDeviceSynchronize();

    cudaEventRecord(stop,0); 
    cudaEventSynchronize(start);    
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&gpu_timer,start,stop);

    total_gpu1+=gpu_timer;
    //printf("The total gpu run time is %f ms.\n",costtime2);


    // 2.Matrix mul using shared memory kernel
    cudaEventRecord(start,0);

    mul_kernel_shared<<<blocknum, threads>>>(N,N,N,d_a,d_b,c2,sharedsize);
    cudaDeviceSynchronize();

    cudaEventRecord(stop,0); 
    cudaEventSynchronize(start);    
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&gpu_timer,start,stop);

    total_gpu2+=gpu_timer;
    //printf("The total gpu (use shared memory) run time is %f ms.\n",costtime);


    // 3.Matrix mul using shared memory and csr kernel

    cudaEventRecord(start,0);

    mul_kernel_shared_csr<<<blocknum, threads>>>(N,N,N,d_row,d_val,d_b,c3,sharedsize);
    cudaDeviceSynchronize();

    cudaEventRecord(stop,0); 
    cudaEventSynchronize(start);    
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&gpu_timer,start,stop);
    total_gpu3+=gpu_timer;
    //printf("The total gpu (using csr and shared memory) run time is %f ms.\n",costtime3);

    // 4.Use register
    cudaEventRecord(start,0);

    mul_kernel_shared_csr_reg<<<blocknum, threads>>>(N,N,N,d_row,d_val,d_b,c4,sharedsize);
    cudaDeviceSynchronize();

    cudaEventRecord(stop,0); 
    cudaEventSynchronize(start);    
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&gpu_timer,start,stop);
    total_gpu4+=gpu_timer;
    //printf("The total gpu (using csr by register and shared memory) run time is %f ms.\n",costtime5);

    // 5.Matrix using cublas function call
    float alpha = 1;
    float beta = 0;
    int M=1;                // B->vector

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEventRecord(start,0);

    // matrix cublas call
    cublasSgemm(handle,
        CUBLAS_OP_T,  
        CUBLAS_OP_N,   
        M,                    // row of B
        N,                    // col of A
        N,                    // row of B
        &alpha,           
        d_b,            
        M,                    
        d_a,         
        N,         
        &beta,          
        c5,           
        M);
    cudaDeviceSynchronize();

    cudaEventRecord(stop,0); 
    cudaEventSynchronize(start);    
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&gpu_timer,start,stop);
    total_gpu5+=gpu_timer;
    //printf("The total gpu (using cublas) run time is %f ms.\n",costtime4);


    // Correct test Part
    printf("Test %d:  ",tt);

    bool res;
    res=resultcompare(d_c,c1,1e-4f);
    if(res) printf("1P! ");
    else printf("1F! ");

    res=resultcompare(d_c,c2,1e-4f);
    if(res) printf("2P! ");
    else printf("2F! ");

    res=resultcompare(d_c,c3,1e-4f);
    if(res) printf("3P! ");
    else printf("3F! ");

    res=resultcompare(d_c,c4,1e-4f);
    if(res) printf("4P! ");
    else printf("4F! ");

    res=resultcompare(d_c,c5,1e-4f);
    if(res) printf("5P!\n");
    else printf("5F!\n");


    // test diff
    /*for(int i=0;i<TEST_TIMES*N;i++)
    {
        printf("c=%f\to1=%f\to2=%f\to3=%f\to4=%f\to5=%f\n",*(ref+i),*(h_c1+i),*(h_c2+i),*(h_c3+i),*(h_c4+i),*(h_c5+i));
    }*/


    }

    // Statistic Total Time
    printf("Matrxi: %d*%d, S: %d,Test Times: %d\n",N,N,S,TEST_TIMES);
    printf("The csr decoding time is %.4lf ms.\n",total_csr*1000);
    printf("The matrix trans time is %.4lf ms.\n",total_trans*1000);
    printf("The total cpu run time is %.4lf ms.\n",total_cpu*1000);

    printf("The total gpu run time is %f ms.\n",total_gpu1);
    printf("The total gpu (use shared memory) run time is %f ms.\n",total_gpu2);
    printf("The total gpu (using csr and shared memory) run time is %f ms.\n",total_gpu3);
    printf("The total gpu (using csr by register and shared memory) run time is %f ms.\n",total_gpu4);
    printf("The total gpu (using cublas) run time is %f ms.\n",total_gpu5);

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(c1);
    cudaFree(c2);
    cudaFree(c3);
    cudaFree(c4);
    cudaFree(c5);

    free(sma_a_col);
    free(sma_a_val);
    return 0;
}