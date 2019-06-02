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
#define S (N/4)                     // sparse matrix's nonzero elements per col (75% currently)
#define BLOCK_SIZE 32              // block size ( 256->16 / 1024->32 )
#define TEST_TIMES 100             // mul module caculate times (for easy record the running time)
#define TEST_REPEAT 100

static void __global__ mul_kernel_shared_csr(int rowsize,int colsize,int colpitch,const int *d_row,const float *d_val,const float *d_b,float *d_c,const int sharedsize,int s)
{
    __shared__ float s_b[N];
    float temp=0.0f;
    uint index= threadIdx.x + blockIdx.x * blockDim.x;

    if(index>=colsize) return;

    // load shared memory (vec)
    for(int start=0;start<rowsize;start+=sharedsize)
    {
        for(int i=threadIdx.x;i<sharedsize&&(i+start)<rowsize;i+=blockDim.x)
        {
            s_b[i]=d_b[i+start];
        }
        __syncthreads();
    }
    

    // Main loop unrolling
    for(int i=0;i<s;i++)
    {
        temp+=d_val[index+N*i]*s_b[d_row[index+i*N]];
    }
    
    if(index<colsize)
        d_c[index]=temp;

}

static void __global__ mul_kernel_shared_csr_rolling(int rowsize,int colsize,int colpitch,const int *d_row,const float *d_val,const float *d_b,float *d_c,const int sharedsize)
{
    __shared__ float s_b[N];
    float temp=0.0f;
    uint index= threadIdx.x + blockIdx.x * blockDim.x;

    if(index>=colsize) return;

    // load shared memory (vec)
    for(int start=0;start<rowsize;start+=sharedsize)
    {
        for(int i=threadIdx.x;i<sharedsize&&(i+start)<rowsize;i+=blockDim.x)
        {
            s_b[i]=d_b[i+start];
        }
        __syncthreads();
    }
    
    temp+=d_val[index+N*0]*s_b[d_row[index+N*0]];
        temp+=d_val[index+N*1]*s_b[d_row[index+N*1]];
        temp+=d_val[index+N*2]*s_b[d_row[index+N*2]];
        temp+=d_val[index+N*3]*s_b[d_row[index+N*3]];
        temp+=d_val[index+N*4]*s_b[d_row[index+N*4]];
        temp+=d_val[index+N*5]*s_b[d_row[index+N*5]];
        temp+=d_val[index+N*6]*s_b[d_row[index+N*6]];
        temp+=d_val[index+N*7]*s_b[d_row[index+N*7]];
        temp+=d_val[index+N*8]*s_b[d_row[index+N*8]];
        temp+=d_val[index+N*9]*s_b[d_row[index+N*9]];
        temp+=d_val[index+N*10]*s_b[d_row[index+N*10]];
        temp+=d_val[index+N*11]*s_b[d_row[index+N*11]];
        temp+=d_val[index+N*12]*s_b[d_row[index+N*12]];
        temp+=d_val[index+N*13]*s_b[d_row[index+N*13]];
        temp+=d_val[index+N*14]*s_b[d_row[index+N*14]];
        temp+=d_val[index+N*15]*s_b[d_row[index+N*15]];
        temp+=d_val[index+N*16]*s_b[d_row[index+N*16]];
        temp+=d_val[index+N*17]*s_b[d_row[index+N*17]];
        temp+=d_val[index+N*18]*s_b[d_row[index+N*18]];
        temp+=d_val[index+N*19]*s_b[d_row[index+N*19]];
        temp+=d_val[index+N*20]*s_b[d_row[index+N*20]];
        temp+=d_val[index+N*21]*s_b[d_row[index+N*21]];
        temp+=d_val[index+N*22]*s_b[d_row[index+N*22]];
        temp+=d_val[index+N*23]*s_b[d_row[index+N*23]];
        temp+=d_val[index+N*24]*s_b[d_row[index+N*24]];
        temp+=d_val[index+N*25]*s_b[d_row[index+N*25]];
        temp+=d_val[index+N*26]*s_b[d_row[index+N*26]];
        temp+=d_val[index+N*27]*s_b[d_row[index+N*27]];
        temp+=d_val[index+N*28]*s_b[d_row[index+N*28]];
        temp+=d_val[index+N*29]*s_b[d_row[index+N*29]];
        temp+=d_val[index+N*30]*s_b[d_row[index+N*30]];
        temp+=d_val[index+N*31]*s_b[d_row[index+N*31]];
        temp+=d_val[index+N*32]*s_b[d_row[index+N*32]];
        temp+=d_val[index+N*33]*s_b[d_row[index+N*33]];
        temp+=d_val[index+N*34]*s_b[d_row[index+N*34]];
        temp+=d_val[index+N*35]*s_b[d_row[index+N*35]];
        temp+=d_val[index+N*36]*s_b[d_row[index+N*36]];
        temp+=d_val[index+N*37]*s_b[d_row[index+N*37]];
        temp+=d_val[index+N*38]*s_b[d_row[index+N*38]];
        temp+=d_val[index+N*39]*s_b[d_row[index+N*39]];
        temp+=d_val[index+N*40]*s_b[d_row[index+N*40]];
        temp+=d_val[index+N*41]*s_b[d_row[index+N*41]];
        temp+=d_val[index+N*42]*s_b[d_row[index+N*42]];
        temp+=d_val[index+N*43]*s_b[d_row[index+N*43]];
        temp+=d_val[index+N*44]*s_b[d_row[index+N*44]];
        temp+=d_val[index+N*45]*s_b[d_row[index+N*45]];
        temp+=d_val[index+N*46]*s_b[d_row[index+N*46]];
        temp+=d_val[index+N*47]*s_b[d_row[index+N*47]];
        temp+=d_val[index+N*48]*s_b[d_row[index+N*48]];
        temp+=d_val[index+N*49]*s_b[d_row[index+N*49]];
        temp+=d_val[index+N*50]*s_b[d_row[index+N*50]];
        temp+=d_val[index+N*51]*s_b[d_row[index+N*51]];
        temp+=d_val[index+N*52]*s_b[d_row[index+N*52]];
        temp+=d_val[index+N*53]*s_b[d_row[index+N*53]];
        temp+=d_val[index+N*54]*s_b[d_row[index+N*54]];
        temp+=d_val[index+N*55]*s_b[d_row[index+N*55]];
        temp+=d_val[index+N*56]*s_b[d_row[index+N*56]];
        temp+=d_val[index+N*57]*s_b[d_row[index+N*57]];
        temp+=d_val[index+N*58]*s_b[d_row[index+N*58]];
        temp+=d_val[index+N*59]*s_b[d_row[index+N*59]];
        temp+=d_val[index+N*60]*s_b[d_row[index+N*60]];
        temp+=d_val[index+N*61]*s_b[d_row[index+N*61]];
        temp+=d_val[index+N*62]*s_b[d_row[index+N*62]];
        temp+=d_val[index+N*63]*s_b[d_row[index+N*63]];
        temp+=d_val[index+N*64]*s_b[d_row[index+N*64]];
        temp+=d_val[index+N*65]*s_b[d_row[index+N*65]];
        temp+=d_val[index+N*66]*s_b[d_row[index+N*66]];
        temp+=d_val[index+N*67]*s_b[d_row[index+N*67]];
        temp+=d_val[index+N*68]*s_b[d_row[index+N*68]];
        temp+=d_val[index+N*69]*s_b[d_row[index+N*69]];
        temp+=d_val[index+N*70]*s_b[d_row[index+N*70]];
        temp+=d_val[index+N*71]*s_b[d_row[index+N*71]];
        temp+=d_val[index+N*72]*s_b[d_row[index+N*72]];
        temp+=d_val[index+N*73]*s_b[d_row[index+N*73]];
        temp+=d_val[index+N*74]*s_b[d_row[index+N*74]];
        temp+=d_val[index+N*75]*s_b[d_row[index+N*75]];
        temp+=d_val[index+N*76]*s_b[d_row[index+N*76]];
        temp+=d_val[index+N*77]*s_b[d_row[index+N*77]];
        temp+=d_val[index+N*78]*s_b[d_row[index+N*78]];
        temp+=d_val[index+N*79]*s_b[d_row[index+N*79]];
        temp+=d_val[index+N*80]*s_b[d_row[index+N*80]];
        temp+=d_val[index+N*81]*s_b[d_row[index+N*81]];
        temp+=d_val[index+N*82]*s_b[d_row[index+N*82]];
        temp+=d_val[index+N*83]*s_b[d_row[index+N*83]];
        temp+=d_val[index+N*84]*s_b[d_row[index+N*84]];
        temp+=d_val[index+N*85]*s_b[d_row[index+N*85]];
        temp+=d_val[index+N*86]*s_b[d_row[index+N*86]];
        temp+=d_val[index+N*87]*s_b[d_row[index+N*87]];
        temp+=d_val[index+N*88]*s_b[d_row[index+N*88]];
        temp+=d_val[index+N*89]*s_b[d_row[index+N*89]];
        temp+=d_val[index+N*90]*s_b[d_row[index+N*90]];
        temp+=d_val[index+N*91]*s_b[d_row[index+N*91]];
        temp+=d_val[index+N*92]*s_b[d_row[index+N*92]];
        temp+=d_val[index+N*93]*s_b[d_row[index+N*93]];
        temp+=d_val[index+N*94]*s_b[d_row[index+N*94]];
        temp+=d_val[index+N*95]*s_b[d_row[index+N*95]];
        temp+=d_val[index+N*96]*s_b[d_row[index+N*96]];
        temp+=d_val[index+N*97]*s_b[d_row[index+N*97]];
        temp+=d_val[index+N*98]*s_b[d_row[index+N*98]];
        temp+=d_val[index+N*99]*s_b[d_row[index+N*99]];
        temp+=d_val[index+N*100]*s_b[d_row[index+N*100]];
        temp+=d_val[index+N*101]*s_b[d_row[index+N*101]];
        temp+=d_val[index+N*102]*s_b[d_row[index+N*102]];
        temp+=d_val[index+N*103]*s_b[d_row[index+N*103]];
        temp+=d_val[index+N*104]*s_b[d_row[index+N*104]];
        temp+=d_val[index+N*105]*s_b[d_row[index+N*105]];
        temp+=d_val[index+N*106]*s_b[d_row[index+N*106]];
        temp+=d_val[index+N*107]*s_b[d_row[index+N*107]];
        temp+=d_val[index+N*108]*s_b[d_row[index+N*108]];
        temp+=d_val[index+N*109]*s_b[d_row[index+N*109]];
        temp+=d_val[index+N*110]*s_b[d_row[index+N*110]];
        temp+=d_val[index+N*111]*s_b[d_row[index+N*111]];
        temp+=d_val[index+N*112]*s_b[d_row[index+N*112]];
        temp+=d_val[index+N*113]*s_b[d_row[index+N*113]];
        temp+=d_val[index+N*114]*s_b[d_row[index+N*114]];
        temp+=d_val[index+N*115]*s_b[d_row[index+N*115]];
        temp+=d_val[index+N*116]*s_b[d_row[index+N*116]];
        temp+=d_val[index+N*117]*s_b[d_row[index+N*117]];
        temp+=d_val[index+N*118]*s_b[d_row[index+N*118]];
        temp+=d_val[index+N*119]*s_b[d_row[index+N*119]];
        temp+=d_val[index+N*120]*s_b[d_row[index+N*120]];
        temp+=d_val[index+N*121]*s_b[d_row[index+N*121]];
        temp+=d_val[index+N*122]*s_b[d_row[index+N*122]];
        temp+=d_val[index+N*123]*s_b[d_row[index+N*123]];
        temp+=d_val[index+N*124]*s_b[d_row[index+N*124]];
        temp+=d_val[index+N*125]*s_b[d_row[index+N*125]];
        temp+=d_val[index+N*126]*s_b[d_row[index+N*126]];
        temp+=d_val[index+N*127]*s_b[d_row[index+N*127]];
        temp+=d_val[index+N*128]*s_b[d_row[index+N*128]];
        temp+=d_val[index+N*129]*s_b[d_row[index+N*129]];
        temp+=d_val[index+N*130]*s_b[d_row[index+N*130]];
        temp+=d_val[index+N*131]*s_b[d_row[index+N*131]];
        temp+=d_val[index+N*132]*s_b[d_row[index+N*132]];
        temp+=d_val[index+N*133]*s_b[d_row[index+N*133]];
        temp+=d_val[index+N*134]*s_b[d_row[index+N*134]];
        temp+=d_val[index+N*135]*s_b[d_row[index+N*135]];
        temp+=d_val[index+N*136]*s_b[d_row[index+N*136]];
        temp+=d_val[index+N*137]*s_b[d_row[index+N*137]];
        temp+=d_val[index+N*138]*s_b[d_row[index+N*138]];
        temp+=d_val[index+N*139]*s_b[d_row[index+N*139]];
        temp+=d_val[index+N*140]*s_b[d_row[index+N*140]];
        temp+=d_val[index+N*141]*s_b[d_row[index+N*141]];
        temp+=d_val[index+N*142]*s_b[d_row[index+N*142]];
        temp+=d_val[index+N*143]*s_b[d_row[index+N*143]];
        temp+=d_val[index+N*144]*s_b[d_row[index+N*144]];
        temp+=d_val[index+N*145]*s_b[d_row[index+N*145]];
        temp+=d_val[index+N*146]*s_b[d_row[index+N*146]];
        temp+=d_val[index+N*147]*s_b[d_row[index+N*147]];
        temp+=d_val[index+N*148]*s_b[d_row[index+N*148]];
        temp+=d_val[index+N*149]*s_b[d_row[index+N*149]];
        temp+=d_val[index+N*150]*s_b[d_row[index+N*150]];
        temp+=d_val[index+N*151]*s_b[d_row[index+N*151]];
        temp+=d_val[index+N*152]*s_b[d_row[index+N*152]];
        temp+=d_val[index+N*153]*s_b[d_row[index+N*153]];
        temp+=d_val[index+N*154]*s_b[d_row[index+N*154]];
        temp+=d_val[index+N*155]*s_b[d_row[index+N*155]];
        temp+=d_val[index+N*156]*s_b[d_row[index+N*156]];
        temp+=d_val[index+N*157]*s_b[d_row[index+N*157]];
        temp+=d_val[index+N*158]*s_b[d_row[index+N*158]];
        temp+=d_val[index+N*159]*s_b[d_row[index+N*159]];
        temp+=d_val[index+N*160]*s_b[d_row[index+N*160]];
        temp+=d_val[index+N*161]*s_b[d_row[index+N*161]];
        temp+=d_val[index+N*162]*s_b[d_row[index+N*162]];
        temp+=d_val[index+N*163]*s_b[d_row[index+N*163]];
        temp+=d_val[index+N*164]*s_b[d_row[index+N*164]];
        temp+=d_val[index+N*165]*s_b[d_row[index+N*165]];
        temp+=d_val[index+N*166]*s_b[d_row[index+N*166]];
        temp+=d_val[index+N*167]*s_b[d_row[index+N*167]];
        temp+=d_val[index+N*168]*s_b[d_row[index+N*168]];
        temp+=d_val[index+N*169]*s_b[d_row[index+N*169]];
        temp+=d_val[index+N*170]*s_b[d_row[index+N*170]];
        temp+=d_val[index+N*171]*s_b[d_row[index+N*171]];
        temp+=d_val[index+N*172]*s_b[d_row[index+N*172]];
        temp+=d_val[index+N*173]*s_b[d_row[index+N*173]];
        temp+=d_val[index+N*174]*s_b[d_row[index+N*174]];
        temp+=d_val[index+N*175]*s_b[d_row[index+N*175]];
        temp+=d_val[index+N*176]*s_b[d_row[index+N*176]];
        temp+=d_val[index+N*177]*s_b[d_row[index+N*177]];
        temp+=d_val[index+N*178]*s_b[d_row[index+N*178]];
        temp+=d_val[index+N*179]*s_b[d_row[index+N*179]];
        temp+=d_val[index+N*180]*s_b[d_row[index+N*180]];
        temp+=d_val[index+N*181]*s_b[d_row[index+N*181]];
        temp+=d_val[index+N*182]*s_b[d_row[index+N*182]];
        temp+=d_val[index+N*183]*s_b[d_row[index+N*183]];
        temp+=d_val[index+N*184]*s_b[d_row[index+N*184]];
        temp+=d_val[index+N*185]*s_b[d_row[index+N*185]];
        temp+=d_val[index+N*186]*s_b[d_row[index+N*186]];
        temp+=d_val[index+N*187]*s_b[d_row[index+N*187]];
        temp+=d_val[index+N*188]*s_b[d_row[index+N*188]];
        temp+=d_val[index+N*189]*s_b[d_row[index+N*189]];
        temp+=d_val[index+N*190]*s_b[d_row[index+N*190]];
        temp+=d_val[index+N*191]*s_b[d_row[index+N*191]];
        temp+=d_val[index+N*192]*s_b[d_row[index+N*192]];
        temp+=d_val[index+N*193]*s_b[d_row[index+N*193]];
        temp+=d_val[index+N*194]*s_b[d_row[index+N*194]];
        temp+=d_val[index+N*195]*s_b[d_row[index+N*195]];
        temp+=d_val[index+N*196]*s_b[d_row[index+N*196]];
        temp+=d_val[index+N*197]*s_b[d_row[index+N*197]];
        temp+=d_val[index+N*198]*s_b[d_row[index+N*198]];
        temp+=d_val[index+N*199]*s_b[d_row[index+N*199]];
        temp+=d_val[index+N*200]*s_b[d_row[index+N*200]];
        temp+=d_val[index+N*201]*s_b[d_row[index+N*201]];
        temp+=d_val[index+N*202]*s_b[d_row[index+N*202]];
        temp+=d_val[index+N*203]*s_b[d_row[index+N*203]];
        temp+=d_val[index+N*204]*s_b[d_row[index+N*204]];
        temp+=d_val[index+N*205]*s_b[d_row[index+N*205]];
        temp+=d_val[index+N*206]*s_b[d_row[index+N*206]];
        temp+=d_val[index+N*207]*s_b[d_row[index+N*207]];
        temp+=d_val[index+N*208]*s_b[d_row[index+N*208]];
        temp+=d_val[index+N*209]*s_b[d_row[index+N*209]];
        temp+=d_val[index+N*210]*s_b[d_row[index+N*210]];
        temp+=d_val[index+N*211]*s_b[d_row[index+N*211]];
        temp+=d_val[index+N*212]*s_b[d_row[index+N*212]];
        temp+=d_val[index+N*213]*s_b[d_row[index+N*213]];
        temp+=d_val[index+N*214]*s_b[d_row[index+N*214]];
        temp+=d_val[index+N*215]*s_b[d_row[index+N*215]];
        temp+=d_val[index+N*216]*s_b[d_row[index+N*216]];
        temp+=d_val[index+N*217]*s_b[d_row[index+N*217]];
        temp+=d_val[index+N*218]*s_b[d_row[index+N*218]];
        temp+=d_val[index+N*219]*s_b[d_row[index+N*219]];
        temp+=d_val[index+N*220]*s_b[d_row[index+N*220]];
        temp+=d_val[index+N*221]*s_b[d_row[index+N*221]];
        temp+=d_val[index+N*222]*s_b[d_row[index+N*222]];
        temp+=d_val[index+N*223]*s_b[d_row[index+N*223]];
        temp+=d_val[index+N*224]*s_b[d_row[index+N*224]];
        temp+=d_val[index+N*225]*s_b[d_row[index+N*225]];
        temp+=d_val[index+N*226]*s_b[d_row[index+N*226]];
        temp+=d_val[index+N*227]*s_b[d_row[index+N*227]];
        temp+=d_val[index+N*228]*s_b[d_row[index+N*228]];
        temp+=d_val[index+N*229]*s_b[d_row[index+N*229]];
        temp+=d_val[index+N*230]*s_b[d_row[index+N*230]];
        temp+=d_val[index+N*231]*s_b[d_row[index+N*231]];
        temp+=d_val[index+N*232]*s_b[d_row[index+N*232]];
        temp+=d_val[index+N*233]*s_b[d_row[index+N*233]];
        temp+=d_val[index+N*234]*s_b[d_row[index+N*234]];
        temp+=d_val[index+N*235]*s_b[d_row[index+N*235]];
        temp+=d_val[index+N*236]*s_b[d_row[index+N*236]];
        temp+=d_val[index+N*237]*s_b[d_row[index+N*237]];
        temp+=d_val[index+N*238]*s_b[d_row[index+N*238]];
        temp+=d_val[index+N*239]*s_b[d_row[index+N*239]];
        temp+=d_val[index+N*240]*s_b[d_row[index+N*240]];
        temp+=d_val[index+N*241]*s_b[d_row[index+N*241]];
        temp+=d_val[index+N*242]*s_b[d_row[index+N*242]];
        temp+=d_val[index+N*243]*s_b[d_row[index+N*243]];
        temp+=d_val[index+N*244]*s_b[d_row[index+N*244]];
        temp+=d_val[index+N*245]*s_b[d_row[index+N*245]];
        temp+=d_val[index+N*246]*s_b[d_row[index+N*246]];
        temp+=d_val[index+N*247]*s_b[d_row[index+N*247]];
        temp+=d_val[index+N*248]*s_b[d_row[index+N*248]];
        temp+=d_val[index+N*249]*s_b[d_row[index+N*249]];
        temp+=d_val[index+N*250]*s_b[d_row[index+N*250]];
        temp+=d_val[index+N*251]*s_b[d_row[index+N*251]];
        temp+=d_val[index+N*252]*s_b[d_row[index+N*252]];
        temp+=d_val[index+N*253]*s_b[d_row[index+N*253]];
        temp+=d_val[index+N*254]*s_b[d_row[index+N*254]];
        temp+=d_val[index+N*255]*s_b[d_row[index+N*255]];

    if(index<colsize)
        d_c[index]=temp;

}

// cpu matrix mul
void mul_cpu(float *a,float *b,float *c)
{
    for(int i=0;i<N;i++)
    {
        c[i]=0;
        for(int j=0;j<N;j++)
            c[i]+=(*(a+i*N+j)**(b+j));
    }
}

// test cpu and gpu mul result
bool resultcompare(float *ref,float *test,float accu)
{
    for(int i=0;i<N;i++)
    {
        if(fabs(*(ref+i)-*(test+i))>accu) return false;
    }
    return true;
}



int main()
{
    srand(time(0));

    // Memory Manage

    int s=S;
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

    float *c1,*c2;               // Mul result C (on GPU : 1-4)
    cudaMallocManaged(&c1, sizeof(float)*N);
    cudaMallocManaged(&c2, sizeof(float)*N);
    

    // Init setting
    bool a_row[N];                          // nozero element flag
    int pos;
    float temp;
    int rand_temp;

    // CPU timer
    clock_t begin,end;
    double timer;
    double total_csr=0,total_trans=0,total_cpu=0;  // Save total time of each work

    // GPU timer
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpu_timer;
    float total_gpu1=0,total_gpu2=0;

    // GPU set threads & blocks
    uint threads=256;
    int sharedsize=N;
    int blocknum=(N+threads-1)/threads;


    // * Test begin *

    for(int tt=0; tt<TEST_TIMES; tt++)
    {

    //printf("Test %d:  \n",tt);

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
                rand_temp=rand()%10;
                while(rand_temp==0)
                rand_temp=rand()%10;
                *(d_a+i*N+k)=rand_temp;
                //printf("row:%d val:%f \n",*(sma_a_col+pos),*(sma_a_val+pos));
                pos++;
            }
        }
        
    }

    for (int i = 0; i < N; i++)
        *(d_b+i) = rand() % 10;

    // * Recording csr encoding time *
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

    for(int jj=0; jj<TEST_REPEAT; jj++)
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
            //printf("[%d,%d]d_row=%d,d_val=%f\n",i,j,*(d_row+j*N+i),*(d_val+j*N+i));
        }   
    end=clock();
    timer=(double)(end-begin)/CLOCKS_PER_SEC;
    total_trans+=timer;


    // * GPU Caculation Part *
    // 1.Matrix mul using shared memory and csr kernel

    cudaEventRecord(start,0);

    for(int jj=0; jj<TEST_REPEAT; jj++)
        mul_kernel_shared_csr<<<blocknum, threads>>>(N,N,N,d_row,d_val,d_b,c1,sharedsize,s);

    cudaDeviceSynchronize();

    cudaEventRecord(stop,0); 
    cudaEventSynchronize(start);    
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&gpu_timer,start,stop);
    total_gpu1+=gpu_timer;
    //printf("The total gpu (using csr and shared memory) run time is %f ms.\n",gpu_timer);

    // 2.Matrix mul using shared memory and csr kernel (loop rolling)

    cudaEventRecord(start,0);

    for(int jj=0; jj<TEST_REPEAT; jj++)
        mul_kernel_shared_csr_rolling<<<blocknum, threads>>>(N,N,N,d_row,d_val,d_b,c2,sharedsize);

    cudaDeviceSynchronize();

    cudaEventRecord(stop,0); 
    cudaEventSynchronize(start);    
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&gpu_timer,start,stop);
    total_gpu2+=gpu_timer;
    //printf("The total gpu (using csr and shared memory) run time is %f ms.\n",gpu_timer);


    // Correct test Part
    printf("Test %d:  ",tt);

    bool res;
    res=resultcompare(d_c,c1,1e-4f);
    if(res) printf("1P! ");
    else printf("1F! ");

    res=resultcompare(d_c,c2,1e-4f);
    if(res) printf("2P!\n");
    else printf("2F!\n");


    // test diff
    /*for(int i=0;i<TEST_TIMES*N;i++)
    {
        printf("c=%f\to1=%f\to2=%f\to3=%f\to4=%f\to5=%f\n",*(d_c+i),*(c1+i),*(c2+i),*(c3+i),*(c4+i),*(c5+i));
    }*/


    }

    // Statistic Total Time
    printf("Matrxi: %d*%d, S: %d,Test Times: %d\n",N,N,S,TEST_TIMES);
    printf("The csr encoding time is %.4lf ms.\n",total_csr*1000);
    printf("The matrix trans time is %.4lf ms.\n",total_trans*1000);
    printf("The total cpu run time is %.4lf ms.\n",total_cpu*1000);

    printf("The unrolling gpu run time is %f ms.\n",total_gpu1);
    printf("The rolling gpu run time is %f ms.\n",total_gpu2);

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(c1);
    cudaFree(c2);

    free(sma_a_col);
    free(sma_a_val);
    return 0;
}