#include<iostream>
#include<stdio.h>
#define N 256
int main()
{
    for(int i=0; i< N; i++)
        printf("\ttemp+=d_val[index+N*%d]*s_b[d_row[index+N*%d]];\n",i,i);
}