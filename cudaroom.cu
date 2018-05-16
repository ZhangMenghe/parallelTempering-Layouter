#include<stdio.h>

__device__
void externalFun(){
    if(threadIdx.x == 0 )
    printf("block %d:hello from cudaroom\n", blockIdx.x );
}
