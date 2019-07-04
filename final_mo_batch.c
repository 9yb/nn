#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include "nn.h"
#include "MT.h"

void relu(int n,const float *x,float *y){
    for(int i=0;i<n;i++){
        y[i] = (x[i]>0) ? x[i] : 0;
    }
}

void fc(int m,int n,const float *x,const float *A,const float *b,float *y){
    for(int i=0;i<m;i++){
        y[i] = b[i];
        for(int j=0;j<n;j++){
            y[i] += A[n*i+j] * x[j];
        }
    }
}

void softmax(int n,const float *x,float *y){
    float x_max = 0;
    float exp_sum = 0;
    for(int i=0;i<n;i++){
        if(x_max < x[i]){
            x_max = x[i];
        }   
    }
    for(int i=0;i<n;i++){
        exp_sum += exp(x[i]-x_max);
    }
    for(int i=0;i<n;i++){
        y[i] = exp(x[i]-x_max) / exp_sum;
    }
}

int inference6(const float *A1,const float *A2,const float *A3,
const float *b1,const float *b2,const float *b3,float *x,float *y){
    float *y1 = malloc(sizeof(float)*50);
    float *y2 = malloc(sizeof(float)*100);
    float temp = 0;
    int index;

    fc(50,784,x,A1,b1,y1);
    relu(50,y1,y1);
    fc(100,50,y1,A2,b2,y2);
    relu(100,y2,y2);
    fc(10,100,y2,A3,b3,y);
    softmax(10,y,y);

    for(int i=0;i<=9;i++){
        if(temp < y[i]){
            temp = y[i];
            index = i;
        }
    }
    free(y1);free(y2);
    return index;
}

void softmaxwithloss_bwd(int n,const float *y,unsigned char t,float *dEdx){
    for(int i=0;i<n;i++){
        dEdx[i] = (i==t)? y[i] - 1.0 : y[i];
    }
}

void relu_bwd(int n,const float *x,const float *dEdy,float *dEdx){
    for(int i=0;i<n;i++){
        dEdx[i] = (x[i]>0)? dEdy[i] : 0;
    }
}

void fc_bwd(int m,int n,const float *x,const float *dEdy,const float *A,float *dEdA,float *dEdb,float *dEdx){
    for(int i=0;i<m;i++){
        dEdb[i] = dEdy[i];
        for(int j=0;j<n;j++){
            dEdA[n*i+j] = dEdy[i] * x[j];
        }
    }

    for(int i=0;i<n;i++){
        dEdx[i] = 0;
        for(int j=0;j<m;j++){
            dEdx[i] += A[n*j+i] * dEdy[j];
        }
    }
}

void backward6(const float *A1,const float *A2,const float *A3,
const float *b1,const float *b2,const float *b3,const float *x,
unsigned char t,float *y,float *dEdA1,float *dEdA2,float *dEdA3,float *dEdb1,float *dEdb2,float *dEdb3){
    float *relu1_before = malloc(sizeof(float)*50);
    float *relu2_before = malloc(sizeof(float)*100);
    float *fc2_before = malloc(sizeof(float)*50);
    float *fc3_before = malloc(sizeof(float)*100);

    fc(50,784,x,A1,b1,relu1_before);
    relu(50,relu1_before,fc2_before);
    fc(100,50,fc2_before,A2,b2,relu2_before);
    relu(100,relu2_before,fc3_before);
    fc(10,100,fc3_before,A3,b3,y);
    softmax(10,y,y);

    float *dx3 = malloc(sizeof(float)*10);
    float *dx2 = malloc(sizeof(float)*100);
    float *dx1 = malloc(sizeof(float)*50);
    float *dx0 = malloc(sizeof(float)*784);

    softmaxwithloss_bwd(10,y,t,dx3);
    fc_bwd(10,100,fc3_before,dx3,A3,dEdA3,dEdb3,dx2);
    relu_bwd(100,relu2_before,dx2,dx2);
    fc_bwd(100,50,fc2_before,dx2,A2,dEdA2,dEdb2,dx1);
    relu_bwd(50,relu1_before,dx1,dx1);
    fc_bwd(50,784,x,dx1,A1,dEdA1,dEdb1,dx0);

    free(relu1_before);free(relu2_before);
    free(fc2_before);free(fc3_before);
    free(dx0);free(dx1);free(dx2);free(dx3);
}

void shuffle(int n,int *x){
    int t = 0;
    for(int i=0;i<n;i++){
        t = genrand_int32() % n;
        int temp = x[i];
        x[i] = x[t];
        x[t] = temp;
    }
}

float cross_entropy_error(const float *y,int t){
    return -1*log(y[t]+1e-7);
}

void add(int n,const float *x,float *o){
    for (int i = 0; i < n;i++){
        o[i] = x[i] + o[i];
    }
}

void scale(int n,float x,float *o){
    for (int i = 0; i < n;i++){
        o[i] = o[i] * x;
    }
}

void init(int n,float x,float *o){
    for (int i = 0; i < n;i++){
        o[i] = x;
    }
}

void rand_init(int n,float *o){
    for (int i = 0; i < n;i++){
        o[i] = (float)(rand() - (RAND_MAX / 2)) / (RAND_MAX / 2); //[-1:1]
    }
}



double rand_normal( double mu, double sigma ){
    double z=sqrt( -2.0*log(genrand_real3())) * sin( 2.0*M_PI*genrand_real3());
    return mu + sigma*z;
}

void he_init(int n,float *o){
    for(int i=0;i<n;i++){
        o[i]=rand_normal(0,sqrt(2.0/n));
    }
}

void momentum(int n,float *v,float *o,float *t){
    float mu = 0.9;
    float r = 0.1;
    for(int i=0;i<n;i++){
        v[i] = mu * v[i] - r * o[i];
    }
    add(n,v,t);
}

void save(const char *filename,int m,int n,const float *A,const float *b){
    FILE *fp;
    fp = fopen(filename,"wb");
    fwrite(A,sizeof(float),m*n,fp);
    fwrite(A,sizeof(float),m*n,fp);
    fclose(fp);
}

int main(void)
{
    float *train_x = NULL;
    unsigned char *train_y = NULL;
    int train_count = -1;

    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;

    int width = -1;
    int height = -1;

    load_mnist(&train_x, &train_y, &train_count,
                &test_x, &test_y, &test_count,
                &width, &height);

    // これ以降，３層NN の係数 A_784x10 および b_784x10 と，
    // 訓練データ train_x + 784*i (i=0,...,train_count-1), train_y[0]～train_y[train_count-1],
    // テストデータ test_x + 784*i (i=0,...,test_count-1), test_y[0]～test_y[test_count-1],
    // を使用することができる．

    srand(time(NULL));

    int epoch = 500;
    int batch_size = 100;

    float *y = malloc(sizeof(float)*10);

    float *dEdA1 = malloc(sizeof(float)*784*50);
    float *dEdb1 = malloc(sizeof(float)*50);
    float *dEdA2 = malloc(sizeof(float)*50*100);
    float *dEdb2 = malloc(sizeof(float)*100);
    float *dEdA3 = malloc(sizeof(float)*100*10);
    float *dEdb3 = malloc(sizeof(float)*10);

    float *vA1 = malloc(sizeof(float)*784*50);
    float *vb1 = malloc(sizeof(float)*50);
    float *vA2 = malloc(sizeof(float)*50*100);
    float *vb2 = malloc(sizeof(float)*100);
    float *vA3 = malloc(sizeof(float)*100*10);
    float *vb3 = malloc(sizeof(float)*10);

    float *dEdA1_ave = malloc(sizeof(float)*784*50);
    float *dEdb1_ave = malloc(sizeof(float)*50);
    float *dEdA2_ave = malloc(sizeof(float)*50*100);
    float *dEdb2_ave = malloc(sizeof(float)*100);
    float *dEdA3_ave = malloc(sizeof(float)*100*10);
    float *dEdb3_ave = malloc(sizeof(float)*10);

    float *A1 = malloc(sizeof(float)*784*50);
    float *b1 = malloc(sizeof(float)*50);
    float *A2 = malloc(sizeof(float)*50*100);
    float *b2 = malloc(sizeof(float)*100);
    float *A3 = malloc(sizeof(float)*100*10);
    float *b3 = malloc(sizeof(float)*10);

    int *index = malloc(sizeof(int)*train_count);

    float Loss_ave = 0;
    float Accuracy = 0;
    float max_accuracy = 0;

    he_init(784*50,A1);
    he_init(50,b1);
    he_init(50*100,A2);
    he_init(100,b2);
    he_init(100*10,A3);
    he_init(10,b3);

    fprintf(stderr, "0%%       50%%       100%%\n");
    fprintf(stderr, "+---------+---------+\n");

    for(int i=0;i<epoch;i++){
        for(int o=0;o<train_count;o++){
            index[o] = o;
        }
        shuffle(train_count,index);

        init(784*50,0,vA1);
        init(50,0,vb1);
        init(50*100,0,vA2);
        init(100,0,vb2);
        init(100*10,0,vA3);
        init(10,0,vb3);

        for(int j=0;j<train_count/batch_size;j++){
            init(784*50,0,dEdA1_ave);
            init(50,0,dEdb1_ave);
            init(50*100,0,dEdA2_ave);
            init(100,0,dEdb2_ave);
            init(100*10,0,dEdA3_ave);
            init(10,0,dEdb3_ave);
            
            for(int k=0;k<batch_size;k++){
                backward6(A1,A2,A3,b1,b2,b3,train_x + 784*index[batch_size*j+k],train_y[index[batch_size*j+k]],y,dEdA1,dEdA2,dEdA3,dEdb1,dEdb2,dEdb3);
                add(784*50,dEdA1,dEdA1_ave);
                add(50,dEdb1,dEdb1_ave);
                add(50*100,dEdA2,dEdA2_ave);
                add(100,dEdb2,dEdb2_ave);
                add(100*10,dEdA3,dEdA3_ave);
                add(10,dEdb3,dEdb3_ave);
            }
            scale(784*50,1/((float)batch_size),dEdA1_ave);
            scale(50,1/((float)batch_size),dEdb1_ave);
            scale(50*100,1/((float)batch_size),dEdA2_ave);
            scale(100,1/((float)batch_size),dEdb2_ave);
            scale(100*10,1/((float)batch_size),dEdA3_ave);
            scale(10,1/((float)batch_size),dEdb3_ave);

            momentum(784*50,vA1,dEdA1_ave,A1);
            momentum(50,vb1,dEdb1_ave,b1);
            momentum(50*100,vA2,dEdA2_ave,A2);
            momentum(100,vb2,dEdb2_ave,b2);
            momentum(100*10,vA3,dEdA3_ave,A3);
            momentum(10,vb3,dEdb3_ave,b3);

            for (int m=0; m < (j+1)/(train_count/(20*batch_size)); m++) {
                fprintf(stderr, "#");
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "Epoch%2d:%3d%%\n", i,100*j/(train_count/batch_size-1));
            fprintf(stderr, "\033[2A");    

        }
           

        float sum = 0;
        float loss_sum = 0;
        
        for(int m=0;m<test_count;m++){
            if(inference6(A1,A2,A3,b1,b2,b3,test_x+m*784,y) == test_y[m]){
                sum++;
            }
            loss_sum += cross_entropy_error(y,test_y[m]);
        }
        fprintf(stderr, "\033[2B");
        fprintf(stderr,"Loss Average: %f (%+.3f)\n",loss_sum/test_count,loss_sum/test_count-Loss_ave);
        fprintf(stderr,"Accuracy: %f (%+.2f)\n",sum*100.0/test_count,sum*100.0/test_count-Accuracy);
        if(sum*100.0/test_count > max_accuracy){
            max_accuracy = sum*100.0/test_count;
            save("fc1.dat",50,784,A1,b1);
            save("fc2.dat",100,50,A2,b2);
            save("fc3.dat",10,100,A3,b3);
        }
        fprintf(stderr,"Max Accuracy: %.2f%%\n",max_accuracy+0.001);
        fprintf(stderr, "\033[5A");
        fprintf(stderr, "                     \n");
        fprintf(stderr, "\033[1A");

        Loss_ave = loss_sum/test_count;
        Accuracy = sum*100.0/test_count;   

    }
    /*
    save("fc1.dat",50,784,A1,b1);
    save("fc2.dat",100,50,A2,b2);
    save("fc3.dat",10,100,A3,b3);
    */

    fprintf(stderr, "\n\n\n\n\nfinish!\n");

  return 0;
}
