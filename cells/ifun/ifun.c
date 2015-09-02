/*
This code, "ClosedLoopRoessertEtAl", is a derivative of "internalclock" by Takeru Honda and Tadashi Yamazaki used under CC-BY (http://creativecommons.org/licenses/by/3.0/). 
The code of "internalclock" was downloaded from https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=115966
"ClosedLoopRoessertEtAl" is licensed under CC BY by Christian Rössert.
*/

//
// Simulation program
//

#include "ifun.h"

#define wid(i,j) ((j)+N*(i))
#define zid(t,i) ((i)+N*(t))

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Period parameters */  
#define NR 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UMASK 0x80000000UL /* most significant w-r bits */
#define LMASK 0x7fffffffUL /* least significant r bits */
#define MIXBITS(u,v) ( ((u) & UMASK) | ((v) & LMASK) )
#define TWIST(u,v) ((MIXBITS(u,v) >> 1) ^ ((v)&1UL ? MATRIX_A : 0UL))

static unsigned long state[NR]; /* the array for the state vector  */
static int left = 1;
static int initf = 0;
static unsigned long *next;

/* initializes state[N] with a seed */
void init_genrand(unsigned long s)
{
    int j;
    state[0]= s & 0xffffffffUL;
    for (j=1; j<NR; j++) {
        state[j] = (1812433253UL * (state[j-1] ^ (state[j-1] >> 30)) + j); 
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array state[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        state[j] &= 0xffffffffUL;  /* for >32 bit machines */
    }
    left = 1; initf = 1;
}

static void next_state(void)
{
    unsigned long *p=state;
    int j;

    /* if init_genrand() has not been called, */
    /* a default initial seed is used         */
    if (initf==0) init_genrand(5489UL);

    left = NR;
    next = state;
    
    for (j=NR-M+1; --j; p++) 
        *p = p[M] ^ TWIST(p[0], p[1]);

    for (j=M; --j; p++) 
        *p = p[M-NR] ^ TWIST(p[0], p[1]);

    *p = p[M-NR] ^ TWIST(p[0], state[0]);
}

/* generates a random number on [0,1)-real-interval */
double genrand_real2(void)
{
    unsigned long y;

    if (--left == 0) next_state();
    y = *next++;

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return (double)y * (1.0/4294967296.0); 
    /* divided by 2^32 */
}

double random_normal(void)  /* normal distribution, centered on 0, std dev 1 */
{
  return sqrt(-2*log(genrand_real2())) * cos(2*M_PI*genrand_real2());
}


// This function takes a seed for the random number generator and
// returns the random matrix w_{ij} (Eq.(2.1) in p.1033).
// The matrix is the size of N*N, and the i th row represents the list
// of indices for presynaptic neurons of neuron i.  The connection
// probability is Pr.
// Each row terminates with -1 so that the program can know
// the end of the list.
//const unsigned long

int *random_matrix_index(double *r, int N, double Pr) 
{
  int i, j, n;
  int *w;

  w = (int *)malloc(N*N*sizeof(int));
  for(i = 0; i < N; i++){
    n = 0;
    for(j = 0; j < N; j++){
      if (genrand_real2() < Pr){  //r[wid(i,j)]
	w[wid(i,n)] = j;
	n++;
        //fprintf(stdout, "r: %f", r[wid(i,j)]);
        //fprintf(stdout, "conn: %i", j);
      }
    }
    w[wid(i,n)] = -1;
  }
  return w;
}

// This function returns the T*N vector of neural activity z(t,i) (p. 1033).
// The vector is accessed as a T*N matrix through the function zid(t,i).
double *activity_pattern(int T, int N)
{
  int t, i;
  double *z;

  z = (double *)malloc(T*N*sizeof(double));
  for(t = 0; t < T; t++){
    for(i = 0; i < N; i++){
      z[zid(t,i)] = 0;
    }
  }
  return z;
}


// This function takes the random matrix w and the empty array of
// the neural activity z, and fill the array z.
void run(const int *w, double *z, double *ih0, double *I, double tau, int N, double kappa, int T, double *rand)
{
  int t, i, n;
  double u[N], q[N], f[N], ih[N]; // coef[N];
  double r, tempih, tempih_pos, coeftemp;

  const double decay = exp(-1.0/tau);
  const double coef = 2.0*kappa/N; //
  const double varinh0 = ih0[3];

  double *coefinh;
  coefinh = (double *)malloc(N*N*sizeof(double));

  for(i = 0; i < N; i++){
    q[i] = 0;

    if (genrand_real2() < ih0[2]){ // random ipis/contralateral mf input   rand[i]
	f[i] = -1;
    }else{
        f[i] = 1;
    } 

    ih[i] = ih0[0] + (ih0[1]*ih0[0])*random_normal();
    
    //fprintf(stdout, "ih: %i: %f\n ", i, ih[i]);
    //coef[i] = 2.0 * ( kappa + (kappa/2)*random_normal() ) / N;

    for(n = 0; n < N; n++)
    {  
      coeftemp = coef + (varinh0*coef)*random_normal();
      coefinh[wid(i,n)] = (coeftemp > 0) ? coeftemp : 0;
      //fprintf(stdout, "uinh: %i: %i %f %f\n ", n, c, coeftemp, coefinh[widu(n,c)]);
    }
  }



  // Iterative calculation of Eq. (2.1)
  for(t = 1; t < T; t++){
    for(i = 0; i < N; i++){
      q[i] = z[zid(t-1,i)] + decay*q[i];
    }
    for(i = 0; i < N; i++){

      r = 0;
      // the list of presynaptic neurons is terminated with -1.
      for(n = 0; w[wid(i,n)] >= 0; n++){
	r += coefinh[wid(i,n)]*q[w[wid(i,n)]]; //coef[i]
      }
      tempih = ih[i] + f[i]*ih[i]*I[t]; //ih0[0] 	
      tempih_pos = (tempih > 0) ? tempih : 0;
      u[i] = tempih_pos - r + rand[1]*ih0[0]*random_normal(); //zid(t,i)
    }
    for(i = 0; i < N; i++){
      z[zid(t,i)] = (u[i] > 0) ? u[i] : 0;
    }
  }
}

double *ifun(double *r, int N, int T, double Pr, double tau, double kappa, double *ih, double *I) //const unsigned long
{
  double *z;
  int *w;

  //fprintf(stdout, "start");

  init_genrand(r[0]);

  w = random_matrix_index(r, N, Pr);
  z = activity_pattern(T, N);
  
  run(w, z, ih, I, tau, N, kappa, T, r);

  free(w);

  //fprintf(stdout, "end");
  //fprintf(stdout, "test: %s", z[0]);
  return z;
}
