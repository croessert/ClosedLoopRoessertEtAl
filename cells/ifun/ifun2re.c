/*
This code, "ClosedLoopRoessertEtAl", is a derivative of "internalclock" by Takeru Honda and Tadashi Yamazaki used under CC-BY (http://creativecommons.org/licenses/by/3.0/). 
The code of "internalclock" was downloaded from https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=115966
"ClosedLoopRoessertEtAl" is licensed under CC BY by Christian Rössert.
*/

//
// Simulation program
//

#include "ifun2re.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Period parameters */
#define NR 624
#define MR 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UMASK 0x80000000UL /* most significant w-r bits */
#define LMASK 0x7fffffffUL /* least significant r bits */
#define MIXBITS(u,v) ( ((u) & UMASK) | ((v) & LMASK) )
#define TWIST(u,v) ((MIXBITS(u,v) >> 1) ^ ((v)&1UL ? MATRIX_A : 0UL))

#define iid(i,t) ((t)+T*(i))
#define zid(t,i) ((i)+NN*(t))
#define wid(n,c) ((n)+N*(c))
#define widv(n,c) ((n)+Nv*(c))
#define widu(n,c) ((n)+Nu*(c))

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

    for (j=NR-MR+1; --j; p++)
        *p = p[MR] ^ TWIST(p[0], p[1]);

    for (j=MR; --j; p++)
        *p = p[MR-NR] ^ TWIST(p[0], p[1]);

    *p = p[MR-NR] ^ TWIST(p[0], state[0]);
}

/* generates a random number on [0,0xffffffff]-interval */
unsigned long genrand_int32(void)
{
    unsigned long y;

    if (--left == 0) next_state();
    y = *next++;

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
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

int *random_matrix_index(int M, int N, int C) // M->N
{
  int c, n, x, i;
  int *w;
  int cells[M];

  w = (int *)malloc(N*C*sizeof(int));

  for(n = 0; n < N; n++)
  {

    for(x=0;x<M;x++)
        cells[x] = 0;

    for(c = 0; c < C; c++)
    {
      for(;;)                 /* loop until a valid card is drawn */
      {
        i = genrand_int32() % M;     /* generate random drawn */
        if(cells[i] == 0)        /* has cell been drawn? */
        {
          cells[i] = 1;        /* show that cell is drawn */
          w[wid(n,c)] = i;    /* save in vector */
          //fprintf(stdout, " wid(n,c):%i ",wid(n,c));
          //fprintf(stdout, " %i:%i:%i ", n,c,i);
          break;              /* end the loop */
        }
      }                       /* repeat loop until valid card found */
    }
  }
  //fprintf(stdout, "\n");
  return w;
}


// This function returns the T*N vector of neural activity z(t,i) (p. 1033).
// The vector is accessed as a T*N matrix through the function zid(t,i).
double *activity_pattern(int T, int *N)
{
  int t, i;
  double *z;
  int NN;
  NN = (N[0]+N[1]);

  z = (double *)malloc(T*NN*sizeof(double));
  for(t = 0; t < T; t++){
    for(i = 0; i < NN; i++){
      z[zid(t,i)] = 0;
    }
  }
  return z;
}


// This function takes the random matrix w and the empty array of
// the neural activity z, and fill the array z.
void run(const int *wv, const int *wu, double *z, double *ih, double *I, double *tau, int *N, double *kappa, int *C, int T, double *rand, double *x, int *nfilt, double *It)
{
  int t, n, c, f;
  int Nu, Nv, NN;
  Nu = N[0];
  Nv = N[1];
  NN = Nu+Nv;

  double decayinh, decayex, decaym, coefex0, coefinh0, coefm0, varex0, varinh0, varm0, coeftemp;
  decayinh = exp(-1.0/tau[0]);
  decayex = exp(-1.0/tau[1]);
  decaym = exp(-1.0/tau[2]);

  coefinh0 = 2.0*kappa[0]/C[0];
  varinh0 = ih[6];
  coefex0 = 2.0*kappa[1]/C[1];
  varex0 = ih[7];
  coefm0 = 2.0*kappa[2]/C[1];
  varm0 = ih[8];

  double qinh[Nv], qex[Nu], qm[Nu];
  double fu[Nu], fv[Nv];
  double ihu[Nu], ihv[Nv];
  double tempih, tempih_pos, rec;
  double rec_out[nfilt[0]];

  double u[Nu], v[Nv];
  double r;
  int Cinh, Cex;
  Cinh = (int) C[0];
  Cex = (int) C[1];

  double *coefinh, *coefex, *coefm;
  coefinh = (double *)malloc(Nu*Cinh*sizeof(double));
  coefex = (double *)malloc(Nv*Cex*sizeof(double));
  coefm = (double *)malloc(Nv*Cex*sizeof(double));

  double *reu, *rev;
  reu = (double *)malloc(Nu*nfilt[0]*sizeof(double));
  rev = (double *)malloc(Nv*nfilt[0]*sizeof(double));

  for(n = 0; n < Nu; n++)
  {
    qex[n] = 0;
    qm[n] = 0;
    init_genrand((n+1)*1);
    if (genrand_real2() < ih[2]) // random ipis/contralateral mf input
    {
	    fu[n] = -1;
    }else{
        fu[n] = 1;
    }
    ihu[n] = ih[0] + (ih[1]*ih[0])*random_normal();
    //fprintf(stdout, "ihu: %i: %f\n ", n, ihu[n]);

    for(c = 0; c < Cinh; c++)
    {
      coeftemp = coefinh0 + (varinh0*coefinh0)*random_normal();
      coefinh[widu(n,c)] = (coeftemp > 0) ? coeftemp : 0;
      //fprintf(stdout, "uinh: %i: %i %f %f\n ", n, c, coeftemp, coefinh[widu(n,c)]);
    }
    for(f = 0; f < nfilt[0]; f++)
    {
        init_genrand((n+1)*nfilt[f+1]*Nu);
        reu[widu(n,f)] = ih[13] + (ih[13]*ih[15])*random_normal();
        if (genrand_real2() > ih[11]) // is inhibitory
            { reu[widu(n,f)] *= -1; }
        if (genrand_real2() > ih[9]) // no feedback
            { reu[widu(n,f)] = 0; }

        //fprintf(stdout, "reu: %i %i: %f\n ", n, f, reu[widu(n,f)]);
    }

  }

  for(n = 0; n < Nv; n++)
  {
    qinh[n] = 0;

    init_genrand((n+1)*10);
    if (genrand_real2() < ih[5]) // random ipis/contralateral mf input
    {
	    fv[n] = -1;
    }else{
        fv[n] = 1;
    }
    ihv[n] = ih[3] + (ih[4]*ih[3])*random_normal();
    //fprintf(stdout, "ihv: %i: %f\n ", n, ihv[n]);

    for(c = 0; c < Cex; c++)
    {
      coeftemp = coefex0 + (varex0*coefex0)*random_normal();
      coefex[widv(n,c)] = (coeftemp > 0) ? coeftemp : 0;
      //fprintf(stdout, "vex: %i: %i %f %f\n ", n, c, coeftemp, coefex[widv(n,c)]);

      coeftemp = coefm0 + (varm0*coefm0)*random_normal();
      coefm[widv(n,c)] = (coeftemp > 0) ? coeftemp : 0;
      //fprintf(stdout, "vm: %i: %i %f %f\n ", n, c, coeftemp, coefm[widv(n,c)]);
    }

    for(f = 0; f < nfilt[0]; f++)
    {
        init_genrand((n+1)*nfilt[f+1]*Nv);
        rev[widv(n,f)] = ih[14] + (ih[14]*ih[16])*random_normal();
        if (genrand_real2() > ih[12]) // is inhibitory
            { rev[widv(n,f)] *= -1; }
        if (genrand_real2() > ih[10]) // no feedback
            { rev[widv(n,f)] = 0; }

        //fprintf(stdout, "rev: %i %i: %f\n ", n, f, rev[widv(n,f)]);
    }
  }

  // Iterative calculation of Eq. (2.1)
  for(t = 1; t < T; t++)
  {
    //fprintf(stdout, "I[iid(2,t)] %i: %f\n ", t, I[iid(2,t)]);

    for(f = 0; f < nfilt[0]; f++)
    {
        rec_out[f] = 0;
    }

    for(n = 0; n < Nu; n++)
    {
      qex[n] = z[zid(t-1,n)] + decayex*qex[n];
      qm[n] = z[zid(t-1,n)] + decaym*qm[n];

      //if (t > 100) {rec_out = rec_out + x[n]*z[zid(t-100,n)];}

      for(f = 0; f < nfilt[0]; f++)
      {
          rec_out[f] = rec_out[f] + x[widu(n,f)] * z[zid(t-1,n)];
      }
      //if (t == 1) {fprintf(stdout, "x[n] %i: %f\n ", n, x[n]);}
    }

    for(f = 0; f < nfilt[0]; f++)
    {
        if (t == 1000) {fprintf(stdout, "rec_out[%i] %i: %f\n ", f, t, rec_out[f]);}
    }

    for(n = 0; n < Nv; n++)
    {
      qinh[n] = z[zid(t-1,n+Nu)] + decayinh*qinh[n];
    }

    for(n = 0; n < Nu; n++)
    {
      r = 0;
      // the list of presynaptic neurons is terminated with -1.
      for(c = 0; c < Cinh; c++)
      {
	      r += coefinh[widu(n,c)] * qinh[wv[widu(n,c)]];
      }
      tempih = ihu[n] + fu[n]*ihu[n]*I[iid(0,t)] + reu[n]; //ih[0]
      tempih_pos = (tempih > 0) ? tempih : 0;

      rec = 0;
      for(f = 0; f < nfilt[0]; f++)
      {
          rec -= reu[widu(n,f)] * (It[iid(f,t)] + It[iid(f,0)]*random_normal() + rec_out[f]);
          //if (t == 1000) {fprintf(stdout, "u: rec %i: %f\n ", f, rec);}
      }

      u[n] = tempih_pos - r + rand[1]*ih[0]*random_normal() + rec;
    }

    for(n = 0; n < Nu; n++)
    {
      z[zid(t,n)] = (u[n] > 0) ? u[n] : 0;
    }

    for(n = 0; n < Nv; n++)
    {
      r = 0;
      // the list of presynaptic neurons is terminated with -1.
      for(c = 0; c < Cex; c++)
      {
	      r += (coefex[widv(n,c)] * qex[wu[widv(n,c)]] - coefm[widv(n,c)] * qm[wu[widv(n,c)]]);
      }
      tempih = ihv[n] + fv[n]*ihv[n]*I[iid(1,t)]; //ih[3]
      tempih_pos = (tempih > 0) ? tempih : 0;

      rec = 0;
      for(f = 0; f < nfilt[0]; f++)
      {
          rec -= rev[widv(n,f)] * (It[iid(f,t)] + It[iid(f,0)]*random_normal() + rec_out[f]);
          //if (t == 1000) {fprintf(stdout, "v: rec %i: %f\n ", f, rec);}
      }

      v[n] = tempih_pos + r + rand[2]*ih[3]*random_normal() + rec;
    }
    for(n = 0; n < Nv; n++)
    {
      z[zid(t,n+Nu)] = (v[n] > 0) ? v[n] : 0;
    }
  }
}


double *ifun2re(double *r, int *N, int T, int *C, double *tau, double *kappa, double *ih, double *I, double *x, int *nfilt, double *It) //const unsigned long
{
  double *z;
  int *wu, *wv;

  //fprintf(stdout, "start");

  init_genrand(r[0]);

  int N1, N0, C0, C1;
  N1 = (int) N[1];
  N0 = (int) N[0];
  C0 = (int) C[0];
  C1 = (int) C[1];

  wv = random_matrix_index(N1, N0, C0);
  wu = random_matrix_index(N0, N1, C1);

  z = activity_pattern(T, N);

  run(wv, wu, z, ih, I, tau, N, kappa, C, T, r, x, nfilt, It);

  free(wu);
  free(wv);
  //fprintf(stdout, "end");
  //fprintf(stdout, "test: %s", z[0]);
  return z;
}
