#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void init(double *x, int N, double val);
int check(double *y, int N, double r);

void daxpy(int N, double a, double const *x, double *y) {
  for (int i = 0; i < N; ++i)
  y[i] += a * x[i];
}

int main(int argc, char *argv[]) {

  int N = 1000 * 1000;
  int ntimes = 1000;

  // If command line argument, then set the length of the array
  if (argc == 2) {
    N = atoi(argv[1]);
  }
  if (argc == 3) {
    N = atoi(argv[1]);
    ntimes = atoi(argv[2]);
  }
  printf("running daxpy with %d elements %d times\n", N, ntimes);

  double * x = malloc(sizeof(double) * N);
  double * y = malloc(sizeof(double) * N);

  // Initalise
  init(x, N, 1.0);
  init(y, N, 0.0);

  double tic = omp_get_wtime();
  for (int t = 0; t < ntimes; ++t) {
    daxpy(N, 2.0, x, y);
  }
  double toc = omp_get_wtime();

  int correct = check(y, N, (double)ntimes * (0.0 + 2.0 * 1.0));
  if (!correct) printf("ERROR: RESULT INCORRECT\n");
  else printf("CORRECT\n");

  printf("daxpy took %lfs\n", toc-tic);

  free(x);
  free(y);

  return EXIT_SUCCESS;
}

void init(double *x, int N, double val) {
  for (int i = 0; i < N; ++i) {
    x[i] = val;
  }
}

int check(double *y, int N, double r) {
  for (int i = 0; i < N; ++i) {
    if (fabs(y[i] - r) > 1.0E-12) {
      return 0;
    }
  }
  return 1;
}

