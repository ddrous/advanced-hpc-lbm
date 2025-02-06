/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8

** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdbool.h>
#include <mpi.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float* speeds[NSPEEDS];
} t_speed;

typedef struct {
    int row_start;   // Starting index of the rows this process is responsible for
    int row_end;     // Ending index of the rows this process is responsible for (exclusive)
    int row_work;    // Total number of rows this process is handling
    int up_rank;     // Rank of the process that is logically "above" this one (for sending data up and receiving from down)
    int down_rank;   // Rank of the process that is logically "below" this one (for sending data down and receiving from up)
} m_info;

/*
** function prototypes
*/
int init_params(const char* paramfile, t_param* params);
/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* obstaclefile,
               t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, m_info rank_info, t_speed** collate_cells_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
void init_m_info(m_info *rank_info, const t_param params, int rank, int size);
int exchange_halos(const t_param params, const t_speed* cells, t_speed* tmp_cells, m_info rank_info);
float timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, m_info rank_info, int tot_cells);
float pro_reb_col_avg(const t_param params, t_speed* __restrict__ cells, t_speed* __restrict__ tmp_cells, const int* __restrict__ obstacles, m_info rank_info, int tot_cells);
static int accelerate_flow(const t_param params, t_speed* __restrict__ cells, const int* __restrict__ obstacles, m_info rank_info);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);
/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_speed** collate_cells_ptr);
int nb_unoccupied_cells(const t_param params, t_speed* cells, int* obstacles, m_info rank_info);


/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);


/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  t_speed* collate_cells = NULL;
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */
  int rank;              /* the rank of this process */
  int size;              /* number of processes in the communicator */
  int tag = 0;           /* scope for adding extra information to a message */
  MPI_Status status;     /* struct used by MPI_Recv */
  m_info rank_info;

  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;

  init_params(paramfile, &params);
  init_m_info(&rank_info, params, rank, size);
  initialise(obstaclefile, params, &cells, &tmp_cells, &obstacles, &av_vels, rank_info, &collate_cells);

  float local_av_vels[params.maxIters];


  /* Computes the number of non-obstructed cells local to each rank */
  int local_unoc_cells=0, global_unoc_cells=0;
  local_unoc_cells = nb_unoccupied_cells(params, cells, obstacles, rank_info);
  /* Computes the global number of non-obstructed cells */
  MPI_Allreduce(&local_unoc_cells, &global_unoc_cells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);


  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;
  for (int tt = 0; tt < params.maxIters; tt++)
  {
    __assume_aligned(cells, 64);
    __assume_aligned(tmp_cells, 64);
    //local_av_vels[tt] = timestep(params, cells, tmp_cells, obstacles, rank_info);
    //float av_vel = 0.0f;
// Accelerate flow
    if (rank_info.row_start <= params.ny-2 && params.ny-2 < rank_info.row_end){
      accelerate_flow(params, cells, obstacles, rank_info);
    }

  // Exchange halos
    exchange_halos(params, cells, tmp_cells, rank_info);
// Main timestep operations (propagate, rebound, collision, and average velocity)
    local_av_vels[tt] = pro_reb_col_avg(params, cells, tmp_cells, obstacles, rank_info, global_unoc_cells);

  // Swapp pointers
    t_speed tmp = *cells;
    *cells = *tmp_cells;
    *tmp_cells = tmp;
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }
  
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 

  // MPI_Reduce(local_av_vels, av_vels, params.maxIters, MPI_FLOAT,
  //         MPI_SUM, 0, MPI_COMM_WORLD);
  
  int* recvcounts = NULL;
  int* displs = NULL;

  recvcounts = (int*)malloc(size * sizeof(int));
  displs = (int*)malloc(size * sizeof(int));
  int rows_per_rank = params.ny / size;
  int remainder = params.ny % size;

  int *collate_obst = malloc(params.nx * params.ny * sizeof(int));
  
  float collate_vels[params.maxIters];

  if (!recvcounts || !displs || !collate_obst) {
    // Handle allocation failure
    fprintf(stderr, "Memory allocation failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
}
  for (int i = 0; i < size; i++) {
    recvcounts[i] = params.nx * ((i < remainder) ? rows_per_rank + 1 : rows_per_rank);
    displs[i] = params.nx * ((i < remainder) ? (rows_per_rank + 1) * i : (remainder * (rows_per_rank + 1)) + (i - remainder) * rows_per_rank);
  }

  for (int kk = 0; kk < NSPEEDS; ++kk) {
      MPI_Gatherv(&(cells->speeds[kk][params.nx]), rank_info.row_work * params.nx, MPI_FLOAT, &(collate_cells->speeds[kk][0]), recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  }
  

  MPI_Gatherv(obstacles, rank_info.row_work * params.nx, MPI_INT, collate_obst, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Reduce(local_av_vels, collate_vels, params.maxIters, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {

    for (int ii = 0; ii < params.maxIters; ii++) {
      av_vels[ii] = collate_vels[ii] / global_unoc_cells;
    }
    /* Total/collate time stops here.*/
    gettimeofday(&timstr, NULL);
    col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    tot_toc = col_toc;
  
    /* write final values and free memory */
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
    write_values(params, collate_cells, obstacles, av_vels);
    free(recvcounts);
    free(displs);
  }
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, &collate_cells);
  free(collate_obst);
  collate_obst = NULL;
  MPI_Finalize();
  return EXIT_SUCCESS;
}


/**
 * @brief Computes the number of non-blocked cells in this rank's subgrid
 */
int nb_unoccupied_cells(const t_param params, t_speed* cells, int* obstacles, m_info rank_info)
{
  int tot_cells = 0;  /* no. of cells used in calculation */

  /* loop over all non-blocked cells */
  #pragma omp parallel for reduction(+:tot_cells)
  for (int jj = rank_info.row_start; jj < rank_info.row_end; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
        if (!obstacles[ii + jj*params.nx - rank_info.row_start*params.nx])
        ++tot_cells;
    }
  }

  return tot_cells;
}



void init_m_info(m_info *rank_info, const t_param params, int rank, int size) {
    // Calculate rows per process and remainder
    int rows_per_rank = params.ny / size;
    int remainder = params.ny % size;

    // Determine the row start and row count for this process
    if (rank < remainder) {
        rank_info->row_start = rank * (rows_per_rank + 1);
        rank_info->row_work = rows_per_rank + 1;
    } else {
        rank_info->row_start = rank * rows_per_rank + remainder;
        rank_info->row_work = rows_per_rank;
    }

    // Calculate the row end
    rank_info->row_end = rank_info->row_start + rank_info->row_work;

    // Set up neighbor ranks
    rank_info->up_rank = (rank == 0) ? size - 1 : rank - 1;
    rank_info->down_rank = (rank + 1) % size;
}

int exchange_halos(const t_param params, const t_speed* cells, t_speed* tmp_cells, m_info rank_info){


  /* Send up and receive from down */
  for (int kk = 0; kk < NSPEEDS; kk++){
    MPI_Sendrecv( &cells->speeds[kk][rank_info.row_work*params.nx] , 
                  params.nx , 
                  MPI_FLOAT , 
                  rank_info.up_rank , 
                  kk , 
                  &cells->speeds[kk][0] , 
                  params.nx , 
                  MPI_FLOAT , 
                  rank_info.down_rank , 
                  kk , 
                  MPI_COMM_WORLD , MPI_STATUS_IGNORE);
  }


  /* Send down and receive from up */
  for (int kk = 0; kk < NSPEEDS; kk++){
    MPI_Sendrecv( &cells->speeds[kk][1*params.nx] , 
                  params.nx , 
                  MPI_FLOAT , 
                  rank_info.down_rank , 
                  kk , 
                  &cells->speeds[kk][(rank_info.row_work+1)*params.nx] , 
                  params.nx , 
                  MPI_FLOAT , 
                  rank_info.up_rank , 
                  kk , 
                  MPI_COMM_WORLD , MPI_STATUS_IGNORE);
  }


  return 0;
}


float timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, m_info rank_info, int tot_cells)
{
  float av_vel = 0.0f;
// Accelerate flow
  if (rank_info.row_start <= params.ny-2 && params.ny-2 < rank_info.row_end)
    accelerate_flow(params, cells, obstacles, rank_info);


  // Exchange halos
  exchange_halos(params, cells, tmp_cells, rank_info);
// Main timestep operations (propagate, rebound, collision, and average velocity)
  av_vel = pro_reb_col_avg(params, cells, tmp_cells, obstacles, rank_info, tot_cells);


  // Swapp pointers
  t_speed tmp = *cells;
  *cells = *tmp_cells;
  *tmp_cells = tmp;


  return av_vel;
}

float pro_reb_col_avg(const t_param params, t_speed* __restrict__ cells, t_speed* __restrict__ tmp_cells, const int* __restrict__ obstacles, m_info rank_info, int tot_cells)
{
  const float c_sq = 3.f; /* square of speed of sound */
  const float csq =  c_sq * 0.5;
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  float tot_u = 0.f;          /* accumulated magnitudes of velocity for each cell */
  /* initialise */
  __assume((params.nx)%128==0);
  __assume((params.ny)%128==0);
  __assume_aligned(cells, 64);
  __assume_aligned(tmp_cells, 64);
  for (int jj = 1; jj < rank_info.row_work + 1; jj++)
  {
    #pragma omp simd reduction(+:tot_u)
    for (int ii = 0; ii < params.nx; ii++)
    {
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj - 1 + params.ny) % params.ny;
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      int current = ii + jj*params.nx;

      float currentspeeds[NSPEEDS];
      currentspeeds[0] = cells->speeds[0][current];
      currentspeeds[1] = cells->speeds[1][x_w + jj*params.nx];
      currentspeeds[2] = cells->speeds[2][ii + y_s*params.nx];
      currentspeeds[3] = cells->speeds[3][x_e + jj*params.nx];
      currentspeeds[4] = cells->speeds[4][ii + y_n*params.nx];
      currentspeeds[5] = cells->speeds[5][x_w + y_s*params.nx];
      currentspeeds[6] = cells->speeds[6][x_e + y_s*params.nx];
      currentspeeds[7] = cells->speeds[7][x_e + y_n*params.nx];
      currentspeeds[8] = cells->speeds[8][x_w + y_n*params.nx];
      /* compute local density total */
      float local_density = 0.f;
      
      #pragma omp simd
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        local_density += currentspeeds[kk];
      }

      /* compute x velocity component */
      float u_x = (currentspeeds[1]
                    + currentspeeds[5]
                    + currentspeeds[8]
                    - (currentspeeds[3]
                       + currentspeeds[6]
                       + currentspeeds[7]));
      /* compute y velocity component */
      float u_y = (currentspeeds[2]
                    + currentspeeds[5]
                    + currentspeeds[6]
                    - (currentspeeds[4]
                       + currentspeeds[7]
                        + currentspeeds[8]));

      /* velocity squared */
      float u_sq = u_x * u_x + u_y * u_y;

      /* directional velocity components */
      float u[NSPEEDS];
      u[1] =   u_x;        /* east */
      u[2] =         u_y;  /* north */
      u[3] = - u_x;        /* west */
      u[4] =       - u_y;  /* south */
      u[5] =   u_x + u_y;  /* north-east */
      u[6] = - u_x + u_y;  /* north-west */
      u[7] = - u_x - u_y;  /* south-west */
      u[8] =   u_x - u_y;  /* south-east */

      float inv_ld = 1.f / local_density;
      float common = local_density - u_sq * csq * inv_ld;

      float ux = u_x * inv_ld;
      float uy = u_y * inv_ld;
      
      bool mask = obstacles[current];

      //tot_cells_ += !mask;
      tmp_cells->speeds[0][current] = mask * currentspeeds[0] + !mask * (currentspeeds[0]
                                                + params.omega
                                                * (w0 * common
                                                 - currentspeeds[0]));

      tmp_cells->speeds[1][current] = mask * currentspeeds[3] + !mask * (currentspeeds[1]
                                               + params.omega
                                               * (w1 * (common + (u[1] * c_sq) * (1 + u[1] * csq * inv_ld))
                                                - currentspeeds[1]));
        
      tmp_cells->speeds[2][current] = mask * currentspeeds[4] + !mask * (currentspeeds[2]
                                               + params.omega
                                               * (w1 * (common + (u[2] * c_sq) * (1 + u[2] * csq * inv_ld))
                                                - currentspeeds[2]));
                                            
      tmp_cells->speeds[3][current] = mask * currentspeeds[1] + !mask * (currentspeeds[3]
                                               + params.omega
                                               * (w1 * (common + (u[3] * c_sq) * (1 + u[3] * csq * inv_ld))
                                                - currentspeeds[3]));
        
      tmp_cells->speeds[4][current] = mask * currentspeeds[2] + !mask * (currentspeeds[4]
                                               + params.omega
                                               * (w1 * (common + (u[4] * c_sq) * (1 + u[4] * csq * inv_ld))
                                                - currentspeeds[4]));

      tmp_cells->speeds[5][current] = mask * currentspeeds[7] + !mask * (currentspeeds[5]
                                                + params.omega
                                                * (w2 * (common + (u[5] * c_sq) * (1 + u[5] * csq * inv_ld))
                                                 - currentspeeds[5]));

      tmp_cells->speeds[6][current] = mask * currentspeeds[8] + !mask * (currentspeeds[6]
                                                + params.omega
                                                * (w2 * (common + (u[6] * c_sq) * (1 + u[6] * csq * inv_ld))
                                                 - currentspeeds[6]));
        
      tmp_cells->speeds[7][current] = mask * currentspeeds[5] + !mask * (currentspeeds[7]
                                                + params.omega
                                                * (w2 * (common + (u[7] * c_sq) * (1 + u[7] * csq * inv_ld))
                                                 - currentspeeds[7]));

      tmp_cells->speeds[8][current] = mask * currentspeeds[6] + !mask * (currentspeeds[8]
                                                + params.omega
                                                * (w2 * (common + (u[8] * c_sq) * (1 + u[8] * csq * inv_ld))
                                                 - currentspeeds[8]));

      tot_u += !mask * sqrtf((ux * ux) + (uy * uy));
      // if (tot_u != 0)
      // {
      //   printf("totu: %f", tot_u);
      // }
      }
    }
  return tot_u / tot_cells;
}

static int accelerate_flow(const t_param params, t_speed* __restrict__ cells, const int* __restrict__ obstacles, m_info rank_info)
{
    /* compute weighting factors */
  const float w1 = params.density * params.accel / 9.f;
  const float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  const int jj = params.ny - 2 - rank_info.row_start;

  // printf("accelerating row: %d \n row to accelerate: %d \n", jj, params.ny - 2);
  
  __assume_aligned(cells, 64);
  __assume((params.nx)%128==0);
  #pragma omp simd
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    int current = ii + jj*params.nx;
    if (!obstacles[current]
        && (cells->speeds[3][current] - w1) > 0.f
        && (cells->speeds[6][current] - w2) > 0.f
        && (cells->speeds[7][current] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->speeds[1][current] += w1;
      cells->speeds[5][current] += w2;
      cells->speeds[8][current] += w2;
      /* decrease 'west-side' densities */
      cells->speeds[3][current] -= w1;
      cells->speeds[6][current] -= w2;
      cells->speeds[7][current] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  __assume_aligned(cells, 64);
  __assume((params.nx)%128==0);
  __assume((params.ny)%128==0);
  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      int current = ii + jj*params.nx;
      /* ignore occupied cells */
      if (!obstacles[current])
      {
        /* local density total */
        float local_density = 0.f;
        
        #pragma omp simd
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells->speeds[kk][current];
        }

        /* x-component of velocity */
        float u_x = (cells->speeds[1][current]
                      + cells->speeds[5][current]
                      + cells->speeds[8][current]
                      - (cells->speeds[3][current]
                         + cells->speeds[6][current]
                         + cells->speeds[7][current]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells->speeds[2][current]
                      + cells->speeds[5][current]
                      + cells->speeds[6][current]
                      - (cells->speeds[4][current]
                         + cells->speeds[7][current]
                         + cells->speeds[8][current]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase current of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int init_params(const char* paramfile, t_param* params)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    retval;         /* to hold return value for checking */
    fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);
  return EXIT_SUCCESS;
}

int initialise(const char* obstaclefile,
               t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, m_info rank_info, t_speed** collate_cells_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */
  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs./
  */

  /* main grid */
  *cells_ptr = (t_speed*)_mm_malloc(sizeof(t_speed), 64);
  if (*cells_ptr == NULL) die("cannot allocate memory for t_speed structure", __LINE__, __FILE__);

  *tmp_cells_ptr = (t_speed*)_mm_malloc(sizeof(t_speed), 64);
  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for t_speed structure", __LINE__, __FILE__);
  
  *collate_cells_ptr = (t_speed*)_mm_malloc(sizeof(t_speed), 64);
  if (*collate_cells_ptr == NULL) die("cannot allocate memory for t_speed structure", __LINE__, __FILE__);

  for (int i = 0; i < NSPEEDS; ++i) {
    (*cells_ptr)->speeds[i] = (float*)_mm_malloc((rank_info.row_work + 2) * params.nx * sizeof(float), 64);
    if ((*cells_ptr)->speeds[i] == NULL) die("cannot allocate memory for cells speeds", __LINE__, __FILE__);

    (*tmp_cells_ptr)->speeds[i] = (float*)_mm_malloc((rank_info.row_work + 2) * params.nx * sizeof(float), 64);
    if ((*tmp_cells_ptr)->speeds[i] == NULL) die("cannot allocate memory for tmp_cells speeds", __LINE__, __FILE__);
    
    (*collate_cells_ptr)->speeds[i] = (float*)_mm_malloc(params.ny * params.nx * sizeof(float), 64);
    if ((*collate_cells_ptr)->speeds[i] == NULL) die("cannot allocate memory for collate_cells speeds", __LINE__, __FILE__);
  }

  /* the map of obstacles */
  *obstacles_ptr =  malloc(sizeof(int) * ((rank_info.row_work + 2) * params.nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params.density * 4.f / 9.f;
  float w1 = params.density      / 9.f;
  float w2 = params.density      / 36.f;
  __assume_aligned(*cells_ptr, 64);
  __assume((params.nx)%128==0);
  __assume((params.ny)%128==0);
  for (int jj = 0; jj < rank_info.row_work + 2; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      if (jj == rank_info.row_work + 1 || jj == 0)
      {
        (*cells_ptr)->speeds[0][ii + jj * params.nx] = 0;
        (*cells_ptr)->speeds[1][ii + jj * params.nx] = 0;
        (*cells_ptr)->speeds[2][ii + jj * params.nx] = 0;
        (*cells_ptr)->speeds[3][ii + jj * params.nx] = 0;
        (*cells_ptr)->speeds[4][ii + jj * params.nx] = 0;
        (*cells_ptr)->speeds[5][ii + jj * params.nx] = 0;
        (*cells_ptr)->speeds[6][ii + jj * params.nx] = 0;
        (*cells_ptr)->speeds[7][ii + jj * params.nx] = 0;
        (*cells_ptr)->speeds[8][ii + jj * params.nx] = 0;
        
        (*tmp_cells_ptr)->speeds[0][ii + jj * params.nx] = 0;
        (*tmp_cells_ptr)->speeds[1][ii + jj * params.nx] = 0;
        (*tmp_cells_ptr)->speeds[2][ii + jj * params.nx] = 0;
        (*tmp_cells_ptr)->speeds[3][ii + jj * params.nx] = 0;
        (*tmp_cells_ptr)->speeds[4][ii + jj * params.nx] = 0;
        (*tmp_cells_ptr)->speeds[5][ii + jj * params.nx] = 0;
        (*tmp_cells_ptr)->speeds[6][ii + jj * params.nx] = 0;
        (*tmp_cells_ptr)->speeds[7][ii + jj * params.nx] = 0;
        (*tmp_cells_ptr)->speeds[8][ii + jj * params.nx] = 0;
      }
      (*cells_ptr)->speeds[0][ii + jj * params.nx] = w0;
      (*cells_ptr)->speeds[1][ii + jj * params.nx] = w1;
      (*cells_ptr)->speeds[2][ii + jj * params.nx] = w1;
      (*cells_ptr)->speeds[3][ii + jj * params.nx] = w1;
      (*cells_ptr)->speeds[4][ii + jj * params.nx] = w1;
      (*cells_ptr)->speeds[5][ii + jj * params.nx] = w2;
      (*cells_ptr)->speeds[6][ii + jj * params.nx] = w2;
      (*cells_ptr)->speeds[7][ii + jj * params.nx] = w2;
      (*cells_ptr)->speeds[8][ii + jj * params.nx] = w2;

      (*tmp_cells_ptr)->speeds[0][ii + jj * params.nx] = w0;
      (*tmp_cells_ptr)->speeds[1][ii + jj * params.nx] = w1;
      (*tmp_cells_ptr)->speeds[2][ii + jj * params.nx] = w1;
      (*tmp_cells_ptr)->speeds[3][ii + jj * params.nx] = w1;
      (*tmp_cells_ptr)->speeds[4][ii + jj * params.nx] = w1;
      (*tmp_cells_ptr)->speeds[5][ii + jj * params.nx] = w2;
      (*tmp_cells_ptr)->speeds[6][ii + jj * params.nx] = w2;
      (*tmp_cells_ptr)->speeds[7][ii + jj * params.nx] = w2;
      (*tmp_cells_ptr)->speeds[8][ii + jj * params.nx] = w2;
    }
  }
  
  __assume((params.nx)%128==0);
  __assume((params.ny)%128==0);
  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < rank_info.row_work; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params.nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params.nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params.ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    if (yy >= rank_info.row_end || yy < rank_info.row_start) continue;

    /* assign to array */
    int conv_yy = yy - rank_info.row_start;
    (*obstacles_ptr)[xx + conv_yy*params.nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params.maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_speed** collate_cells_ptr)
{
  /*
  ** free up allocated memory
  */
  for (int i = 0; i < NSPEEDS; ++i) {
    _mm_free((*cells_ptr)->speeds[i]);
    (*cells_ptr)->speeds[i] = NULL;

    _mm_free((*tmp_cells_ptr)->speeds[i]);
    (*tmp_cells_ptr)->speeds[i] = NULL;
    
    _mm_free((*collate_cells_ptr)->speeds[i]);
    (*collate_cells_ptr)->speeds[i] = NULL;
    }

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  __assume_aligned(cells, 64);
  __assume((params.nx)%128==0);
  __assume((params.ny)%128==0);
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells->speeds[kk][ii + jj*params.nx];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  __assume((params.nx)%128==0);
  __assume((params.ny)%128==0);
  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        #pragma omp simd
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells->speeds[kk][ii + jj*params.nx];
        }

        /* compute x velocity component */
        u_x = (cells->speeds[1][ii + jj*params.nx]
               + cells->speeds[5][ii + jj*params.nx]
               + cells->speeds[8][ii + jj*params.nx]
               - (cells->speeds[3][ii + jj*params.nx]
                  + cells->speeds[6][ii + jj*params.nx]
                  + cells->speeds[7][ii + jj*params.nx]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells->speeds[2][ii + jj*params.nx]
               + cells->speeds[5][ii + jj*params.nx]
               + cells->speeds[6][ii + jj*params.nx]
               - (cells->speeds[4][ii + jj*params.nx]
                  + cells->speeds[7][ii + jj*params.nx]
                  + cells->speeds[8][ii + jj*params.nx]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
