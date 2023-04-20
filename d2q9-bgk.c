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
**
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
#include <mm_malloc.h>
#include <omp.h>
#include <mpi.h>


#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

#define ICC        /* If using icc compiler, accounts for _mm_malloc, etc. */
// #define DEBUG      /* For debugging */


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


/* struct to hold the 'speed' values - a Structure of Arrays*/
typedef struct
{
  float * speeds[NSPEEDS];
} s_speed;

/* struct to hold all necessary subgrid information for a particular rank */
typedef struct
{
  int rank, size, remainder, start, end, row_work, row_start, row_end, down_rank, up_rank;
} m_info;


/*
** function prototypes: all function apply to each rank's __subgrid__ unless specified so
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, s_speed* cells_ptr, s_speed* tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, m_info* rank_info);

/* initilises the obstacle array, but for the entire grid (done by a single rank)*/
int initialise_global_obstacles(const char* obstaclefile, t_param* params, int** obstacles_ptr);

int compute_rank_info(int rank, int size, m_info* rank_info, t_param params);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
*/
float timestep(const t_param params, s_speed* restrict cells, s_speed* restrict tmp_cells, int* obstacles, m_info rank_info, int tot_cells);
int accelerate_flow(const t_param params, const s_speed* restrict cells, const int* obstacles, m_info rank_info);
int exchange_halos(const t_param params, const s_speed* cells, s_speed* tmp_cells, m_info rank_info);
float pro_reb_col_avg(const t_param params, const s_speed* cells, s_speed* tmp_cells, const int* obstacles, m_info rank_info, int tot_cells);


int collate_data(const t_param params, s_speed* cells, m_info rank_info);
int collate_vels(const t_param params, float** av_vels, m_info rank_info);

int write_values(const t_param params, s_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, s_speed* cells_ptr, s_speed* tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, const s_speed* restrict cells);

/* compute number of unoccupied cells */
int nb_unoccupied_cells(const t_param params, s_speed* cells, int* obstacles, m_info rank_info);

/* compute average velocity */
float av_velocity(const t_param params, s_speed* cells, int* obstacles);

/* calculate Reynolds number */
// float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);



/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{

  MPI_Init( &argc , &argv);


  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  s_speed cells     ;    /* grid containing fluid densities */
  s_speed tmp_cells ;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */
  m_info rank_info;           /* all the information this MPI rank needs for its computation */

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


  /* Allocate space for space local to each MPI rank */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &rank_info);

  /* Print information about the work each rank does (horizontal split) */
  // printf("=== Rank Info === \nRank %-6d TotalRanks %-6d Remainder %-6d RowWork %-6d RowStart %-6d RowEnd %-6d CellStart %-6d   CellEnd %-6d \n", rank_info.rank, rank_info.size, rank_info.remainder, rank_info.row_work, rank_info.row_start, rank_info.row_end, rank_info.start, rank_info.end);


  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;


  /* Computes the number of non-obstructed cells local to each rank */
  int local_unoc_cells=0, global_unoc_cells=0;
  local_unoc_cells = nb_unoccupied_cells(params, &cells, obstacles, rank_info);
  /* Computes the global number of non-obstructed cells */
  MPI_Allreduce(&local_unoc_cells, &global_unoc_cells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  /* ==Main loop==: each rank timesteps and computes a "partial" average velocity */
  for (int tt = 0; tt < params.maxIters; tt++)
    av_vels[tt] = timestep(params, &cells, &tmp_cells, obstacles, rank_info, global_unoc_cells);


  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic = comp_toc;


  /* Collate data from all ranks into rank 0 */
  collate_data(params, &cells, rank_info);
  /* Sum the "partial" average velocities from all ranks */
  collate_vels(params, &av_vels, rank_info);

  /* Print "total" average velocities */
  if (rank_info.rank==0){
    for (int tt = 0; tt < params.maxIters; tt++)
    {
      #ifdef DEBUG
          printf("== AFTER COLLATE : %d==\n", tt);
          printf("av velocity: %.12E\n", av_vels[tt]);
          printf("tot density: %.12E\n", total_density(params, &cells));
      #endif
    }
  }


  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;


  MPI_Finalize();


  if (rank_info.rank==0){
    /* write final values and free memory */
    printf("\n==done==\n");
    // printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);

    /* Make global obstacle pointer and read the entire obstacle file */
    initialise_global_obstacles(obstaclefile, &params, &obstacles);
    write_values(params, &cells, obstacles, av_vels);
  }

  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;

}



/**
 * @brief Computes the subgrid information for a particular rank
 */
int compute_rank_info(int rank, int size, m_info* rank_info, t_param params){

  rank_info->rank = rank;
  rank_info->size = size;

  /* Base number of rows per rank */
  rank_info->row_work = params.ny / size;

  rank_info->row_start = rank * rank_info->row_work;
  rank_info->row_end = rank_info->row_start + rank_info->row_work;

  rank_info->remainder = params.ny % size;

  if (rank_info->remainder != 0){
    if (rank < rank_info->remainder) {
      rank_info->row_start += rank_info->rank;
      rank_info->row_end = rank_info->row_start + rank_info->row_work + 1;
    }
    else {
      rank_info->row_start += rank_info->remainder;
      rank_info->row_end = rank_info->row_start + rank_info->row_work;
    }
  }

  /* Update to get the effective number of rows for this rank */
  rank_info->row_work = rank_info->row_end - rank_info->row_start;

  rank_info->down_rank = (rank == 0) ? size-1: rank - 1;
  rank_info->up_rank = (rank == size-1) ? 0: rank + 1;

  rank_info->start = params.nx * rank_info->row_start;
  rank_info->end = params.nx * rank_info->row_end;

  return 0;
}



/**
 * @brief Computes the number of non-blocked cells in this rank's subgrid
 */
int nb_unoccupied_cells(const t_param params, s_speed* cells, int* obstacles, m_info rank_info)
{
  int tot_cells = 0;  /* no. of cells used in calculation */

  /* loop over all non-blocked cells */
  #pragma omp parallel for reduction(+:tot_cells)
  for (int jj = rank_info.row_start; jj < rank_info.row_end; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
        if (!obstacles[ii + jj*params.nx - rank_info.start])
        ++tot_cells;
    }
  }

  return tot_cells;
}



/**
 * @brief Main loop of the program: accelerates flow, then exhanges halos, then propagates, rebounds and collides
 */
float timestep(const t_param params, s_speed* cells, s_speed* tmp_cells, int* obstacles, m_info rank_info, int tot_cells)
{

  struct timeval timstr;
  double s_tic, s_toc, p_tic, p_toc;

  gettimeofday(&timstr, NULL);
  s_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  if (rank_info.row_start <= params.ny-2 && params.ny-2 < rank_info.row_end)
    accelerate_flow(params, cells, obstacles, rank_info);

  gettimeofday(&timstr, NULL);
  s_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  exchange_halos(params, cells, tmp_cells, rank_info);

  gettimeofday(&timstr, NULL);
  p_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  float av_vel = pro_reb_col_avg(params, cells, tmp_cells, obstacles, rank_info, tot_cells);

  // Swapp pointers
  s_speed tmp = *cells;
  *cells = *tmp_cells;
  *tmp_cells = tmp;

  gettimeofday(&timstr, NULL);
  p_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);


  printf("\n==Amdhal's times==\n");
  printf("Elapsed Serial time:\t\t\t%.6lf (s)\n",    s_toc - s_tic);
  printf("Elapsed Parallel time:\t\t\t%.6lf (s)\n", p_toc - p_tic);

  return av_vel;

}



/**
 * @brief Accelerates flow in the second row of the grid by the corresponding rank only
 */
int accelerate_flow(const t_param params, const s_speed* restrict cells, const int* restrict obstacles, m_info rank_info){

  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  #ifdef ICC
    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      __assume_aligned(cells->speeds[kk], 64);
    }
    __assume_aligned(obstacles, 64);
    __assume((params.nx)%2==0);
  #endif

  /* modify the 2nd row of the grid */
  const int jj = params.ny - 2;

  #pragma vector aligned
  #pragma omp parallel for simd
  for (int ii = 0; ii < params.nx; ii++)
  {
    const int global_id = ii + jj*params.nx;
    const int id = (global_id - rank_info.start) + params.nx;   // local cell id (add params.nx) to account for halo

    int cond = ((!obstacles[id - params.nx])                    // NOTE: no halos in obstacles 
        && ((cells->speeds[3][id] - w1) > 0.f)
        && ((cells->speeds[6][id] - w2) > 0.f)
        && ((cells->speeds[7][id] - w2) > 0.f));

      /* increase 'east-side' densities */
      cells->speeds[1][id] = cond ? cells->speeds[1][id]+ w1: cells->speeds[1][id];
      cells->speeds[5][id] = cond ? cells->speeds[5][id]+ w2: cells->speeds[5][id] ;
      cells->speeds[8][id] = cond ? cells->speeds[8][id]+ w2: cells->speeds[8][id];
      /* decrease 'west-side' densities */
      cells->speeds[3][id] = cond ? cells->speeds[3][id]- w1: cells->speeds[3][id];
      cells->speeds[6][id] = cond ? cells->speeds[6][id]- w2: cells->speeds[6][id];
      cells->speeds[7][id] = cond ? cells->speeds[7][id]- w2: cells->speeds[7][id];
  }

  return 0;

}


/**
 * @brief Fused loop combination of propagate flow, rebound, and collide; a "partial" average velocity is returned 
 */
float pro_reb_col_avg(const t_param params, const s_speed* restrict cells, s_speed* restrict tmp_cells, const int* obstacles, m_info rank_info, int tot_cells)
{

  #ifdef ICC
    for (int kk = 0; kk < NSPEEDS; kk++)
    {
    __assume_aligned(cells->speeds[kk], 64);
    __assume_aligned(tmp_cells->speeds[kk], 64);
    }

    __assume_aligned(obstacles, 64);
    __assume((params.nx)%2==0);
  #endif

  /* compute weighting factors */
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0_ = 4.f / 9.f;  /* weighting factor */
  const float w1_ = 1.f / 9.f;  /* weighting factor */
  const float w2_ = 1.f / 36.f; /* weighting factor */
  const float val1 = 2.f * c_sq * c_sq;
  const float val2 = 2.f * c_sq;

  float tot_u = 0.f;          /* accumulated magnitudes of velocity for each cell */

  /* Fused Loop */
  #pragma vector aligned
  #pragma omp parallel for collapse(1) reduction(+:tot_u)
  for (int jj = rank_info.row_start; jj < rank_info.row_end; jj++)
  {
    float tot_u_tmp = 0.f;    /*temporary mangitude of velocities to allow vectorisation */

    #pragma omp simd reduction(+:tot_u_tmp)
    for (int ii = 0; ii < params.nx; ii++)
    {
      float tmp_speeds[NSPEEDS];    // To hold the temporary speeds for this cell
      const int r_id = (jj - rank_info.row_start) + 1;            // local row id
      const int global_id = ii + jj*params.nx;                    // global cell id
      const int id = (global_id - rank_info.start) + params.nx;   // local cell id = (ii + r_id*params.nx)

      // PROPAGATE

      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = r_id + 1;
      int x_e = (ii + 1) % params.nx;
      int y_s = r_id - 1;
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);

      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_speeds[0] = cells->speeds[0][ii + r_id*params.nx];  /* central cell, no movement */
      tmp_speeds[1] = cells->speeds[1][x_w + r_id*params.nx]; /* east */
      tmp_speeds[2] = cells->speeds[2][ii + y_s*params.nx];   /* north */
      tmp_speeds[3] = cells->speeds[3][x_e + r_id*params.nx]; /* west */
      tmp_speeds[4] = cells->speeds[4][ii + y_n*params.nx];   /* south */
      tmp_speeds[5] = cells->speeds[5][x_w + y_s*params.nx];  /* north-east */
      tmp_speeds[6] = cells->speeds[6][x_e + y_s*params.nx];  /* north-west */
      tmp_speeds[7] = cells->speeds[7][x_e + y_n*params.nx];  /* south-west */
      tmp_speeds[8] = cells->speeds[8][x_w + y_n*params.nx];  /* south-east */


      if (obstacles[id - params.nx])    // Remember: obstacles don't have halos, so decrement id
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cells->speeds[1][id] = tmp_speeds[3];
        tmp_cells->speeds[2][id] = tmp_speeds[4];
        tmp_cells->speeds[3][id] = tmp_speeds[1];
        tmp_cells->speeds[4][id] = tmp_speeds[2];
        tmp_cells->speeds[5][id] = tmp_speeds[7];
        tmp_cells->speeds[6][id] = tmp_speeds[8];
        tmp_cells->speeds[7][id] = tmp_speeds[5];
        tmp_cells->speeds[8][id] = tmp_speeds[6];
      }

      // COLLISION

      /* don't consider occupied cells */
      else    // if (!obstacles[id- params.nx])
      {
        float local_density = tmp_speeds[0] + tmp_speeds[1] + tmp_speeds[2] + tmp_speeds[3]
                          + tmp_speeds[4] + tmp_speeds[5] + tmp_speeds[6] + tmp_speeds[7] + tmp_speeds[8];

        /* compute x velocity component */
        float u_x = (tmp_speeds[1]
                      + tmp_speeds[5]
                      + tmp_speeds[8]
                      - (tmp_speeds[3]
                         + tmp_speeds[6]
                         + tmp_speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (tmp_speeds[2]
                      + tmp_speeds[5]
                      + tmp_speeds[6]
                      - (tmp_speeds[4]
                         + tmp_speeds[7]
                         + tmp_speeds[8]))
                     / local_density;

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

        /* equilibrium densities */
        float d_equ[NSPEEDS];

        /* zero velocity density: weight w0_ */
        d_equ[0] = w0_ * local_density
                   * (1.f - u_sq / (2.f * c_sq));

        // #pragma omp simd
        for (int kk = 1; kk < 5; kk++)
        {
          d_equ[kk] = w1_ * local_density * (1.f + u[kk] / c_sq
                                          + (u[kk] * u[kk]) / val1
                                          - u_sq / val2);
          d_equ[kk+4] = w2_ * local_density * (1.f + u[kk+4] / c_sq
                                          + (u[kk+4] * u[kk+4]) / val1
                                          - u_sq / val2);
        }

        // #pragma omp simd
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          tmp_cells->speeds[kk][id] = tmp_speeds[kk]
                                                  + params.omega
                                                  * (d_equ[kk] - tmp_speeds[kk]);
        }

        // AVERAGE VELOCITY
        tot_u_tmp += sqrtf(u_sq);

      }
    }
      tot_u += tot_u_tmp;

  }

  return tot_u / (float)tot_cells;

}



/**
 * @brief Re-initialise the obstacles array; globally this time ! 
 *        This avoids gathering back local obstacles at the moment of writing results to file
 */
int initialise_global_obstacles(const char* obstaclefile, t_param* params, int** obstacles_ptr){

  #ifdef ICC
    _mm_free(*obstacles_ptr);
    *obstacles_ptr = (int *)_mm_malloc(sizeof(int) * (params->ny * params->nx), 64);
    // int * tmp_ptr = (int *)_mm_realloc(sizeof(int) * (params->ny * params->nx), 64);
    // *obstacles_ptr = tmp_ptr;
  #else
    int * tmp_ptr = (int *)realloc(*obstacles_ptr, sizeof(int) * (params->ny * params->nx));
    *obstacles_ptr = tmp_ptr;
  #endif
  if (*obstacles_ptr == NULL) die("cannot allocate memory for obstacles", __LINE__, __FILE__);

  /* first set all cells in obstacle array to zero */
  #pragma omp parallel for
  for (int jj = 0; jj < params->ny; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;   // Obstacles don't deal with halos
    }
  }

  /* open the obstacle data file */
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

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

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  return 0;
}


/**
 * @brief Gather all cell data into main rank
 */
int collate_data(const t_param params, s_speed* cells, m_info rank_info){

  s_speed global_cells;    /* contains arrays big enough to handle the entire grid */
  int received_counts[rank_info.size], displacements[rank_info.size];

  if (rank_info.rank == 0){ 
    for (int kk = 0; kk < NSPEEDS; kk++)
      {
        #ifdef ICC
          global_cells.speeds[kk] = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
        #else
          global_cells.speeds[kk] = (float*)malloc(sizeof(float) * (params.ny * params.nx));
        #endif
      }
    if (global_cells.speeds == NULL)
      die("cannot allocate memory for global cells", __LINE__, __FILE__);


    /* Compute count and displacement info for collating data */
    for (int src = 0; src < rank_info.size; src++)
    {
      m_info src_info;
      compute_rank_info(src, rank_info.size, &src_info, params);
      received_counts[src] = src_info.row_work*params.nx;
      displacements[src] = src_info.start;
    }
  }

  for (int kk = 0; kk < NSPEEDS; kk++){
    MPI_Gatherv(&cells->speeds[kk][params.nx], 
                rank_info.row_work*params.nx,
                MPI_FLOAT, 
                &global_cells.speeds[kk][0], 
                received_counts,
                displacements, 
                MPI_FLOAT,
                0, 
                MPI_COMM_WORLD);
  }

  if (rank_info.rank == 0){ 
    /* Swapp local and global cells (which is freed), 
    while cells will be used for writing results to file*/
    s_speed tmp = *cells;
    *cells = global_cells;
    global_cells = tmp;

    for (int kk = 0; kk < NSPEEDS; kk++){
      #ifdef ICC
        _mm_free(global_cells.speeds[kk]);
      #else
        free(global_cells.speeds[kk]);
      #endif
      global_cells.speeds[kk] = NULL;
    }
  }

  return 0;

}


/**
 * @brief Sums the average velocities
 */
int collate_vels(const t_param params, float** av_vels, m_info rank_info){

    float* av_vels_total = NULL;   /* Per-timestep av_vels representative of the whole grid */

    if (rank_info.rank == 0)
      av_vels_total = (float*)malloc(sizeof(float) * params.maxIters);

    MPI_Reduce( *av_vels, av_vels_total, params.maxIters, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD );

    if (rank_info.rank == 0){
      /* Swap av_vels with av_vels_total (for writing to file later)*/
      float* tmp = *av_vels;
      *av_vels = av_vels_total;
      av_vels_total = tmp;

      free(av_vels_total);
      av_vels_total = NULL;
    }

  return 0;

}




/**
 * @brief Exchange halo regions: up (north) and down (south) rows 
 */
int exchange_halos(const t_param params, const s_speed* cells, s_speed* tmp_cells, m_info rank_info){

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






/**
 * @brief Computes average velocity for current state of the cells (final time, not needed !)
 */
float av_velocity(const t_param params, s_speed* cells, int* obstacles)
{
  int tot_cells = 0;    /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells->speeds[kk][ii + jj*params.nx];
        }

        /* x-component of velocity */
        float u_x = (cells->speeds[1][ii + jj*params.nx]
                      + cells->speeds[5][ii + jj*params.nx]
                      + cells->speeds[8][ii + jj*params.nx]
                      - (cells->speeds[3][ii + jj*params.nx]
                         + cells->speeds[6][ii + jj*params.nx]
                         + cells->speeds[7][ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells->speeds[2][ii + jj*params.nx]
                      + cells->speeds[5][ii + jj*params.nx]
                      + cells->speeds[6][ii + jj*params.nx]
                      - (cells->speeds[4][ii + jj*params.nx]
                         + cells->speeds[7][ii + jj*params.nx]
                         + cells->speeds[8][ii + jj*params.nx]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / tot_cells;
}




/**
 * @brief Initialise all arrays using their rank information appropriately
 */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, s_speed* cells_ptr, s_speed* tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, m_info* rank_info)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
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



  /* Initialize MPI information */
  int size, rank;
  MPI_Comm_size( MPI_COMM_WORLD , &size);
  MPI_Comm_rank( MPI_COMM_WORLD , &rank);
  compute_rank_info(rank, size, rank_info, *params);


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
  */

  /* main grid */
  for (int i = 0; i < NSPEEDS; i++)
  {
    #ifdef ICC
      cells_ptr->speeds[i] = (float*)_mm_malloc(sizeof(float) * ((rank_info->row_work+2) * params->nx), 64);
    #else
      cells_ptr->speeds[i] = (float*)malloc(sizeof(float) * ((rank_info->row_work+2) * params->nx));
    #endif
  }
  if (cells_ptr->speeds == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  for (int i = 0; i < NSPEEDS; i++)
  {
    #ifdef ICC
      tmp_cells_ptr->speeds[i] = (float*)_mm_malloc(sizeof(float) * ((rank_info->row_work+2) * params->nx), 64);
    #else
      tmp_cells_ptr->speeds[i] = (float*)malloc(sizeof(float) * ((rank_info->row_work+2) * params->nx));
    #endif
  }
  if (tmp_cells_ptr->speeds == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* the map of obstacles */
  #ifdef ICC
      *obstacles_ptr = (int *)_mm_malloc(sizeof(int) * ((rank_info->row_work) * params->nx), 64);
  #else
      *obstacles_ptr = (int *)malloc(sizeof(int) * ((rank_info->row_work) * params->nx));
  #endif
  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  /* Each thread initialises it's data: first touch implementation */
  #pragma omp parallel for
  for (int jj = rank_info->row_start; jj < rank_info->row_end; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params->nx; ii++)
    {
      int global_id = ii + jj*params->nx;                           // id for the entire grid
      int local_id = (global_id - rank_info->start) + params->nx;  	// accounts for shift for halo region

      /* centre */
      cells_ptr->speeds[0][local_id] = w0;
      /* axis directions */
      cells_ptr->speeds[1][local_id] = w1;
      cells_ptr->speeds[2][local_id] = w1;
      cells_ptr->speeds[3][local_id] = w1;
      cells_ptr->speeds[4][local_id] = w1;
      /* diagonals */
      cells_ptr->speeds[5][local_id] = w2;
      cells_ptr->speeds[6][local_id] = w2;
      cells_ptr->speeds[7][local_id] = w2;
      cells_ptr->speeds[8][local_id] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  #pragma omp parallel for
  for (int jj = rank_info->row_start; jj < rank_info->row_end; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx - rank_info->start] = 0;   // Obstacles don't deal with halos
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

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    int local_id = xx + yy*params->nx - rank_info->start;
    if (0 <= local_id && local_id < rank_info->row_work*params->nx)
      (*obstacles_ptr)[xx + yy*params->nx - rank_info->start] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}



/**
 * @brief Mainly frees allocated arrays
 */
int finalise(const t_param* params, s_speed* cells_ptr, s_speed* tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  for (int i = 0; i < NSPEEDS; i++)
  {
    #ifdef ICC
     _mm_free(cells_ptr->speeds[i]);
     _mm_free(tmp_cells_ptr->speeds[i]);
    #else
     free(cells_ptr->speeds[i]);
     free(tmp_cells_ptr->speeds[i]);
    #endif
     cells_ptr->speeds[i] = NULL;
     tmp_cells_ptr->speeds[i] = NULL;
  }

  #ifdef ICC
    _mm_free(*obstacles_ptr);
    _mm_free(*av_vels_ptr);
  #else
    free(*obstacles_ptr);
    free(*av_vels_ptr);
  #endif

  *obstacles_ptr = NULL;
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}

/**
 * @brief Not needed !
 */
float calc_reynolds(const t_param params, s_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, const s_speed* restrict cells)
{
  float total = 0.f;  /* accumulator */

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

/**
 * @brief Write all values to output file: done by a single rank
 */
int write_values(const t_param params, s_speed* cells, int* obstacles, float* av_vels)
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

  for (int jj = 0; jj < params.ny; jj++)
  {
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
