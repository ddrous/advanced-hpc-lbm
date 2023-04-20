#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mm_malloc.h>
#include <omp.h>
#include <mpi.h>
#define NSPEEDS 9
#define FINALSTATEFILE "final_state.dat"
#define AVVELSFILE "av_vels.dat"
#define ICC 
typedef struct
{
  int nx;
  int ny;
  int maxIters;
  int reynolds_dim;
  float density;
  float accel;
  float omega;
} t_param;
typedef struct
{
  float * speeds[NSPEEDS];
} s_speed;
typedef struct
{
  int rank, size, remainder, start, end, row_work, row_start, row_end, down_rank, up_rank;
} m_info;
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, s_speed* cells_ptr, s_speed* tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, m_info* rank_info);
int initialise_global_obstacles(const char* obstaclefile, t_param* params, int** obstacles_ptr);
int compute_rank_info(int rank, int size, m_info* rank_info, t_param params);
float timestep(const t_param params, s_speed* restrict cells, s_speed* restrict tmp_cells, int* obstacles, m_info rank_info, int tot_cells);
int accelerate_flow(const t_param params, const s_speed* restrict cells, const int* obstacles, m_info rank_info);
int exchange_halos(const t_param params, const s_speed* cells, s_speed* tmp_cells, m_info rank_info);
float pro_reb_col_avg(const t_param params, const s_speed* cells, s_speed* tmp_cells, const int* obstacles, m_info rank_info, int tot_cells);
int collate_data(const t_param params, s_speed* cells, m_info rank_info);
int collate_vels(const t_param params, float** av_vels, m_info rank_info);
int write_values(const t_param params, s_speed* cells, int* obstacles, float* av_vels);
int finalise(const t_param* params, s_speed* cells_ptr, s_speed* tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);
float total_density(const t_param params, const s_speed* restrict cells);
int nb_unoccupied_cells(const t_param params, s_speed* cells, int* obstacles, m_info rank_info);
float av_velocity(const t_param params, s_speed* cells, int* obstacles);
void die(const char* message, const int line, const char* file);
void usage(const char* exe);
int main(int argc, char* argv[])
{
  MPI_Init( &argc , &argv);
  char* paramfile = NULL;
  char* obstaclefile = NULL;
  t_param params;
  s_speed cells ;
  s_speed tmp_cells ;
  int* obstacles = NULL;
  float* av_vels = NULL;
  struct timeval timstr;
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc;
  m_info rank_info;
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &rank_info);
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;
  int local_unoc_cells=0, global_unoc_cells=0;
  local_unoc_cells = nb_unoccupied_cells(params, &cells, obstacles, rank_info);
  MPI_Allreduce(&local_unoc_cells, &global_unoc_cells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  for (int tt = 0; tt < params.maxIters; tt++)
    av_vels[tt] = timestep(params, &cells, &tmp_cells, obstacles, rank_info, global_unoc_cells);
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic = comp_toc;
  collate_data(params, &cells, rank_info);
  collate_vels(params, &av_vels, rank_info);
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
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  MPI_Finalize();
  if (rank_info.rank==0){
    printf("\n==done==\n");
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n", init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n", tot_toc - tot_tic);
    initialise_global_obstacles(obstaclefile, &params, &obstacles);
    write_values(params, &cells, obstacles, av_vels);
  }
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
  return EXIT_SUCCESS;
}
int compute_rank_info(int rank, int size, m_info* rank_info, t_param params){
  rank_info->rank = rank;
  rank_info->size = size;
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
  rank_info->row_work = rank_info->row_end - rank_info->row_start;
  rank_info->down_rank = (rank == 0) ? size-1: rank - 1;
  rank_info->up_rank = (rank == size-1) ? 0: rank + 1;
  rank_info->start = params.nx * rank_info->row_start;
  rank_info->end = params.nx * rank_info->row_end;
  return 0;
}
int nb_unoccupied_cells(const t_param params, s_speed* cells, int* obstacles, m_info rank_info)
{
  int tot_cells = 0;
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
float timestep(const t_param params, s_speed* cells, s_speed* tmp_cells, int* obstacles, m_info rank_info, int tot_cells)
{
  if (rank_info.row_start <= params.ny-2 && params.ny-2 < rank_info.row_end)
    accelerate_flow(params, cells, obstacles, rank_info);
  exchange_halos(params, cells, tmp_cells, rank_info);
  float av_vel = pro_reb_col_avg(params, cells, tmp_cells, obstacles, rank_info, tot_cells);
  s_speed tmp = *cells;
  *cells = *tmp_cells;
  *tmp_cells = tmp;
  return av_vel;
}
int accelerate_flow(const t_param params, const s_speed* restrict cells, const int* restrict obstacles, m_info rank_info){
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
  const int jj = params.ny - 2;
  #pragma vector aligned
  #pragma omp parallel for simd
  for (int ii = 0; ii < params.nx; ii++)
  {
    const int global_id = ii + jj*params.nx;
    const int id = (global_id - rank_info.start) + params.nx;
    int cond = ((!obstacles[id - params.nx])
        && ((cells->speeds[3][id] - w1) > 0.f)
        && ((cells->speeds[6][id] - w2) > 0.f)
        && ((cells->speeds[7][id] - w2) > 0.f));
      cells->speeds[1][id] = cond ? cells->speeds[1][id]+ w1: cells->speeds[1][id];
      cells->speeds[5][id] = cond ? cells->speeds[5][id]+ w2: cells->speeds[5][id] ;
      cells->speeds[8][id] = cond ? cells->speeds[8][id]+ w2: cells->speeds[8][id];
      cells->speeds[3][id] = cond ? cells->speeds[3][id]- w1: cells->speeds[3][id];
      cells->speeds[6][id] = cond ? cells->speeds[6][id]- w2: cells->speeds[6][id];
      cells->speeds[7][id] = cond ? cells->speeds[7][id]- w2: cells->speeds[7][id];
  }
  return 0;
}
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
  const float c_sq = 1.f / 3.f;
  const float w0_ = 4.f / 9.f;
  const float w1_ = 1.f / 9.f;
  const float w2_ = 1.f / 36.f;
  const float val1 = 2.f * c_sq * c_sq;
  const float val2 = 2.f * c_sq;
  float tot_u = 0.f;
  #pragma vector aligned
  #pragma omp parallel for collapse(1) reduction(+:tot_u)
  for (int jj = rank_info.row_start; jj < rank_info.row_end; jj++)
  {
    float tot_u_tmp = 0.f;
    #pragma omp simd reduction(+:tot_u_tmp)
    for (int ii = 0; ii < params.nx; ii++)
    {
      float tmp_speeds[NSPEEDS];
      const int r_id = (jj - rank_info.row_start) + 1;
      const int global_id = ii + jj*params.nx;
      const int id = (global_id - rank_info.start) + params.nx;
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = r_id + 1;
      int x_e = (ii + 1) % params.nx;
      int y_s = r_id - 1;
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      tmp_speeds[0] = cells->speeds[0][ii + r_id*params.nx];
      tmp_speeds[1] = cells->speeds[1][x_w + r_id*params.nx];
      tmp_speeds[2] = cells->speeds[2][ii + y_s*params.nx];
      tmp_speeds[3] = cells->speeds[3][x_e + r_id*params.nx];
      tmp_speeds[4] = cells->speeds[4][ii + y_n*params.nx];
      tmp_speeds[5] = cells->speeds[5][x_w + y_s*params.nx];
      tmp_speeds[6] = cells->speeds[6][x_e + y_s*params.nx];
      tmp_speeds[7] = cells->speeds[7][x_e + y_n*params.nx];
      tmp_speeds[8] = cells->speeds[8][x_w + y_n*params.nx];
      if (obstacles[id - params.nx])
      {
        tmp_cells->speeds[1][id] = tmp_speeds[3];
        tmp_cells->speeds[2][id] = tmp_speeds[4];
        tmp_cells->speeds[3][id] = tmp_speeds[1];
        tmp_cells->speeds[4][id] = tmp_speeds[2];
        tmp_cells->speeds[5][id] = tmp_speeds[7];
        tmp_cells->speeds[6][id] = tmp_speeds[8];
        tmp_cells->speeds[7][id] = tmp_speeds[5];
        tmp_cells->speeds[8][id] = tmp_speeds[6];
      }
      else
      {
        float local_density = tmp_speeds[0] + tmp_speeds[1] + tmp_speeds[2] + tmp_speeds[3]
                          + tmp_speeds[4] + tmp_speeds[5] + tmp_speeds[6] + tmp_speeds[7] + tmp_speeds[8];
        float u_x = (tmp_speeds[1]
                      + tmp_speeds[5]
                      + tmp_speeds[8]
                      - (tmp_speeds[3]
                         + tmp_speeds[6]
                         + tmp_speeds[7]))
                     / local_density;
        float u_y = (tmp_speeds[2]
                      + tmp_speeds[5]
                      + tmp_speeds[6]
                      - (tmp_speeds[4]
                         + tmp_speeds[7]
                         + tmp_speeds[8]))
                     / local_density;
        float u_sq = u_x * u_x + u_y * u_y;
        float u[NSPEEDS];
        u[1] = u_x;
        u[2] = u_y;
        u[3] = - u_x;
        u[4] = - u_y;
        u[5] = u_x + u_y;
        u[6] = - u_x + u_y;
        u[7] = - u_x - u_y;
        u[8] = u_x - u_y;
        float d_equ[NSPEEDS];
        d_equ[0] = w0_ * local_density
                   * (1.f - u_sq / (2.f * c_sq));
        for (int kk = 1; kk < 5; kk++)
        {
          d_equ[kk] = w1_ * local_density * (1.f + u[kk] / c_sq
                                          + (u[kk] * u[kk]) / val1
                                          - u_sq / val2);
          d_equ[kk+4] = w2_ * local_density * (1.f + u[kk+4] / c_sq
                                          + (u[kk+4] * u[kk+4]) / val1
                                          - u_sq / val2);
        }
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          tmp_cells->speeds[kk][id] = tmp_speeds[kk]
                                                  + params.omega
                                                  * (d_equ[kk] - tmp_speeds[kk]);
        }
        tot_u_tmp += sqrtf(u_sq);
      }
    }
      tot_u += tot_u_tmp;
  }
  return tot_u / (float)tot_cells;
}
int initialise_global_obstacles(const char* obstaclefile, t_param* params, int** obstacles_ptr){
  #ifdef ICC
    _mm_free(*obstacles_ptr);
    *obstacles_ptr = (int *)_mm_malloc(sizeof(int) * (params->ny * params->nx), 64);
  #else
    int * tmp_ptr = (int *)realloc(*obstacles_ptr, sizeof(int) * (params->ny * params->nx));
    *obstacles_ptr = tmp_ptr;
  #endif
  if (*obstacles_ptr == NULL) die("cannot allocate memory for obstacles", __LINE__, __FILE__);
  #pragma omp parallel for
  for (int jj = 0; jj < params->ny; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }
  char message[1024];
  FILE* fp;
  int xx, yy;
  int blocked;
  int retval;
  fp = fopen(obstaclefile, "r");
  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);
    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);
    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }
  fclose(fp);
  return 0;
}
int collate_data(const t_param params, s_speed* cells, m_info rank_info){
  s_speed global_cells;
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
int collate_vels(const t_param params, float** av_vels, m_info rank_info){
    float* av_vels_total = NULL;
    if (rank_info.rank == 0)
      av_vels_total = (float*)malloc(sizeof(float) * params.maxIters);
    MPI_Reduce( *av_vels, av_vels_total, params.maxIters, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD );
    if (rank_info.rank == 0){
      float* tmp = *av_vels;
      *av_vels = av_vels_total;
      av_vels_total = tmp;
      free(av_vels_total);
      av_vels_total = NULL;
    }
  return 0;
}
int exchange_halos(const t_param params, const s_speed* cells, s_speed* tmp_cells, m_info rank_info){
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
float av_velocity(const t_param params, s_speed* cells, int* obstacles)
{
  int tot_cells = 0;
  float tot_u;
  tot_u = 0.f;
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      if (!obstacles[ii + jj*params.nx])
      {
        float local_density = 0.f;
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells->speeds[kk][ii + jj*params.nx];
        }
        float u_x = (cells->speeds[1][ii + jj*params.nx]
                      + cells->speeds[5][ii + jj*params.nx]
                      + cells->speeds[8][ii + jj*params.nx]
                      - (cells->speeds[3][ii + jj*params.nx]
                         + cells->speeds[6][ii + jj*params.nx]
                         + cells->speeds[7][ii + jj*params.nx]))
                     / local_density;
        float u_y = (cells->speeds[2][ii + jj*params.nx]
                      + cells->speeds[5][ii + jj*params.nx]
                      + cells->speeds[6][ii + jj*params.nx]
                      - (cells->speeds[4][ii + jj*params.nx]
                         + cells->speeds[7][ii + jj*params.nx]
                         + cells->speeds[8][ii + jj*params.nx]))
                     / local_density;
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        ++tot_cells;
      }
    }
  }
  return tot_u / tot_cells;
}
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, s_speed* cells_ptr, s_speed* tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, m_info* rank_info)
{
  char message[1024];
  FILE* fp;
  int xx, yy;
  int blocked;
  int retval;
  fp = fopen(paramfile, "r");
  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }
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
  fclose(fp);
  int size, rank;
  MPI_Comm_size( MPI_COMM_WORLD , &size);
  MPI_Comm_rank( MPI_COMM_WORLD , &rank);
  compute_rank_info(rank, size, rank_info, *params);
# 560 "d2q9-bgk_nonewline.c"
  for (int i = 0; i < NSPEEDS; i++)
  {
    #ifdef ICC
      cells_ptr->speeds[i] = (float*)_mm_malloc(sizeof(float) * ((rank_info->row_work+2) * params->nx), 64);
    #else
      cells_ptr->speeds[i] = (float*)malloc(sizeof(float) * ((rank_info->row_work+2) * params->nx));
    #endif
  }
  if (cells_ptr->speeds == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  for (int i = 0; i < NSPEEDS; i++)
  {
    #ifdef ICC
      tmp_cells_ptr->speeds[i] = (float*)_mm_malloc(sizeof(float) * ((rank_info->row_work+2) * params->nx), 64);
    #else
      tmp_cells_ptr->speeds[i] = (float*)malloc(sizeof(float) * ((rank_info->row_work+2) * params->nx));
    #endif
  }
  if (tmp_cells_ptr->speeds == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  #ifdef ICC
      *obstacles_ptr = (int *)_mm_malloc(sizeof(int) * ((rank_info->row_work) * params->nx), 64);
  #else
      *obstacles_ptr = (int *)malloc(sizeof(int) * ((rank_info->row_work) * params->nx));
  #endif
  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density / 9.f;
  float w2 = params->density / 36.f;
  #pragma omp parallel for
  for (int jj = rank_info->row_start; jj < rank_info->row_end; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params->nx; ii++)
    {
      int global_id = ii + jj*params->nx;
      int local_id = (global_id - rank_info->start) + params->nx;
      cells_ptr->speeds[0][local_id] = w0;
      cells_ptr->speeds[1][local_id] = w1;
      cells_ptr->speeds[2][local_id] = w1;
      cells_ptr->speeds[3][local_id] = w1;
      cells_ptr->speeds[4][local_id] = w1;
      cells_ptr->speeds[5][local_id] = w2;
      cells_ptr->speeds[6][local_id] = w2;
      cells_ptr->speeds[7][local_id] = w2;
      cells_ptr->speeds[8][local_id] = w2;
    }
  }
  #pragma omp parallel for
  for (int jj = rank_info->row_start; jj < rank_info->row_end; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx - rank_info->start] = 0;
    }
  }
  fp = fopen(obstaclefile, "r");
  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);
    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);
    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);
    int local_id = xx + yy*params->nx - rank_info->start;
    if (0 <= local_id && local_id < rank_info->row_work*params->nx)
      (*obstacles_ptr)[xx + yy*params->nx - rank_info->start] = blocked;
  }
  fclose(fp);
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);
  return EXIT_SUCCESS;
}
int finalise(const t_param* params, s_speed* cells_ptr, s_speed* tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
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
float calc_reynolds(const t_param params, s_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}
float total_density(const t_param params, const s_speed* restrict cells)
{
  float total = 0.f;
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
int write_values(const t_param params, s_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;
  const float c_sq = 1.f / 3.f;
  float local_density;
  float pressure;
  float u_x;
  float u_y;
  float u;
  fp = fopen(FINALSTATEFILE, "w");
  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      else
      {
        local_density = 0.f;
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells->speeds[kk][ii + jj*params.nx];
        }
        u_x = (cells->speeds[1][ii + jj*params.nx]
               + cells->speeds[5][ii + jj*params.nx]
               + cells->speeds[8][ii + jj*params.nx]
               - (cells->speeds[3][ii + jj*params.nx]
                  + cells->speeds[6][ii + jj*params.nx]
                  + cells->speeds[7][ii + jj*params.nx]))
              / local_density;
        u_y = (cells->speeds[2][ii + jj*params.nx]
               + cells->speeds[5][ii + jj*params.nx]
               + cells->speeds[6][ii + jj*params.nx]
               - (cells->speeds[4][ii + jj*params.nx]
                  + cells->speeds[7][ii + jj*params.nx]
                  + cells->speeds[8][ii + jj*params.nx]))
              / local_density;
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        pressure = local_density * c_sq;
      }
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
