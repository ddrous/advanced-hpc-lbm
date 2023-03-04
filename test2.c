#include <stdlib.h>

int func(float* const restrict cells, const int nx){

    int obstacles[nx];
    int jj = 0;

    float w1=0.1f, w2=0.3f;

    #pragma simd
    for (int ii = 0; ii < nx; ii++)
    {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    // if ((!obstacles[ii + jj*nx])
    //     && ((cells[ii + jj*nx] - w1) > 0.f)
    //     && ((cells[ii + jj*nx] - w2) > 0.f)
    //     && ((cells[ii + jj*nx] - w2) > 0.f))
    // {
    //     /* increase 'east-side' densities */
    //     cells[ii + jj*nx] += w1;
    //     cells[ii + jj*nx] += w2;
    //     cells[ii + jj*nx] += w2;
    //     /* decrease 'west-side' densities */
    //     cells[ii + jj*nx] -= w1;
    //     cells[ii + jj*nx] -= w2;
    //     cells[ii + jj*nx] -= w2;
    // }

    int cond = ((!obstacles[ii + jj*nx])
        && ((cells[ii + jj*nx] - w1) > 0.f)
        && ((cells[ii + jj*nx] - w2) > 0.f)
        && ((cells[ii + jj*nx] - w2) > 0.f));

    // int cond  = obstacles[ii];
    cells[ii] = cond ? cells[ii]+0.2: cells[ii];

    // cells[ii] += 0.2f;

    }

    return 0;

}



int main(){
const int nx = 100;
float *cells = (float*)malloc(sizeof(float)*nx);

// func(cells, nx);
}
