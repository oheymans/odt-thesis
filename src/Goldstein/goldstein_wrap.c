#include "goldstein.h"
#include "util.h"
#include "grad.h"
#include "pi.h"
#include <stdlib.h>
#include <omp.h>

void goldstein_unwrap_array(const float *wrapped,
                            float *unwrapped,
                            int xsize,
                            int ysize)
{
    int length = xsize * ysize;
    float *phase, *soln, *grady, *gradx, *mask;
    unsigned char *bitflags;
    int *path_order, *list;

    AllocateFloat(&phase, length, "phase data");
    AllocateFloat(&soln, length, "solution array");
    AllocateFloat(&grady, length, "vertical gradient");
    AllocateFloat(&gradx, length, "horizontal gradient");
    AllocateFloat(&mask, length, "mask array");
    AllocateByte(&bitflags, length, "flag array");
    AllocateInt(&path_order, length, "path order");
    AllocateInt(&list, 2*(xsize+ysize), "list");

    // copy wrapped input
    for (int k=0; k<length; k++) phase[k] = wrapped[k];

    // init mask and bitflags
    for (int k=0; k<length; k++) { mask[k] = 1; bitflags[k] = 0; }

    // gradients
    Gradxy(phase, gradx, grady, xsize, ysize);

    // residues
    int NumRes = Residues_parallel(phase, bitflags, xsize, ysize);

    // branch cuts
    int MaxCutLen = (xsize + ysize)/2;
    GoldsteinBranchCuts_parallel(bitflags, MaxCutLen, NumRes, xsize, ysize);

    // unwrap
    UnwrapAroundCutsFrontier(phase, bitflags, soln,
                             xsize, ysize, path_order,
                             grady, gradx, list, length);

    // copy result
    for (int k=0; k<length; k++) unwrapped[k] = soln[k] * TWOPI;

    free(phase); free(soln); free(grady); free(gradx);
    free(mask); free(bitflags); free(path_order); free(list);
}
