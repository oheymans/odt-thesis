#include <stdio.h>
#include <stdlib.h>
#include "goldstein_core.h"   // declare prototypes there

// Thin wrapper to expose a simple API for Python
void unwrap_phase_QGPU(float *phase, float *soln, int xsize, int ysize) {
    unsigned char *bitflags;
    float *gradx, *grady;
    int *path_order, *list;

    int length = xsize * ysize;

    // Allocate working arrays
    bitflags = (unsigned char*) calloc(length, sizeof(unsigned char));
    gradx = (float*) calloc(length, sizeof(float));
    grady = (float*) calloc(length, sizeof(float));
    path_order = (int*) calloc(length, sizeof(int));
    list = (int*) calloc(2 * (xsize + ysize), sizeof(int));

    // Algorithm steps from goldstein_core.c
    Gradxy(phase, gradx, grady, xsize, ysize);
    int NumRes = Residues_parallel(phase, bitflags, xsize, ysize);
    int MaxCutLen = (xsize + ysize) / 2;
    GoldsteinBranchCuts_parallel(bitflags, MaxCutLen, NumRes, xsize, ysize);
    UnwrapAroundCutsFrontier(phase, bitflags, soln, xsize, ysize,
                             path_order, grady, gradx, list, length);

    // Free
    free(bitflags);
    free(gradx);
    free(grady);
    free(path_order);
    free(list);
}