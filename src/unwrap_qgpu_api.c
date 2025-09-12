// unwrap_qgpu_api.cpp
extern "C" {
    __declspec(dllexport)
    void unwrap_phase_QGPU(float *phase, float *soln, int xsize, int ysize);
}

#include "util.h"
#include "grad.h"
#include "extract.h"
#include "pi.h"
#include <omp.h>
#include <stdlib.h>
#include <string.h>

// Forward declarations from the library
int Residues_serial(float *phase, unsigned char *bitflags, int xsize, int ysize);
void GoldsteinBranchCuts_serial(unsigned char *bitflags, int MaxCutLen, int NumRes, int xsize, int ysize);
int UnwrapAroundCutsGoldstein(float *phase, unsigned char *bitflags, float *soln,
                              int xsize, int ysize, int *path_order);

void unwrap_phase_QGPU(float *phase, float *soln, int xsize, int ysize)
{
    int length = xsize * ysize;

    // Allocate helpers
    unsigned char *bitflags = (unsigned char *)calloc(length, sizeof(unsigned char));
    int *path_order = (int *)calloc(length, sizeof(int));

    // Step 1: detect residues
    int NumRes = Residues_serial(phase, bitflags, xsize, ysize);

    // Step 2: place branch cuts
    int MaxCutLen = (xsize + ysize) / 2;
    GoldsteinBranchCuts_serial(bitflags, MaxCutLen, NumRes, xsize, ysize);

    // Step 3: unwrap around cuts
    UnwrapAroundCutsGoldstein(phase, bitflags, soln, xsize, ysize, path_order);

    // Free memory
    free(bitflags);
    free(path_order);
}
