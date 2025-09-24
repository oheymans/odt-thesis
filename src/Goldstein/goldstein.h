// goldstein.h
#ifndef GOLDSTEIN_H
#define GOLDSTEIN_H

void Gradxy(float *phase, float *gradx, float *grady, int xsize, int ysize);

int Residues_parallel(float *phase, unsigned char *bitflags, int xsize, int ysize);
int Residues_serial(float *phase, unsigned char *bitflags, int xsize, int ysize);

void GoldsteinBranchCuts_parallel(unsigned char *bitflags,
                                  int MaxCutLen, int NumRes,
                                  int xsize, int ysize);

int UnwrapAroundCutsFrontier(float *phase,
                             unsigned char *bitflags,
                             float *soln,
                             int xsize, int ysize,
                             int *path_order,
                             float *grady, float *gradx,
                             int *list, int length);

int UnwrapAroundCutsGoldstein(float *phase,
                              unsigned char *bitflags,
                              float *soln,
                              int xsize, int ysize,
                              int *path_order);

#endif
