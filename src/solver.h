#pragma once
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int SolverGS_HQresidual(double *U_double, double *iRho_double, double *jRho_double,
                        double *iOmega_double, double *jOmega_double,
                        double *Msk_double, double *iW_double, double *jW_double,
                        double mu, double lambda, double tol,
                        int numIter, int ban_OmegaInit, int imageW, int imageH);

void computeOmega_HQ(double *iOmega, double *jOmega, double *U,
                     double *iRho, double *jRho, double *Msk,
                     double *iW, double *jW,
                     double mu, double lambda, int imageW, int imageH);

void ColumnRowBackwardDifferences(double *D_column, double *D_row, double *Data,
                                  int imageW, int imageH);

void MatrixAddition(double *Dst_C, double *Src_A, double *Src_B,
                    int imageW, int imageH);

double *UnwrapARM(double *G, double *Msk, double *iW, double *jW, double *F,
                  double mu, double lambda, int numIter, int ban_OmegaInit,
                  int imageW, int imageH);
