#define _USE_MATH_DEFINES // for C
#define SubstractMeanPhasePlane
#include <math.h>

#include <stdio.h>
#include <string.h>
//OpenCV
#include <opencv2/opencv.hpp>

//Serial and Parallel Solvers
#define Solver_Serial
//#define Solver_OpenMP
//#define Solver_XeonPhi
//#define Solver_GPU


#ifdef Solver_Serial
	#include "solver.h"
#endif // Solver_Serial
#ifdef Solver_OpenMP
	#include "solver_OpenMP.h"
#endif // Solver_OpenMP
#ifdef Solver_XeonPhi
	//#include "../XeonPhi_PhaseUnwrappingLibrary/solver_XeonPhi.h"
	#include "../XeonPhi_PhaseUnwrappingLibrary/solver_XeonPhi_simd.h"
#endif // Solver_XeonPhi
#ifdef Solver_GPU
	#include "solver.h"
	#include "../GPU_PhaseUnwrappingLibrary/solver_GPU.h"
#endif // Solver_GPU

class PhaseUnwrapping
{
public:
	PhaseUnwrapping();
	~PhaseUnwrapping();

	void CreateWindows();
	void Show_Results();
	void Save_Results(int argc, char **argv);
	void ProcessPhaseUnwrapping(cv::Mat Src_WrapPhase, cv::Mat Src_Mask, double lambda, double mu, int numIter, int numlevels, int ban_OmegaInit);

	void NestedMultigridUnwrapARM(double lambda, double mu, int numIter, int numlevels, int ban_OmegaInit);
	void NestedMultigridUnwrapARM(unsigned char *UnWrapPhase, unsigned char *iW, unsigned char *jW, unsigned char *WrapResidual, unsigned char *WrapPhase, unsigned char *UnWrapPhaseInit, unsigned char *Mask, unsigned char *phasePlane, double lambda, double mu, int numIter, int numlevels, int ban_OmegaInit, int imageW, int imageH);
	
	void ComputeWdata(cv::Mat phi, double mu);
	void ColumnRowBackwardDifferences(cv::Mat D_column, cv::Mat D_row, cv::Mat Data, int imageW, int imageH);
	void ColumnRowForwardDifferences(cv::Mat D_column, cv::Mat D_row, cv::Mat Data, int imageW, int imageH);
	cv::Mat MultigridUnwrapARM(cv::Mat WrapResidual, cv::Mat Mask, cv::Mat iW, cv::Mat jW, cv::Mat UnWrapInit, double mu, double lambda, int numIter, int numlevels, int ban_OmegaInit, int imageW, int imageH);
	cv::Mat UnwrapARM(cv::Mat G, cv::Mat Msk, cv::Mat iW, cv::Mat jW, cv::Mat F, double mu, double lambda, int numIter, int ban_OmegaInit, int imageW, int imageH);
	void  SubstratePhase(cv::Mat WrapResidual, cv::Mat phaseMap, cv::Mat PhaseRef, cv::Mat Mask, int imageW, int imageH);
	void ReWrappedPhase(cv::Mat ReWrapPhase, cv::Mat UnWrapPhase,cv::Mat Mask, int imageW, int imageH);
	void SubSampleMask(cv::Mat Dst, cv::Mat Src, int imageW, int imageH, int imageW_sub, int imageH_sub);
	void CircShif_x(cv::Mat Dst, cv::Mat Src, int imageW, int imageH);
	void CircShif_y(cv::Mat Dst, cv::Mat Src, int imageW, int imageH);
	void CircShif_xy(cv::Mat Dst, cv::Mat Src, int imageW, int imageH);
	void releaseImages();
	
private:
	//Variables
	int imageH;
	int imageW;
	cv::Mat WrapPhase;
	cv::Mat phasePlane;
	cv::Mat Mask;
	cv::Mat UnWrapPhase;
	cv::Mat UnWrapPhaseInit;
	cv::Mat iW,jW;
	cv::Mat ReWrapPhase;
	cv::Mat WrapResidual;
};