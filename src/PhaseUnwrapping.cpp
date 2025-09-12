/*
Please cite the following if you use this software in any publications:

Hernandez-Lopez, F. J., Rivera, M., Salazar-Garibay A., & Legarda-Saenz R. (2018).
Comparison of multi-hardware parallel implementations for a phase unwrapping algorithm.
Optical Engineering.

Rivera, M., Hernandez-Lopez, F. J., & Gonzalez, A. (2015).
Phase unwrapping by accumulation of residual maps.
Optics and Lasers in Engineering, 64, 51-58. ISSN: 0143-8166.
*/

#include "PhaseUnwrapping.h"

PhaseUnwrapping::PhaseUnwrapping(){

}


void PhaseUnwrapping::CreateWindows(){
	cv::namedWindow("Wrapped phase",CV_WINDOW_NORMAL);
	cv::namedWindow("UnWrapped phase", CV_WINDOW_NORMAL);
	cv::namedWindow("ReWrapped phase", CV_WINDOW_NORMAL);
	cv::namedWindow("Residual phase", CV_WINDOW_NORMAL);
	/*cv::namedWindow("Wrapped phase", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("UnWrapped phase", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("ReWrapped phase", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("Residual phase", CV_WINDOW_AUTOSIZE);*/

	cv::namedWindow("phasePlane", CV_WINDOW_NORMAL);
}
void PhaseUnwrapping::Show_Results(){
	cv::normalize(WrapPhase, WrapPhase, 1.0, 0.0, CV_MINMAX);//to visualization
	cv::normalize(UnWrapPhase, UnWrapPhase, 1.0, 0.0, CV_MINMAX);
	cv::normalize(ReWrapPhase, ReWrapPhase, 1.0, 0.0, CV_MINMAX);
	cv::normalize(WrapResidual, WrapResidual, 1.0, 0.0, CV_MINMAX);
	cv::imshow("Wrapped phase", WrapPhase);
	cv::imshow("UnWrapped phase", UnWrapPhase);
	cv::imshow("ReWrapped phase", ReWrapPhase);
	cv::imshow("Residual phase", WrapResidual);

	cv::normalize(phasePlane, phasePlane, 1.0, 0.0, CV_MINMAX);
	cv::imshow("phasePlane", phasePlane);
}
void PhaseUnwrapping::Save_Results(int argc, char **argv){
	cv::normalize(UnWrapPhase, UnWrapPhase, 255.0, 0.0, CV_MINMAX);
	cv::normalize(ReWrapPhase, ReWrapPhase, 255.0, 0.0, CV_MINMAX);
	cv::normalize(WrapResidual, WrapResidual, 255.0, 0.0, CV_MINMAX);
	/*cv::imwrite("../results/UnWrapPhase.png", UnWrapPhase);
	cv::imwrite("../results/ReWrapPhase.png", ReWrapPhase);
	cv::imwrite("../results/WrapResidual.png", WrapResidual);*/
	//cv::normalize(phasePlane, phasePlane, 255.0, 0.0, CV_MINMAX);
	//cv::imwrite("../results/phasePlane.png", phasePlane);
	char name_Unwrap[100];
	char name_Rewrap[100];
	char input[100];
	sprintf(input,"%s", argv[1]);
	std::string temp(input);
	int pos_s=temp.find_last_of("/");
	#ifdef Solver_Serial
	    strcpy(name_Unwrap, "../results/Serial_UnWrap_");
	    strcpy(name_Rewrap, "../results/Serial_ReWrap_");
	#endif // Solver_Serial
	#ifdef Solver_OpenMP
	    strcpy(name_Unwrap, "../results/multiCore_UnWrap_");
	    strcpy(name_Rewrap, "../results/multiCore_ReWrap_");
	#endif // Solver_OpenMP
	#ifdef Solver_XeonPhi
	    strcpy(name_Unwrap, "../results/XeonPhi_UnWrap_");
	    strcpy(name_Rewrap, "../results/XeonPhi_ReWrap_");
	#endif // Solver_XeonPhi
	#ifdef Solver_GPU
	    strcpy(name_Unwrap, "../results/GPU_UnWrap_");
	    strcpy(name_Rewrap, "../results/GPU_ReWrap_");
	#endif // Solver_GPU
	strcat(name_Unwrap, &(input[pos_s+1]));
	strcat(name_Rewrap, &(input[pos_s + 1]));
	cv::imwrite(name_Unwrap, UnWrapPhase);
	cv::imwrite(name_Rewrap, ReWrapPhase);
}
//Process in double precision
void PhaseUnwrapping::ProcessPhaseUnwrapping(cv::Mat Src_WrapPhase, cv::Mat Src_Mask, double lambda, double mu, int numIter, int numlevels, int ban_OmegaInit){
	imageH = Src_WrapPhase.rows;
	imageW = Src_WrapPhase.cols;
	//Create memory

	WrapPhase.create(imageH, imageW, CV_64FC(1));
	Mask.create(imageH, imageW, CV_64FC(1));
	UnWrapPhaseInit.create(imageH, imageW, CV_64FC(1));
	UnWrapPhase = cv::Mat::zeros(imageH, imageW, CV_64FC(1));
	ReWrapPhase = cv::Mat::zeros(imageH, imageW, CV_64FC(1));
	iW.create(imageH, imageW, CV_64FC(1));
	jW.create(imageH, imageW, CV_64FC(1));
	WrapResidual.create(imageH, imageW, CV_64FC(1));

	phasePlane.create(imageH, imageW, CV_64FC(1));

	//Convert Image
	Src_WrapPhase.convertTo(WrapPhase, CV_64FC(1), 1.0, 0.0);//UChar to double
	Src_Mask.convertTo(Mask, CV_64FC(1), 1.0/255, 0.0);//between (0,1)

	cv::normalize(WrapPhase, WrapPhase, M_PI, -M_PI, CV_MINMAX);//between (-pi,pi)
	cv::multiply(WrapPhase, Mask, WrapPhase, 1.0); 

	//NestedMultigridUnwrapARM(lambda, mu, numIter, numlevels,ban_OmegaInit);

	NestedMultigridUnwrapARM(UnWrapPhase.data, iW.data, jW.data, WrapResidual.data, WrapPhase.data, UnWrapPhaseInit.data, Mask.data,phasePlane.data, lambda, mu, numIter, numlevels, ban_OmegaInit, imageW, imageH);

#ifdef SubstractMeanPhasePlane
	//Add phasePlane to UnWrapPhase
	UnWrapPhase += phasePlane;
#endif // SubstractMeanPhasePlane

	//UnWrapPhase += .9*M_PI;
	ReWrappedPhase(ReWrapPhase, UnWrapPhase,Mask, imageW, imageH);
	//cv::multiply(UnWrapPhase, Mask, UnWrapPhase, 1.0);
}
void PhaseUnwrapping::NestedMultigridUnwrapARM(unsigned char *UnWrapPhase, unsigned char *iW, unsigned char *jW, unsigned char *WrapResidual, unsigned char *WrapPhase, unsigned char *UnWrapPhaseInit, unsigned char *Mask,unsigned char *phasePlane, double lambda, double mu, int numIter, int numlevels, int ban_OmegaInit,int imageW,int imageH){
	double *UnWrapPhase_double = reinterpret_cast<double *>(UnWrapPhase);
	double *iW_double = reinterpret_cast<double *>(iW);
	double *jW_double = reinterpret_cast<double *>(jW);
	double *WrapResidual_double = reinterpret_cast<double *>(WrapResidual);
	double *WrapPhase_double = reinterpret_cast<double *>(WrapPhase);
	double *UnWrapPhaseInit_double = reinterpret_cast<double *>(UnWrapPhaseInit);
	double *Mask_double = reinterpret_cast<double *>(Mask);

	double *phasePlane_double= reinterpret_cast<double *>(phasePlane);

#ifdef Solver_Serial
	#ifdef SubstractMeanPhasePlane
		SubstratePlane(WrapPhase_double, phasePlane_double, WrapPhase_double, Mask_double, imageW, imageH);
	#endif // SubstractMeanPhasePlane
	Solver_NestedMultigridUnwrapARM(UnWrapPhase_double, iW_double, jW_double, WrapResidual_double, WrapPhase_double, UnWrapPhaseInit_double, Mask_double, lambda, mu, numIter, numlevels, ban_OmegaInit, imageW, imageH);
#endif // Solver_Serial

#ifdef Solver_OpenMP
	#ifdef SubstractMeanPhasePlane
		SubstratePlane_OpenMP(WrapPhase_double, phasePlane_double, WrapPhase_double, Mask_double, imageW, imageH);
	#endif // SubstractMeanPhasePlane
	Solver_NestedMultigridUnwrapARM_OpenMP(UnWrapPhase_double, iW_double, jW_double, WrapResidual_double, WrapPhase_double, UnWrapPhaseInit_double, Mask_double, lambda, mu, numIter, numlevels, ban_OmegaInit, imageW, imageH);
#endif // Solver_OpenMP

#ifdef Solver_XeonPhi
	#ifdef SubstractMeanPhasePlane
		SubstratePlane(WrapPhase_double, phasePlane_double, WrapPhase_double, Mask_double, imageW, imageH);
		//SubstratePlane_OpenMP(WrapPhase_double, phasePlane_double, WrapPhase_double, Mask_double, imageW, imageH);
	#endif // SubstractMeanPhasePlane
	Solver_NestedMultigridUnwrapARM_XeonPhi(UnWrapPhase_double, iW_double, jW_double, WrapResidual_double, WrapPhase_double, UnWrapPhaseInit_double, Mask_double, lambda, mu, numIter, numlevels, ban_OmegaInit, imageW, imageH);
#endif // Solver_XeonPhi

#ifdef Solver_GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nRunning in GPU: %s", prop.name);
	#ifdef SubstractMeanPhasePlane
		SubstratePlane(WrapPhase_double, phasePlane_double, WrapPhase_double, Mask_double, imageW, imageH);
		//SubstratePlane_OpenMP(WrapPhase_double, phasePlane_double, WrapPhase_double, Mask_double, imageW, imageH);
	#endif // SubstractMeanPhasePlane
	Solver_NestedMultigridUnwrapARM_GPU(UnWrapPhase_double, iW_double, jW_double, WrapResidual_double, WrapPhase_double, UnWrapPhaseInit_double, Mask_double, lambda, mu, numIter, numlevels, ban_OmegaInit, imageW, imageH);
#endif // Solver_GPU
}
void PhaseUnwrapping::NestedMultigridUnwrapARM(double lambda, double mu, int numIter, int numlevels, int ban_OmegaInit){
	/*%ARM Unwrapping
	% numIter:       number of iteration(heat eq.solver)
	% numlevels:     number of upper levels in the pyramid
	% ban_OmegaInit: initial condition for Omegas
	% example
	% Unwrap = NestedMultigridUnwrapARM(1e-1, pi / 10, 500, 5,1,0)
	%
	%-------------------------------------------------------------------- -
	% Corresponds to Algorithm 3 in
	% Cite:
	% Mariano Rivera, Francisco Hernandez - Lopez and Adonai Gonzalez, "Phase 
	% unwrapping by accumulation of residual maps, " to appear in Optics 
	% and Lasers in Engineering, 2014.
	%
	%-------------------------------------------------------------------- -*/
	int superUplevel;
	cv::Mat Zeros;
	WrapPhase.copyTo(WrapResidual);

	Zeros = cv::Mat::zeros(imageH, imageW, CV_64FC(1));
	
	ComputeWdata(WrapPhase, M_PI / 4);
	//Zeros.copyTo(UnWrapPhaseInit);
	for (superUplevel = numlevels; superUplevel > 0; superUplevel--){
		//printf("\nSuper Up Level: %d", superUplevel);
		Zeros.copyTo(UnWrapPhaseInit);
		
		UnWrapPhase = UnWrapPhase + MultigridUnwrapARM(WrapResidual, Mask, iW, jW, UnWrapPhaseInit, mu, lambda, numIter / 2, numlevels, ban_OmegaInit, imageW, imageH);
		
		SubstratePhase(WrapResidual, WrapPhase, UnWrapPhase, Mask, imageW, imageH);
	}

	//free memory
	Zeros.release();
}

void  PhaseUnwrapping::SubstratePhase(cv::Mat WrapResidual,cv::Mat phaseMap,cv::Mat PhaseRef,cv::Mat Mask,int imageW,int imageH){
/*% PARAMETERS
% phaseMap:     Wrapped phase
% PhaseRed : Phase to substract
% Mask : Binary mask of the region of interest
%
% OUTPUT
% phaseMap2 : Wraped phase with out the best fitted plane
% -------------------------------------------------------------------- -
% Mariano Rivera
% 14 enero 2011
*/
	int i, j;
	double sIphase, cIphase;
	double sref, cref;
	double num, den;
	for (i = 0; i < imageH; i++){
		for (j = 0; j < imageW; j++){
			//% analytic signal for the phase map
			sIphase = sin(phaseMap.at<double>(i, j));
			cIphase = cos(phaseMap.at<double>(i, j));
			//% reference phase
			sref = sin(PhaseRef.at<double>(i,j));
			cref = cos(PhaseRef.at<double>(i,j));
			/*% substract a phase(wrapped or unwrapped) to the wrappped phase : implements
			% Eq. (19) in ARM method paper :
			% \sin(a - b) = \sin a \cos b - \cos a \sin b
			% \cos(a - b) = \cos a \cos b + \sin a \sin b*/
			num = sIphase*cref - cIphase*sref;    //% numerador sin la referencia
			den = cIphase*cref + sIphase*sref;  //% denumerador sin la referencia

			WrapResidual.at<double>(i,j) = Mask.at<double>(i,j)*atan2(num, den);
		}
	}
}

void PhaseUnwrapping::ComputeWdata(cv::Mat phi, double mu){
	/*% Quality map from the data
	% Mariano Rivera
	% 22 may 2014*/
	int i, j;
	double dx;
	double dy;
	double wrap_dx;
	double wrap_dy;
	//wrapped finite diferences of the wrapped phase
	cv::Mat Dx, Dy;
	Dx.create(imageH, imageW, CV_64FC(1));
	Dy.create(imageH, imageW, CV_64FC(1));

	//Diferences by row and column
	ColumnRowBackwardDifferences(Dx, Dy, phi, imageW, imageH);

	for (i = 0; i < imageH; i++){
		for (j = 0; j < imageW; j++){
			dx = Dx.at<double>(i, j);
			dy = Dy.at<double>(i, j);
			wrap_dx = atan2(sin(dx), cos(dx));
			wrap_dy = atan2(sin(dy), cos(dy));
			iW.at<double>(i, j) = mu / (mu + wrap_dx*wrap_dx);
			jW.at<double>(i, j) = mu / (mu + wrap_dy*wrap_dy);
		}
	}
	//free memory
	Dx.release();
	Dy.release();
}

cv::Mat PhaseUnwrapping::MultigridUnwrapARM(cv::Mat Wrap, cv::Mat Msk, cv::Mat iW, cv::Mat jW, cv::Mat UnWrapInit, double mu, double lambda, int numIter, int numlevels, int ban_OmegaInit, int imageW, int imageH){
	cv::Mat UnWrap;
	
	if (numlevels != 0){
		//% down sampling
		cv::Mat subWrap;
		cv::Mat subUnWrapInit;
		cv::Mat subMsk;
		cv::Mat subiW;
		cv::Mat subjW;
		cv::Mat subUnWrap;

		int imageW_sub = imageW / 2;
		int imageH_sub = imageH / 2;
		if (imageW_sub > 7 && imageH_sub > 7){
			subWrap.create(imageH_sub, imageW_sub, CV_64FC(1));
			subUnWrapInit.create(imageH_sub, imageW_sub, CV_64FC(1));
			subMsk.create(imageH_sub, imageW_sub, CV_64FC(1));
			subiW.create(imageH_sub, imageW_sub, CV_64FC(1));
			subjW.create(imageH_sub, imageW_sub, CV_64FC(1));
			cv::resize(Wrap, subWrap, cv::Size(imageW_sub, imageH_sub), 0.0, 0.0, CV_INTER_NN);
			cv::resize(UnWrapInit, subUnWrapInit, cv::Size(imageW_sub, imageH_sub), 0.0, 0.0, CV_INTER_NN);
			
			/*cv::resize(Msk, subMsk, cv::Size(imageW_sub, imageH_sub), 0.0, 0.0, CV_INTER_NN);//Pendiente:subsampleMask 
			cv::resize(iW, subiW, cv::Size(imageW_sub, imageH_sub), 0.0, 0.0, CV_INTER_NN);
			cv::resize(jW, subjW, cv::Size(imageW_sub, imageH_sub), 0.0, 0.0, CV_INTER_NN);*/
			SubSampleMask(subMsk,Msk,imageW,imageH,imageW_sub,imageH_sub);
			SubSampleMask(subiW, iW, imageW, imageH, imageW_sub, imageH_sub);
			SubSampleMask(subjW, jW, imageW, imageH, imageW_sub, imageH_sub);

			//% call uper level
			subUnWrap = MultigridUnwrapARM(subWrap, subMsk, subiW, subjW, subUnWrapInit,
				                                 mu, lambda / 4, numIter, numlevels - 1, ban_OmegaInit, 
												 imageW_sub, imageH_sub);
			//% up sampling
			cv::resize(subUnWrap, UnWrapInit, cv::Size(imageW, imageH), 0.0, 0.0,CV_INTER_CUBIC);
			//cv::resize(subUnWrap, UnWrapInit, cv::Size(imageW, imageH), 0.0, 0.0,CV_INTER_NN);
			//cv::multiply(UnWrapInit, Msk, UnWrapInit, 1.0);
		}
	}
	UnWrap = UnwrapARM(Wrap, Msk, iW, jW, UnWrapInit, mu, lambda, numIter, ban_OmegaInit, imageW,imageH);

	return(UnWrap);
}


void PhaseUnwrapping::SubSampleMask(cv::Mat Dst, cv::Mat Src, int imageW, int imageH,int imageW_sub, int imageH_sub){
	
	cv::resize(Src, Dst, cv::Size(imageW_sub, imageH_sub), 0.0, 0.0, CV_INTER_NN);
	
	cv::Mat circshift_x;
	cv::Mat circshift_y;
	cv::Mat circshift_xy;
	cv::Mat subcircshift_x;
	cv::Mat subcircshift_y;
	cv::Mat subcircshift_xy;

	circshift_x.create(imageH, imageW, CV_64FC(1));
	circshift_y.create(imageH, imageW, CV_64FC(1));
	circshift_xy.create(imageH, imageW, CV_64FC(1));
	subcircshift_x.create(imageH_sub, imageW_sub, CV_64FC(1));
	subcircshift_y.create(imageH_sub, imageW_sub, CV_64FC(1));
	subcircshift_xy.create(imageH_sub, imageW_sub, CV_64FC(1));

	CircShif_x(circshift_x,Src,imageW,imageH);
	CircShif_y(circshift_y, Src, imageW, imageH);
	CircShif_xy(circshift_xy, Src, imageW, imageH);

	cv::resize(circshift_x, subcircshift_x, cv::Size(imageW_sub, imageH_sub), 0.0, 0.0, CV_INTER_NN);
	cv::resize(circshift_y, subcircshift_y, cv::Size(imageW_sub, imageH_sub), 0.0, 0.0, CV_INTER_NN);
	cv::resize(circshift_xy, subcircshift_xy, cv::Size(imageW_sub, imageH_sub), 0.0, 0.0, CV_INTER_NN);
	
	cv::multiply(Dst, subcircshift_x,Dst, 1.0);
	cv::multiply(Dst, subcircshift_y, Dst, 1.0);
	cv::multiply(Dst, subcircshift_xy, Dst, 1.0);

	circshift_x.release();
	circshift_y.release();
	circshift_xy.release();
	subcircshift_x.release();
	subcircshift_y.release();
	subcircshift_xy.release();

}
void PhaseUnwrapping::CircShif_x(cv::Mat Dst,cv::Mat Src,int imageW,int imageH){
	int i, j;
	//Forward
	for (i = 0; i < imageH; i++){
		Dst.at<double>(i, imageW - 1) = Src.at<double>(i, 0);
	}
	
	for (i = 0; i < imageH; i++){
		for (j = 0; j < imageW-1; j++){
			Dst.at<double>(i, j) = Src.at<double>(i, j + 1);
		}
	}
}
void PhaseUnwrapping::CircShif_y(cv::Mat Dst, cv::Mat Src, int imageW, int imageH){
	int i, j;
	//Forward
	for (j = 0; j < imageW; j++){
		Dst.at<double>(imageH - 1, j) = Src.at<double>(0, j);
	}

	for (i = 0; i < imageH-1; i++){
		for (j = 0; j < imageW; j++){
			Dst.at<double>(i, j) = Src.at<double>(i + 1, j);
		}
	}
}

void PhaseUnwrapping::CircShif_xy(cv::Mat Dst, cv::Mat Src, int imageW, int imageH){
	CircShif_x(Dst, Src, imageW, imageH);
	CircShif_y(Dst, Dst, imageW, imageH);
}

cv::Mat PhaseUnwrapping::UnwrapARM(cv::Mat G,cv::Mat Msk,cv::Mat iW,cv::Mat jW,cv::Mat F,
	double mu, double lambda, int numIter, int ban_OmegaInit, int imageW, int imageH){

	//% uses the Gauss - Seidel solver for the heat equation
	double tol = 1e-3;
	int i, j;
	double dx_G;
	double dy_G;
	double wrap_dx_G;
	double wrap_dy_G;
	int iterations;

	//% differencias finitas envueltas menos la diferencia inicial
	cv::Mat iRho_aux, jRho_aux;
	cv::Mat Dx_G, Dy_G, Dx_F, Dy_F, iRho, jRho, iOmega, jOmega,U;
	Dx_G.create(imageH, imageW, CV_64FC(1));
	Dy_G.create(imageH, imageW, CV_64FC(1));
	Dx_F.create(imageH, imageW, CV_64FC(1));
	Dy_F.create(imageH, imageW, CV_64FC(1));
	iRho.create(imageH, imageW, CV_64FC(1));
	jRho.create(imageH, imageW, CV_64FC(1));

	iOmega = cv::Mat::ones(imageH, imageW, CV_64FC(1));
	jOmega = cv::Mat::ones(imageH, imageW, CV_64FC(1));

	U = cv::Mat::zeros(imageH, imageW, CV_64FC(1));

	//Diferences by row and column
	ColumnRowBackwardDifferences(Dx_G, Dy_G, G, imageW, imageH);
	ColumnRowBackwardDifferences(Dx_F, Dy_F, F, imageW, imageH);

	for (i = 0; i < imageH; i++){
		for (j = 0; j < imageW; j++){
			dx_G = Dx_G.at<double>(i, j);
			dy_G = Dy_G.at<double>(i, j);
			wrap_dx_G = atan2(sin(dx_G), cos(dx_G));
			wrap_dy_G = atan2(sin(dy_G), cos(dy_G));
			iRho.at<double>(i, j) = wrap_dx_G - Dx_F.at<double>(i, j);
			jRho.at<double>(i, j) = wrap_dy_G - Dy_F.at<double>(i, j);
		}
	}
#ifdef Solver_Serial
	//% Solver Single Core
	iterations = SolverGS_HQresidual(U.data, iRho.data, jRho.data, iOmega.data, jOmega.data,
		         Msk.data, iW.data, jW.data, mu, lambda, tol, numIter, ban_OmegaInit, imageW, imageH);
	U = U + F;
#endif
	
	Dx_G.release();
	Dy_G.release();
	Dx_F.release();
	Dy_F.release();
	iRho.release();
	jRho.release();
	iOmega.release();
	jOmega.release();
	return(U);
}

void PhaseUnwrapping::ColumnRowBackwardDifferences(cv::Mat D_column, cv::Mat D_row, cv::Mat Data, int imageW, int imageH){
	int i, j;

	//Backward
	D_column.at<double>(0, 0) = Data.at<double>(0, 0) - Data.at<double>(0, imageW - 1);
	D_row.at<double>(0, 0) = Data.at<double>(0, 0) - Data.at<double>(imageH - 1, 0);
	for (i = 1; i < imageH; i++){
		D_column.at<double>(i, 0) = Data.at<double>(i, 0) - Data.at<double>(i, imageW - 1);
		D_row.at<double>(i, 0) = Data.at<double>(i, 0) - Data.at<double>(i - 1, 0);
	}
	for (j = 1; j < imageW; j++){
		D_row.at<double>(0, j) = Data.at<double>(0, j) - Data.at<double>(imageH - 1, j);
		D_column.at<double>(0, j) = Data.at<double>(0, j) - Data.at<double>(0, j - 1);
	}
	for (i = 1; i < imageH; i++){
		for (j = 1; j < imageW; j++){
			D_column.at<double>(i, j) = Data.at<double>(i, j) - Data.at<double>(i, j - 1);
			D_row.at<double>(i, j) = Data.at<double>(i, j) - Data.at<double>(i - 1, j);
		}
	}
}
void PhaseUnwrapping::ColumnRowForwardDifferences(cv::Mat D_column, cv::Mat D_row, cv::Mat Data, int imageW, int imageH){
	int i, j;

	//Forward
	D_column.at<double>(imageH - 1, imageW - 1) = Data.at<double>(imageH - 1, imageW - 1) - Data.at<double>(imageH - 1, 0);
	D_row.at<double>(imageH - 1, imageW - 1) = Data.at<double>(imageH - 1, imageW - 1) - Data.at<double>(0, imageW - 1);
	for (i = 0; i < imageH-1; i++){
		D_column.at<double>(i, imageW - 1) = Data.at<double>(i, imageW - 1) - Data.at<double>(i, 0);
		D_row.at<double>(i, imageW - 1) = Data.at<double>(i, imageW - 1) - Data.at<double>(i + 1, imageW - 1);
	}
	for (j = 0; j < imageW-1; j++){
		D_row.at<double>(imageH - 1, j) = Data.at<double>(imageH - 1, j) - Data.at<double>(0, j);
		D_column.at<double>(imageH - 1, j) = Data.at<double>(imageH - 1, j) - Data.at<double>(imageH - 1, j + 1);
	}
	for (i = 0; i < imageH-1; i++){
		for (j = 0; j < imageW-1; j++){
			D_column.at<double>(i, j) = Data.at<double>(i, j) - Data.at<double>(i, j + 1);
			D_row.at<double>(i, j) = Data.at<double>(i, j) - Data.at<double>(i + 1, j);
		}
	}
}
void PhaseUnwrapping::ReWrappedPhase(cv::Mat ReWrapPhase, cv::Mat UnWrapPhase, cv::Mat Mask, int W, int H){
	int i, j;
	double unwrap;
	for (i = 0; i < H; i++){
		for (j = 0; j < W; j++){
			unwrap = UnWrapPhase.at<double>(i, j);
			ReWrapPhase.at<double>(i, j) = atan2(sin(unwrap), cos(unwrap))*Mask.at<double>(i, j);
		}
	}
}
void PhaseUnwrapping::releaseImages(){
	if (WrapPhase.data)
		WrapPhase.release();
	if (Mask.data)
		Mask.release();
	if (UnWrapPhase.data)
		UnWrapPhase.release();
	if (iW.data)
		iW.release();
	if (jW.data)
		jW.release();
	if (WrapResidual.data)
		WrapResidual.release();
	if (ReWrapPhase.data)
		ReWrapPhase.release();
	if (UnWrapPhaseInit.data)
		UnWrapPhaseInit.release();
}


PhaseUnwrapping::~PhaseUnwrapping(){
	releaseImages();
}
