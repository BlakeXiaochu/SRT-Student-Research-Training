#include <stdio.h>
#include <math.h>
#ifdef USEOMP
#include <omp.h>
#endif

typedef unsigned char uint8;
typedef unsigned int uint32;

/*Compile setting*/
#ifdef _MSC_VER
    #define DLL_EXPORT __declspec( dllexport ) 
#else
    #define DLL_EXPORT
#endif

/*Compute the Cumulative Sums of Quantized features weights within nBins*/
/*
	FtrsVec - 2D array.Each row represents a sample(a feature vector), and each column represents a feature.
	Wt 		- Samples' weights
	N 		- The number of samples 
	FtrId 	- Feature id(0 ~ F-1)
	nBins 	- Quantization parameter
	WtCumSum 	- Returned values.
*/
void FtrsWtCumSum(uint8** FtrsVec, double* Wt, int SampNum, uint32 FtrId, int nBins, double* WtCumSum)
{
	int i = 0;
	/*Initialize*/
	for(i = 0; i < nBins; i++)
	{
		WtCumSum[i] = 0;
	}

	for(i = 0; i < SampNum; i++)
	{
		WtCumSum[FtrsVec[i][FtrId]] += Wt[i];
	}

	for(i = 1; i < nBins; i++)
	{
		WtCumSum[i] += WtCumSum[i - 1];
	}

	return;
}

/*Compute minimum errors and corresponding thresholds of each feature*/
/*
	PosFtrsVec 	- [NPxF] negative feature vectors(Each row represents a sample, and each column represents a feature)
	NegFtrsVec  - [NNxF] positive feature vectors(Each row represents a sample, and each column represents a feature)
	PosWt       - [NPx1] positive samples weights
	NegWt       - [NNx1] negitive samples weights
	NP 			- The number of positive samples
	NN 			- The number of negative samples
	SampFtrsId 	- Subsample feature id array
	SampNum 	- The number of subsample
	prior 		- Node Weight
	nBins 		- Quantization parameter

	err  		- Return values. Minimum errors of each feature.
	thrs 		- Return values. Corresponding thresholds of each feature
*/
DLL_EXPORT void BestStump(uint8** PosFtrsVec, uint8** NegFtrsVec, double* PosWt, double* NegWt, int NP, int NN, uint32* SampFtrsId, int SampNum, double prior, int nBins, double* err, uint8* thrs)
{
	int i, j;
	uint8 Thrs1, Thrs2;
	double Err, MinErr1, MinErr2;
	double PosWtCumSum[256], NegWtCumSum[256];
	
	#ifdef USEOMP
	nThreads = min(nThreads,omp_get_max_threads());
	#pragma omp parallel for num_threads(nThreads)
	#endif

	/*Go through every subsamples of feature*/
	for(i = 0; i < SampNum; i++)
	{
		Thrs1 = Thrs2 = 0;
		MinErr1 = MinErr2 = 10.0;

		FtrsWtCumSum(PosFtrsVec, PosWt, NP, SampFtrsId[i], nBins, PosWtCumSum);
		FtrsWtCumSum(NegFtrsVec, NegWt, NN, SampFtrsId[i], nBins, NegWtCumSum);

		/*Put Positive samples to left child(< thrs)*/
		for(j = 0; j < nBins; j++)
		{
			/*Put Positive samples to left child(> thrs)*/
			Err = prior - PosWtCumSum[j] + NegWtCumSum[j];			/*prior = PosWtCumSum[255], 1 - prior = NegWtCumSum[255]*/
			if(Err < MinErr1)
			{
				MinErr1 = Err;
				Thrs1 = j;
			}
			/*Put Positive samples to right child(> thrs)*/
			if(1 - Err < MinErr2)
			{
				MinErr2 = 1 - Err;
				Thrs2 = j;
			}
		}

		MinErr1 = (MinErr1 < MinErr2) ? MinErr1 : MinErr2;
		Thrs1 = (MinErr1 < MinErr2) ? Thrs1 : Thrs2;
		
		if(prior < 0.5)
		{
			/*Judge wheather classification does acquire a lower error*/
			if(MinErr1 >= prior) {thrs[i] = nBins - 1; err[i] = prior;}
			else {thrs[i] = Thrs1; err[i] = MinErr1;}
		}
		else
		{
			if(MinErr1 >= 1 - prior) {thrs[i] = nBins - 1; err[i] = prior;}
			else {thrs[i] = Thrs1; err[i] = MinErr1;}
		}
	}

	return;
}