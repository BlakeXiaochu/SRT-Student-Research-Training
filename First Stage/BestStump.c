#include <math.h>
#ifdef USEOMP
#include <omp.h>
#endif

#define min(x,y) ((x) < (y) ? (x) : (y))

typedef unsigned char uint8;
typedef unsigned int uint32;

#ifdef _MSC_VER
    #define DLL_EXPORT __declspec( dllexport ) 
#else
    #define DLL_EXPORT
#endif

/*Compute the Cumulative Sums of Quantized features within nBins*/
void FtrsWtCumSum(uint8** FtrsVec, double* Wt, int FtrsNum, int* SampFtrsId, int SampNum, int nBins, uint32* HistBins)
{
	return;
}


DLL_EXPORT void BestStump(uint8** PosFtrsVec, uint8** NegFtrsVec, double* PosWt, double* NegWt, int NP, int NN, int* SampFtrsId, int SampNum, int prior, int nBins, double* err, uint8* thrs)
{
	return;
}