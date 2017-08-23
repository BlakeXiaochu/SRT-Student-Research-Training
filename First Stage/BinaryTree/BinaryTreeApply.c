#ifdef USEOMP
#include <omp.h>
#endif

/*Compile setting*/
#ifdef _MSC_VER
    #define DLL_EXPORT __declspec( dllexport ) 
#else
    #define DLL_EXPORT
#endif

typedef unsigned char uint8;
typedef unsigned int uint32;

DLL_EXPORT void BinaryTreeApply(double** Samples, uint32 SampleNum, uint32* Fids, double* Thrs, uint32* Child, double* NodeAlpha, int nThreads, double* SamplesAlpha)
{
	int i, j, cr;

	#ifdef USEOMP
		nThreads = min(nThreads,omp_get_max_threads());
	#pragma omp parallel for num_threads(nThreads)
	#endif

	for(i = 0; i < SampleNum; i++)
	{
		j = 0;
		while(1)
		{
			/*Leaf Node*/
			if(Child[j] == 0) {SamplesAlpha[i] = NodeAlpha[j]; break;}
			/*Split Node*/
			else
			{
				cr = Samples[i][Fids[j]] < Thrs[j] ? -1 : 1;
				if(cr == -1) j = Child[j];
				else j = Child[j] + 1;
			}
		}
	}
}