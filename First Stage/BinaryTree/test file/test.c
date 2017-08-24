#ifdef _MSC_VER
    #define DLL_EXPORT __declspec( dllexport ) 
#else
    #define DLL_EXPORT
#endif

DLL_EXPORT int test(int **a)
{
	return(a[2][1]);
}