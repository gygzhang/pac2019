#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include "DataStruct_Array.h"
//add following two header file
#include <omp.h>
#include <immintrin.h>
#define F 2.2E3
#define Time 1E6
using namespace std;
using namespace FYSPACE;

const int ONE_D = 1;
const int TWO_D = 2;
const int THREE_D = 3;
const int ni = 500;
const int nj = 400;
const int nk = 300;

typedef double RDouble;
typedef FYArray<RDouble, 3> RDouble3D;
typedef FYArray<RDouble, 4> RDouble4D;

int preccheck(RDouble4D dqdx_4d, RDouble4D dqdy_4d, RDouble4D dqdz_4d);

inline unsigned long long rdtsc(void)
{
	unsigned long hi = 0, lo = 0;

	__asm__ __volatile__("lfence;rdtsc" : "=a"(lo), "=d"(hi));

	return (((unsigned long long)lo)) | (((unsigned long long)hi) << 32);
}

int main()
{
	double start, end, elapsed;
	const int nDim = THREE_D;
	const double fourth = 0.25;
	int mst = 0;
	int med = 3;

	//-1,501
	Range I(-1, ni + 1);
	//-1,401
	Range J(-1, nj + 1);

	Range K(-1, nk + 1);
	RDouble3D x(I, J, K, fortranArray);
	RDouble3D y(I, J, K, fortranArray);
	RDouble3D z(I, J, K, fortranArray);
	for (int k = -1; k <= nk + 1; ++k)
	{
		for (int j = -1; j <= nj + 1; ++j)
		{
			for (int i = -1; i <= ni + 1; ++i)
			{
				x(i, j, k) = i * 0.1;
				y(i, j, k) = j * 0.2;
				z(i, j, k) = k * 0.3;
			}
		}
	}


	// 申请变量空间
	I = Range(-1, ni + 1);
	J = Range(-1, nj + 1);
	K = Range(-1, nk + 1);
	Range D(1, 3);
	RDouble4D xfn(I, J, K, D, fortranArray);  // 网格单元↙左下面法向，D为方向
	RDouble4D yfn(I, J, K, D, fortranArray);
	RDouble4D zfn(I, J, K, D, fortranArray);
	RDouble4D area(I, J, K, D, fortranArray);  // 网格单元↙左下面面积
	RDouble3D vol(I, J, K, fortranArray);  // 网格单元体积

	Range M(0, 3); // 4个变量：速度u、v、w，温度T
	RDouble4D q_4d(I, J, K, M, fortranArray); // 存储流场量，位置在单元中心
	RDouble4D dqdx_4d(I, J, K, M, fortranArray); // 存储流场量计算得到的梯度偏x
	RDouble4D dqdy_4d(I, J, K, M, fortranArray); // 存储流场量计算得到的梯度偏y
	RDouble4D dqdz_4d(I, J, K, M, fortranArray); // 存储流场量计算得到的梯度偏z

	// 计算网格单元几何数据 xfn、fn、zfn、area、vol
	// 速度u、v、w，温度T 流场变量赋值，存储在q_4d中，便于后面速度、温度界面梯度计算
	// 程序每执行一个迭代步，流场变量被更新。此处给初场值u=1.0，v=0.0，w=0.0，T=1.0
	for (int k = -1; k <= nk + 1; ++k)
	{
		for (int j = -1; j <= nj + 1; ++j)
		{
			for (int i = -1; i <= ni + 1; ++i)
			{
				xfn(i, j, k, 1) = 1.0;
				xfn(i, j, k, 2) = 0.0;
				xfn(i, j, k, 3) = 0.0;
				yfn(i, j, k, 1) = 0.0;
				yfn(i, j, k, 2) = 1.0;
				yfn(i, j, k, 3) = 0.0;
				zfn(i, j, k, 1) = 0.0;
				zfn(i, j, k, 2) = 0.0;
				zfn(i, j, k, 3) = 1.0;
				area(i, j, k, 1) = 0.06;
				area(i, j, k, 2) = 0.03;
				area(i, j, k, 3) = 0.02;
				vol(i, j, k) = 0.006;
			}
		}
	}
	for (int k = -1; k <= nk + 1; ++k)
	{
		for (int j = -1; j <= nj + 1; ++j)
		{
			for (int i = -1; i <= ni + 1; ++i)
			{
				q_4d(i, j, k, 0) = (x(i, j, k) * x(i, j, k) + y(i, j, k)*y(i, j, k) - 1.3164) / 2.1547; // u = a*x*x+b*y*y
				q_4d(i, j, k, 1) = (z(i, j, k)*z(i, j, k) - 0.2157) * 0.137; // v=c*z*z
				q_4d(i, j, k, 2) = (2.0*x(i, j, k) + 1.737) / 3.14; // w=d*x
				q_4d(i, j, k, 3) = x(i, j, k) + y(i, j, k) + 1.3765; // T = x + y
			}
		}
	}
	start = rdtsc();
	//以上为数据初始化部分，不可修改！
	// --------------------------------------------------------------------
	// 求解速度、温度在“单元界面”上的梯度，i、j、k三个方向依次求解
	// 在程序中是“耗时部分”，每一个迭代步都会求解，以下为未优化代码
	// 希望参赛队伍在理解该算法的基础上，实现更高效的界面梯度求解，提升程序执行效率
	// --------------------------------------------------------------------
	// 此处开始统计计算部分代码运行时间
	int n_threads = omp_get_max_threads();

	for (int nsurf = 1; nsurf <= 3; ++nsurf)
	{
		int index[] = { 1,2,3,1,2 };
		int ns1 = nsurf;
		int ns2 = index[nsurf];
		int ns3 = index[nsurf + 1];

		int i, j, k, m;


#pragma omp parallel num_threads(n_threads)
		{
#pragma omp  for collapse(3) private(i,j,k,m) schedule(static)
			for (m = mst; m < med; ++m)
			{
				for (k = 1; k < nk + 1; ++k)
				{
					for (j = 1; j < nj + 1; ++j)
					{
						for (i = 1; i < ni + 1 - 8; i += 8)
						{
							__m512d zeo = _mm512_setzero_pd();
							_mm512_store_pd(&dqdx_4d(i, j, k, m), zeo);
							_mm512_store_pd(&dqdx_4d(i, j, k, m), zeo);
							_mm512_store_pd(&dqdx_4d(i, j, k, m), zeo);
						}
						__m256d zeo = _mm256_setzero_pd();
						_mm256_store_pd(&dqdx_4d(i, j, k, m), zeo);
						_mm256_store_pd(&dqdy_4d(i, j, k, m), zeo);
						_mm256_store_pd(&dqdz_4d(i, j, k, m), zeo);
					}
				}
			}
		}

		Range IW(-1, ni + 1);
		Range JW(-1, nj + 1);
		Range KW(-1, nk + 1);

		RDouble3D worksx(IW, JW, KW, fortranArray);
		RDouble3D worksy(IW, JW, KW, fortranArray);
		RDouble3D worksz(IW, JW, KW, fortranArray);
		RDouble3D workqm(IW, JW, KW, fortranArray);
#pragma omp parallel num_threads(n_threads)
		{
			if (nsurf == 1)
			{

				double nt1 = rdtsc();
				int i, j, k, m, kj;
				__m512d xfn1, yfn1, zfn1, area1, q_4d1, dqdy_4d1, xfn2, yfn2, zfn2, area2, q_4d2, dqdy_4d2, temp1, temp2, temp3, temp4, temp5, temp6, temp8, temp9, temp10, temp11, temp12;
				__m512d t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, u1, u2, u3, u4, u5, u6, u7, u8, u9, st3;
				__m256d ts1, ts2, ts3, ts4, ts5, ts6, ts7, ts8, ts9, ts10, us0, us1, us2, us3, us4, us5, us6, us7, us8, us9, bl1;
#pragma omp  for collapse(3)\
 private(m,i,j,k,xfn1,yfn1, zfn1, area1, q_4d1, dqdy_4d1, xfn2, yfn2, zfn2, area2, q_4d2, dqdy_4d2,temp1,temp2,temp3,temp4,temp5,temp6) schedule(static)
				for (m = mst; m <= med; ++m)
				{
					for (k = 1; k < nk + 1; ++k)
					{
						for (j = 1; j < nj + 1; ++j)
						{
							for (i = 1; i < ni + 1 - 8; i += 8)
							{

								xfn1 = _mm512_load_pd(&xfn(i, j, k, ns1));
								area1 = _mm512_load_pd(&area(i, j, k, ns1));
								xfn2 = _mm512_load_pd(&xfn(i - 1, j, k, ns1));
								area2 = _mm512_load_pd(&area(i - 1, j, k, ns1));

								temp1 = _mm512_mul_pd(xfn1, area1);
								temp3 = -_mm512_fmadd_pd(xfn2, area2, temp1);
								//temp2 = _mm512_mul_pd(xfn2, area2);
								//temp3 = -_mm512_add_pd(temp1, temp2);
								//temp3 = -(xfn1*area1 + xfn2 * area2);
								temp4 = _mm512_set1_pd((double)-1);
								//temp3 = _mm512_mul_pd(temp3, temp4);
								temp5 = _mm512_load_pd(&q_4d(i - 1, j, k, m));
								temp6 = _mm512_load_pd(&dqdx_4d(i - 1, j, k, m));
								temp5 = _mm512_mul_pd(temp3, temp5);
								temp6 = _mm512_sub_pd(temp6, temp5);
								_mm512_store_pd(&dqdx_4d(i, j, k, m), temp5);
								_mm512_store_pd(&dqdx_4d(i - 1, j, k, m), temp6);

								yfn1 = _mm512_load_pd(&yfn(i, j, k, ns1));
								yfn2 = _mm512_load_pd(&yfn(i - 1, j, k, ns1));
								temp1 = _mm512_mul_pd(yfn1, area1);
								temp3 = -_mm512_fmadd_pd(yfn2, area2, temp1);
								//temp2 = _mm512_mul_pd(yfn2, area2);
								//temp3 = _mm512_add_pd(temp1, temp2);
								//temp3 = _mm512_mul_pd(temp3, temp4);
								temp5 = _mm512_load_pd(&q_4d(i - 1, j, k, m));
								temp5 = _mm512_mul_pd(temp3, temp5);
								_mm512_store_pd(&dqdy_4d(i, j, k, m), temp5);
								temp6 = _mm512_load_pd(&dqdy_4d(i - 1, j, k, m));
								temp6 = _mm512_sub_pd(temp6, temp5);
								_mm512_store_pd(&dqdy_4d(i - 1, j, k, m), temp6);


								zfn1 = _mm512_load_pd(&zfn(i, j, k, ns1));
								zfn2 = _mm512_load_pd(&zfn(i - 1, j, k, ns1));
								temp1 = _mm512_mul_pd(zfn1, area1);
								temp3 = -_mm512_fmadd_pd(zfn2, area2, temp1);
								//temp2 = _mm512_mul_pd(zfn2, area2);
								//temp3 = _mm512_add_pd(temp1, temp2);
								//temp3 = _mm512_mul_pd(temp3, temp4);
								temp5 = _mm512_load_pd(&q_4d(i - 1, j, k, m));
								temp5 = _mm512_mul_pd(temp3, temp5);
								_mm512_store_pd(&dqdz_4d(i, j, k, m), temp5);
								temp6 = _mm512_load_pd(&dqdz_4d(i - 1, j, k, m));
								temp6 = _mm512_sub_pd(temp6, temp5);
								_mm512_store_pd(&dqdz_4d(i - 1, j, k, m), temp6);
								//--------------------------------
								const __m512d tmep7 = _mm512_load_pd(&q_4d(i, j, k, m));
								temp8 = _mm512_load_pd(&q_4d(i - 1, j, k, m));
								temp9 = _mm512_load_pd(&q_4d(i, j - 1, k, m));
								temp10 = _mm512_load_pd(&q_4d(i - 1, j - 1, k, m));
								temp11 = _mm512_set1_pd(fourth);
								temp12 = _mm512_add_pd(tmep7, temp8);
								temp12 = _mm512_add_pd(temp9, temp12);
								temp12 = _mm512_add_pd(temp10, temp12);
								temp12 = _mm512_mul_pd(temp11, temp12);

								xfn1 = _mm512_load_pd(&xfn(i, j, k, ns2));
								area1 = _mm512_load_pd(&area(i, j, k, ns2));
								xfn2 = _mm512_load_pd(&xfn(i - 1, j, k, ns2));
								area2 = _mm512_load_pd(&area(i - 1, j, k, ns2));
								temp1 = _mm512_mul_pd(xfn1, area1);
								temp3 = _mm512_fmadd_pd(xfn2, area2, temp1);
								//temp2 = _mm512_mul_pd(xfn2, area2);
								//temp3 = _mm512_add_pd(temp1, temp2);
								temp5 = _mm512_load_pd(&dqdx_4d(i, j, k, m));
								temp5 = _mm512_fmsub_pd(temp3, temp12, temp5);
								temp5 = _mm512_mul_pd(temp5, temp4);
								_mm512_store_pd(&dqdx_4d(i, j, k, m), temp5);
								temp5 = _mm512_load_pd(&dqdx_4d(i, j - 1, k, m));
								temp5 = _mm512_fmadd_pd(temp3, temp12, temp5);
								_mm512_store_pd(&dqdx_4d(i, j - 1, k, m), temp5);
								// worksx(i,j,k) = xfn(i,j,k,ns2) * area(i,j,k,ns2) + xfn(i-1,j,k,ns2) * area(i-1,j,k,ns2);
								// dqdx_4d(i,j,k,m) -= worksx(i,j,k) * workqm(i,j,k);
								// dqdx_4d(i,j-1,k,m) += worksx(i,j,k) * workqm(i,j,k);

								yfn1 = _mm512_load_pd(&yfn(i, j, k, ns2));
								yfn2 = _mm512_load_pd(&yfn(i - 1, j, k, ns2));
								temp1 = _mm512_mul_pd(yfn1, area1);
								temp3 = _mm512_fmadd_pd(yfn2, area2, temp1);
								//temp2 = _mm512_mul_pd(xfn2, area2);
								//temp3 = _mm512_add_pd(temp1, temp2);
								temp5 = _mm512_load_pd(&dqdy_4d(i, j, k, m));
								temp5 = _mm512_fmsub_pd(temp3, temp12, temp5);
								temp5 = _mm512_mul_pd(temp5, temp4);
								_mm512_store_pd(&dqdy_4d(i, j, k, m), temp5);
								temp5 = _mm512_load_pd(&dqdy_4d(i, j - 1, k, m));
								temp5 = _mm512_fmadd_pd(temp3, temp12, temp5);
								_mm512_store_pd(&dqdy_4d(i, j - 1, k, m), temp5);

								zfn1 = _mm512_load_pd(&zfn(i, j, k, ns2));
								zfn2 = _mm512_load_pd(&zfn(i - 1, j, k, ns2));
								temp1 = _mm512_mul_pd(zfn1, area1);
								temp3 = _mm512_fmadd_pd(zfn2, area2, temp1);
								//temp2 = _mm512_mul_pd(xfn2, area2);
								//temp3 = _mm512_add_pd(temp1, temp2);
								temp5 = _mm512_load_pd(&dqdz_4d(i, j, k, m));
								temp5 = _mm512_fmsub_pd(temp3, temp12, temp5);
								temp5 = _mm512_mul_pd(temp5, temp4);
								_mm512_store_pd(&dqdz_4d(i, j, k, m), temp5);
								temp5 = _mm512_load_pd(&dqdz_4d(i, j - 1, k, m));
								temp5 = _mm512_fmadd_pd(temp3, temp12, temp5);
								_mm512_store_pd(&dqdz_4d(i, j - 1, k, m), temp5);


								// worksx(i,j,k) = xfn(i,j,k,ns3) * area(i,j,k,ns3) + xfn(i-1,j,k,ns3) * area(i-1,j,k,ns3);
								// worksy(i,j,k) = yfn(i,j,k,ns3) * area(i,j,k,ns3) + yfn(i-1,j,k,ns3) * area(i-1,j,k,ns3);
								// worksz(i,j,k) = zfn(i,j,k,ns3) * area(i,j,k,ns3) + zfn(i-1,j,k,ns3) * area(i-1,j,k,ns3);
								// workqm(i,j,k) = fourth * (q_4d(i,j,k,m) + q_4d(i-1,j,k,m) + q_4d(i,j,k-1,m) + q_4d(i-1,j,k-1,m));
								// dqdx_4d(i,j,k,m) -= worksx(i,j,k) * workqm(i,j,k);
								// dqdy_4d(i,j,k,m) -= worksy(i,j,k) * workqm(i,j,k);
								// dqdz_4d(i,j,k,m) -= worksz(i,j,k) * workqm(i,j,k);
								// dqdx_4d(i,j,k-1,m) += worksx(i,j,k) * workqm(i,j,k);
								// dqdy_4d(i,j,k-1,m) += worksy(i,j,k) * workqm(i,j,k);
								// dqdz_4d(i,j,k-1,m) += worksz(i,j,k) * workqm(i,j,k);

								t1 = _mm512_load_pd(&q_4d(i, j, k, m));
								t2 = _mm512_load_pd(&q_4d(i - 1, j, k, m));
								t3 = _mm512_load_pd(&q_4d(i, j, k - 1, m));
								t4 = _mm512_load_pd(&q_4d(i - 1, j, k - 1, m));
								t5 = _mm512_set1_pd(fourth);
								t6 = _mm512_add_pd(t1, t2);
								t6 = _mm512_add_pd(t3, t6);
								t6 = _mm512_add_pd(t4, t6);
								t6 = _mm512_mul_pd(t5, t6);

								u1 = _mm512_load_pd(&xfn(i, j, k, ns3));
								u2 = _mm512_load_pd(&area(i, j, k, ns3));
								u3 = _mm512_load_pd(&xfn(i - 1, j, k, ns3));
								u4 = _mm512_load_pd(&area(i - 1, j, k, ns3));
								u5 = _mm512_mul_pd(u1, u2);
								u6 = _mm512_mul_pd(u3, u4);
								u7 = _mm512_add_pd(u5, u6);
								u8 = _mm512_load_pd(&dqdx_4d(i, j, k, m));
								st3 = _mm512_set1_pd((double)-1);
								u8 = _mm512_fmsub_pd(u7, t6, u8);
								u8 = _mm512_mul_pd(u8, st3);
								_mm512_store_pd(&dqdx_4d(i, j, k, m), u8);
								u8 = _mm512_load_pd(&dqdx_4d(i, j, k - 1, m));
								u8 = _mm512_fmadd_pd(u7, t6, u8);
								_mm512_store_pd(&dqdx_4d(i, j, k - 1, m), u8);

								u1 = _mm512_load_pd(&yfn(i, j, k, ns3));
								u3 = _mm512_load_pd(&yfn(i - 1, j, k, ns3));
								u5 = _mm512_mul_pd(u1, u2);
								u6 = _mm512_mul_pd(u3, u4);
								u7 = _mm512_add_pd(u5, u6);
								u8 = _mm512_load_pd(&dqdy_4d(i, j, k, m));
								u8 = _mm512_fmsub_pd(u7, t6, u8);
								u8 = _mm512_mul_pd(u8, st3);
								_mm512_store_pd(&dqdy_4d(i, j, k, m), u8);
								u8 = _mm512_load_pd(&dqdy_4d(i, j, k - 1, m));
								u8 = _mm512_fmadd_pd(u7, t6, u8);
								_mm512_store_pd(&dqdy_4d(i, j, k - 1, m), u8);

								u1 = _mm512_load_pd(&zfn(i, j, k, ns3));
								u3 = _mm512_load_pd(&zfn(i - 1, j, k, ns3));
								u5 = _mm512_mul_pd(u1, u2);
								u6 = _mm512_mul_pd(u3, u4);
								u7 = _mm512_add_pd(u5, u6);
								u8 = _mm512_load_pd(&dqdz_4d(i, j, k, m));
								u8 = _mm512_fmsub_pd(u7, t6, u8);
								u8 = _mm512_mul_pd(u8, st3);
								_mm512_store_pd(&dqdz_4d(i, j, k, m), u8);
								u8 = _mm512_load_pd(&dqdz_4d(i, j, k - 1, m));
								u8 = _mm512_fmadd_pd(u7, t6, u8);
								_mm512_store_pd(&dqdz_4d(i, j, k - 1, m), u8);
							}
							__m256d xfn1 = _mm256_load_pd(&xfn(i, j, k, ns1));
							__m256d area1 = _mm256_load_pd(&area(i, j, k, ns1));
							__m256d xfn2 = _mm256_load_pd(&xfn(i - 1, j, k, ns1));
							__m256d area2 = _mm256_load_pd(&area(i - 1, j, k, ns1));
							__m256d r1 = _mm256_mul_pd(xfn1, area1);
							__m256d r2 = _mm256_mul_pd(xfn2, area2);
							__m256d r3 = _mm256_add_pd(r1, r2);
							__m256d r4 = _mm256_set1_pd((double)-1);
							r3 = _mm256_mul_pd(r3, r4);
							__m256d r5 = _mm256_load_pd(&q_4d(i - 1, j, k, m));
							r5 = _mm256_mul_pd(r3, r5);
							_mm256_store_pd(&dqdx_4d(i, j, k, m), r5);
							__m256d r6 = _mm256_load_pd(&dqdx_4d(i - 1, j, k, m));
							r6 = _mm256_sub_pd(r6, r5);
							_mm256_store_pd(&dqdx_4d(i - 1, j, k, m), r6);

							__m256d yfn1 = _mm256_load_pd(&yfn(i, j, k, ns1));
							__m256d yfn2 = _mm256_load_pd(&yfn(i - 1, j, k, ns1));
							r1 = _mm256_mul_pd(yfn1, area1);
							r2 = _mm256_mul_pd(yfn2, area2);
							r3 = _mm256_add_pd(r1, r2);
							r3 = _mm256_mul_pd(r3, r4);
							r5 = _mm256_load_pd(&q_4d(i - 1, j, k, m));
							r5 = _mm256_mul_pd(r3, r5);
							_mm256_store_pd(&dqdy_4d(i, j, k, m), r5);
							r6 = _mm256_load_pd(&dqdy_4d(i - 1, j, k, m));
							r6 = _mm256_sub_pd(r6, r5);
							_mm256_store_pd(&dqdy_4d(i - 1, j, k, m), r6);

							__m256d zfn1 = _mm256_load_pd(&zfn(i, j, k, ns1));
							__m256d zfn2 = _mm256_load_pd(&zfn(i - 1, j, k, ns1));
							r1 = _mm256_mul_pd(zfn1, area1);
							r3 = _mm256_fmadd_pd(zfn2, area2, r1);
							//r2 = _mm256_mul_pd(xfn2, area2);
							//r3 = _mm256_add_pd(r1, r2);
							r3 = _mm256_mul_pd(r3, r4);
							r5 = _mm256_load_pd(&q_4d(i - 1, j, k, m));
							r5 = _mm256_mul_pd(r3, r5);
							_mm256_store_pd(&dqdz_4d(i, j, k, m), r5);
							r6 = _mm256_load_pd(&dqdz_4d(i - 1, j, k, m));
							r6 = _mm256_sub_pd(r6, r5);
							_mm256_store_pd(&dqdz_4d(i - 1, j, k, m), r6);
							//--------------------------------
							__m256d r7 = _mm256_load_pd(&q_4d(i, j, k, m));
							__m256d r8 = _mm256_load_pd(&q_4d(i - 1, j, k, m));
							__m256d r9 = _mm256_load_pd(&q_4d(i, j - 1, k, m));
							__m256d r10 = _mm256_load_pd(&q_4d(i - 1, j - 1, k, m));
							__m256d r11 = _mm256_set1_pd(fourth);
							__m256d r12 = _mm256_add_pd(r7, r8);
							r12 = _mm256_add_pd(r9, r12);
							r12 = _mm256_add_pd(r10, r12);
							r12 = _mm256_mul_pd(r11, r12);

							xfn1 = _mm256_load_pd(&xfn(i, j, k, ns2));
							area1 = _mm256_load_pd(&area(i, j, k, ns2));
							xfn2 = _mm256_load_pd(&xfn(i - 1, j, k, ns2));
							area2 = _mm256_load_pd(&area(i - 1, j, k, ns2));
							r1 = _mm256_mul_pd(xfn1, area1);
							r3 = _mm256_fmadd_pd(xfn2, area2, r1);
							//r2 = _mm256_mul_pd(xfn2, area2);
							//r3 = _mm256_add_pd(r1, r2);
							r5 = _mm256_load_pd(&dqdx_4d(i, j, k, m));
							r5 = _mm256_fmsub_pd(r3, r12, r5);
							r5 = _mm256_mul_pd(r5, r4);
							_mm256_store_pd(&dqdx_4d(i, j, k, m), r5);
							r5 = _mm256_load_pd(&dqdx_4d(i, j - 1, k, m));
							r5 = _mm256_fmadd_pd(r3, r12, r5);
							_mm256_store_pd(&dqdx_4d(i, j - 1, k, m), r5);

							yfn1 = _mm256_load_pd(&yfn(i, j, k, ns2));
							yfn2 = _mm256_load_pd(&yfn(i - 1, j, k, ns2));
							r1 = _mm256_mul_pd(yfn1, area1);
							r3 = _mm256_fmadd_pd(yfn2, area2, r1);
							//r2 = _mm256_mul_pd(yfn2, area2);
							//r3 = _mm256_add_pd(r1, r2);
							r5 = _mm256_load_pd(&dqdy_4d(i, j, k, m));
							r5 = _mm256_fmsub_pd(r3, r12, r5);
							r5 = _mm256_mul_pd(r5, r4);
							_mm256_store_pd(&dqdy_4d(i, j, k, m), r5);
							r5 = _mm256_load_pd(&dqdy_4d(i, j - 1, k, m));
							r5 = _mm256_fmadd_pd(r3, r12, r5);
							_mm256_store_pd(&dqdy_4d(i, j - 1, k, m), r5);

							zfn1 = _mm256_load_pd(&zfn(i, j, k, ns2));
							zfn2 = _mm256_load_pd(&zfn(i - 1, j, k, ns2));
							r1 = _mm256_mul_pd(zfn1, area1);
							r3 = _mm256_fmadd_pd(zfn2, area2, r1);
							//r2 = _mm256_mul_pd(zfn2, area2);
							//r3 = _mm256_add_pd(r1, r2);
							r5 = _mm256_load_pd(&dqdz_4d(i, j, k, m));
							r5 = _mm256_fmsub_pd(r3, r12, r5);
							r5 = _mm256_mul_pd(r5, r4);
							_mm256_store_pd(&dqdz_4d(i, j, k, m), r5);
							r5 = _mm256_load_pd(&dqdz_4d(i, j - 1, k, m));
							r5 = _mm256_fmadd_pd(r3, r12, r5);
							_mm256_store_pd(&dqdz_4d(i, j - 1, k, m), r5);


							ts1 = _mm256_load_pd(&q_4d(i, j, k, m));
							ts2 = _mm256_load_pd(&q_4d(i - 1, j, k, m));
							ts3 = _mm256_load_pd(&q_4d(i, j, k - 1, m));
							ts4 = _mm256_load_pd(&q_4d(i - 1, j, k - 1, m));
							ts5 = _mm256_set1_pd(fourth);
							ts6 = _mm256_add_pd(ts1, ts2);
							ts6 = _mm256_add_pd(ts3, ts6);
							ts6 = _mm256_add_pd(ts4, ts6);
							ts6 = _mm256_mul_pd(ts5, ts6);

							us1 = _mm256_load_pd(&xfn(i, j, k, ns3));
							us2 = _mm256_load_pd(&area(i, j, k, ns3));
							us3 = _mm256_load_pd(&xfn(i - 1, j, k, ns3));
							us4 = _mm256_load_pd(&area(i - 1, j, k, ns3));
							us5 = _mm256_mul_pd(us1, us2);
							us6 = _mm256_mul_pd(us3, us4);
							us7 = _mm256_add_pd(us5, us6);
							us8 = _mm256_load_pd(&dqdx_4d(i, j, k, m));
							us9 = _mm256_set1_pd((double)-1);
							us8 = _mm256_fmsub_pd(us7, ts6, us8);
							us8 = _mm256_mul_pd(us8, us9);
							_mm256_store_pd(&dqdx_4d(i, j, k, m), us8);
							us8 = _mm256_load_pd(&dqdx_4d(i, j, k - 1, m));
							us8 = _mm256_fmadd_pd(us7, ts6, us8);
							_mm256_store_pd(&dqdx_4d(i, j, k - 1, m), us8);

							us1 = _mm256_load_pd(&yfn(i, j, k, ns3));
							us3 = _mm256_load_pd(&yfn(i - 1, j, k, ns3));
							us5 = _mm256_mul_pd(us1, us2);
							us6 = _mm256_mul_pd(us3, us4);
							us7 = _mm256_add_pd(us5, us6);
							us8 = _mm256_load_pd(&dqdy_4d(i, j, k, m));
							us8 = _mm256_fmsub_pd(us7, ts6, us8);
							us8 = _mm256_mul_pd(us8, us9);
							_mm256_store_pd(&dqdy_4d(i, j, k, m), us8);
							us8 = _mm256_load_pd(&dqdy_4d(i, j, k - 1, m));
							us8 = _mm256_fmadd_pd(us7, ts6, us8);
							_mm256_store_pd(&dqdy_4d(i, j, k - 1, m), us8);

							us1 = _mm256_load_pd(&zfn(i, j, k, ns3));
							us3 = _mm256_load_pd(&zfn(i - 1, j, k, ns3));
							us5 = _mm256_mul_pd(us1, us2);
							us6 = _mm256_mul_pd(us3, us4);
							us7 = _mm256_add_pd(us5, us6);
							us8 = _mm256_load_pd(&dqdz_4d(i, j, k, m));
							us8 = _mm256_fmsub_pd(us7, ts6, us8);
							us8 = _mm256_mul_pd(us8, us9);
							_mm256_store_pd(&dqdz_4d(i, j, k, m), us8);
							us8 = _mm256_load_pd(&dqdz_4d(i, j, k - 1, m));
							us8 = _mm256_fmadd_pd(us7, ts6, us8);
							_mm256_store_pd(&dqdz_4d(i, j, k - 1, m), us8);
						}
					}
				}
				//0.26
#pragma omp for private(k,j,kj,i) schedule(static)
				for (kj = 1; kj < ((nj + 1) * (nk + 1)); kj++) {
					k = kj / (nj + 1);
					j = kj % (nj + 1);
					for (i = 1; i < ni + 1 - 8; i += 8)
					{
						t1 = _mm512_load_pd(&vol(i, j, k));
						t2 = _mm512_load_pd(&vol(i - 1, j, k));
						t3 = _mm512_set1_pd((double)1.0);
						t4 = _mm512_add_pd(t1, t2);
						t4 = _mm512_div_pd(t3, t4);
						_mm512_store_pd(&workqm(i, j, k), t4);
					}
					ts1 = _mm256_load_pd(&vol(i, j, k));
					ts2 = _mm256_load_pd(&vol(i - 1, j, k));
					ts3 = _mm256_set1_pd((double)1.0);
					ts4 = _mm256_add_pd(ts1, ts2);
					ts4 = _mm256_div_pd(ts3, ts4);
					_mm256_store_pd(&workqm(i, j, k), ts4);
				}
			}
			else if (nsurf == 2)
			{
				double nt1 = rdtsc();

				int i, j, k, m, kj;
				__m512d t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, u1, u2, u3, u4, u5, u6, u7, u8, u9;
				__m256d ts1, ts2, ts3, ts4, ts5, ts6, ts7, ts8, ts9, ts10, us0, us1, us2, us3, us4, us5, us6, us7, us8, us9, bl1;
#pragma omp  for collapse(3) private(i,j,k,m) schedule(static)
				for (m = mst; m <= med; ++m)
				{
					for (k = 1; k < nk + 1; ++k)
					{
						for (j = 1; j < nj + 1; ++j)
						{
							for (i = 1; i < ni + 1 - 8; i += 8)
							{

								t1 = _mm512_load_pd(&xfn(i, j, k, ns1));
								t2 = _mm512_load_pd(&area(i, j, k, ns1));
								t3 = _mm512_load_pd(&xfn(i, j - 1, k, ns1));
								t4 = _mm512_load_pd(&area(i, j - 1, k, ns1));
								t5 = _mm512_mul_pd(t1, t2);
								t6 = _mm512_mul_pd(t3, t4);
								t7 = _mm512_add_pd(t5, t6);
								t8 = _mm512_set1_pd((double)-1);
								t7 = _mm512_mul_pd(t7, t8);
								t9 = _mm512_load_pd(&q_4d(i, j - 1, k, m));
								t9 = _mm512_mul_pd(t7, t9);
								_mm512_store_pd(&dqdx_4d(i, j, k, m), t9);
								t10 = _mm512_load_pd(&dqdx_4d(i, j - 1, k, m));
								t10 = _mm512_sub_pd(t10, t9);
								_mm512_store_pd(&dqdx_4d(i, j - 1, k, m), t10);

								t1 = _mm512_load_pd(&yfn(i, j, k, ns1));
								t3 = _mm512_load_pd(&yfn(i, j - 1, k, ns1));
								t5 = _mm512_mul_pd(t1, t2);
								t6 = _mm512_mul_pd(t3, t4);
								t7 = _mm512_add_pd(t5, t6);
								t7 = _mm512_mul_pd(t7, t8);
								t9 = _mm512_load_pd(&q_4d(i, j - 1, k, m));
								t9 = _mm512_mul_pd(t7, t9);
								_mm512_store_pd(&dqdy_4d(i, j, k, m), t9);
								t10 = _mm512_load_pd(&dqdy_4d(i, j - 1, k, m));
								t10 = _mm512_sub_pd(t10, t9);
								_mm512_store_pd(&dqdy_4d(i, j - 1, k, m), t10);

								t1 = _mm512_load_pd(&zfn(i, j, k, ns1));
								t3 = _mm512_load_pd(&zfn(i, j - 1, k, ns1));
								t5 = _mm512_mul_pd(t1, t2);
								t6 = _mm512_mul_pd(t3, t4);
								t7 = _mm512_add_pd(t5, t6);
								t7 = _mm512_mul_pd(t7, t8);
								t9 = _mm512_load_pd(&q_4d(i, j - 1, k, m));
								t9 = _mm512_mul_pd(t7, t9);
								_mm512_store_pd(&dqdz_4d(i, j, k, m), t9);
								t10 = _mm512_load_pd(&dqdz_4d(i, j - 1, k, m));
								t10 = _mm512_sub_pd(t10, t9);
								_mm512_store_pd(&dqdz_4d(i, j - 1, k, m), t10);
								//--------------------------------
								u1 = _mm512_load_pd(&q_4d(i, j, k, m));
								u2 = _mm512_load_pd(&q_4d(i, j - 1, k, m));
								u3 = _mm512_load_pd(&q_4d(i, j, k - 1, m));
								u4 = _mm512_load_pd(&q_4d(i, j - 1, k - 1, m));
								u5 = _mm512_set1_pd(fourth);
								u6 = _mm512_add_pd(u1, u2);
								u6 = _mm512_add_pd(u3, u6);
								u6 = _mm512_add_pd(u4, u6);
								u6 = _mm512_mul_pd(u5, u6);

								t1 = _mm512_load_pd(&xfn(i, j, k, ns2));
								t2 = _mm512_load_pd(&area(i, j, k, ns2));
								t3 = _mm512_load_pd(&xfn(i, j - 1, k, ns2));
								t4 = _mm512_load_pd(&area(i, j - 1, k, ns2));
								t5 = _mm512_mul_pd(t1, t2);
								t6 = _mm512_mul_pd(t3, t4);
								t7 = _mm512_add_pd(t5, t6);
								t9 = _mm512_load_pd(&dqdx_4d(i, j, k, m));
								t9 = _mm512_fmsub_pd(t7, u6, t9);
								t9 = _mm512_mul_pd(t9, t8);
								_mm512_store_pd(&dqdx_4d(i, j, k, m), t9);
								t9 = _mm512_load_pd(&dqdx_4d(i, j, k - 1, m));
								t9 = _mm512_fmadd_pd(t7, u6, t9);
								_mm512_store_pd(&dqdx_4d(i, j, k - 1, m), t9);

								t1 = _mm512_load_pd(&yfn(i, j, k, ns2));
								t3 = _mm512_load_pd(&yfn(i, j - 1, k, ns2));
								t5 = _mm512_mul_pd(t1, t2);
								t6 = _mm512_mul_pd(t3, t4);
								t7 = _mm512_add_pd(t5, t6);
								t9 = _mm512_load_pd(&dqdy_4d(i, j, k, m));
								t9 = _mm512_fmsub_pd(t7, u6, t9);
								t9 = _mm512_mul_pd(t9, t8);
								_mm512_store_pd(&dqdy_4d(i, j, k, m), t9);
								t9 = _mm512_load_pd(&dqdy_4d(i, j, k - 1, m));
								t9 = _mm512_fmadd_pd(t7, u6, t9);
								_mm512_store_pd(&dqdy_4d(i, j, k - 1, m), t9);

								t1 = _mm512_load_pd(&zfn(i, j, k, ns2));
								t3 = _mm512_load_pd(&zfn(i, j - 1, k, ns2));
								t5 = _mm512_mul_pd(t1, t2);
								t6 = _mm512_mul_pd(t3, t4);
								t7 = _mm512_add_pd(t5, t6);
								t9 = _mm512_load_pd(&dqdz_4d(i, j, k, m));
								t9 = _mm512_fmsub_pd(t7, u6, t9);
								t9 = _mm512_mul_pd(t9, t8);
								_mm512_store_pd(&dqdz_4d(i, j, k, m), t9);
								t9 = _mm512_load_pd(&dqdz_4d(i, j, k - 1, m));
								t9 = _mm512_fmadd_pd(t7, u6, t9);
								_mm512_store_pd(&dqdz_4d(i, j, k - 1, m), t9);




								t1 = _mm512_load_pd(&q_4d(i, j, k, m));
								t2 = _mm512_load_pd(&q_4d(i, j - 1, k, m));
								t3 = _mm512_load_pd(&q_4d(i - 1, j, k, m));
								t4 = _mm512_load_pd(&q_4d(i - 1, j - 1, k, m));
								t5 = _mm512_set1_pd(fourth);
								t6 = _mm512_add_pd(t1, t2);
								t6 = _mm512_add_pd(t3, t6);
								t6 = _mm512_add_pd(t4, t6);
								t6 = _mm512_mul_pd(t5, t6);

								u1 = _mm512_load_pd(&xfn(i, j, k, ns3));
								u2 = _mm512_load_pd(&area(i, j, k, ns3));
								u3 = _mm512_load_pd(&xfn(i, j - 1, k, ns3));
								u4 = _mm512_load_pd(&area(i, j - 1, k, ns3));
								u5 = _mm512_mul_pd(u1, u2);
								u6 = _mm512_mul_pd(u3, u4);
								u7 = _mm512_add_pd(u5, u6);
								u8 = _mm512_load_pd(&dqdx_4d(i, j, k, m));
								u9 = _mm512_set1_pd((double)-1);
								u8 = _mm512_fmsub_pd(u7, t6, u8);
								u8 = _mm512_mul_pd(u8, u9);
								_mm512_store_pd(&dqdx_4d(i, j, k, m), u8);
								u8 = _mm512_load_pd(&dqdx_4d(i - 1, j, k, m));
								u8 = _mm512_fmadd_pd(u7, t6, u8);
								_mm512_store_pd(&dqdx_4d(i - 1, j, k, m), u8);

								u1 = _mm512_load_pd(&yfn(i, j, k, ns3));
								u3 = _mm512_load_pd(&yfn(i, j - 1, k, ns3));
								u5 = _mm512_mul_pd(u1, u2);
								u6 = _mm512_mul_pd(u3, u4);
								u7 = _mm512_add_pd(u5, u6);
								u8 = _mm512_load_pd(&dqdy_4d(i, j, k, m));
								u8 = _mm512_fmsub_pd(u7, t6, u8);
								u8 = _mm512_mul_pd(u8, u9);
								_mm512_store_pd(&dqdy_4d(i, j, k, m), u8);
								u8 = _mm512_load_pd(&dqdy_4d(i - 1, j, k, m));
								u8 = _mm512_fmadd_pd(u7, t6, u8);
								_mm512_store_pd(&dqdy_4d(i - 1, j, k, m), u8);

								u1 = _mm512_load_pd(&zfn(i, j, k, ns3));
								u3 = _mm512_load_pd(&zfn(i, j - 1, k, ns3));
								u5 = _mm512_mul_pd(u1, u2);
								u6 = _mm512_mul_pd(u3, u4);
								u7 = _mm512_add_pd(u5, u6);
								u8 = _mm512_load_pd(&dqdz_4d(i, j, k, m));
								u8 = _mm512_fmsub_pd(u7, t6, u8);
								u8 = _mm512_mul_pd(u8, u9);
								_mm512_store_pd(&dqdz_4d(i, j, k, m), u8);
								u8 = _mm512_load_pd(&dqdz_4d(i - 1, j, k, m));
								u8 = _mm512_fmadd_pd(u7, t6, u8);
								_mm512_store_pd(&dqdz_4d(i - 1, j, k, m), u8);
							}
							ts1 = _mm256_load_pd(&xfn(i, j, k, ns1));
							ts2 = _mm256_load_pd(&area(i, j, k, ns1));
							ts3 = _mm256_load_pd(&xfn(i, j - 1, k, ns1));
							ts4 = _mm256_load_pd(&area(i, j - 1, k, ns1));
							ts5 = _mm256_mul_pd(ts1, ts2);
							ts6 = _mm256_mul_pd(ts3, ts4);
							ts7 = _mm256_add_pd(ts5, ts6);
							ts8 = _mm256_set1_pd((double)-1);
							ts7 = _mm256_mul_pd(ts7, ts8);
							ts9 = _mm256_load_pd(&q_4d(i, j - 1, k, m));
							ts9 = _mm256_mul_pd(ts7, ts9);
							_mm256_store_pd(&dqdx_4d(i, j, k, m), ts9);
							ts10 = _mm256_load_pd(&dqdx_4d(i, j - 1, k, m));
							ts10 = _mm256_sub_pd(ts10, ts9);
							_mm256_store_pd(&dqdx_4d(i, j - 1, k, m), ts10);

							ts1 = _mm256_load_pd(&yfn(i, j, k, ns1));
							ts3 = _mm256_load_pd(&yfn(i, j - 1, k, ns1));
							ts5 = _mm256_mul_pd(ts1, ts2);
							ts6 = _mm256_mul_pd(ts3, ts4);
							ts7 = _mm256_add_pd(ts5, ts6);
							ts7 = _mm256_mul_pd(ts7, ts8);
							ts9 = _mm256_load_pd(&q_4d(i, j - 1, k, m));
							ts9 = _mm256_mul_pd(ts7, ts9);
							_mm256_store_pd(&dqdy_4d(i, j, k, m), ts9);
							ts10 = _mm256_load_pd(&dqdy_4d(i, j - 1, k, m));
							ts10 = _mm256_sub_pd(ts10, ts9);
							_mm256_store_pd(&dqdy_4d(i, j - 1, k, m), ts10);

							ts1 = _mm256_load_pd(&zfn(i, j, k, ns1));
							ts3 = _mm256_load_pd(&zfn(i, j - 1, k, ns1));
							ts5 = _mm256_mul_pd(ts1, ts2);
							ts6 = _mm256_mul_pd(ts3, ts4);
							ts7 = _mm256_add_pd(ts5, ts6);
							ts7 = _mm256_mul_pd(ts7, ts8);
							ts9 = _mm256_load_pd(&q_4d(i, j - 1, k, m));
							ts9 = _mm256_mul_pd(ts7, ts9);
							_mm256_store_pd(&dqdz_4d(i, j, k, m), ts9);
							ts10 = _mm256_load_pd(&dqdz_4d(i, j - 1, k, m));
							ts10 = _mm256_sub_pd(ts10, ts9);
							_mm256_store_pd(&dqdz_4d(i, j - 1, k, m), ts10);
							us0 = _mm256_load_pd(&q_4d(i, j, k, m));
							us1 = _mm256_load_pd(&q_4d(i, j - 1, k, m));
							us2 = _mm256_load_pd(&q_4d(i, j, k - 1, m));
							us3 = _mm256_load_pd(&q_4d(i, j - 1, k - 1, m));
							us4 = _mm256_set1_pd(fourth);
							us5 = _mm256_add_pd(us0, us1);
							us5 = _mm256_add_pd(us2, us5);
							us5 = _mm256_add_pd(us3, us5);
							us5 = _mm256_mul_pd(us4, us5);

							ts1 = _mm256_load_pd(&xfn(i, j, k, ns2));
							ts2 = _mm256_load_pd(&area(i, j, k, ns2));
							ts3 = _mm256_load_pd(&xfn(i, j - 1, k, ns2));
							ts4 = _mm256_load_pd(&area(i, j - 1, k, ns2));
							ts5 = _mm256_mul_pd(ts1, ts2);
							ts6 = _mm256_mul_pd(ts3, ts4);
							ts7 = _mm256_add_pd(ts5, ts6);
							ts9 = _mm256_load_pd(&dqdx_4d(i, j, k, m));
							ts9 = _mm256_fmsub_pd(ts7, us5, ts9);
							ts9 = _mm256_mul_pd(ts9, ts8);
							_mm256_store_pd(&dqdx_4d(i, j, k, m), ts9);
							ts9 = _mm256_load_pd(&dqdx_4d(i, j, k - 1, m));
							ts9 = _mm256_fmadd_pd(ts7, us5, ts9);
							_mm256_store_pd(&dqdx_4d(i, j, k - 1, m), ts9);

							ts1 = _mm256_load_pd(&yfn(i, j, k, ns2));
							ts3 = _mm256_load_pd(&yfn(i, j - 1, k, ns2));
							ts5 = _mm256_mul_pd(ts1, ts2);
							ts6 = _mm256_mul_pd(ts3, ts4);
							ts7 = _mm256_add_pd(ts5, ts6);
							ts9 = _mm256_load_pd(&dqdy_4d(i, j, k, m));
							ts9 = _mm256_fmsub_pd(ts7, us5, ts9);
							ts9 = _mm256_mul_pd(ts9, ts8);
							_mm256_store_pd(&dqdy_4d(i, j, k, m), ts9);
							ts9 = _mm256_load_pd(&dqdy_4d(i, j, k - 1, m));
							ts9 = _mm256_fmadd_pd(ts7, us5, ts9);
							_mm256_store_pd(&dqdy_4d(i, j, k - 1, m), ts9);

							ts1 = _mm256_load_pd(&zfn(i, j, k, ns2));
							ts3 = _mm256_load_pd(&zfn(i, j - 1, k, ns2));
							ts5 = _mm256_mul_pd(ts1, ts2);
							ts6 = _mm256_mul_pd(ts3, ts4);
							ts7 = _mm256_add_pd(ts5, ts6);
							ts9 = _mm256_load_pd(&dqdz_4d(i, j, k, m));
							ts9 = _mm256_fmsub_pd(ts7, us5, ts9);
							ts9 = _mm256_mul_pd(ts9, ts8);
							_mm256_store_pd(&dqdz_4d(i, j, k, m), ts9);
							ts9 = _mm256_load_pd(&dqdz_4d(i, j, k - 1, m));
							ts9 = _mm256_fmadd_pd(ts7, us5, ts9);
							_mm256_store_pd(&dqdz_4d(i, j, k - 1, m), ts9);




							bl1 = _mm256_load_pd(&q_4d(i, j, k, m));
							ts1 = _mm256_load_pd(&q_4d(i, j - 1, k, m));
							ts2 = _mm256_load_pd(&q_4d(i - 1, j, k, m));
							ts3 = _mm256_load_pd(&q_4d(i - 1, j - 1, k, m));
							ts4 = _mm256_set1_pd(fourth);
							ts5 = _mm256_add_pd(bl1, ts1);
							ts5 = _mm256_add_pd(ts2, ts5);
							ts5 = _mm256_add_pd(ts3, ts5);
							ts5 = _mm256_mul_pd(ts4, ts5);

							us1 = _mm256_load_pd(&xfn(i, j, k, ns3));
							us2 = _mm256_load_pd(&area(i, j, k, ns3));
							us3 = _mm256_load_pd(&xfn(i, j - 1, k, ns3));
							us4 = _mm256_load_pd(&area(i, j - 1, k, ns3));
							us5 = _mm256_mul_pd(us1, us2);
							us6 = _mm256_mul_pd(us3, us4);
							us7 = _mm256_add_pd(us5, us6);
							us8 = _mm256_load_pd(&dqdx_4d(i, j, k, m));
							us9 = _mm256_set1_pd((double)-1);
							us8 = _mm256_fmsub_pd(us7, ts5, us8);
							us8 = _mm256_mul_pd(us8, us9);
							_mm256_store_pd(&dqdx_4d(i, j, k, m), us8);
							us8 = _mm256_load_pd(&dqdx_4d(i - 1, j, k, m));
							us8 = _mm256_fmadd_pd(us7, ts5, us8);
							_mm256_store_pd(&dqdx_4d(i - 1, j, k, m), us8);

							us1 = _mm256_load_pd(&yfn(i, j, k, ns3));
							us3 = _mm256_load_pd(&yfn(i, j - 1, k, ns3));
							us5 = _mm256_mul_pd(us1, us2);
							us6 = _mm256_mul_pd(us3, us4);
							us7 = _mm256_add_pd(us5, us6);
							us8 = _mm256_load_pd(&dqdy_4d(i, j, k, m));
							us8 = _mm256_fmsub_pd(us7, ts5, us8);
							us8 = _mm256_mul_pd(us8, us9);
							_mm256_store_pd(&dqdy_4d(i, j, k, m), us8);
							us8 = _mm256_load_pd(&dqdy_4d(i - 1, j, k, m));
							us8 = _mm256_fmadd_pd(us7, ts5, us8);
							_mm256_store_pd(&dqdy_4d(i - 1, j, k, m), us8);

							us1 = _mm256_load_pd(&zfn(i, j, k, ns3));
							us3 = _mm256_load_pd(&zfn(i, j - 1, k, ns3));
							us5 = _mm256_mul_pd(us1, us2);
							us6 = _mm256_mul_pd(us3, us4);
							us7 = _mm256_add_pd(us5, us6);
							us8 = _mm256_load_pd(&dqdz_4d(i, j, k, m));
							us8 = _mm256_fmsub_pd(us7, ts5, us8);
							us8 = _mm256_mul_pd(us8, us9);
							_mm256_store_pd(&dqdz_4d(i, j, k, m), us8);
							us8 = _mm256_load_pd(&dqdz_4d(i - 1, j, k, m));
							us8 = _mm256_fmadd_pd(us7, ts5, us8);
							_mm256_store_pd(&dqdz_4d(i - 1, j, k, m), us8);
						}
					}
				}
				//__m512d t1, t2, t3, t4, t5, t6, u1, u2, u3, u4, u5, u6, u7, u8, u9;
				//__m256d bl1, ts1, ts2, ts3, ts4, ts5, us1, us2, us3, us4, us5, us6, us7, us8, us9;
#pragma omp for private(k,j,kj,i)
				for (kj = 1; kj < ((nj + 1) * (nk + 1)); kj++) {
					k = kj / (nj + 1);
					j = kj % (nj + 1);
					for (i = 1; i < ni + 1 - 8; i += 8)
					{
						t1 = _mm512_load_pd(&vol(i, j, k));
						t2 = _mm512_load_pd(&vol(i, j - 1, k));
						t3 = _mm512_set1_pd((double)1.0);
						t4 = _mm512_add_pd(t1, t2);
						t4 = _mm512_div_pd(t3, t4);
						_mm512_store_pd(&workqm(i, j, k), t4);
					}
					ts1 = _mm256_load_pd(&vol(i, j, k));
					ts2 = _mm256_load_pd(&vol(i, j - 1, k));
					ts3 = _mm256_set1_pd((double)1.0);
					ts4 = _mm256_add_pd(ts1, ts2);
					ts4 = _mm256_div_pd(ts3, ts4);
					_mm256_store_pd(&workqm(i, j, k), ts4);
				}


			}
			else
			{
				double nt1 = rdtsc();
				int i, j, k, m, kj;
				__m512d t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10;
				__m256d ts1, ts2, ts3, ts4, ts5, ts6, ts7, ts8, ts9, ts10, us0, us1, us2, us3, us4, us5, us6, us7, us8, us9, us10, bl1;
#pragma omp  for collapse(2) private(j,i,k,m) 
				for (m = mst; m <= med; ++m)
				{
					for (j = 1; j < nj + 1; ++j)
					{
						for (k = 1; k < nk + 1; ++k)
						{
							for (i = 1; i < ni + 1 - 8; i += 8)
							{
								t1 = _mm512_load_pd(&xfn(i, j, k, ns1));
								t2 = _mm512_load_pd(&area(i, j, k, ns1));
								t3 = _mm512_load_pd(&xfn(i, j, k - 1, ns1));
								t4 = _mm512_load_pd(&area(i, j, k - 1, ns1));
								t5 = _mm512_mul_pd(t1, t2);
								t6 = _mm512_mul_pd(t3, t4);
								t7 = _mm512_add_pd(t5, t6);
								t8 = _mm512_set1_pd((double)-1);
								t7 = _mm512_mul_pd(t7, t8);
								t9 = _mm512_load_pd(&q_4d(i, j, k - 1, m));
								t9 = _mm512_mul_pd(t7, t9);
								_mm512_store_pd(&dqdx_4d(i, j, k, m), t9);
								t10 = _mm512_load_pd(&dqdx_4d(i, j, k - 1, m));
								t10 = _mm512_sub_pd(t10, t9);
								_mm512_store_pd(&dqdx_4d(i, j, k - 1, m), t10);

								t1 = _mm512_load_pd(&yfn(i, j, k, ns1));
								t3 = _mm512_load_pd(&yfn(i, j, k - 1, ns1));
								t5 = _mm512_mul_pd(t1, t2);
								t6 = _mm512_mul_pd(t3, t4);
								t7 = _mm512_add_pd(t5, t6);
								t7 = _mm512_mul_pd(t7, t8);
								t9 = _mm512_load_pd(&q_4d(i, j, k - 1, m));
								t9 = _mm512_mul_pd(t7, t9);
								_mm512_store_pd(&dqdy_4d(i, j, k, m), t9);
								t10 = _mm512_load_pd(&dqdy_4d(i, j, k - 1, m));
								t10 = _mm512_sub_pd(t10, t9);
								_mm512_store_pd(&dqdy_4d(i, j, k - 1, m), t10);

								t1 = _mm512_load_pd(&zfn(i, j, k, ns1));
								t3 = _mm512_load_pd(&zfn(i, j, k - 1, ns1));
								t5 = _mm512_mul_pd(t1, t2);
								t6 = _mm512_mul_pd(t3, t4);
								t7 = _mm512_add_pd(t5, t6);
								t7 = _mm512_mul_pd(t7, t8);
								t9 = _mm512_load_pd(&q_4d(i, j, k - 1, m));
								t9 = _mm512_mul_pd(t7, t9);
								_mm512_store_pd(&dqdz_4d(i, j, k, m), t9);
								t10 = _mm512_load_pd(&dqdz_4d(i, j, k - 1, m));
								t10 = _mm512_sub_pd(t10, t9);
								_mm512_store_pd(&dqdz_4d(i, j, k - 1, m), t10);
								u1 = _mm512_load_pd(&q_4d(i, j, k, m));
								u2 = _mm512_load_pd(&q_4d(i, j, k - 1, m));
								u3 = _mm512_load_pd(&q_4d(i - 1, j, k, m));
								u4 = _mm512_load_pd(&q_4d(i - 1, j, k - 1, m));
								u5 = _mm512_set1_pd(fourth);
								u6 = _mm512_add_pd(u1, u2);
								u6 = _mm512_add_pd(u3, u6);
								u6 = _mm512_add_pd(u4, u6);
								u6 = _mm512_mul_pd(u5, u6);

								t1 = _mm512_load_pd(&xfn(i, j, k, ns2));
								t2 = _mm512_load_pd(&area(i, j, k, ns2));
								t3 = _mm512_load_pd(&xfn(i, j, k - 1, ns2));
								t4 = _mm512_load_pd(&area(i, j, k - 1, ns2));
								t5 = _mm512_mul_pd(t1, t2);
								t6 = _mm512_mul_pd(t3, t4);
								t7 = _mm512_add_pd(t5, t6);
								t9 = _mm512_load_pd(&dqdx_4d(i, j, k, m));
								t9 = _mm512_fmsub_pd(t7, u6, t9);
								t9 = _mm512_mul_pd(t9, t8);
								_mm512_store_pd(&dqdx_4d(i, j, k, m), t9);
								t9 = _mm512_load_pd(&dqdx_4d(i - 1, j, k, m));
								t9 = _mm512_fmadd_pd(t7, u6, t9);
								_mm512_store_pd(&dqdx_4d(i - 1, j, k, m), t9);

								t1 = _mm512_load_pd(&yfn(i, j, k, ns2));
								t3 = _mm512_load_pd(&yfn(i, j, k - 1, ns2));
								t5 = _mm512_mul_pd(t1, t2);
								t6 = _mm512_mul_pd(t3, t4);
								t7 = _mm512_add_pd(t5, t6);
								t9 = _mm512_load_pd(&dqdy_4d(i, j, k, m));
								t9 = _mm512_fmsub_pd(t7, u6, t9);
								t9 = _mm512_mul_pd(t9, t8);
								_mm512_store_pd(&dqdy_4d(i, j, k, m), t9);
								t9 = _mm512_load_pd(&dqdy_4d(i - 1, j, k, m));
								t9 = _mm512_fmadd_pd(t7, u6, t9);
								_mm512_store_pd(&dqdy_4d(i - 1, j, k, m), t9);

								t1 = _mm512_load_pd(&zfn(i, j, k, ns2));
								t3 = _mm512_load_pd(&zfn(i, j, k - 1, ns2));
								t5 = _mm512_mul_pd(t1, t2);
								t6 = _mm512_mul_pd(t3, t4);
								t7 = _mm512_add_pd(t5, t6);
								t9 = _mm512_load_pd(&dqdz_4d(i, j, k, m));
								t9 = _mm512_fmsub_pd(t7, u6, t9);
								t9 = _mm512_mul_pd(t9, t8);
								_mm512_store_pd(&dqdz_4d(i, j, k, m), t9);
								t9 = _mm512_load_pd(&dqdz_4d(i - 1, j, k, m));
								t9 = _mm512_fmadd_pd(t7, u6, t9);
								_mm512_store_pd(&dqdz_4d(i - 1, j, k, m), t9);
							}
							ts1 = _mm256_load_pd(&xfn(i, j, k, ns1));
							ts2 = _mm256_load_pd(&area(i, j, k, ns1));
							ts3 = _mm256_load_pd(&xfn(i, j, k - 1, ns1));
							ts4 = _mm256_load_pd(&area(i, j, k - 1, ns1));
							ts5 = _mm256_mul_pd(ts1, ts2);
							ts6 = _mm256_mul_pd(ts3, ts4);
							ts7 = _mm256_add_pd(ts5, ts6);
							ts8 = _mm256_set1_pd((double)-1);
							ts7 = _mm256_mul_pd(ts7, ts8);
							ts9 = _mm256_load_pd(&q_4d(i, j, k - 1, m));
							ts9 = _mm256_mul_pd(ts7, ts9);
							_mm256_store_pd(&dqdx_4d(i, j, k, m), ts9);
							ts10 = _mm256_load_pd(&dqdx_4d(i, j, k - 1, m));
							ts10 = _mm256_sub_pd(ts10, ts9);
							_mm256_store_pd(&dqdx_4d(i, j, k - 1, m), ts10);

							ts1 = _mm256_load_pd(&yfn(i, j, k, ns1));
							ts3 = _mm256_load_pd(&yfn(i, j, k - 1, ns1));
							ts5 = _mm256_mul_pd(ts1, ts2);
							ts6 = _mm256_mul_pd(ts3, ts4);
							ts7 = _mm256_add_pd(ts5, ts6);
							ts7 = _mm256_mul_pd(ts7, ts8);
							ts9 = _mm256_load_pd(&q_4d(i, j, k - 1, m));
							ts9 = _mm256_mul_pd(ts7, ts9);
							_mm256_store_pd(&dqdy_4d(i, j, k, m), ts9);
							ts10 = _mm256_load_pd(&dqdy_4d(i, j, k - 1, m));
							ts10 = _mm256_sub_pd(ts10, ts9);
							_mm256_store_pd(&dqdy_4d(i, j, k - 1, m), ts10);

							ts1 = _mm256_load_pd(&zfn(i, j, k, ns1));
							ts3 = _mm256_load_pd(&zfn(i, j, k - 1, ns1));
							ts5 = _mm256_mul_pd(ts1, ts2);
							ts6 = _mm256_mul_pd(ts3, ts4);
							ts7 = _mm256_add_pd(ts5, ts6);
							ts7 = _mm256_mul_pd(ts7, ts8);
							ts9 = _mm256_load_pd(&q_4d(i, j, k - 1, m));
							ts9 = _mm256_mul_pd(ts7, ts9);
							_mm256_store_pd(&dqdz_4d(i, j, k, m), ts9);
							ts10 = _mm256_load_pd(&dqdz_4d(i, j, k - 1, m));
							ts10 = _mm256_sub_pd(ts10, ts9);
							_mm256_store_pd(&dqdz_4d(i, j, k - 1, m), ts10);
							us1 = _mm256_load_pd(&q_4d(i, j, k, m));
							us2 = _mm256_load_pd(&q_4d(i, j, k - 1, m));
							us3 = _mm256_load_pd(&q_4d(i - 1, j, k, m));
							us4 = _mm256_load_pd(&q_4d(i - 1, j, k - 1, m));
							us5 = _mm256_set1_pd(fourth);
							us6 = _mm256_add_pd(us1, us2);
							us6 = _mm256_add_pd(us3, us6);
							us6 = _mm256_add_pd(us4, us6);
							us6 = _mm256_mul_pd(us5, us6);

							ts1 = _mm256_load_pd(&xfn(i, j, k, ns2));
							ts2 = _mm256_load_pd(&area(i, j, k, ns2));
							ts3 = _mm256_load_pd(&xfn(i, j, k - 1, ns2));
							ts4 = _mm256_load_pd(&area(i, j, k - 1, ns2));
							ts5 = _mm256_mul_pd(ts1, ts2);
							ts6 = _mm256_mul_pd(ts3, ts4);
							ts7 = _mm256_add_pd(ts5, ts6);
							ts9 = _mm256_load_pd(&dqdx_4d(i, j, k, m));
							ts9 = _mm256_fmsub_pd(ts7, us6, ts9);
							ts9 = _mm256_mul_pd(ts9, ts8);
							_mm256_store_pd(&dqdx_4d(i, j, k, m), ts9);
							ts9 = _mm256_load_pd(&dqdx_4d(i - 1, j, k, m));
							ts9 = _mm256_fmadd_pd(ts7, us6, ts9);
							_mm256_store_pd(&dqdx_4d(i - 1, j, k, m), ts9);

							ts1 = _mm256_load_pd(&yfn(i, j, k, ns2));
							ts3 = _mm256_load_pd(&yfn(i, j, k - 1, ns2));
							ts5 = _mm256_mul_pd(ts1, ts2);
							ts6 = _mm256_mul_pd(ts3, ts4);
							ts7 = _mm256_add_pd(ts5, ts6);
							ts9 = _mm256_load_pd(&dqdy_4d(i, j, k, m));
							ts9 = _mm256_fmsub_pd(ts7, us6, ts9);
							ts9 = _mm256_mul_pd(ts9, ts8);
							_mm256_store_pd(&dqdy_4d(i, j, k, m), ts9);
							ts9 = _mm256_load_pd(&dqdy_4d(i - 1, j, k, m));
							ts9 = _mm256_fmadd_pd(ts7, us6, ts9);
							_mm256_store_pd(&dqdy_4d(i - 1, j, k, m), ts9);

							ts1 = _mm256_load_pd(&zfn(i, j, k, ns2));
							ts3 = _mm256_load_pd(&zfn(i, j, k - 1, ns2));
							ts5 = _mm256_mul_pd(ts1, ts2);
							ts6 = _mm256_mul_pd(ts3, ts4);
							ts7 = _mm256_add_pd(ts5, ts6);
							ts9 = _mm256_load_pd(&dqdz_4d(i, j, k, m));
							ts9 = _mm256_fmsub_pd(ts7, us6, ts9);
							ts9 = _mm256_mul_pd(ts9, ts8);
							_mm256_store_pd(&dqdz_4d(i, j, k, m), ts9);
							ts9 = _mm256_load_pd(&dqdz_4d(i - 1, j, k, m));
							ts9 = _mm256_fmadd_pd(ts7, us6, ts9);
							_mm256_store_pd(&dqdz_4d(i - 1, j, k, m), ts9);
						}
					}
				}
#pragma omp for collapse(3) private(m,j,k,i) schedule(static) 
				for (m = mst; m <= med; ++m)
				{
					for (k = 1; k < nk + 1; ++k)
					{
						for (j = 1; j < nj + 1; ++j)
						{
							for (i = 1; i < ni + 1 - 8; i += 8)
							{
								// worksx(i,j,k) = xfn(i,j,k,ns3) * area(i,j,k,ns3) + xfn(i,j,k-1,ns3) * area(i,j,k-1,ns3);
								// worksy(i,j,k) = yfn(i,j,k,ns3) * area(i,j,k,ns3) + yfn(i,j,k-1,ns3) * area(i,j,k-1,ns3);
								// worksz(i,j,k) = zfn(i,j,k,ns3) * area(i,j,k,ns3) + zfn(i,j,k-1,ns3) * area(i,j,k-1,ns3);
								// workqm(i,j,k) = fourth * ( q_4d(i,j,k,m) + q_4d(i,j,k-1,m) + q_4d(i,j-1,k,m) + q_4d(i,j-1,k-1,m) );
								// dqdx_4d(i,j,k,m) -= worksx(i,j,k) * workqm(i,j,k);
								// dqdy_4d(i,j,k,m) -= worksy(i,j,k) * workqm(i,j,k);
								// dqdz_4d(i,j,k,m) -= worksz(i,j,k) * workqm(i,j,k);
								// dqdx_4d(i,j-1,k,m) += worksx(i,j,k) * workqm(i,j,k);
								// dqdy_4d(i,j-1,k,m) += worksy(i,j,k) * workqm(i,j,k);
								// dqdz_4d(i,j-1,k,m) += worksz(i,j,k) * workqm(i,j,k);
								t1 = _mm512_load_pd(&q_4d(i, j, k, m));
								t2 = _mm512_load_pd(&q_4d(i, j, k - 1, m));
								t3 = _mm512_load_pd(&q_4d(i, j - 1, k, m));
								t4 = _mm512_load_pd(&q_4d(i, j - 1, k - 1, m));
								t5 = _mm512_set1_pd(fourth);
								t6 = _mm512_add_pd(t1, t2);
								t6 = _mm512_add_pd(t3, t6);
								t6 = _mm512_add_pd(t4, t6);
								t6 = _mm512_mul_pd(t5, t6);

								u1 = _mm512_load_pd(&xfn(i, j, k, ns3));
								u2 = _mm512_load_pd(&area(i, j, k, ns3));
								u3 = _mm512_load_pd(&xfn(i, j, k - 1, ns3));
								u4 = _mm512_load_pd(&area(i, j, k - 1, ns3));
								u5 = _mm512_mul_pd(u1, u2);
								u7 = _mm512_mul_pd(u3, u4);
								u8 = _mm512_add_pd(u5, u7);
								u9 = _mm512_load_pd(&dqdx_4d(i, j, k, m));
								u10 = _mm512_set1_pd((double)-1);
								u9 = _mm512_fmsub_pd(u8, t6, u9);
								u9 = _mm512_mul_pd(u9, u10);
								_mm512_store_pd(&dqdx_4d(i, j, k, m), u9);
								u9 = _mm512_load_pd(&dqdx_4d(i, j - 1, k, m));
								u9 = _mm512_fmadd_pd(u8, t6, u9);
								_mm512_store_pd(&dqdx_4d(i, j - 1, k, m), u9);

								u1 = _mm512_load_pd(&yfn(i, j, k, ns3));
								u3 = _mm512_load_pd(&yfn(i, j, k - 1, ns3));
								u5 = _mm512_mul_pd(u1, u2);
								u7 = _mm512_mul_pd(u3, u4);
								u8 = _mm512_add_pd(u5, u7);
								u9 = _mm512_load_pd(&dqdy_4d(i, j, k, m));
								u9 = _mm512_fmsub_pd(u8, t6, u9);
								u9 = _mm512_mul_pd(u9, u10);
								_mm512_store_pd(&dqdy_4d(i, j, k, m), u9);
								u9 = _mm512_load_pd(&dqdy_4d(i, j - 1, k, m));
								u9 = _mm512_fmadd_pd(u8, t6, u9);
								_mm512_store_pd(&dqdy_4d(i, j - 1, k, m), u9);

								u1 = _mm512_load_pd(&zfn(i, j, k, ns3));
								u3 = _mm512_load_pd(&zfn(i, j, k - 1, ns3));
								u5 = _mm512_mul_pd(u1, u2);
								u7 = _mm512_mul_pd(u3, u4);
								u8 = _mm512_add_pd(u5, u7);
								u9 = _mm512_load_pd(&dqdz_4d(i, j, k, m));
								u9 = _mm512_fmsub_pd(u8, t6, u9);
								u9 = _mm512_mul_pd(u9, u10);
								_mm512_store_pd(&dqdz_4d(i, j, k, m), u9);
								u9 = _mm512_load_pd(&dqdz_4d(i, j - 1, k, m));
								u9 = _mm512_fmadd_pd(u8, t6, u9);
								_mm512_store_pd(&dqdz_4d(i, j - 1, k, m), u9);
							}
							ts1 = _mm256_load_pd(&q_4d(i, j, k, m));
							ts2 = _mm256_load_pd(&q_4d(i, j, k - 1, m));
							ts3 = _mm256_load_pd(&q_4d(i, j - 1, k, m));
							ts4 = _mm256_load_pd(&q_4d(i, j - 1, k - 1, m));
							ts5 = _mm256_set1_pd(fourth);
							ts6 = _mm256_add_pd(ts1, ts2);
							ts6 = _mm256_add_pd(ts3, ts6);
							ts6 = _mm256_add_pd(ts4, ts6);
							ts6 = _mm256_mul_pd(ts5, ts6);

							us1 = _mm256_load_pd(&xfn(i, j, k, ns3));
							us2 = _mm256_load_pd(&area(i, j, k, ns3));
							us3 = _mm256_load_pd(&xfn(i, j, k - 1, ns3));
							us4 = _mm256_load_pd(&area(i, j, k - 1, ns3));
							us5 = _mm256_mul_pd(us1, us2);
							us7 = _mm256_mul_pd(us3, us4);
							us8 = _mm256_add_pd(us5, us7);
							us9 = _mm256_load_pd(&dqdx_4d(i, j, k, m));
							us10 = _mm256_set1_pd((double)-1);
							us9 = _mm256_fmsub_pd(us8, ts6, us9);
							us9 = _mm256_mul_pd(us9, us10);
							_mm256_store_pd(&dqdx_4d(i, j, k, m), us9);
							us9 = _mm256_load_pd(&dqdx_4d(i, j - 1, k, m));
							us9 = _mm256_fmadd_pd(us8, ts6, us9);
							_mm256_store_pd(&dqdx_4d(i, j - 1, k, m), us9);

							us1 = _mm256_load_pd(&yfn(i, j, k, ns3));
							us3 = _mm256_load_pd(&yfn(i, j, k - 1, ns3));
							us5 = _mm256_mul_pd(us1, us2);
							us7 = _mm256_mul_pd(us3, us4);
							us8 = _mm256_add_pd(us5, us7);
							us9 = _mm256_load_pd(&dqdy_4d(i, j, k, m));
							us9 = _mm256_fmsub_pd(us8, ts6, us9);
							us9 = _mm256_mul_pd(us9, us10);
							_mm256_store_pd(&dqdy_4d(i, j, k, m), us9);
							us9 = _mm256_load_pd(&dqdy_4d(i, j - 1, k, m));
							us9 = _mm256_fmadd_pd(us8, ts6, us9);
							_mm256_store_pd(&dqdy_4d(i, j - 1, k, m), us9);

							us1 = _mm256_load_pd(&zfn(i, j, k, ns3));
							us3 = _mm256_load_pd(&zfn(i, j, k - 1, ns3));
							us5 = _mm256_mul_pd(us1, us2);
							us7 = _mm256_mul_pd(us3, us4);
							us8 = _mm256_add_pd(us5, us7);
							us9 = _mm256_load_pd(&dqdz_4d(i, j, k, m));
							us9 = _mm256_fmsub_pd(us8, ts6, us9);
							us9 = _mm256_mul_pd(us9, us10);
							_mm256_store_pd(&dqdz_4d(i, j, k, m), us9);
							us9 = _mm256_load_pd(&dqdz_4d(i, j - 1, k, m));
							us9 = _mm256_fmadd_pd(us8, ts6, us9);
							_mm256_store_pd(&dqdz_4d(i, j - 1, k, m), us9);
						}
					}
				}

#pragma omp for private(k,j,kj,i) schedule(static)
				for (kj = 1; kj < ((nj + 1) * (nk + 1)); kj++) {
					k = kj / (nj + 1);
					j = kj % (nj + 1);
					for (i = 1; i < ni + 1 - 8; i += 8)
					{
						t1 = _mm512_load_pd(&vol(i, j, k));
						t2 = _mm512_load_pd(&vol(i, j, k - 1));
						t3 = _mm512_set1_pd((double)1.0);
						t4 = _mm512_add_pd(t1, t2);
						t4 = _mm512_div_pd(t3, t4);
						_mm512_store_pd(&workqm(i, j, k), t4);
					}
					ts1 = _mm256_load_pd(&vol(i, j, k));
					ts2 = _mm256_load_pd(&vol(i, j, k - 1));
					ts3 = _mm256_set1_pd((double)1.0);
					ts4 = _mm256_add_pd(ts1, ts2);
					ts4 = _mm256_div_pd(ts3, ts4);
					_mm256_store_pd(&workqm(i, j, k), ts4);
				}

#pragma omp for collapse(3) private(k,j,kj,i) schedule(static)
				for (m = mst; m <= med; ++m)
				{

					for (k = 1; k < nk + 1; ++k)
					{
						for (j = 1; j < nj + 1; ++j)
						{
							for (i = 1; i < ni + 1 - 8; i += 8)
							{
								__m512d t1 = _mm512_load_pd(&workqm(i, j, k));
								__m512d t2 = _mm512_load_pd(&dqdx_4d(i, j, k, m));
								__m512d t3 = _mm512_load_pd(&dqdy_4d(i, j, k, m));
								__m512d t4 = _mm512_load_pd(&dqdz_4d(i, j, k, m));
								t2 = _mm512_mul_pd(t2, t1);
								t3 = _mm512_mul_pd(t3, t1);
								t4 = _mm512_mul_pd(t4, t1);
								_mm512_store_pd(&dqdx_4d(i, j, k, m), t2);
								_mm512_store_pd(&dqdy_4d(i, j, k, m), t3);
								_mm512_store_pd(&dqdz_4d(i, j, k, m), t4);
							}
							ts1 = _mm256_load_pd(&workqm(i, j, k));
							ts2 = _mm256_load_pd(&dqdx_4d(i, j, k, m));
							ts3 = _mm256_load_pd(&dqdy_4d(i, j, k, m));
							ts4 = _mm256_load_pd(&dqdz_4d(i, j, k, m));
							ts2 = _mm256_mul_pd(ts2, ts1);
							ts3 = _mm256_mul_pd(ts3, ts1);
							ts4 = _mm256_mul_pd(ts4, ts1);
							_mm256_store_pd(&dqdx_4d(i, j, k, m), ts2);
							_mm256_store_pd(&dqdy_4d(i, j, k, m), ts3);
							_mm256_store_pd(&dqdz_4d(i, j, k, m), ts4);
						}
					}
				}


			}
		}





		// 该方向界面梯度值被计算出来后，会用于粘性通量计算，该值使用后下一方向会重新赋0计算
	}
	//----------------------------------------------------
	//以下为正确性对比部分，不可修改！
	//----------------------------------------------------
	end = rdtsc();
	elapsed = (end - start) / (F*Time);
	cout << "The programe elapsed " << elapsed << setprecision(8) << " s" << endl;
	if (!preccheck(dqdx_4d, dqdy_4d, dqdz_4d))
		cout << "Result check passed!" << endl;
	return 0;
}

int preccheck(RDouble4D dqdx_4d, RDouble4D dqdy_4d, RDouble4D dqdz_4d)
{
	double tmp, real;
	ifstream file("check.txt", std::ofstream::binary);
	if (!file)
	{
		cout << "Error opening check file! ";
		exit(1);
	}
	for (int i = 0; i < ni; ++i)
	{
		for (int j = 0; j < nj; ++j)
		{
			for (int k = 0; k < nk; ++k)
			{
				for (int m = 0; m < 3; ++m)
				{
					file.read(reinterpret_cast<char*>(&tmp), sizeof(double));
					if (fabs(dqdx_4d(i, j, k, m) - tmp) > 1e-6)
					{
						real = dqdx_4d(i, j, k, m);
						cout << "Precision check failed !" << endl;
						cout << "Your result is " << setprecision(15) << real << endl;
						cout << "The Standard result is " << setprecision(15) << tmp << endl;
						cout << "The wrong position is " << endl;
						cout << "i=" << i << ",j=" << j << ",k=" << k << ",m=" << m << endl;
						exit(1);
					}

					file.read(reinterpret_cast<char*>(&tmp), sizeof(double));
					if (fabs(dqdy_4d(i, j, k, m) - tmp) > 1e-6)
					{
						real = dqdy_4d(i, j, k, m);
						cout << "Precision check failed !" << endl;
						cout << "Your result is " << setprecision(15) << real << endl;
						cout << "The Standard result is " << setprecision(15) << tmp << endl;
						cout << "The wrong position is " << endl;
						cout << "i=" << i << ",j=" << j << ",k=" << k << ",m=" << m << endl;
						exit(1);
					}

					file.read(reinterpret_cast<char*>(&tmp), sizeof(double));
					if (fabs(dqdz_4d(i, j, k, m) - tmp) > 1e-6)
					{
						real = dqdz_4d(i, j, k, m);
						cout << "Precision check failed !" << endl;
						cout << "Your result is " << setprecision(15) << real << endl;
						cout << "The Standard result is " << setprecision(15) << tmp << endl;
						cout << "The wrong position is " << endl;
						cout << "i=" << i << ",j=" << j << ",k=" << k << ",m=" << m << endl;
						exit(1);
					}
				}
			}
		}
	}
	file.close();
	return 0;
}
