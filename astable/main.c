/*
 * main.c
 *
 *  Created on: Jul 11, 2013
 *      Author: Daniel O'Malley (omalled@lanl.gov)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "astable.h"

int main( int argc, char **argv )
{
	double alpha;
	double beta;
	double x;
	double y;
	double z;
	int i;
	struct integrand_params p;
	alpha = 1.5;
	beta = 1.0;
	/*
	for(x = 1e0; x <= 1e4; x += 1e0)
	{
		printf("%g\n", standard_astable_pdf(x, alpha, beta))
	}
	*/
	z = 0;
	for( i = 0; i < 1e9; i++ )
	{
		//astable_cdf_interp( 1e5 * drand48(), 1. + (i % 3) / 3., ((i % 9) - (i % 3)) / 9., 0., 1., &y);
		symmetric_astable_cdf_interp( 1e5 * drand48(), 0.1 + 1.9 * drand48(), 0., 1., &y );
		z += y;
		//standard_astable_cdf( 1e5 * drand48() - 5e4, 2 * drand48(), 2 * drand48() - 1 );
		//standard_astable_pdf(1e5 * drand48() - 5e4, 2, 2 * drand48() - 1);
		//alpha = drand48();
	}
	printf( "%g\n", log10( z ) );
	return 0;
}
