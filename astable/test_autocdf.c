/*
 * testcdf.c
 *
 *  Created on: Jul 16, 2013
 *      Author: Daniel O'Malley (omalled@lanl.gov)
 */

#include <stdio.h>
#include <stdlib.h>
#include "astable.h"
#include <math.h>

int main( int argc, char **argv )
{
	double a;
	//if(argc == 1) astable_cdf_interp( 0, 1.0, 0.1, 0., 1., &a);
	if( argc == 1 ) printf( "%g\n", standard_astable_cdf( 0, 1., 0.1 ) );
	else amain( argc, argv );
	return 0;
}

int amain( int argc, char **argv )
{
	double alpha;
	double beta;
	double x;
	double percentile;
	double my_percentile;
	double max[11];
	double max_error_percentile[11];
	double my_error_percentile[11];
	int i, j;
	alpha = atof( argv[1] );
	for( j = 0; j < 11; j++ )
	{
		max[j] = -1;
		max_error_percentile[j] = -1.;
	}
	for( i = 0; i < 171; i++ )
	{
		scanf( "%lg", &percentile );
		for( j = 0; j < 11; j++ )
		{
			scanf( "\t%lg", &x );
			//printf("%g, %g, %g, %g\n", percentile, j / 10., max, standard_astable_cdf(x[j], alpha, j / 10.));
			//my_percentile = standard_astable_cdf(x, alpha, j / 10.);
			//my_percentile = astable_cdf_interp( x, alpha, j / 10., 0., 1. );
			if( j == 0 ) symmetric_astable_cdf_interp( x, alpha, 0., 1., &my_percentile );
			else my_percentile = astable_cdf( x, alpha, j / 10., 0., 1. );
			if( max[j] < fabs( my_percentile - percentile ) )
			{
				//printf("percentile: %g, my value: %g, x: %g, beta: %g, difference: %g\n", percentile, standard_astable_cdf(x, alpha, j / 10.), x, j / 10., fabs(percentile - my_percentile));
				max[j] = my_percentile - percentile;
				max_error_percentile[j] = percentile;
				my_error_percentile[j] = my_percentile;
			}
		}
		scanf( "\n" );
	}
	for( j = 0; j < 11; j++ )
	{
		printf( "beta = %g, max error = %g, max error percentile = %g, my percentile: %g\n", j / 10., max[j], max_error_percentile[j], my_error_percentile[j] );
	}
	return 0;
	/*
	for(x = 0.; x < 1.01; x += 0.1)
	{
		printf("%g\n", standard_astable_pdf(x, alpha, beta));
	}
	*/
	//for(x = -5; x < 5.001; x += 0.1) printf("%g\n", standard_astable_pdf(x, alpha, beta));
	//for(x = -6; x < 6.001; x += .25) printf("%g:\t%g\t%g\n", x, standard_astable_cdf(x, alpha, beta), standard_astable_cdf(x, alpha, beta));
	/*
	for(x = 0; x < 1e4; x++)
	{
		//printf("%d: ", (int)x);
		standard_astable_cdf(100*(drand48() - .5), 0.1 + 1.8*drand48(), .99*(drand48() - .5));
	}
	*/
	//printf("%g\n", standard_astable_cdf(-3.0, alpha, beta));
	return 0;
}
