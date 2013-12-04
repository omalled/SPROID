#include <math.h>
#include "astable/astable.h"

double sym_log_likelihood(double alpha, double gamma, double lambda, const double *dt, const double *dx, int length)
{
	int i;
	double sum;
	double btan;
	double z;
	double retval;

	sum = 0.;
	if(alpha == 1.)
	{
		for(i = 0; i < length; i++)
		{
			symmetric_astable_pdf_interp(dx[i], alpha, gamma * dt[i], lambda * pow(dt[i], 1. / alpha), &retval);
			if(retval == 0) return -1. / 0.;
			sum += log(retval);
		}
	}
	else
	{
		btan = 0. * tan(M_PI * alpha / 2);
		for(i = 0; i < length; i++)
		{
			z = dx[i] + (btan * lambda - gamma) * dt[i];
			z /= lambda * pow(dt[i], 1. / alpha);
			printf("a\n");
			symmetric_astable_pdf_interp(z, alpha, 0., 1., &retval);
			printf("b\n");
			z = standard_astable_pdf(z, alpha, 0.);
			retval /= lambda * pow(dt[i], 1. / alpha);
			if(retval == 0) return -1. / 0.;
			sum += log(retval);
		}
	}

	return sum;
}

double log_likelihood(double alpha, double beta, double gamma, double lambda, const double *dt, const double *dx, int length)
{
	int i;
	double sum;
	double btan;
	double z;

	sum = 0.;
	if(alpha == 1.)
	{
		for(i = 0; i < length; i++)
		{
			z = astable_pdf(dx[i], alpha, beta, gamma * dt[i], lambda * pow(dt[i], 1. / alpha));
			if(z == 0) return -1. / 0.;
			sum += log(z);
		}
	}
	else
	{
		btan = beta * tan(M_PI * alpha / 2);
		for(i = 0; i < length; i++)
		{
			z = dx[i] + (btan * lambda - gamma) * dt[i];
			z /= lambda * pow(dt[i], 1. / alpha);
			z = standard_astable_pdf(z, alpha, beta);
			z /= lambda * pow(dt[i], 1. / alpha);
			if(z == 0) return -1. / 0.;
			sum += log(z);
		}
	}

	return sum;
}
