#include <math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/* Build the cholesky decomposition of the covariance matrix of
 * a fractional Brownian motion assuming sigma=1.
 * */
gsl_matrix *get_chol_covmat(double H, const double *t, int length)
{
	gsl_matrix *covmat;
	double cov;
	double *t_pow_h;
	double twoH;
	int i, j;

	twoH = 2 * H;
	//Build the covariance matrix -- note we don't multiply by 0.5 * sigma, because we only need to do it once when we evaluate x \cdot \Sigma^{-1} x
	covmat = gsl_matrix_alloc(length, length);
	//Do the diagonal
	//Build up an array of the form t^{2H}, because we will use the calculations repeatedly
	//And we might as well set up gsl_x while we're at it
	t_pow_h = (double *)malloc(sizeof(double) * length);
	for(i = 0; i < length; i++)
	{
		t_pow_h[i] = pow(t[i], twoH);
		gsl_matrix_set(covmat, i, i, 2 * t_pow_h[i]);
	}
	//Do the off-diagonal
	for(i = 1; i < length; i++)
	{
		for(j = 0; j < i; j++)
		{
			cov = t_pow_h[i] + t_pow_h[j] - pow(fabs(t[i] - t[j]), twoH);
			gsl_matrix_set(covmat, i, j, cov);
			/* We don't need to set the upper triangular part, because gsl_linalg_cholesky_decomp doesn't look at that part
			gsl_matrix_set(covmat, j, i, cov);
			*/
		}
	}

	//After the next line, covmat contains the cholesky decomposition of the covmat, rather than the covmat
	gsl_linalg_cholesky_decomp(covmat);

	free(t_pow_h);

	return covmat;
}

/*
double **getTrajectories(double sigma, double H, const double *t, int length, int num_trajs)
{
	const gsl_rng_type *T;
	gsl_rng *r;
	gsl_matrix *chol_covmat;
	gsl_vector *traj;
	gsl_vector *rand_norms;
	double **trajs;
	int i, j;

	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc(T);

	chol_covmat = get_chol_covmat(H, t, length);

	trajs = (double **)malloc(num_trajs * sizeof(double *));
	rand_norms = gsl_vector_alloc(length);
	for(i = 0; i < num_trajs; i++)
	{
		trajs[i] = (double *)malloc(length * sizeof(double));
		for(j = 0; j < length; j++)
		{
			gsl_vector_set(rand_norms, j, gsl_ran_gaussian(r, sigma));
		}
		gsl_blas_ddot(rand_norms, chol_covmat, &traj);
		for(j = 0; j < length; j++)
		{
			trajs[i][j] = gsl_vector_get(traj, j);
		}
		gsl_vector_free(traj);
	}

	gsl_matrix_free(chol_covmat);
	gsl_vector_free(rand_norms);
	gsl_rng_free(r);

	return trajs;
}
*/

double log_likelihood(double sigma, double H, const double *t, const double *x, int length)
{
	gsl_matrix *chol_covmat;
	gsl_vector *gsl_x;
	gsl_vector *covinv_x;
	double cov;
	double *t_pow_h;
	double twoH;
	double logdet;
	double x_covinv_x;
	double numerator, denominator;//these are really the log of the numerator, denominator
	int i, j;

	twoH = 2 * H;
	gsl_x = gsl_vector_alloc(length);
	for(i = 0; i < length; i++)
	{
		gsl_vector_set(gsl_x, i, x[i]);
	}

	chol_covmat = get_chol_covmat(H, t, length);

	logdet = 0.;
	for(i = 0; i < length; i++)
	{
		logdet += log(fabs(gsl_matrix_get(chol_covmat, i, i)));
	}
	logdet += length * log(sqrt(0.5) * sigma);//we didn't multiply by 0.5*sigma^2 when building the covariance matrix, so we add sqrt(0.5*sigma^2) here (the cholesky decomposition is the square root of the matrix, so each term in the sum above essentially should be multiplied by that factor)
	logdet *= 2;

	covinv_x = gsl_vector_alloc(length);
	gsl_linalg_cholesky_solve(chol_covmat, gsl_x, covinv_x);
	gsl_blas_ddot(gsl_x, covinv_x, &x_covinv_x);

	gsl_vector_free(covinv_x);
	gsl_vector_free(gsl_x);
	gsl_matrix_free(chol_covmat);

	denominator = 0.5 * (length * log(2 * M_PI) + logdet);
	numerator = -0.5 * x_covinv_x / (0.5 * sigma * sigma);//here we divide by 0.5*sigma^2, because we didn't multiply by that when building the covariance matrix

	return numerator - denominator;
}
