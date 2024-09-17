# SCC Replication: Documentation + Review

## Files included:
- `main.f90`
- `prep.f90`
- `drawSingle.f90`
- `draw.f90`
- `compute_single.f90`
- `compute_gen.f90`
- `compute_reg.f90`
- `HelperModules/dotops.f90`
- `HelperModules/myfuncs.f90`

## Documentation

### `main.f90`


### `prep.f90`
- `sets2grid`: Sets up grids s2grid and s2alpha
- `phifunc(j,s)`: Calculates $\phi(j,s)=\sqrt{2}*cos(\pi*s*j)$ if $j \neq 0$; $\phi(j,s)=1$ otherwise
- `initFw`: Initializes matrices and vectors and allocates space for them in the memory
- `setSfromga(ga,S)`: Construcs a symmetric matrix S from an array ga
- `mkgammas`: Generates random numbers based on the Faure sequence 
- `initSFs`: Initializes matrices and parameters for structured factor models.
- `initSUs`: Initializes several matrices and parameters used in the computation of certain statistical models
- `getlfweights(sel,w)`: Computes local factor weights based on the selection criteria sel. 
- `setregionwA(r,ir,sel)`: Sets the weight matrix w for a region r based on the selection criteria sel and calculates the corresponding matrices A and B used in regional linear regression
- `setG`: Sets the values of a matrix G
- `loadregions`: loads regional data from files

### `drawSingle.f90`
- `draw_muU`: Uniformly draws mu
- `draw_rho`: Draws rho
- `draw_CoCrho`

### `draw.f90`


### `compute_single.f90`


### `compute_gen.f90`


### `compute_reg.f90`


### `HelperModules/dotops.f90`
- `colscalarr(a,b)`: Takes 2 real numbers a, b as inputs and returns an array containing a and b
- `col_vecr(a,b)`: Takes 2 vectors a and b and returns a 2D array where the first row is a and the second row is b
- `col_vecc(a,b)`: Takes 2 vectors a and b and returns a 2D array where the first column is a and the second column is b
- `col_leftright_v(A,b)`: Takes a matrix A and a vector b and appends b to A as the last column
- `col_leftright_vx(b,A)`: Takes a matrix A and a vector b and prepends b to A as the first column
- `col_leftright_m(A, B)`: Takes two matrices A and B and appends the columns of B to the columns of A
- `col_updown_v(A,b)`: Takes a matrix A and a vector b and appends b to A as the last row
- `col_updown_vx(b,A)`: Takes a vector b and a matrix A and prepends b to A as the first row
- `col_updown_m(A,B)`: Takes two matrices A and B and appends the rows of B to the rows of A
- `dotplusr(A,b)`: Takes a matrix A and a vector b and adds b to each row of A
- `dotplusrx(b,A)`: Takes a vector b and a matrix A and adds b to each row of A
- `dotplusc(A,b)`: Takes a matrix A and a vector b and adds b to each column of A
- `dotpluscx(b,A)`: Takes a vector b and a matrix A and adds b to each column of A
- `dotminusr(A,b)`: Takes a matrix A and vector b and substracts b from each row of A
- `dotminusrx(b,A)`: Takes a vector b and matrix A and substracts b from each row of A
- `dotminusc(A,b)`: Takes a matrix A and a vector b and substracts b from each column of A
- `dotminuscx(b,A)`: Takes a vector b and a matrix A and substracts b from each column of A
- `dottimesr(A,b)`: Takes a matrix A and a vector b and multiplies b with each row of A
- `dottimesrx(b,A)`: Takes a vector b and a matrix A and multiplies b with each row of A
- `dottimesc(A,b)`: Takes a matrix A and a vector b and multiplies b with each column of A
- `dottimescx(b,A)`: Takes a vector b and a matrix A and multiplies b with each column of A 
- `dotdivider(A,b)`: Takes a matrix A and a vector b and divides each row of A by b (elemnt-wise)
- `dotdeviderx(b,A)`: Takes a vector b and a matrix A and divides each row of A by b (elemnt-wise)
- `dotdividec(A,b)`: Takes a matrix A and a vector b and divides each column of A by b (elemnt-wise)
- `dotdividecx(b,A)`: Takes a vector b and a matrix A and divides each column of A by b (elemnt-wise)

### `HelperModules/myfuncs.f90`
- `mkGQxw(GQxw)`: Computes a quadrature rule of a 2d matrix
- `gausscdfinv_s(x)`: Calculates the inverse CDF for a standard normal distribution. It converts its input value x to the corresponding z-score of the standard normal distribution
- `gausscdfinv_v(x)`: Converts an array of probability values (x) to their corresponding z-scores of the standard normal distribution
- `gausscdf(x)`: Converts a given quantile x into the corresponding probability value under the standard normal distribution.
- `gausscdfvec(x)`: Converts each element of the vector x (quantiles) to probability values under the standard normal distribution
- `gaussdens(x)`: Given a scalar or array, computes the probability density function of the standard normal distribution for the scalar or for all elements of the array respectively
- `gausspdf(x)`: Given a scalar or array, computes the probability density function of the standard normal distribution for the scalar or for all elements of the array respectively
- `rotmat(phi)`: Constructs a 2x2 rotation matrix for a given angle phi
- `setbounds_m(m,l,u)`: Bounds all values of a matrix m between a lower bound l and an upper bound u
- `setbounds_v(v,l,u)`: Bounds all values of a vector v between a lower bound l and an upper bound u
- `inittime()`: Initializes a timer
- `printtime()`: Print the time elapsed since the timer was initialized
- `isfinite(x)`: Checks whether a float is finite (not infinite and not NaN)
- `boole(flag)`: Returns 1 if the flag argument is true; false otherwise
- `rboole(flag)`: Returns 1 if the flag is true; false otherwise (as a real number)
- `getquadmin(x,y)`: Estimates the location of the minimum for a quadratic function that passes through (xi,yi), i=1,2,3
- `getquadint(x,y,x0)`: Interpolated value of a quadratic function at a given point x0, based on points (xi,yi), i=1,2,3
- `getlinint(x,y,x0)`: Finds the value of a linear function on x0, given that the function passes through points (xi,yi), i=1,2
- `logit(x)`: Calculates logistic sigmoid for x
- `invlogit(x)`: Calculates the inverse logistic sigmoid for x
- `invertpd(A)`: Computes the inverse of a positive definite matrix A
- `invertgen(A)`: Computes the inverse of a real general matrix
- `conpd(A)`: Computes the L1 condition number of a real symmetric positive definite matrix A
- `choleski(A)`: Computes the lower triangular Cholesky factorization of a real symmetric positive definite matrix A
- `choleskird(A)`: Computes the lower triangular Cholesky factorization of a real symmetric positive definite matrix A
- `outerprod(v1,v2)`: Calculates the outer product of two vectors v1 and v2
- `diagonal(A)`: Returns the diagonal of a matrix A as a vector
- `eye(k)`: Returns the matrix $I_k$
- `zeros(k)`: Returns a zero vector of length k
- `ones(k)`: Returns a ones vector of length k
- `diagmat(vec)`: Returns a matrix with the vector vec as its diagonal; every other value is 0
- `bandmat(vec)`: Computes the band matrix of a vector. Each column is a shifted version of a segment of the input vector
- `getdiagonal(mat)`: Gets the diagonal of a matrix A as a column vector
- `detpd(A)`: Computes the determinant of a positive definite matrix A using the Cholesky decomposition
- `logdetpd(A)`: Computes the logarithm of the determinant of a positive definite matrix A using the Cholesky decomposition
- `detgetn(A)`: Computes the determinant of a real general matrix
- `logabsdetgen(A)`: Computes the logarithm of the absolute value of the determinant of a real general matrix
- `rann_m(n,m)`: Generates an nxm matrix of pseudorandom numbers drawn from a standard normal distribution
- `rann_v(n)`: Generates a vector of n entries from a standard normal distribution
- `rann_s()`: Generates a scalar drawn from a standard normal distribution
- `ranu_s()`: Generates a scalar from a uniform (0,1) distribution
- `ranu_v(n)`: Generates a vector of n entries from a uniform (0,1) distribution
- `ranui_v(n,k)`: Generates a vector of n entries from a discrete uniform (1,2,...,k) distribution
- `ranui_s(n)`: Generates a scalar from a discrete uniform (1,2,...,k) distribution
- `logmeanexp(v)`: Computes the logarithm of the mean of the exponentiates elements of a vector v
- `logsumexp(v)`: Computes the logarithm of the sum of exponentiated elements of a given vector v
- `orderstat(k,a)`: Finds the k-th smallest element in an array a
- `quantile_v(X,ps)`: Calculates the quantiles of a dataset X based on given probabilities ps
- `quantile_s(X,p)`: Calculates a single quantile of a dataset X based on a given probability p
- `sort(X)`: Returns a sorted version of X
- `sortind(X)`: Returns an argsort of X
- `stddev(X)`: Computes the standard deviation of an array X
- `getcorr(X,Y)`: Computes the correlation coefficient of arrays X, Y
- `selectifc(X,cond)`: Selects columns from a matrix X that fulfill a boolean condition cond
- `selectifr(X,cond)`: Selects rows from a matrix X that fulfill a boolean condition cond
- `selectif_r(v,cond)`: Selects elements from a vector v that fulfill a boolean condition cond
- `reversec(A)`: Reverses tje columns of a matrix A
- `reverser(A)`: Reverses the rows of a matrix A
- `kron(A,B)`: Calculates the kron product of two matrices A and B
- `trace(A)`: Calculates the trace of a matrix A
- `vech(A)`: Flattens the matrix A into a vector
- `loadcsv(filename, headerflag)`: Reads data from a csv file
- `loadstrings(filename)`: Reads strings from a file
- `loadvec(flename)`: Reads a vector from a file
- `loadmat(filename)`: Reads a matrix from a file
- `savemat(filename, mat)`: Saves a matrix mat to a specified file
- `savevec(filename, vec)`: Saves a vector vec to a specific file
- `mydispnames(vec)`: Pritns a vector in a formatted manner
- `mydispmat(mat)`: Prints a matrix in a formatted manner
- `mydispscalar(x)`: Prints a scalar x in a formatted manner
- `mydispstring(str)`: Prints a string str in a formatted manner
- `mydispvec(vec)`: Prints a vector in a formatted manner
- `mydispline(vec)`: prints a line in a formatted manner
- `numstr(x)`: Converts a real number x to a sequence of characters
- `mydispivec(vec)`: Prints an array of integers in a formatted manner
- `convtos(i,k)`: Converts an integer i into a string 