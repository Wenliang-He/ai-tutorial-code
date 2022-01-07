
import numpy as np

############################
### Part 1. Numpy Arrays ###
############################

### 1.1 1D Array (a.k.a. Vector)
data = [1, 2, 3, 4]; data
arr = np.array(data); arr               # 1D-array is often called a 'vector' or simply just 'array'
isinstance(arr, np.ndarray)             # np.ndarray is short for 'numpy n-dimensional array'
arr.ndim
arr.shape                               # 1D array
arr.dtype
arr.size
arr.T                                   # T for transpose
arr.T.shape

### 1.2 2D Array (a.k.a. Matrix)
data = [1, 2, 3, 4]; data
data = [data, data]; data               # 2D-array is often called a 'matrix'
mat = np.array(data); mat
mat.ndim
mat.shape                               # 2D array
mat.dtype
mat.size
mat.T                                   # a transposed 2D matrix
mat.T.shape

### 1.3 3D Array
data = [1, 2, 3, 4]; data
data = [[data, data],[data, data]]; data
mat = np.array(data); mat
mat.ndim
mat.shape                               # 3D array
mat.dtype
mat.size
mat.T
mat.T.shape

### 1.4 Arrays vs. Lists
lst = [1, 2, 3, 4]; lst
arr = np.array(lst); arr

## vectorized solutions
# - same elements
# - agreeable dimensions
[x*2 for x in lst]
arr * 2

# same elements
arr = np.array([False, 1, 2]); arr
arr * 2
arr = np.array([False, 1, 2, 'a', 'b']); arr
arr * 2

# agreeable dimensions
lst = [[1, 2, 3], [1, 2, 3]]; lst
mat = np.array(lst); mat
mat * 2
[[x*2 for x in l] for l in lst]

lst = [[1, 2, 3], [1, 2]]; lst
mat = np.array(lst); mat
mat * 2

### 1.5 Array Indexing
arr = np.arange(12) + 1; arr

## indexing with 1D arrays
# positional indexing - calling by position
arr[2]              # forward selection
arr[-2]             # backward selection
arr[[2, 3]]         # new to array
arr[1:]             # slicing
arr[:3]
arr[1:3]
arr[::-1]

# logical indexing - calling by logic
flag = (arr <= 5)   # create a logical flag
arr[flag]           # new to array

flag = (arr>=2) & (arr<=5)
arr[flag]

flag = ~(arr <= 5)
arr[flag]

## indexing with 2D arrays
mat = np.arange(12).reshape(3, 4); mat
mat[0]              # prioritize over rows
mat[[0,2]]
mat[:2]

mat[2, ]            # specify calling out rows
mat[2, :]           # recommended usage
mat[:, 2]           # recommended usage
mat[, 2]            # won't work

2,                  # treated as a tuple
,2                  # grammatically wrong
mat[(2,)]           # works
mat[2, ]

mat[2, 2]
mat[[1, 2], 2]
mat[[1, 2], 1:3]

# selection vs. slicing
mat = np.arange(12).reshape(3, 4); mat
a = mat[0, ]; a         # single selection reduces the dimensionality
a.shape
b = mat[0:1, ]; b       # slicing always gives output of identical dimensions
b.shape

arr = np.arange(5); arr
arr[0]
arr[0:1]

### 1.6 Special np.array Data Types
arr = np.array([1, np.NaN, np.Inf, -np.Inf]); arr
arr.dtype
np.isnan(arr)
np.isfinite(arr)
np.isinf(arr)

arr.isnan()         # won’t work
arr.isfinite()      # won't work
arr.isinf()         # won't work

arr = np.array([1, None, np.NaN, np.Inf]); arr
np.isnan(arr)
np.isfinite(arr)
np.isinf(arr)



#############################
### Part 2. Create Arrays ###
#############################
import numpy as np

### 2.1 Orderly Arrays
np.arange(4)
a = np.arange(2, 12, 2); a
np.arange(2, 12)

list(range(4))
b = list(range(2, 12, 2)); b

np.zeros(4)
np.zeros((3, 4))

np.ones(4)
np.ones((3, 4))
mat = np.ones((3, 4)) * 2.1; mat

np.ones_like(mat)
np.ones_like(mat) * 1.2

np.zeros_like(mat)

np.full_like(mat, 1.2)
np.full_like(mat, 1.2, dtype=float)

np.eye(4)       # I - eye - Identity matrix
np.eye(4) * 2.1
np.diag(np.arange(4))


### 2.2 Random Arrays
np.random.permutation([1, 2, 3, 4, 5])	    # randomly permute a ndarray
np.random.permutation(10)				    # permutation(np.arange(10)); identical
np.random.permutation(np.arange(10))

arr = np.arange(5); arr
ans = np.random.permutation(arr)
ans
arr

arr = np.arange(5); arr
ans = np.random.shuffle(arr)                # shuffle in-place
ans
arr

mat = np.arange(12).reshape((4, 3)); mat
np.random.shuffle(mat)					    # only shuffle the 1st dimension (row-wise)
mat										    # shuffle is done in-place

np.random.choice([1,2,3,4], 10)
np.random.choice([1,2,3,4], 10, replace=True)   # sample with replacement
np.random.choice([1,2,3,4], 10, replace=False)  # sample without replacement

np.random.choice([1,2,3,4], 4, replace=True)
np.random.choice([1,2,3,4], 4, replace=False)
np.random.choice([1,2,3,4])

np.random.rand(3, 4)						# a 3-by-4 probabilities [0, 1) ndarray
np.random.randn(3, 4)					    # a 3-by-4 ndarray from the standard normal distribution, where m=0, std=1

np.random.uniform(size=(3,4))			    # uniform [0, 1) by default
np.random.uniform(-2, 10, size=(3,4))	    # draw from uniform [-2, 10)
np.random.randint(-2, 10, 5)				# a int ndarray of length 5 from unif(-10, 10)
np.random.randint(11, size=(4,3))			# a int 4-by-3 ndarray values from unif(0, 10)
np.random.normal(size=(3,4))				# the standard normal by default
np.random.normal(2, 3, size=(3,4))		    # normal distribution with m = 2 & sd = 3

# set random seed
np.random.seed(2021)						# set seed for reproducibility
np.random.rand(10)                          # create some random numbers

# set random state
rng = np.random.RandomState(2021)			# an alternative way to set seed
rng.rand(10)								# identical results



################################
### Part 3. Array Operations ###
################################
import numpy as np

### 3.1 Reshape Arrays
arr = np.arange(12); arr
arr.shape

mat = arr.reshape(3, 4); mat
mat = arr.reshape((3, 4)); mat

mat.reshape(mat.size)
mat.reshape(-1)
mat.reshape(-1, order='C')      # flatten row-wise like C language
mat.reshape(-1, order='F')      # flatten column-wise like Fortune language

arr = np.arange(24); arr
mat = arr.reshape(-1, 3, 4); mat
mat.reshape(2, -1, order='C')
mat.reshape(2, -1, order='F')

## add a new dimension
arr = np.arange(4); arr
arr.reshape(1, -1)
arr[None, :]
arr[np.newaxis, :]

mat = np.arange(12).reshape(3, 4); mat
mat.reshape(1, -1)
mat.reshape(1, len(mat), -1)
mat[None, :]
mat[np.newaxis, :]

## remove a useless dimension
mat = mat.reshape(1, len(mat), -1); mat
mat.shape
mat = mat.squeeze(); mat
mat.shape


### 3.2 Transpose Arrays
mat = np.arange(12).reshape(3, 4); mat
mat.T
mat.swapaxes(0, 1)
mat.transpose()

mat = np.arange(12).reshape(2, 2, 3); mat
mat.T
mat.swapaxes(0, 2)
mat.transpose()
mat.transpose([2, 1, 0])

mat.swapaxes(1, 2)
mat.transpose([0, 2, 1])

mat.swapaxes(0, 1)
mat.transpose([1, 0, 2])


### 3.3 Combine Arrays
# dimensional arrays (matrixes)
m = np.arange(8).reshape(2,4); m

np.vstack((m,m))						# row-wise concatenation - v for vertical
np.concatenate((m,m), axis=0)			# identical
np.r_[m,m]								# identical; r for row-wise

np.hstack((m,m))						# column-wise concatenation - h for horizontal
np.concatenate((m,m), axis=1)			# identical
np.c_[m,m]								# identical; c for column-wise

# dimensionaless arrays
a = np.array([1,2,3]); a				# a dimensionless vector
b = np.array([4,5,6]); a				# inconsistent behavior appears

np.hstack([a,b]) 						# a & b treated as if they are row vectors
np.concatenate([a,b], axis=0)			# identical; a,b only have the first dimension
np.r_[a,b]								# identical

np.vstack([a,b])						# a & b treated as if they are row vectors
np.concatenate([a,b], axis=1)			# Error due to absence of column index
np.c_[a,b]								# np.vstack([a,b]).T


### 3.4 Set Operations
x = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4]); x
y = np.array([1, 6, 7, 8, 3]); y

np.unique(x)                        # set(x)
np.intersect1d(x,y)					# set(x).intersection(y)
np.union1d(x,y)						# set(x).union(y)
np.setdiff1d(x,y)					# set(x).difference(y)
np.setxor1d(x,y)                    # set(x) ^ set(y)

np.in1d(x,y)						# a logical flag
x[~np.in1d(x,y)]                    # use logical indexing

x == 3
(x==3) | (x==2)
x == np.array([2,3])
np.in1d(x, [2,3])


### 3.5 Statistics
z = np.arange(12).reshape((3,4)); z

np.sum(z)								# get the grand sum
z.sum()									# identical

z.sum(0); z.sum(axis=0);				# column sums - 0 > sum across rows
z.sum(1); z.sum(axis=1);				# row sums    - 1 > sum across columns

z.mean(0)								# column means
z.mean(1)								# row means
										# try cumsum(), cumprod(), min(), max()
z.std(0)								# column-wise std with df = n by default
z.std(0, ddof=1)						# df = n-1; ddof - degree of freedom
z.min(0)								# column minimums
z.max(0)

z.cumsum(0)                             # cumulative sum
z.cumprod(0)                            # cumulative product

np.all([True, True, False])
np.any([True, True, False])
(z>=6).all(1)
(z>=6).any(1)

z = z.astype(float); z					# NaN is treated as float, not integer
z[0,0] = np.nan							# insert a missing value
z.sum(0)								# column(s) with NaN return NaN

# Question: how to compute column means excluding np.nan?


### 3.6 User-defined Functions
func = lambda x: (x[0] + x[-1]) * 0.5		# define a lambda function
mat = np.arange(12).reshape(4,3); mat		# create a 4-by-3 ndarray
np.apply_along_axis(func, 0, mat)			# apply func column-wise
np.apply_along_axis(func, 1, mat)			# apply func row-wise



#############################################
### Part 4. Other Commonly Used Functions ###
#############################################
# Check out a list of numpy routines:
# https://numpy.org/doc/stable/reference/routines.html

import numpy as np

### 4.1 Sorting & Searching

# np.argmax / np.argmin
arr = np.random.permutation(12); arr
np.max(arr)

idx = np.argmax(arr); idx
idx = np.argmin(arr); idx
arr[idx]

mat = arr.reshape(3, 4); mat
mat.argmax(0)
mat.argmax(1)

# np.sort / np.argsort
np.sort(arr)

np.sort(mat)
np.sort(mat, axis=-1)           # default - sort along the last axis
np.sort(mat, 1)                 # identical
np.sort(mat, 0)

idx = np.argsort(arr); idx
arr[idx]

idx = np.argsort(mat); idx
mat[0][idx[0]]
mat[1][idx[1]]
mat[2][idx[2]]

# np.where
arr = np.arange(12); arr
np.where(arr<=5, arr, -arr)             # conditional mapping
[x if x<=5 else -x  for x in arr]       # almost identical

mat = arr.reshape(3, 4); mat
np.where(mat<=5, mat, -1)               # difficult to apply list comprehension

# np.clip
np.clip(arr, 2, 10)
np.clip(mat, 2, 10)


### 4.2 Creating Orderly Arrays

# np.repeat / np.tile
np.repeat(3, 4)
np.array([3] * 4)

np.repeat([1, 2, 3], 4)
np.array([1, 2, 3] * 4)

np.repeat([1, 2, 3], [3, 2, 1])

mat = np.arange(12).reshape(3,4); mat
np.repeat(mat, 2)
np.repeat(mat, 2, axis=0)
np.repeat(mat, 2, axis=1)

np.vstack([mat] * 2)
np.hstack([mat] * 2)

np.repeat(mat, [1, 2, 3], axis=0)

arr = np.array([0, 1, 2])
np.tile(arr, 2)
np.tile(arr, (1, 2))
np.tile(arr, (2, 1))
np.tile(arr, (2, 2))
np.tile(arr, (2, 2, 2))

mat = np.array([[1, 2], [3, 4]]); mat
np.tile(mat, 2)
np.tile(mat, (1, 2))
np.tile(mat, (2, 1))
np.tile(mat, (2, 2))
np.tile(mat, (2, 2, 2))

# np.linspace / np.meshgrid
np.linspace(2.0, 3.0, num=5)            # linear spacing
np.arange(2.0, 3.0, step=0.25)
np.arange(2.0, 3.0+0.25, step=0.25)


xx = np.arange(1, 10); xx
yy = np.arange(1, 10); yy
gx, gy = np.meshgrid(xx, yy)        # outer product with 1D arrays
gx
gy
prod = gx * gy; prod

ans = np.zeros_like(prod); ans
for iy, y in enumerate(yy):
    for ix, x in enumerate(xx):
        ans[iy, ix] = x * y
ans


### 4.3 Matrix & Linear Algebra

# np.diag / np.diagonal
mat = np.array([[1,2,3,4,5]] * 5); mat
np.diagonal(mat)            # extract diagonal elements
np.diag(mat)

np.diag([1,2,3])            # restore elements to the diagonal
np.diagonal([1,2,3])

mat
np.diag(mat, 1)             # extract diagonal elements at other positions
np.diag(mat, 2)
np.diag(mat, -1)
np.diag(mat, -2)

np.diag([1,2,3], 1)         # restore elements to the diagonal at other positions
np.diag([1,2,3], 2)
np.diag([1,2,3], -1)
np.diag([1,2,3], -2)

# np.tri / np.triu / np.tril    # triangle - upper - lower
np.tri(3, 5)                    # create a matrix
np.tri(5, 5)
np.tri(5, 5, 1)
np.tri(5, 5, -2)
np.tri(5, 5).T

np.triu(mat)                    # extract upper triangular matrix
np.triu(mat, 1)
np.triu(mat, -2)

np.tril(mat)                    # extract lower triangular matrix
np.tril(mat, 1)
np.tril(mat, -2)

# np.inner / np.dot
x = np.arange(4); x
y = np.arange(4) + 4; y
np.inner(x, y)
sum(x * y)

x = np.arange(12).reshape(3, 4); x
np.inner(x, y)

x = np.arange(24).reshape((2, 3, 4)); x
y = np.arange(4); y
np.inner(x, y)

matx = np.arange(12).reshape(3, 4); matx
maty = matx.T; maty
np.dot(matx, maty)              # 3*4 x 4*3 = 3*3
np.dot(maty, matx)              # 4*3 x 3*4 = 4*4

# np.linalg.matrix_rank         # get matrix rank
# np.linalg.det                 # get matrix determinant
# np.linalg.inv                 # get inverse of a matrix
# np.linalg.eig                 # get eigen values of a matrix

