
import numpy as np
import pandas as pd

#########################################
### Part 1. Pandas Series & DataFrame ###
#########################################

### 1.1 1D Pandas Series (a.k.a. Vector)
data = [1, 2, 'a', 'b']; data
arr = np.array(data); arr
vec = pd.Series(data); vec                      # Series created from a list
isinstance(vec, pd.Series)
vec.ndim
vec.shape
vec.dtype                                       # can include data of different types
vec.size
vec.T
vec.T.shape
len(vec)

vec[0], type(vec[0])                            # int
vec[2], type(vec[2])                            # string
vec.dtype

# rename index
vec.index                                       # series has names
list(vec.index)
vec.index = list('abcd')                        # hence can be renamed
vec

pd.Series(data)
pd.Series(data, index=list('abcd'))             # Series created from a list

# created from dictionary
dic = {'a': 1, 'b': 2, 'c': 3, 'd': 4}; dic
pd.Series(dic)
pd.Series(data, index=list('abcd'))             # identical

# importance of index
data = [1, 2, 3, 4]
x = pd.Series(data, index=list('abcd')); x
y = pd.Series(data, index=list('abdc')); y
x + y

x = pd.Series(data, index=list('abcd')); x
y = pd.Series(data, index=list('bcde')); y
x + y


### 1.2 2D Pandas DataFrame
data = [1, 3, 'a', 'b']; data
data = [data, data]; data
mat = np.array(data); mat
dat = pd.DataFrame(data); dat                   # DataFrame created from a list of lists
isinstance(dat, pd.DataFrame)
dat.ndim
dat.shape                                       # a 1D row vector
dat.dtype                                       # cannot query dtype with DataFrame
dat.dtypes
dat.size
dat.T                                           # a 1D column vector
dat.T.shape
len(dat)

# rename row & column names
dat
dat.index
list(dat.index)
dat.columns
list(dat.columns)

dat.index = ['i', 'ii']
dat.columns = list('abcd')
dat

pd.DataFrame(                                   # DataFrame created from a list of lists
    data,
    index=['i', 'ii'],
    columns=list('abcd')
)

# created from dictionary
dic = {'a': 1, 'b': 2, 'c': 'a', 'd': 'b'}; dic
lst = [dic, dic, dic, dic]; lst
xdat = pd.DataFrame(lst); xdat                  # DataFrame created from a list of dictionaries

dic = {
    'a': [1, 2, 3, 4],
    'b': [2, 4, 6, 8],
    'c': list('abcd'),
    'd': list('dcba'),
}
ydat = pd.DataFrame(dic); ydat                  # DataFrame created from a dictionary

# importance of index
xdat = pd.DataFrame(data, index=[0,1]); xdat
ydat = pd.DataFrame(data, index=[1,2]); ydat
pd.concat([xdat, ydat], axis=0)                 # np.concatenate() == pd.concat()
pd.concat([xdat, ydat], axis=1)


# add columns
ids = np.arange(len(xdat)); ids
xdat['id'] = ids; xdat

tdat = pd.DataFrame({'id': ids, 'di': ids[::-1]}); tdat
tdat = pd.DataFrame(zip(*[ids, ids[::-1]])); tdat
xdat[['id', 'di']] = tdat; xdat

# remove columns
del xdat['id']; xdat

xdat[['id', 'di']] = tdat; xdat
xdat = xdat.drop(columns=['id','di']); xdat


### 1.3 Indexing Series
vec = pd.Series(range(5), index=list('abcde')) + 1; vec

# positional indexing - calling by position
vec
vec[0]
vec[[0, 2]]
vec[-1]
vec[[0, -1]]
# logical indexing - calling by logic
vec
vec[vec <= 3]
# nominal indexing - calling by name
vec
vec['a']
vec[['a', 'c']]
vec['a':'c']

### 1.4 Indexing DataFrame
data = [1, 2, 'a', 'b']
dat = pd.DataFrame(
    [data] * 8,
    index=list('abcdefgh'),
    columns=list('abcd'),
); dat
dat['a'] = np.arange(len(dat)) + 1
dat

# calling by position
dat.iloc[1, :]                                  # 'iloc' for integer location - position
dat.iloc[[0, 3], :]
dat.iloc[0:3, :]

dat.iloc[0:3, 0]
dat.iloc[0:3, [0, 3]]
dat.iloc[0:3, 0:3]

# calling by name
dat.loc['a', 'b']                               # 'loc' for location - name
dat.loc[['a', 'b'], ['c', 'd']]
dat.loc['a':'c', 'a':'b']

# calling by position - shortcut
dat[:3]                                         # integer slicing -> rows
dat[0]                                          # why not working?
dat[[0, 3]]                                     # why not working?
# calling by logic - shortcut
flag = np.arange(len(dat)) <= 2; flag
dat[flag]                                       # logical -> rows
flag = np.arange(dat.shape[1]) <= 2; flag
dat[flag]
# calling by name - shortcut
dat['a']                                        # selection -> columns
dat[['a', 'c']]
dat.a


### 1.5 Missing Values Related Operations
vec = pd.Series([1, np.NaN, np.Inf, -np.Inf]); vec
pd.isnull(vec)
vec.isnull()
np.isnan(vec)

pd.notnull(vec)
vec.notnull()

len(vec)
vec.size
vec.count()							            # 3; number of non-missing values
vec.dropna()						            # remove NaNs; a copy, not a view

data = {
    'name': ['wen', 'nic', 'neil', 'mark'],
    'age': [32, 22, np.nan, 60],
    'grade': [92, 90, 96, np.nan],
}
dat = pd.DataFrame(data); dat

dat.dropna()                                    # remove rows if any NaNs are present
dat.dropna(axis=0, how="any")			        # identical
dat.dropna(axis=0, how="all")			        # remove rows if all are NaNs
dat.dropna(axis=1, how="any")			        # remove columns if any NaNs are present
dat.dropna(axis=1, how="all")			        # remove columns if all are NaNs

### 1.6 Last Resort
vec.values
vec.tolist()

dat.values
dat.values.tolist()


##################################
### 2. Commonly Used Functions ###
##################################
import numpy as np
import pandas as pd

### 2.1 Series
vec = pd.Series(['b', 'a', 'c', 'b', 'b', np.nan, 'd', 'c', 'b', np.nan, 'd']); vec
vec.sort_values()                               # sort values in ascending order
vec.sort_values(ascending=True)                 # identical
vec.sort_values(ascending=False)                # NaNs are sorted to the end regardless of sorting order

vec = pd.Series(range(4), index=['d', 'a', 'b', 'c']); vec
vec.sort_index()                                # 1, 2, 3, 0; sort index in ascending order
vec.sort_index(ascending=True)				    # identical
vec.sort_index(ascending=False)

vec = pd.Series(['b', 'a', 'c', 'b', 'b', np.nan, 'd', 'c', 'b', np.nan, 'd']); vec
vec.nunique()                                   # number of unique levels; NaNs ignored
vec.nunique(dropna=True)				        # identical
vec.nunique(dropna=False)

vec.unique()							        # levels by dictionary order; NaNs included
vec.unique(dropna=True)                         # won't work
vec.dropna().unique()
vec.unique().dropna()
pd.Series(vec.unique()).dropna()

vec.unique()                                    # orderless - pandas - faster
np.unique(vec)
np.unique(vec.tolist())                         # orderly   - numpy  - slower
list(set(vec))                                  # can use set() with numpy

vec.duplicated()			                    # Boolean output; first come first unique
vec.drop_duplicates()					        # unique levels; return Series
vec[~vec.duplicated()]                          # identical

vec.value_counts()                              # frequency table in descending order
vec.value_counts(dropna=True)			        # identical
vec.value_counts(dropna=False)
vec.value_counts().sort_index()                 # sort frequency table by index

flag = vec.isin(['b', 'c']); flag				# compare to np.in1d()
vec[flag]								        # filter out 'b' and 'c'
flag.all()								        # False
flag.any()								        # True


### 2.2 DataFrame
dat = pd.DataFrame({
    'var1': ['one']*3 + ['two']*4,
    'var2': [1, 1, 2, 3, 3, 4, 4]
}); dat

dat.duplicated()						        # find duplicate rows; Boolean output
dat.drop_duplicates()					        # drop duplicate rows directly
dat[~dat.duplicated()]				            # identical

dat['id'] = range(7); dat					    # add a new variable thereafter
dat.drop_duplicates()
dat.drop_duplicates('var1')			            # drop duplicates by selected variable(s)
dat.drop_duplicates(['var1', 'var2'])		    # .drop_duplicates(str/list)

# need to shuffle dataframe by row here
idx = np.random.permutation(dat.index); idx
dat = dat.iloc[idx]                            # can also use dat.loc[] in this case

dat.sort_index()                                # sort index in ascending order
dat.sort_index(axis=0)				            # identical
dat.sort_index(axis=1)				            # sort column in ascending order

dat.sort_values(by='var1')                      # sort values by column 'var1' in ascending order
dat.sort_values(by=['var1', 'var2'])            # sort values by column 'var1' first, then by 'var2'
dat.sort_values(by=['var2', 'var1'])            # sort values by column 'var2' first, then by 'var1'



##################################
### Part 3. Compute Statistics ###
##################################
import numpy as np
import pandas as pd

xdat = pd.DataFrame({
    'name': ['wen', 'nic', 'neil', 'mark'],
    'age': [32, 22, np.nan, 60],
    'grade': [92, np.nan, 96, 88],
    'race':['asian', 'white', 'white', np.nan]
}, columns=['name', 'age', 'grade', 'race']); xdat

ndat = xdat[['age', 'grade']] / 3; ndat         # 'n' for numeric

### 3.1 Built-in Statistics Functions
xdat['age'].sum()                               # NaNs are skipped by default
xdat['age'].sum(skipna=True)                    # identical
xdat['age'].sum(skipna=False)

xdat.sum()							            # Pandas function to get column sums
xdat.sum(axis=0)                                # identical
np.sum(xdat)							        # identical
xdat.mean()

xdat.sum(axis=1)						        # perform across-column row-wise operations
xdat.sum(axis=1, skipna=False)		            # include NaNs

xdat.fillna('x').sum()
xdat.fillna({'name': 'x', 'age': 0, 'grade': 0, 'race': 'x'}).sum()

xdat.mean()
xdat.std()
xdat.min()
xdat.max()

xdat.describe()						            # vertical display of descriptive statistics
xdat.describe().T						        # horizontal display

xdat.fillna(xdat.mean())				        # fill NaNs column-wise with column means

### 3.2 User-defined Functions (UDF) - Simple Iterations
# pd.Series.map()
# pd.Series.apply() - extremely infrequent
# pd.DataFrame.applymap()
# pd.DataFrame.apply()

# map
vec = pd.Series([1, 2, 3, 4]); vec
vec * 2
vec.map(lambda x: 'v' + str(x))

# applymap
ndat.round(3)
f = lambda x: '%.3f' % x				        # identical
ndat['age'].map(f)					            # return a Series
ndat.applymap(f)					            # return a DataFrame

# apply - iterate over columns by default
f = lambda vec: vec.max() - vec.min()			# a lambda function to compute range
ndat.apply(f)							        # use axis=1 for row-wise apply
ndat.max() - ndat.min()                         # identical

f = lambda vec: pd.Series([vec.min(), vec.max()], index=['min', 'max'])   # return multiple stats
ndat.apply(f)							        # return a DataFrame
ndat.describe()


dat = pd.DataFrame({
    'var1': list('aAbbccDdeEff'),
    'var2': np.arange(12) + 1
}); dat

rule = {'a': 'one', 'b': 'one', 'c': 'one', 'd': 'two', 'e': 'two', 'f': 'two'}

dat['var1'].map(str.lower).map(rule)	        # map values of var1 based on rule
dat['var1'].str.lower().map(rule)		        # identical
dat['var1'].map(lambda x: rule[x.lower()])      # identical

dat.sum()
dat.apply(np.sum)						        # identical
dat.apply(sum)						            # won’t work
dat.apply(lambda vec: sum(vec))                 # won't work
dat.apply(lambda vec: vec.sum())


dat = pd.DataFrame({
    'var1': np.random.choice([1,2,3,4], 12),
    'var2': np.arange(12) + 1
}); dat

def max_plus(vec, add=1):
    return max(vec) + add

dat[['var1','var2']].apply(max_plus, add=1)     # pass arguments to UDF via apply()


dat.apply(sum)
dat.aggregate(sum)                              # aggregate() is much faster than apply()
dat.agg(sum)                                    # use aggregate() with DataFrame and use agg() with GroupBy object

dat.apply([sum, np.mean])
dat.aggregate([sum, np.mean])
dat.agg([sum, np.mean])


### 3.3 User-defined Functions (UDF) - Advanced Iterations

## Many to One
df = pd.DataFrame({
    'int': [1, 2, 6, 8, -1],
    'float': [0.1, 0.2, 0.2, 10.1, np.nan],
    'str': ['a', 'b', np.nan, 'c', 'a']
}); df

def fun(vec):
    return vec['int'] + vec['float']			# does not have to work with vectors

df['sum1'] = df.apply(fun, axis=1); df

def fun(vec):
    x, y = vec									# “x, y = df” works with elements
    return x + y								# wont’ work with vectors

df['sum2'] = df[['int','float']].apply(fun, axis=1); df


## One to Many
df = pd.DataFrame({
    'int': [1, 2, 6, 8, -1],
    'float': [0.1, 0.2, 0.2, 10.1, np.nan],
    'str': ['a', 'b', np.nan, 'c', 'a']
}); df

# return a list
def fun(x):
    return [x*2, x*3]

ans = pd.DataFrame(df['int'].map(fun).tolist()); ans
df[['a', 'b']] = ans		                    # store in another DataFrame

ans = zip(*df['int'].map(fun)); ans
df['c'], df['d'] = ans		                    # store separately into DataFrame

# return a dictionary
def fun(x):
  return {'a': x*2, 'b': x*3}

pd.DataFrame(df['int'].map(fun).tolist())		# store in another DataFrame

# return a Series
def fun(x):
  return pd.Series({'a': x*2, 'b': x*3})

df['int'].map(fun)					            # won’t work
df['int'].apply(fun)					        # works!


df = pd.DataFrame(np.random.randn(10, 3), columns=["a","b","c"]); df
def fun(vec):
    return pd.Series([max(vec), min(vec)],index = ["max","min"])
df.apply(fun, axis=0)
df.apply(fun, axis=1)


## Many to Many
df = pd.DataFrame({
    'int': [1, 2, 6, 8, -1],
    'float': [0.1, 0.2, 0.2, 10.1, np.nan],
    'str': ['a', 'b', np.nan, 'c', 'a']
}); df

def fun(vec):
    return [vec['int']*2, vec['float']*10]

ans = pd.DataFrame(df.apply(fun, axis=1).tolist()); ans	        # without column names
df[['a', 'b']] = ans

ans = zip(*df.apply(fun, axis=1)); ans
df['c'], df['d'] = ans


def fun(vec):
  return { 'a': vec['int']*2, 'b': vec['float']*10 }

pd.DataFrame(df.apply(fun, axis=1).tolist())                        # with column names


def fun(vec):
    return pd.Series({'c': vec['int']*2, 'd': vec['float']*10})     # cannot be pd.DataFrame

df.apply(fun, axis=1)



######################################
### Part 4. Grouping & Aggregation ###
######################################
import numpy as np
import pandas as pd

### 4.1 The Groupby Class

## Grouping with Series/DFrame
np.random.seed(2012)
dat = pd.DataFrame({
    'var1': np.random.randint(0, 10, 6),
    'var2': np.arange(6) + 1,
    'grp1': list('aaabbb'),
    'grp2': ['one', 'two'] * 3,
}); dat

g1 = dat['var1'].groupby(dat['grp1'])			# return a GroupBy object
ans = g1.sum(); ans							    # means of var1 by grp1
ans.reset_index()

g2 = dat['var1'].groupby([dat['grp1'], dat['grp2']])    # group var1 by (grp1 & grp2)
ans = g2.sum(); ans					            # means of var1 by (grp1 & grp2)
ans.reset_index()
tab = ans.unstack(); tab					    # display in tabular format (i.e. contingency or pivot table)
tab.stack()                                     # convert it back

g3 = dat.groupby(['grp1', 'grp2'])			    # group df by (grp1 & grp2)
ans = g3.sum(); ans							    # means of all numeric vars by (grp1 & grp2)
ans.reset_index()
ans.unstack()

lst = list(g3); lst                             # inspect g3
x = lst[0]; x
type(x), len(x)
x[0]
x[1]

groups = dict(lst)					            # convert into a dict
groups.keys()
groups[('a', 'one')]						    # call values by a tuple key


for idx, val in dat.groupby('grp1'):			# a Groupby object is iterable
    print(idx)
    print(val)
    print()

for idx, val in dat.groupby(['grp1', 'grp2']):	# try ‘for idx, val in g2:’ to compare
    print(idx)								    # print tuple elements together
    print(val)
    print()

g2.sum()
g2.sum().shape
g3.sum()
g3.sum().shape
g3['var1'].sum()							    # identical to g2.mean(), but more flexible
g3[['var1', 'var2']].sum()					    # can decide which var(s) to include later


## Grouping by Things from Elsewhere

# group by list/ndarray/Series
lst = list('aabbba'); lst				        # a list
arr = np.array(lst); arr				        # a ndarray
s1 = pd.Series(arr); s1							# a Series
s2 = pd.Series(arr, index=list('abcdef')); s2	# a Series with named index

dat
dat.groupby(lst).sum()					        # groupby(list)
dat.groupby(arr).sum()					        # groupby(np.array)
dat.groupby(s1).sum()					        # groupby(pd.Series)
dat.groupby(s2).sum()					        # groupby(pd.Series) - fails due to different indexes


g2 = dat['var1'].groupby([dat['grp1'], dat['grp2']])                # a list of two series
g2.sum()
g2 = dat['var1'].groupby([list(dat['grp1']), list(dat['grp2'])])    # a list of two lists
g2.sum()
g2 = dat['var1'].groupby([tuple(dat['grp1']), tuple(dat['grp2'])])  # a list of two tuples

g2 = dat['var1'].groupby(dat[['grp1', 'grp2']])                     # cannot accept a dataframe
g2 = dat['var1'].groupby(dat[['grp1', 'grp2']].values)              # nor a 2D-array
g2.sum()

by = [list(x) for x in list(zip(*dat[['grp1', 'grp2']].values))]; by
g2 = dat['var1'].groupby(by)
g2.sum()

dat[['var1', 'var2']].groupby(by).sum()
dat.groupby(['grp1', 'grp2']).sum()

# group by dictionary
dat = pd.DataFrame(
    [list(range(5))] * 5,
    columns=list('abcde'),
    index=list('abcde'),
); dat

dict = {'a': 'one', 'b': 'one', 'c': 'one', 'd': 'two', 'e': 'two'}		# create a dict for mapping

dat.groupby(dict).sum()					        # row-wise grouping by dictionary
dat.groupby(dict, axis=0).sum()					# identical
dat.groupby(dict, axis=1).sum()			        # column-wise grouping by dictionary

# group by function
dat.index = ['aa', 'bb', 'ccc', 'ddd', 'eee']; dat
dat.groupby(len).sum()
dat.groupby(len, axis=0).sum()
dat.groupby(len, axis=1).sum()

lens = dat.index.map(len); lens
dat.groupby(lens).sum()


### 4.2 GroupBy Methods
np.random.seed(2021)
dat = pd.DataFrame({
    'var1': np.random.randint(0, 10, 6),
    'var2': np.arange(6) + 1,
    'grp1': list('aaabbb'),
    'grp2': ['one', 'two'] * 3,
}); dat


## transform() without aggregation
grouped = dat.groupby('grp1')

out = grouped.transform(lambda df: (df - df.mean()) / df.std()); out
out.groupby(dat.grp1).mean()                    # all means are 0
out.groupby(dat.grp1).std()                     # all std’s are 1

grouped.transform(lambda x: x.max() - x.min())  # broadcast means to fill up DataFrame
grouped.transform(np.mean)                      # broadcast means to fill up DataFrame
grouped.mean()


## agg() & aggregate()

# pd.Series
vec = pd.Series(range(1,5), index=['ace', 'add', 'bat', 'bot' ]); vec
grouped = vec.groupby(lambda x: x[0])                                           # group by the 1st letter of index

grouped.sum()
grouped.agg(sum)                                                                # identical

func_range = lambda x: x.max() - x.min()
grouped.agg([sum, np.mean, func_range])                                         # a list of functions as input
grouped.agg([sum, np.mean, ('range', func_range)])                              # better naming all column names
grouped.agg([('py_sum', sum), ('np_mean', np.mean), ('range', func_range)])     # use tuples to give nicer column names
grouped.agg(['sum', 'mean', ('range', func_range)])                             # use groupby methods directly

grouped.agg(['count', 'min', 'median', 'max', 'std', 'made-up'])                # same examples of groupby methods
[x for x in dir(grouped) if x in ['count', 'min', 'median', 'max', 'std', 'made-up']]


# pd.DataFrame
np.random.seed(2021)
dat = pd.DataFrame({
    'var1': np.random.randint(0, 10, 6),
    'var2': np.arange(6) + 1,
    'grp1': list('aaabbb'),
    'grp2': ['one', 'two'] * 3,
}); dat

grouped = dat.groupby('grp1')

grouped.sum()
grouped.agg(sum)

func_range = lambda x: x.max() - x.min()
grouped.agg([sum, np.mean, func_range])
grouped.agg([('py_sum', sum), ('np_mean', np.mean), ('range', func_range)])     # use tuples to give nicer column names
grouped.agg(['sum', 'mean', ('range', func_range)])
grouped.agg({'var1': sum, 'var2': np.mean})                                     # apply func to specific variables


## apply()
grouped.apply(max)
grouped.apply(max, axis=0)
grouped.apply(max, axis=1)

gmax = lambda df: df.max()
grouped.apply(gmax)
grouped.apply(gmax, axis=0)
grouped.apply(gmax, axis=1)

gmax_gmin = lambda df: pd.DataFrame([df.max(), df.min()])
grouped.apply(gmax_gmin)


def fun(df):
    df["new_col"] = np.cumsum(df["var1"])
    df["new_max"] = np.max(df["new_col"])
    df["new_min"] = np.min(df["new_col"])
    return df

grouped.apply(fun)
dat

fun(dat)
dat


np.random.seed(2021)
dat = pd.DataFrame({
    'var1': np.random.randint(0, 10, 6),
    'var2': np.arange(6) + 1,
    'grp1': list('aaabbb'),
    'grp2': ['one', 'two'] * 3,
}); dat

def fun(df):
    a = df["var1"]
    b = df["var2"]
    return pd.DataFrame({"col1": a+b, "col2": a-b})

adat = dat.groupby("grp1").apply(fun); adat

def fun(df):
    a = df["var1"]
    b = df["var2"]
    return pd.DataFrame(list(zip(a+b, a-b)), columns=["col1", "col2"])

bdat = dat.groupby("grp1").apply(fun); bdat



#############################################
### Part 5. Other Commonly Used Functions ###
#############################################
import numpy as np
import pandas as pd

np.random.seed(2021)
dat = pd.DataFrame({
    'var1': np.random.randint(0, 10, 6),
    'var2': np.arange(6) + 1,
    'grp1': list('aaabbb'),
    'grp2': ['one', 'two'] * 3,
}); dat

## df.rename()
dat.rename(columns={'var1': 'variable1'})
dat.rename(columns={'var1': 'variable1'}, inplace=True)

dat.rename(index={0: 100})
dat.rename(index={0: 100}, inplace=True)

## df.reset_index()
dat.reset_index()
dat.reset_index(inplace=True)

dat.reset_index().rename(columns={'index': 'idx'})

## df.drop()
dat.drop(columns='index')
dat.drop(columns=['index', 'grp2'])

dat.drop(index=0)
dat.drop(index=[0, 5])

## query()
np.random.seed(2021)
dat = pd.DataFrame({
    'var1': np.random.randint(0, 10, 6),
    'var2': np.arange(6) + 1,
    'grp1': list('aaabbb'),
    'grp2': ['one', 'two'] * 3,
}); dat

dat.query("var1>4")
dat.query("var1>4 and var2<6")
dat.query("grp1 == 'a'")
dat.query("grp1 in ('a','b')")

dat[dat['var1'] > 4]
dat[(dat['var1'] > 4) & (dat['var2'] < 6)]
dat[dat['grp1'] == 'a']
dat[dat['grp1'].isin(['a', 'b'])]

## df.assign()
flag = dat["var1"] < 5                  # create a logical flag

df = dat[flag]                          # step1: extract part of a dataframe
df["var1"] = 99                         # step2: assign a new value
df
dat

dat[flag]["var1"] = 99                  # steps1-2: an example of chained assignment
dat

# extract part of a dataframe
df = dat.copy()[flag]                   # method 1
df["var1"] = 99
df
dat

df = dat[flag].assign(var1=99)          # method 2
df
dat

df = df.assign(var1=0, grp3='a')        # change values and add a new column at the same time
df

# perform "chained" assignments
dat.loc[dat["var1"] < 5, "var1"] = 99; dat

## pivot_table()
# 'pivot table' is also known as 'contingency table'
np.random.seed(2021)
dat = pd.DataFrame({
    "id": ["a", "b", "c"] * 5,
    "item": np.random.choice(["beef", "pork", "chicken"], size=15),
    "quantity": np.random.randint(5, 10, 15)
}); dat

dat.pivot_table(index="id", columns="item", values="quantity", aggfunc=sum)
dat.pivot_table(index="id", columns="item", values="quantity", aggfunc=sum, fill_value=-1)
dat.groupby(['id', 'item']).sum()

## merge()
df1 = pd.DataFrame({
    'id': range(4),
    'var1': np.random.randint(0, 10, 4),
    'grp1': list('aabb'),
}); df1

df2 = pd.DataFrame({
    'id': range(2, 6),
    'var2': np.arange(4) + 1,
    'grp2': ['one', 'two'] * 2,
}); df2

df1.merge(df2, how='inner')
df1.merge(df2, how='inner', on='id')
df1.merge(df2, how='outer', on='id')
df1.merge(df2, how='left', on='id')
df1.merge(df2, how='right', on='id')


df1 = pd.DataFrame({
    'id1': range(4),
    'var': np.random.randint(0, 10, 4),
    'grp': list('aabb'),
}); df1

df2 = pd.DataFrame({
    'id2': range(2, 6),
    'var': np.arange(4) + 1,
    'grp': ['one', 'two'] * 2,
}); df2

df1.merge(df2, how='inner', left_on='id1', right_on='id2')
df1.merge(df2, how='inner', left_on='id1', right_on='id2', suffixes=['_df1', '_df2'])

## analytics-related methods
df = pd.DataFrame(np.random.randn(10, 4), columns=['a', 'b', 'c', 'd']); dat

df['a'].quantile(np.arange(0, 1.01, 0.25))
df['a'].hist(bins=5)
df['a'].nlargest(5)
df['a'].nsmallest(5)

bins = np.arange(-2, 3, 1); bins
bins = [-99] + list(bins) + [99]; bins
pd.cut(df['a'], bins)
pd.cut(df['a'], bins, labels=[chr(ord('a')+x) for x in range(len(bins)-1)])

## str-related methods
df = pd.DataFrame({'a': ['value_' + str(i) for i in range(5)]}); df

df['a'].str.upper()
df['a'].str.title()
df['a'].str.endswith('2')
df['a'].str.contains('2')
df['a'].str.replace('_', '-')

df['a'].str.split('_')
df['a'].str.split('_', expand=True)
df['a'].str.split('_').str.join('_')

df['a'].str.match(r'\w*[0-9]')
df['a'].str.extract(r'\w*([0-9])')
