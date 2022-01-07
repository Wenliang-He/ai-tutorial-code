############################
### Part 0. Installation ###
############################

### References
# https://www.programiz.com/python-programming/examples
# https://realpython.com/
# https://leetcode.com/problemset/algorithms/

### Install Python & IDE
# 1. Install Python via Anaconda/Miniconda
#    - python
#    - commonly used python modules/libraries/packages
#    - easy to create new python environments
# 2. Install Python IDE - PyCharm
# 3. Set Python IDE Interpreter
# 4. Set Shortcuts
#    - File >> Settings >> Keymap >> search for 'execute' >>
#    - 'Execute selection in console' >> right-click >> Add Keyboard shortcut >> Ctrl+Enter >> Leave
#    - 'Execute Current Statement in One-Line Console' >> right-click >> Add Keyboard shortcut >> Enter >> Leave

### Understand Key Terms
# python console
# python script
# terminal

1 + 2 + 3

print(1+1)
print(1+2)
print(1+3)

########################
### Part 1. Get Help ###
########################

## Programming: data vs. function
## Object-Oriented Programming (OOP)
# object: class vs. instance
# object: attributes vs. methods

help()      # Ctrl + C to quit
dir()       # check attributes and methods

dir(__builtins__)
help(sum)
?sum        # '?' is a shortcut for help()


###########################################
### Part 2. Four Fundamental Data Types ###
###########################################

### 2.1 Four Data Types
type(True)  # boolean
type(2)     # integer
type(2.0)   # floating point
type('2')   # string
None        # none

help(bool)
help(int)
help(float)
help(str)

dir(str)
help(str.upper)

str.upper('abc')    # 'str' is a class
'abc'.upper()       # 'abc' is an instance of the 'str' class
'abc'.upper().lower()

### 2.2 Data Type Conversion
bool(0)
bool(9.23)
bool(-9.23)

int(9.99)
round(4.50001)
round(4.50001, 3)

float('9.23')

str(True)
str(1)
str(9.23)

int('abc')
bool('abc')
bool('')

### 2.3 Operations

## 2.3.1 arithmatic operations
1 + 1
2 - 1
2 * 3
9 / 2       # arithmetic division
2**3

9 // 2      # integer division
9 % 2       # remainder division (modulo operation)


## 2.3.2 logical operations

# scaler vs. vector
#   not  vs.   ~
#   or   vs.   |
#   and  vs.   &

True or notexists       # short-circuit behavior
False and notexists

True | notexists        # vectorized element-wise evaluation
False & notexists

import numpy as np
~np.array([True, False])
np.array([True, False]) | np.array([True, False])
np.array([True, False]) & np.array([True, False])

## 2.3.3 comparison operations
2 == 1
2 >= 1
2 != 1

a = 1
a
a == 1

a, b = 1, 2
a
b
a,b

a, b = [1, 2]
print(a)
print(b)
print(a, b)

a, b = (1, 2)
print(a)
print(b)
print(a, b)

## 2.3.4 set operations
# please refer to Part 4.3


##################################
### Part 3. More About Strings ###
##################################

### 3.1 String Operations

'ab' + 'cd'             # arithmetic operation
'ab' * 3

"9999" < "Z2"           # comparison operation
"Z2" < "a1"

'bad' in 'abadabara'    # set operation

ord('8')                # method
ord('A')
ord('a')
chr(ord('9'))
chr(ord('A'))
chr(ord('z'))

### 3.2 String Indexing
s = '123456789'
s[0]                # single selection
s[1]
s[0,1]              # cannot do multiple selections
s[-1]

s[1:]               # slicing
s[:1]               # slicing
s[1:5]              # slicing - (start, end]
s[:-1]
s[-5:-1]

s[::2]
s[::3]
s[::-1]
s[::-2]

s[1:8:2]            # s[start:end:step]
s[1:8][::2]

lst = [1,2,3,4]
lst[0]
lst[0:1]

### 3.3 String Methods
str
s = 'a b c d b'
dir(s)

len(s)
s.upper()
ss = s.split(' '); ss  # split string
js = ' '.join(ss); js  # joint string

s = 'abcdb'
list(s)
s.split(' ')

s.find('b')
s.rfind('b')
s.startswith('a')
s.replace('b', 'B')
'2'.rjust(3, '0')
'22'.rjust(3, '0')
'222'.rjust(3, '0')

s = ' \t ' + s + '   \n '  # whitespace = {space, \t, \n}
s.strip()

s = 'abcdb'
s.strip('ab')
s.strip('ba')
s.lstrip('ab') if s.startswith('ab') else s
s[len('ab'):] if s.startswith('ab') else s

## 3.4 String Formatting
# https://pyformat.info/
message = "{} {} {}".format("hello", "world", 2021)
print(message)

## 3.5 Regular Expressions
# not covered here


###########################################
### Part 4. Four Python Data Structures ###
###########################################
# list
# tuple
# set
# dictionary

### 4.1 List
lst = []; lst
lst = list(); lst

lst = [1]; lst
lst = [1, 2, 3]; lst
lst = [1, [2, 3], [1.0, 'a', True]]; lst
len(lst)
len(lst[2])

[1, 2, 3] * 2

## 4.1.1 list indexing
lst = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']; lst
lst = list('abcdefghij'); lst
lst[1]
lst[1:]
lst[1:5]
lst[::2]
lst[::-1]

## 4.1.2 list methods
lst = [True, True, False, False]; lst
len(lst)
any(lst)
all(lst)
sum(lst)

lst = ['a', 'b', 'c', 'd']; lst
len(lst)
any(lst)
all(lst)
sum(lst)
'a' + 'b' + 'c' + 'd'

lst = list('abcd'); lst
lst.append('e'); lst
lst.append(['e','f']); lst

lst = list('abcd'); lst
lst.extend('g'); lst
lst.extend(['g','h']); lst

lst = list('abc'); lst
len(' ')
len('')
lst.append(''); lst     # append ''
lst.extend(''); lst     # append nothing
lst.append(''); lst     # append ''

## 4.1.3 list comprehension
lst = [1, 2, 3, 4, 5, 6]; lst

# traverse a list
for x in lst:
    print(x, x*2)

# multiply each element by 2
ans = []
for x in lst:
    ans.append(x * 2)
ans

# mapping
ans = [x*2 for x in lst]; ans

# conditional mapping
ans = [x*2 if x % 2 == 0 else x for x in lst]; ans

# filtering
ans = [x for x in lst if x % 2 == 0]; ans

# Q: get all string methods that contain 'just'
dir(str)
ans = [x for x in dir(str) if 'just' in x]; ans

# list comprehension with multiple inputs
a = [1,2,3]
b = [4,5,6]

[x-y for x in a for y in b]

ans = []
for x in a:
    for y in b:
        ans.append(x - y)
ans == [x-y for x in a for y in b]

[x-y for x in a for y in b]
[x-y for x in b for y in a]

## 4.1.4 mutability and hence aliasing
lst1 = [1, 2, 3, 4]; lst1
lst2 = lst1; lst2
lst1[0] = 9; lst1       # list is mutable
lst2

lst3 = [x for x in lst1]; lst3
id(lst1)
id(lst2)
id(lst3)
lst1[0] = 1
lst1
lst2
lst3

s = '123'; s
s[0] = '9'              # string is immutable
'9' + s[1:]
s[0] + '9' + s[1:]

### 4.2 Tuple
tup = (); tup
tup = tuple(); tup

tup = (1,); tup
tup = (1, 2, 3, 4); tup

(1,)
type((1,))
len((1,))
(1)
type((1))

tup[0] = 9              # tuple is immutable

len(tup)
sum(tup)

1,
(1, )
, 1
(, 1)
a, b = 1, 2
a, b = (1, 2)


### 4.3 Set
sat = set(); sat

a = {1, 2, 3, 4, 2, 3}; a
a = set([1, 2, 3, 4]); a
b = set([3, 4, 5, 6]); b
len(a)
sum(a)

## 4.3.1 set operations

a.union(b)          # 1. union - set is mutable
a | b               # identical to union
a or b
a or None
None or a

a.intersection(b)   # 2. intersection
a & b               # identical to intersection
a and b
b and a
a and None
None and a

a.difference(b)     # 3. difference
a - b               # identical to difference
a + b

a ^ b               # 4. opposite of intersection
a.union(b) - a.intersection(b)

1 in [1,2,3,4]      # 5. within
2 in {1,2,3,4}
{1,2} in a
{1, 2, 3, 4, [1,2]} # elements in set must be immutable
(1,2) in {1, 2, 3, 4, (1,2)} # tuple is immutable
{1, 2, 3, 4, {1,2}} # elements in set must be immutable

[x in a for x in [1,2,7,8]]
{x in a for x in [1,2,7,8]}

# hash table
1 in [1,2,3,4] # list
1 in {1,2,3,4} # set O(1)

### 4.4 Dictionary
dic = {}
dic = dict()

grades = {'mark': 70, 'wen': 90, 'nic': 95}; grades
grades = dict(mark=70, wen=90, nic=95); grades
grades['wen']

grades = {'mark': 70, 'wen': 90, 'nic': 95, ('mark','wen'): 80}; grades
grades[('mark','wen')]

## 4.4.1 dictionary methods
grades = {'mark': 70, 'wen': 90, 'nic': 95}; grades

len(grades)
max(grades)

'wen' in grades

grades.keys()
grades.values()
grades.items()

grades.update({'mark': 75, 'eric': 70})
grades

## 4.4.2 dictionary comprehension
grades = {'mark': 70, 'wen': 90, 'nic': 95}; grades
len(grades)
sum(grades)

# reverse key-value pairs
ans = {}
for k, v in grades.items():
    ans[v] = k
ans
len(ans)
sum(ans)

{grades[k]:k for k in grades}
{grades[k]:k for k in grades.keys()}
{v:k for k,v in grades.items()}

grades = {'mark': 70, 'wen': 90, 'nic': 90}; grades
{grades[k]:k for k in grades}

# Q: How to reverse a dictionary given presence of conflicting values?


###############################################
### Part 5. Loops & Conditionals & Controls ###
###############################################

### 5.1 For-loops
az = [chr(x) for x in range(ord('a'), ord('z')+1)]; az

for x in az[:10]:                   # iterate over contents
    print(x)

for i in range(10):                 # iterate over indexes
    print(i, az[i])

for i, x in enumerate(az[:10]):     # iterate over indexes & contents
    print(i, x)

### 5.2 While-loops
i = 0
while i < len(az):
    print(i, az[i])
    i += 1
    # i = i + 1

### 5.3 If-else Conditionals

# if only
for i in range(len(az)):
    if i % 2 == 0:
        print("even number:", i, az[i])

_ = [print("even number:", i, x) for i,x in enumerate(az) if i%2 ==0]

# if-else
for i in range(len(az)):
    if i % 2 == 0:
        print("even number:", i, az[i])
    else:
        print("odd number:", i, az[i])

_ = [print("even number:", i, x) if i%2 ==0 else print("odd number:", i, x) for i,x in enumerate(az)]

# if-elif-else
for i in range(len(az)):
    if i % 2 == 0:
        print("can be divided by 2:", i, az[i])
    elif i % 3 == 0:
        print("can be divided by 3:", i, az[i])
    else:
        print("some number:", i, az[i])

# if & if-else
for i in range(len(az)):
    if i % 2 == 0:
        print("can be divided by 2:", i, az[i])

    if i % 3 == 0:
        print("can be divided by 3:", i, az[i])
    else:
        print("some number:", i, az[i])

# if-elif
for i in range(len(az)):
    if i % 2 == 0:
        print("can be divided by 2:", i, az[i])
    elif i % 3 == 0:
        print("can be divided by 3:", i, az[i])

### 5.4 Control Operations
# continue
# break
# pass

imax = 30
for i in range(100):
    if i == imax:
        break           # stop the for-loop altogether
    elif '4' in str(i):
        print("I don't like this number.")
        continue        # skip to the next iteration
    elif i % 5 == 0:
        pass            # do nothing - a placeholder for future code
        print(i, 'divisible by 5')
    elif i % 2 == 0:
        print(i, 'is an even number')
    else:
        print(i, 'is an odd number')

### 5.5 Handle Exceptions
# https://realpython.com/python-exceptions/
az = [chr(x) for x in range(ord('a'), ord('z')+1)]; az

for i in range(100):
    print(i, az[i])

for i in range(100):
    if i < len(az):
        print(i, az[i])

for i in range(100):
    try:
        print(i, az[i])
    except:
        break

for i in range(100):
    try:
        print(i, az[i])
    except Exception as e:      # exception is useful because it does not stop iterations given errors
        print(e)
        break                   # try to remove 'break' and see what happens


################################
### Part 6. Define Functions ###
################################
# good_person  # snake case -> python function/variable name
# GoodPerson   # camel case -> python class name
# good-person  # kabab case

### 6.1 Single-input Functions
def fn(x):                          # def function_name(argument_name):
    ans = x * 2
    return ans                      # 'return' signals the end of function definition

f = lambda x: x*2

fn(3)
f(3)

def fn(x):
    ans1 = x * 2
    ans2 = ans1 + 1
    return (ans1, ans2)

x = fn(3)
x

x, y = fn(3)
x
y

### 6.2 Multi-input Functions
def fn(x, y):
    ans = x - y
    return ans

f = lambda x, y: x - y

fn(2, 3)
fn(x=2, y=3)
fn(y=3, x=2)
fn(3, 2)

f(2, 3)
f(y=3, x=2)


### 6.3 Functions with default values
def fn(x, y=3):     # keyword arguments as defaults
    ans = x - y
    return ans

def fn(x=2, y):     # kwargs must be put in the backwards
    ans = x - y
    return ans

f = lambda x, y=3: x - y

fn(2)
fn(2, 4)
fn(2, y=4)
fn(x=2, 4)          # kwargs must be put in the backwards
fn(x=2, y=4)

f(2)
f(2, 3)
f(2, y=3)
f(x=2, 3)           # kwargs must be put in the backwards
f(x=2, y=3)

### 6.4 Functions with variable-length inputs
# *args & **kwargs                                      # key-word arguments - kwargs
# '*' and '**' are called unpacking operators

x = [1, 2, 3]
print(x)
print(*x)
print(1, 2, 3)

def func(a=1, b=2, *args):      # 'args' can be named anything
    print(type(args))
    sum = 0
    for x in args:
        print(x)
        sum = sum + x
    return (a+b, sum)

func(1, 2, 1, 2, 3, 4)
func(a=1, b=2, 1, 2, 3, 4)      # kwargs must be put in the backwards
func(1, 2, *[1, 2, 3, 4])       # use unpacking operator '*' to handle unknown length
func(1, 2, [1, 2, 3, 4])        # failure
func(1, 2, *[1,2], *[3,4])      # use multiple unpacking operators '*'


def fn(x, y=3):     # keyword arguments as defaults
    ans = x - y
    return ans

def func(a=1, b=2, **kwargs):   # 'kwargs' can be named anything
    print(type(kwargs))
    for k, v in kwargs.items():
        print('key:', k, '  ', 'value:', v)
    diff = fn(**kwargs)
    return (a+b, diff)

func(1, 2, x=3, y=2)
func(x=3, y=2, 1, 2)            # kwargs must be put in the backwards
func(x=3, y=2, a=1, b=2)        # name all inputs
func(1, 2, **{'x':3, 'y':2})    # use unpacking operator '**' to handle unknown length
func(1, 2, {'x':3, 'y':2})      # failure
func(1, 2, **{'x':3}, **{'y':2})    # use multiple unpacking operators '**'

def func(a=1, b=2, *tup, **dic):
    print('type of args  :', type(tup))
    print('type of kwargs:', type(dic))
    summ = sum(tup)
    diff = fn(**dic)
    return (a+b, summ, diff)

func(1, 2, 1,2,3,4, x=10, y=4)
func(1, 2, *[1,2,3,4], x=10, y=4)
func(1, 2, *[1,2,3,4], **{'x':10, 'y':4})
func(1, 2, *[1,2], *[3,4], **{'x':10}, **{'y':4})


##############################
### Part 7. Define a Class ###
##############################
# GoodPerson   # camel case -> python class name
# good_person  # snake case -> python function/variable name
# good-person  # kabab case

### 7.1 Define a Minimalistic Class

# class with attributes
class Person():
    age = 1

class Person(object):
    age = 1

person = Person()
person.age
dir(person)


# class with attributes & methods
class Person(object):
    name = 'wen'
    age = 1
    def say_hi(self):       # try to remove and add 'self'
        s = "hi there"
        print(s)
        return s

person = Person()
person.name
person.age
ans = person.say_hi()
ans

person.name = 'mark'
person.name

### 7.2 Define a Standard Class
# standard class with attributes & methods
class Person(object):
    config_path = "/Users/WHE/OneDrive - MORNINGSTAR INC/documents/recruitment_upskill/training/upskill2021/"

    def __init__(self, name='wen', path='/notexists/'):   # initiatialize
        self.name = name
        self.path = path

    def say_hi(self, name=''):
        if name:
            self.name = name
        # self.name = name if name else self.name
        msg = "Hi there" + ', ' + self.name.capitalize()
        print(msg)
        self.msg = msg
        return None

    def write_it(self, file='candelete.txt'):
        pathfile = self.path + file
        with open(pathfile, 'w') as conn:
            conn.write(self.msg)
            self.pathfile = pathfile
            print("file saved successfully at: ", pathfile)
        return None

Person.config_path

person = Person()
person.config_path
person.name
person.msg
person.say_hi()
person.msg
person.name

person = Person(name='william')
person.name
ans = person.say_hi()
person.name
person.msg
ans = person.write_it()

person = Person(name='william', path=Person.config_path)
ans = person.say_hi()
ans = person.write_it()
person.name
person.pathfile


### 7.3 Inheritance
class Person(object):
    def __init__(self, name='person'):
        self.name = name

    def say_hi(self):
        print("Person says Hi")
        return None

class Kindness(object):
    def __init__(self, name='kindness'):
        self.name = name

    def say_hi(self):
        print("Kindness says Hi")
        return None

    def help(self):
        print("Kindness is helping")
        return None

# single inheritance - example 1
class GoodPerson(Person):
    def __init__(self, skills='cooking'):
        super().__init__()
        self.skills = skills

gp = GoodPerson()
gp.name
gp.skills
gp.say_hi()
gp.help()

# single inheritance - example 2
class GoodPerson(Kindness):
    def __init__(self, skills='cooking'):
        super().__init__()
        self.skills = skills

gp = GoodPerson()
gp.name
gp.skills
gp.say_hi()
gp.help()

# single inheritance - example 3 - redefine attributes & methods
class GoodPerson(Person):
    def __init__(self, name='goodperson', skills='cooking'):
        super().__init__()          # attributes depend on order
        self.name = name            # overwrite after class initialization
        self.skills = skills

    def say_hi(self):               # methods always overwrite
        print("Goodperson says Hi")
        return None

gp = GoodPerson()
gp.name
gp.say_hi()

class GoodPerson(Person):
    def __init__(self, name='goodperson', skills='cooking'):
        self.name = name            # overwrite before class initialization
        self.skills = skills
        super().__init__()          # attributes depend on order

    def say_hi(self):               # methods always overwrite
        print("Goodperson says Hi")
        return None

gp = GoodPerson()
gp.name
gp.say_hi()

# multiple inheritance
class GoodPerson(Person, Kindness):
    def __init__(self, skills='cooking'):
        super().__init__()
        self.skills = skills

class GoodPerson(Kindness, Person):
    def __init__(self, skills='cooking'):
        super().__init__()
        self.skills = skills

gp = GoodPerson()
gp.skills
gp.name
ans = gp.say_hi()
ans = gp.help()

# single inheritance - example 4 - different ways of class initialization
class GoodPerson(Person):
    def __init__(self, option=1):
        if option == 1:
            super().__init__()
        elif option == 2:
            super(GoodPerson, self).__init__()
        elif option == 3:
            Person.__init__(self)   # use this when you inherit multiple classes with different initial arguments

    def help(self):
        print("GoodPerson is helping")
        return None

gp = GoodPerson(option=1)
gp = GoodPerson(option=2)
gp = GoodPerson(option=3)


#########################################
### Part 8. Python Built-in Functions ###
#########################################
dir(__builtins__)

## already covered
# help(), dir(), type()
# bool(), int(), float(), str()
# list(), tuple(), set(), dict()
# ord(), chr(), id()
# len(), max(), min(), sum(), round(),
# range(), enumerate()
# any(), all()


### 8.1 Sorting
lst = [2, 3, 1, 4, 5]
sorted(lst)
sorted(lst, reverse=True)

set = (2, 3, 1, 4, 5); set
sorted(set)

dic = {'a': 4, 'c': 2, 'b': 3, 'd': 1}; dic
sorted(dic)
[dic[k] for k in dic]
[dic[k] for k in sorted(dic)]

out = sorted(dic.items(), key=lambda x: x[1]); out
[x[0] for x in out]

out = sorted(dic.items(), key=lambda x: x[0]); out
[x[0] for x in out]
sorted(dic)

### 8.2 Zipping
a = [ 1,   2,   3 ]
b = ['a', 'b', 'c']
ab = [a, b]; ab
aba = [a, b, a]; aba

zip(a, b)
list(zip(a, b))
list(zip(*ab))      # * is an unpacking operator
list(zip(*aba))     # zipping multiple lists is similar to zipping two

out
tmp = list(zip(*out)); tmp
out = list(zip(*tmp)); out

### 8.3 Set & Get Attributes
s = 'abc'
type(s)

type(s) == str
type(s) in [int, float, str]

isinstance(s, str)
isinstance(s, (int, float, str))

hasattr(s, 'upper')
hasattr(str, 'upper')

s.upper()
str.upper(s)

getattr(s, 'upper')()
getattr(s, 'upper')(s)
getattr(str, 'upper')(s)

s = 'abcABC'; s
fns = ['upper', 'lower']; fns
[getattr(str, fn)(s) for fn in fns]





############################################
### Part 9. Commonly Used Python Modules ###
############################################

# os, shutil, subprocess
# re, time
# functools, collections, itertools
# pickle, csv, json


