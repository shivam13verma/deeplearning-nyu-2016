--Deep earning: HW4
--nngraph_warmup.lua
--Author: Shivam Verma
--
--Problem 1.1 (a)
--
require 'torch'
require 'nngraph'

--given sizes
SIZE_A = 2
SIZE_X = 4
SIZE_Y = 5

x = nn.Linear(SIZE_X,SIZE_A)()
--weight matrix is dim. (A x X)
x.data.module.weight:ones(SIZE_A,SIZE_X)
x.data.module.bias:ones(SIZE_A)
sq_tanh = nn.Square()(nn.Tanh()(x))

y = nn.Linear(SIZE_Y,SIZE_A)()
--weight matrix is dim. (A x Y)
y.data.module.weight:ones(SIZE_A,SIZE_Y)
y.data.module.bias:ones(SIZE_A)
sq_sig = nn.Square()(nn.Sigmoid()(y))

z = nn.Identity()()
mult= nn.CMulTable()({sq_tanh,sq_sig})
sum = nn.CAddTable()({mult,z})
a = nn.gModule({x,y,z},{sum})

--Problem 1.1 (b)

--sample values of x,y,z
t1 = torch.ones(4)
t2 = torch.ones(5)
t3 = torch.ones(2)

--Prints the outputs from forward & backward pass
function print_values(x,y,z)
	print("We are given: \n")
	print("x:\n")
	print (x)
	print("y:\n")
	print (y)
	print("z:\n")
	print(z)
	out = a:forward({x,y,z})
	print("Output from forward pass:\n")
	print(out)
	gradOut = torch.ones(2)
	print("Output from backward pass:\n")
	print(a:backward({x,y,z},gradOut))
end

print_values(t1,t2,t3)
