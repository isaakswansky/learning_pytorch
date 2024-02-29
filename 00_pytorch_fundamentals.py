import torch
print(torch.__version__)

# Creating tensors

# Scalar (single number/ 0-dimensional tensor)
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim) # number of dimensions of a scalar: 0
print(scalar.item()) # get int back from tensor

# Vector: 1-dimensional tensor
vector = torch.tensor([7, 8])
print(vector)
print(vector.ndim)
print(vector.shape)

# Matrix: 2-dimensional tensor
MATRIX = torch.tensor([[7, 8], [9, 10]])
print(MATRIX)
print(MATRIX.ndim)
print(MATRIX.shape)

# Tensor: multi-dimensional
TENSOR = torch.tensor([[[7, 8, 9],
                        [9, 10, 11],
                        [13, 14, 15]]])
print(TENSOR)
print(TENSOR.ndim)
print(TENSOR.shape)
print(TENSOR[0])

# random tensors, neural networks typically use random weights as a starting point
# start with random numbers -> look at data -> adjust weights

# random tensor with shape (3, 4)
random_tensor = torch.rand(3, 4)
print(random_tensor.ndim)

# zeros and ones
zeros = torch.zeros(3, 4)
print(zeros)
ones = torch.ones(3, 4, dtype=torch.int32)
print(ones)
print(zeros * random_tensor)
print(ones.dtype)

# ranges
one_to_ten = torch.arange(start=1, end=11, step=1)
print(one_to_ten)

# Creating tensors like (i.e. in the same shape as another tensor)
ten_zeros = torch.zeros_like(input=one_to_ten)
print(ten_zeros)

# Tensor datatypes
# Tensor datatypes is one of the 3 big errors you'll encounter with PyTorch
# 1. Tensors not right datatype
# 2. Tensors not right shape
# 3. Tensors not on right device (CPU or GPU)
float32_tensor = torch.tensor([1, 2, 3],
                              dtype=torch.float32, # data type of the tensor (default: float32)
                              device="cpu", # device to store the tensor (default: CPU)
                              requires_grad=False # whether to track gradients with this tensor's operations (default: False)
                              )
print(float32_tensor.dtype)
int64_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
print(int64_tensor.dtype)
print(int64_tensor.device)

# Manipulating tensors (tensor operations)
# - Addition
# - Subtraction
# - Multiplication (element-wise)
# - Division (element-wise)
# - Matrix multiplication (dot product)
tensor = torch.tensor([1, 2, 3])
print(tensor + 10) # add to each element
print(tensor - 10) # subtract from each element
print(tensor * 10) # multiply each element by 10
print(tensor / 10) # divide each element by 10
print(tensor ** 2) # square each element
print(torch.mul(tensor, 10)) # multiply each element by 10
print(tensor)

# Matrix multiplication
print(torch.matmul(tensor, tensor)) # dot product
# inner dimensions must match:
# (3, 2) x (2, 3) -> (3, 3)
# the resulting shape is the outer dimensions of the matrices being multiplied

# Transpose: .T
tensor_a = torch.rand(3, 2)
tensor_b = torch.rand(3, 2)
print(torch.mm(tensor_a, tensor_b.T))

# Tensor aggregation (min, max, mean, sum, etc.)
tensor = torch.rand(13, 2)
print("Tensor: ", tensor)
print("Min: ", tensor.min())
print("Max: ", tensor.max())
print("Mean: ", tensor.mean(dtype=torch.float32)) # onl.y works with float or complex
print("Median: ", tensor.median())
print("Sum: ", tensor.sum())
print("argmin: ", tensor.argmin())
print("argmax: ", tensor.argmax())

# Reshaping, stacking, squeezing and unsqueezing
# Reshaping: changing the shape of a tensor
# View: Return a view of an input tensor with the same data but different size or number of elements
# Stacking: Concatenates a sequence of tensors along a new dimension (vstack/hstack)
# Squeezing: Remove single-dimensional entries from the shape of a tensor
# Unsqueezing: Add a dimension with size one
# Permute: Return a view of the original tensor with its dimensions permuted
x = torch.arange(1., 11.)
print(x)
print("original:", x.shape)

# Reshape
x_reshaped = x.reshape(5, 2)
print("reshaped:", x_reshaped)
print(x_reshaped.shape)
x_reshaped[0, 0] = 99 # changing x_reshaped will also change x -> shared memory
print("original:", x)
print("reshaped:", x_reshaped)

# View
x_view = x.view(2, 5)
print("view:", x_view)
print(x_view.shape)
x_view[-1, -1] = 99 # changing x_view will also change x -> shared memory
print(x)
print(x_view)

# Stacking
x_stacked = torch.stack((x, x))
print("stacked dim=0:", x_stacked)
x_stacked = torch.stack((x, x), dim=1)
print("stacked dim=1:", x_stacked)

# Squeezing
x = torch.ones(3, 1, 2)
print(x.shape)
print(torch.squeeze(x).shape)
print(torch.unsqueeze(x, dim=3).shape)

# Permute
x = torch.rand(2, 3, 4)
print(x)
print(x.shape)
y = torch.permute(x, dims=(2, 0, 1)) # returns a view of the original tensor with its dimensions permuted
print(y)
print(y.shape)

# Tensor indexing
x = torch.arange(1., 11.).reshape(5, 2)
print(x)
print(x[0])
print(x[1:5])
print(x[1:2])
print(x[1:5, 1])
print(x[1, 0])

# Interact with numpy
import numpy as np
arr = np.arange(1, 6)
tensor = torch.from_numpy(arr)
print("numpy:", arr)
print("numpy:", arr.dtype)
print("torch:", tensor)
print("torch:", tensor.dtype)
arr += 99 # shared memory -> changing arr will also change tensor
print("numpy:", arr)
print("torch:", tensor)
numpy_tensor = tensor.numpy()
print("numpy:", numpy_tensor)
tensor += 99 # shared memory -> changing tensor will also change numpy_tensor
print("numpy:", numpy_tensor)
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
tensor = tensor + 99 # creates a new tensor, so the changes are not reflected in numpy_tensor
print("numpy:", numpy_tensor)

# Reproducibility
# to reduce randomness in neural networks and PyTorch comes the conceept of a random seed
torch.manual_seed(42) # sets a random seed for reproducibility (works only for one call)
x = torch.rand(2, 3)
y = torch.rand(2, 3)
print(x == y)

torch.manual_seed(42)
x = torch.rand(2, 3)
torch.manual_seed(42) # set seed for each call
y = torch.rand(2, 3)
print(x == y)





