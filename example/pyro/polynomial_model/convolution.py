import pyro
import torch
import itertools

def conv(tensor1: torch.Tensor, tensor2: torch.Tensor):
    shape1 = tensor1.shape
    shape2 = tensor2.shape

    vector_size = max(shape1+shape2)

    conv_shape = list(map(lambda i,j: max(i,j), shape1, shape2))

    tensor_conv = torch.zeros(conv_shape)

    for coordinate in itertools.product(*(range(vec_size) for vec_size in shape2)):
        coordinate_ = list(coordinate)

        slices_conv = tuple(map(lambda i,j: slice(i,min(i+j,vector_size)), coordinate_, shape1))
        slices_tens1 = tuple(map(lambda i,j: slice(0,min(j,vector_size-i)), coordinate_, shape1))

        tensor_conv[slices_conv] += tensor1[slices_tens1]*tensor2[coordinate]

    return tensor_conv

def test():
    t1 = torch.zeros([1,1,3])
    t2 = torch.zeros([1,3,1])
    t3 = torch.zeros([3,1,1])

    t1[0,0] = torch.arange(3)
    t2[0,:,0] = torch.arange(3)
    t3[:,0,0] = 2*torch.arange(3)

    print(t1.shape)
    print(t2.shape)
    print(t3.shape)

    t4 = conv(t1,t2)
    print(t4)
    t5 = conv(t4,t3)

    t6 = conv(t3,t2)
    print(t6)
    t7 = conv(t1,t6)

    print(t5)
    print(t7)

    l1 = torch.arange(1,4)
    l2 = conv(conv(l1,l1),l1)
    print(l2)

def test_time():
    import time
    time_list = []
    for i in range(50,150,10):
        print(i)
        
        t1 = torch.zeros([1,1,i])
        t2 = torch.zeros([1,i,1])
        t3 = torch.zeros([i,1,1])

        t1[0,0] = torch.arange(i)
        t2[0,:,0] = torch.arange(i)
        t3[:,0,0] = 2*torch.arange(i)

        start = time.time()
        t4 = conv(conv(t1,t2),t3)
        t5 = conv(t4,t4)
        time_list.append(time.time()-start)


    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(time_list)
    plt.show()

    
if __name__ == "__main__":
    test()
    #test_time()