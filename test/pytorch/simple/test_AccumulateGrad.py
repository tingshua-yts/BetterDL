import torch
def tensor_status(**kwargs):
    print("\n")
    for arg_name in kwargs:
        t= kwargs[arg_name]
        t_name=arg_name
    print("Status of tensor "+t_name+":\n")
    print("==============")
    print("Data: "+str(t.data))
    print("grad: "+str(t.grad) if t.is_leaf else str(None))
    print("grad_fn: "+str(t.grad_fn))
    print("is_leaf: "+str(t.is_leaf))
    print("requires_grad:"+str(t.requires_grad))
    print("==============\n")
    return ""

#Building the Neural Net
#======================
#Inputs
x=torch.Tensor([5, 3])
print("Inputs:\n",x,"\n")

#Randomly initialized weights
W_x_z=torch.randn(2,2,requires_grad=True)
print("Weights X -> Z:\n",W_x_z,"\n")

W_z_y=torch.randn(2,1,requires_grad=True)
print("Weights Z -> Y:\n",W_z_y,"\n")

#Define output
y=torch.Tensor([7.0])
#======================


#Loss Function
loss_fn=torch.nn.MSELoss()


#FORWARD PROPAGATION
#======================
print("\nForward Propagation:\n")

z = torch.matmul(x , W_x_z )
print("Hidden Layer:\n",z,"\n")

y_pred = torch.matmul(z,W_z_y)
print("Output:\n",y_pred,"\n\nExpected Value :\n",y,"\n\n")
#======================

#Calculate Error(loss)
loss=loss_fn(y_pred,y)
print("Mean Squared Error : " + str(loss.item()))

#BACKPROPAGATION
loss.backward()

print(loss.grad_fn)
print("next_functions : \n",loss.grad_fn.next_functions)

print("\t|\n\t|\n\t|\n\t|\n\t|")
print(loss.grad_fn.next_functions[0][0])
print("next_functions : \n",loss.grad_fn.next_functions[0][0].next_functions)

print("\t|\n\t|\n\t|\n\t|\n\t|")
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])
print("next_functions : \n",loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)

print("\t|\n\t|\n\t|\n\t|\n\t|")
print(loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0])
print("next_functions : \n",loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)

print("\t|\n\t|\n\t|\n\t|\n\t|")
print(loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0])
print("next_functions : \n",loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)

print("\t|\n\t|\n\t|\n\t|\n\t|")
print(loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0])
print("next_functions : \n",loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)
