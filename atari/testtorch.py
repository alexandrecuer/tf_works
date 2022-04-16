import torch

input("utilisation de unsqueeze pour ajouter une dimension")
x = torch.zeros(6)
x[4]=2.0
x[3]=1.1
print(x)
y = x.unsqueeze(-1)
print(y)

input("utilisation de gather")
t = torch.tensor([[1,2],[3,4]])
print("original tensor")
print(t)
index = torch.tensor([[0,0],[1,0]])
print("index")
print(index)
print("résultat")
print(t.gather(1,index))
index=torch.tensor([1,0])
print("index")
print(index)
print("résultat")
print(t.gather(1,index.unsqueeze(-1)).squeeze(-1))

input("utilisation des masques pour mettre les valeurs à 0 sur un choix d'indices")
import numpy as np
mask = np.zeros(4)
mask[2]=1
actions = np.random.rand(4)

mask_tensor = torch.BoolTensor(mask)
actions_tensor = torch.tensor(actions)
print(mask_tensor)
print(actions_tensor)

actions_tensor[mask_tensor] = 0.0
print(actions_tensor)
