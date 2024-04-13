import torch
import torch.nn.functional as F
import pandas as pd
import random
import ast
from torch_geometric.datasets import MovieLens
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


testdataset=MovieLens(root='/tmp/Cora')
### CONSTRAINTS
TEACHER_MAX_HOURS = 20  # Max weekly hours per teacher
WEIGHT_L = 0.8  # Prioritizing teachers workload
WEIGHT_R = 0.1  # Room capacity weight
WEIGHT_C = 0.1  # Constraints weight


# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# data = Data(salam=x, edge_index=edge_index)


# Load data from Excel
#Occasionally adjusting arrays
subjects_data = pd.read_excel("school_data.xlsx", sheet_name="subjects")
weekly_hours_strings = subjects_data["WeeklyHours"].tolist()
weekly_hours_list = [ast.literal_eval(x) for x in weekly_hours_strings]




classes_data = pd.read_excel("school_data.xlsx", sheet_name="classes")
teachers_data = pd.read_excel("school_data.xlsx", sheet_name="teachers")
teachers_expertise_strings = teachers_data["subject_expertise"].tolist()
teachers_expertise_list = [ast.literal_eval(x) for x in teachers_expertise_strings]



#Creating assignment for each possible scenario for the current data, will be updated when hours and classroom data is added
assignment = {}

for lvl_idx, level in enumerate(classes_data["class_id"]):
    for sub_idx, subject in enumerate(subjects_data["lesson_id"]):
                for tch_idx, teacher in enumerate(teachers_data['teacher_id']):
                            key = (lvl_idx, sub_idx, tch_idx)
                            name = f"{level} S:{subject} T:{teacher}"
                            if subject in teachers_expertise_list[tch_idx]:
                                classHoursIndex=0
                                if classes_data["grade_level"][lvl_idx]==5:
                                    classHoursIndex=0
                                elif classes_data["grade_level"][lvl_idx]==6:   
                                    classHoursIndex=1
                                elif classes_data["grade_level"][lvl_idx]==7:
                                    classHoursIndex=2
                                    
                                    
                                if weekly_hours_list[sub_idx][classHoursIndex]!=0:
                                    assignment[key] = 1
                                else:
                                    assignment[key] = 0
                            else:
                                
                                assignment[key] =0
                                
                                
                                                            
# First constraint: Each Class must have the quantity of classes per subject specified in the curriculum. With additional hours data the results will be improved

# Each satisfaction status is implemented inside the arrays that are later implemented into a dataset

c1=[]
for lvl_idx, level in enumerate(classes_data["class_id"]):
    for sub_idx, subject in enumerate(subjects_data["lesson_id"]):
                    classHoursIndex=0
                    if classes_data["grade_level"][lvl_idx]==5:
                                    classHoursIndex=0
                    elif classes_data["grade_level"][lvl_idx]==6:   
                                    classHoursIndex=1
                    elif classes_data["grade_level"][lvl_idx]==7:
                                    classHoursIndex=2
                    if weekly_hours_list[sub_idx][classHoursIndex]!=0:

                        required_slots = weekly_hours_list[sub_idx][classHoursIndex]
                        temp=0
                        for tch_idx, teacher in enumerate(teachers_data['teacher_id']):
                              temp+=assignment[lvl_idx, sub_idx, tch_idx]
                        print("=============")
                        print(temp)
                        print(required_slots)
                        print("=============")

                        c1.append(temp>= required_slots)
                    else: c1.append(False)
                    


#  Maximum work hours for each teacher constraint.

c2=[]

for tch_idx, teacher in enumerate(teachers_data['teacher_id']):
      
    c2.append(sum(
        [
        assignment[lvl_idx, sub_idx, tch_idx]
        for lvl_idx, level in enumerate(classes_data["class_id"])
        for sub_idx, subject in enumerate(subjects_data["lesson_id"])])<= TEACHER_MAX_HOURS)
            



#Creating a custom dataset from these values
num_nodes = 2 
edges = torch.ones((num_nodes, num_nodes), dtype=torch.long)  # Long tensor for edge indices

#As the arrays are not the same length padding is implemented
max_len = max(len(c1), len(c2))  # Find the maximum length
padded_array1 = torch.nn.functional.pad(torch.tensor(c1), (0, max_len - len(c1)), value=False)
padded_array2 = torch.nn.functional.pad(torch.tensor(c2), (0, max_len - len(c2)), value=False)


node_feature1 = padded_array1.unsqueeze(0)
node_feature2 = padded_array2.unsqueeze(0)


labels = torch.tensor([0, 1])

node_features = torch.cat([node_feature1, node_feature2], dim=0)  # Concatenate along node dimension

# Dataset generated
data = Data(edge_index=edges, x=node_features,y=labels)

print(data)


class GNNLayer(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super(GNNLayer, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)

  def forward(self, node_features, edge_index):
    edge_index = edge_index.long()

    messages = F.relu(self.linear1(node_features[edge_index[:, 0]])) 

    num_nodes = node_features.size(0)
    message_matrix = torch.zeros(num_nodes, num_nodes)
    for i in range(edge_index.size(0)):
      source, target = edge_index[i]
      print(target.size())
      
      #error about expand occurs in this line. could you please send feedback about this.
      message_matrix[target, source] = messages[i].expand(max_len, -1) 



    aggregated_messages = message_matrix
    return F.relu(self.linear2(aggregated_messages + node_features))

class ScheduleGNN(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(ScheduleGNN, self).__init__()
    self.gconv1 = GNNLayer(input_dim, hidden_dim)
    self.gconv2 = GNNLayer(hidden_dim, output_dim)

  def forward(self, node_features, edge_index):
    x = self.gconv1(node_features, edge_index)
    x = self.gconv2(x, edge_index)
    return torch.nn.functional.softmax(torch.nn.linear(x, output_dim), dim=1)


model = ScheduleGNN(node_features.shape[1], 16, len(data.y.unique())) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
data.x = data.x.float()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = data.to(device)
model = model.to(device)


for epoch in range(100):  
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)  
    loss = criterion(output, data.y)  
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch: {epoch+1:03d}, Loss: {loss.item():.4f}')

