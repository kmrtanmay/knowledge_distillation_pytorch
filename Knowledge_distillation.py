#Packages imported
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.nn as nn

# check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

##################### Loading Data ########################

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True)
 
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False)

###########################################################



#Defining a convolution block
def conv_block(in_channels,out_channels,*args,**kwargs):
	return nn.Sequential(
			nn.Conv2d(in_channels,out_channels,*args,**kwargs),
		    nn.BatchNorm2d(out_channels),
		    nn.ReLU()
		    )


# Teacher's model having 5 convolution layers
class Teacher_Model(nn.Module):
	def __init__(self,in_channels=3,num_classes=10):
		super(Teacher_Model,self).__init__()

		self.encoder = nn.Sequential(
			conv_block(in_channels,8,kernel_size=3,padding=1),
			nn.MaxPool2d(kernel_size=2, stride=2),
		 	conv_block(8,16,kernel_size=3,padding=1),
            conv_block(16,16,kernel_size=3,padding=1),
			nn.MaxPool2d(kernel_size=2, stride=2),
            conv_block(16,16,kernel_size=3,padding=1),
            conv_block(16,16,kernel_size=3,padding=1),
			nn.MaxPool2d(kernel_size=2, stride=2)
		 	)

		self.decoder = nn.Sequential(
			nn.Linear(28*28*16,1024),
			nn.Sigmoid(),
			nn.Linear(1024,num_classes)
			)

	def forward(self,x):
		x = self.encoder(x)

		x = x.view(x.size(0),-1)

		x = self.decoder(x)

		return x



# Student's model having 3 convolution layers
class Student_Model(nn.Module):
	def __init__(self,in_channels=3,num_classes=10):
		super(Student_Model,self).__init__()

		self.encoder = nn.Sequential(
			conv_block(in_channels,8,kernel_size=3,padding=1),
			nn.MaxPool2d(kernel_size=2, stride=2),
      		conv_block(8,16,kernel_size=3,padding=1),
			nn.MaxPool2d(kernel_size=2, stride=2),
      		conv_block(16,16,kernel_size=3,padding=1),
			nn.MaxPool2d(kernel_size=2, stride=2)
		 	)

		self.decoder = nn.Sequential(
			nn.Linear(28*28*16,1024),
			nn.Sigmoid(),
			nn.Linear(1024,num_classes)
			)

	def forward(self,x):
		x = self.encoder(x)

		x = x.view(x.size(0),-1)

		x = self.decoder(x)

		return x		


teacher = Teacher_Model()
student = Student_Model()

teacher.to(device)
student.to(device)

# optimizers
optimizer_teacher = optim.SGD(teacher.parameters(), lr=0.001, momentum=0.9)
optimizer_student = optim.SGD(student.parameters(), lr=0.001, momentum=0.9)

# loss function
criterion = nn.CrossEntropyLoss()

#### Training function for teacher model
def fit_teacher(model, train_dataloader):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in enumerate(train_dataloader):
        data, target = data[0].to(device), data[1].to(device)
        optimizer_teacher.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer_teacher.step()
    train_loss = train_running_loss/len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(train_dataloader.dataset)
 
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')
    
    return train_loss, train_accuracy
###########################################################################

#####validation function
def validate(model, test_dataloader):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    for int, data in enumerate(test_dataloader):
        data, target = data[0].to(device), data[1].to(device)
        output = model(data)
        loss = criterion(output, target)
        val_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        val_running_correct += (preds == target).sum().item()
    
    val_loss = val_running_loss/len(test_dataloader.dataset)
    val_accuracy = 100. * val_running_correct/len(test_dataloader.dataset)
    
    print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.2f}')

    return val_loss, val_accuracy

# Calculate the loss for training the student model
def distillation(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(F.softmax(y/T,1), F.softmax(teacher_scores/T,1)) * (T*T * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)


##### training function for student model using Knowledge Distillation ###########
def fit_student(student,teacher, train_dataloader,temperature,alpha):
    student.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in enumerate(train_dataloader):
        data, target = data[0].to(device), data[1].to(device)
        optimizer_student.zero_grad()
        y = student(data)
        teacher_scores = teacher(data)
        loss = distillation(y,target,teacher_scores,temperature,alpha)      
        train_running_loss += loss.item()
        _, preds = torch.max(y.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer_student.step()
    train_loss = train_running_loss/len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(train_dataloader.dataset)
 
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')
    
    return train_loss, train_accuracy
######################################################################################

train_loss , train_accuracy = [], []
val_loss , val_accuracy = [], []
start = time.time()
for epoch in range(10):
    train_epoch_loss, train_epoch_accuracy = fit_teacher(teacher, trainloader)
    val_epoch_loss, val_epoch_accuracy = validate(teacher, testloader)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
end = time.time()
 
print((end-start)/60, 'minutes')


train_loss , train_accuracy = [], []
val_loss , val_accuracy = [], []
start = time.time()
for epoch in range(10):
    train_epoch_loss, train_epoch_accuracy = fit_student(student,teacher, trainloader,temperature=10,alpha=0.1)
    val_epoch_loss, val_epoch_accuracy = validate(student, testloader)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
end = time.time()
 
print((end-start)/60, 'minutes')

#########################################################################

##################### Without Knowledge distillation ####################

student_noKD = Student_Model()
student_noKD.to(device)
optimizer_student_noKD = optim.SGD(student_noKD.parameters(), lr=0.001, momentum=0.9)

# training function for student model without Knowledge Distillation

def fit_student_noKD(model, train_dataloader):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in enumerate(train_dataloader):
        data, target = data[0].to(device), data[1].to(device)
        optimizer_student_noKD.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer_student_noKD.step()
    train_loss = train_running_loss/len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(train_dataloader.dataset)
 
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')
    
    return train_loss, train_accuracy


train_loss , train_accuracy = [], []
val_loss , val_accuracy = [], []
start = time.time()
for epoch in range(10):
    train_epoch_loss, train_epoch_accuracy = fit_student_noKD(student1, trainloader)
    val_epoch_loss, val_epoch_accuracy = validate(student1, testloader)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
end = time.time()
 
print((end-start)/60, 'minutes')
