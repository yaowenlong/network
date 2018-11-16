import network
import VOC
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.optim as optim
import torch.nn as nn
criterion = nn.MSELoss()
classes = ('__background__' , # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
xmlpath = "VOC2012/Annotations"
imgpath = "VOC2012/JPEGImages"
transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))

    ]
)
train_data = VOC.MydataSet(xmlpath,imgpath,transform=transform, target_transform=None)
test_data = VOC.MydataSet(xmlpath,imgpath,transform=transform, target_transform=None)
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
net1 = network.vgg13_bn().cuda()
net2 = network.vgg13_bn().cuda()
optimizer = optim.SGD(net1.parameters(), lr=0.001, momentum=0.9)
optimizer2 = optim.SGD(net2.parameters(), lr=0.001, momentum=0.9)
attention2 = 1
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.cuda()
        labels1 = labels[0].cuda()
        labels2 = labels[1].cuda()
        #print(labels)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        net1.getattention(attention2)
        outputs1,attention1 = net1(inputs)
        net2.getattention(attention1)
        outputs2,attention2 = net2(inputs)
        loss = criterion(outputs1, labels1)
        loss.backward(retain_graph=True)
        loss2 = criterion(outputs1, labels2)
        loss2.backward(retain_graph=True)
        optimizer.step()
        optimizer2.step()
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs,_ = net1(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.float()
        total += labels[0].size(0)
        correct += (predicted == labels[0]).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
#print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))