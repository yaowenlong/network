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
net = network.DetectionNetSPP()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        #print(labels)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print(loss.item())
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
#print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))