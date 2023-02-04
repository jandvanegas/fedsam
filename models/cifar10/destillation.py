from fedmd.model_build import build_model
from fedmd.resnet20 import build_resnet20

num_classes = 10
ClientModel1 = build_model(num_classes=num_classes, n1=128, n2=128, n3=192, softmax=True)
ClientModel2 = build_model(num_classes=num_classes, n1=64, n2=64, n3=64, softmax=True)
ClientModel3 = build_model(num_classes=num_classes, n1=128, n2=64, n3=64, softmax=True)
ClientModel4 = build_model(num_classes=num_classes, n1=64, n2=64, n3=128, softmax=True)

ClientModel0 = build_resnet20(num_classes)
