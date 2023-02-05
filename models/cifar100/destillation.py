from fedmd.model_build import build_model
from fedmd.resnet20 import build_resnet20

num_classes = 100
ClientModel1 = build_model(num_classes=num_classes, n1=128, n2=128, n3=192, softmax=False)
ClientModel2 = build_model(num_classes=num_classes, n1=64, n2=64, n3=64, softmax=False)
ClientModel3 = build_model(num_classes=num_classes, n1=128, n2=64, n3=64, softmax=False)
ClientModel4 = build_model(num_classes=num_classes, n1=64, n2=64, n3=128, softmax=False)

ClientModel0 = build_resnet20(num_classes)
