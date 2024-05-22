# To ease the presentation, we consider only certain parameters in this demo. The following set contains the names of them

NAMES_OF_AGGREGATED_PARAMETERS = {'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.weight', 'conv1.weight',
                                  'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var',
                                  'layer1.0.bn1.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean',
                                  'layer1.0.bn2.running_var', 'layer1.0.bn2.weight', 'layer1.0.conv1.weight',
                                  'layer1.0.conv2.weight', 'layer1.1.bn1.bias', 'layer1.1.bn1.running_mean',
                                  'layer1.1.bn1.running_var', 'layer1.1.bn1.weight', 'layer1.1.bn2.bias',
                                  'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var', 'layer1.1.bn2.weight',
                                  'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer2.0.bn1.bias',
                                  'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.bn1.weight',
                                  'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var',
                                  'layer2.0.bn2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight',
                                  'layer2.0.shortcut.0.weight', 'layer2.0.shortcut.1.bias', 'layer2.0.shortcut.1.running_mean',
                                  'layer2.0.shortcut.1.running_var', 'layer2.0.shortcut.1.weight', 'layer2.1.bn1.bias',
                                  'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.bn1.weight',
                                  'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var',
                                  'layer2.1.bn2.weight', 'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer3.0.bn1.bias',
                                  'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.bn1.weight',
                                  'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var',
                                  'layer3.0.bn2.weight', 'layer3.0.conv1.weight', 'layer3.0.conv2.weight',
                                  'layer3.0.shortcut.0.weight', 'layer3.0.shortcut.1.bias', 'layer3.0.shortcut.1.running_mean',
                                  'layer3.0.shortcut.1.running_var', 'layer3.0.shortcut.1.weight', 'layer3.1.bn1.bias',
                                  'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var', 'layer3.1.bn1.weight',
                                  'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var',
                                  'layer3.1.bn2.weight', 'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer4.0.bn1.bias',
                                  'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.weight',
                                  'layer4.0.bn2.bias', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var',
                                  'layer4.0.bn2.weight', 'layer4.0.conv1.weight', 'layer4.0.conv2.weight',
                                  'layer4.0.shortcut.0.weight', 'layer4.0.shortcut.1.bias', 'layer4.0.shortcut.1.running_mean',
                                  'layer4.0.shortcut.1.running_var', 'layer4.0.shortcut.1.weight', 'layer4.1.bn1.bias',
                                  'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var', 'layer4.1.bn1.weight',
                                  'layer4.1.bn2.bias', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var',
                                  'layer4.1.bn2.weight', 'layer4.1.conv1.weight', 'layer4.1.conv2.weight', 'linear.bias',
                                  'linear.weight'}