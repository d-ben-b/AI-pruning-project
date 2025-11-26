import json
import os

file_path = '/home/ben/project/EAI/Lab4 Material/StudentID_Lab4.ipynb'

with open(file_path, 'r') as f:
    notebook = json.load(f)

new_forward_method = [
    "    def forward(self, x):\n",
    "        x = self.conv1(x)\n",
    "        x = self.bn1(x)\n",
    "        x = self.relu(x)\n",
    "        x = self.maxpool(x)\n",
    "\n",
    "        feature1 = self.layer1(x)\n",
    "        feature2 = self.layer2(feature1)\n",
    "        feature3 = self.layer3(feature2)\n",
    "        feature4 = self.layer4(feature3)\n",
    "\n",
    "        x = self.avgpool(feature4)\n",
    "        x = torch.flatten(x, 1)\n",
    "        x = self.fc(x)\n",
    "\n",
    "        return x, [feature1, feature2, feature3, feature4]\n"
]

found = False
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Check if this is the ResNet class cell
        if any('class ResNet(nn.Module):' in line for line in source):
            # Find the start of the forward method
            start_idx = -1
            end_idx = -1
            for i, line in enumerate(source):
                if 'def forward(self, x):' in line:
                    start_idx = i
                    break
            
            if start_idx != -1:
                # We assume the forward method goes until the end of the cell or we can just replace from start_idx
                # In the provided file view, the forward method is at the end of the cell.
                # Let's check if there is anything after it.
                # The view showed it ending with "return x, [feature1, feature2, feature3, feature4]" which is the last line.
                
                # We will replace from start_idx to the end of the source list
                cell['source'] = source[:start_idx] + new_forward_method
                found = True
                break

if found:
    with open(file_path, 'w') as f:
        json.dump(notebook, f, indent=1) # ipynb usually uses indent=1 or similar, but json.dump defaults are different. 
        # To preserve format as much as possible, we might want to be careful, but standard json dump is usually fine for kernels.
        # Actually, let's just use indent=2 or 1 to be safe. The original file looks like it has some indentation.
        # The view_file output showed:
        # 1: {
        # 2:   "cells": [
        # This looks like 2 spaces indentation.
    
    # Re-saving with indent=2
    with open(file_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    print("Successfully modified the notebook.")
else:
    print("Could not find the ResNet class cell.")
