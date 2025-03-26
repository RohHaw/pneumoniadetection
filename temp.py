import sys
import os

path = os.path.abspath("Training/Validator")
print("Added path:", path)
sys.path.append(path)

from xray_validator import ChestXrayValidator
