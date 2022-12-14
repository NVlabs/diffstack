import sys
import torch

# Needed to find model
sys.path.append('./trajectron/trajectron')

args = sys.argv[1:]

for filename in args:
    new_filename = filename + ".new"
    print (f"{filename} -> {new_filename}")

    # Import error can happen here is trying to load checkpoint with old cost_function object.
    # The old pred_metrics folder needs to be copied with environment/nuScenes_data/cost_functions.py     
    model_dict = torch.load(filename, map_location=torch.device('cpu'))

    print (model_dict)
    
    if "planner_cost" in model_dict:
        print ("Removing planner_cost")
        del model_dict["planner_cost"]

    torch.save(model_dict, new_filename)

print ("done")