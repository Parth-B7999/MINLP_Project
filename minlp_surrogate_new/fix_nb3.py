import json
import re

file_path = "DF_model_unsupervised.ipynb"

with open(file_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Use RegEx to cleanly replace the tuple catching regardless of preceding/trailing characters
            line = re.sub(r'_, vr_base, vi_base, rho, A_cut, b_cut = model\(x_test\)', 
                          r'vr_base, vi_base, rho, A_cut, b_cut = model(x_test)', line)
            
            line = re.sub(r'_, vr_base, vi_base, rho, A_cut, b_cut = model\(X_in\)', 
                          r'vr_base, vi_base, rho, A_cut, b_cut = model(X_in)', line)
                          
            new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Regex replace successful.")
