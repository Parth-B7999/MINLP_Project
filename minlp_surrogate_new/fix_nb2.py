import json

file_path = "DF_model_unsupervised.ipynb"
with open(file_path, 'r') as f: nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Fixed the unpacking to remove the unused first tuple param
            line = line.replace("_, vr_base, vi_base, rho, A_cut, b_cut = model(x_test)", "vr_base, vi_base, rho, A_cut, b_cut = model(x_test)")
            line = line.replace("_, vr_base, vi_base, rho, A_cut, b_cut = model(X_in)", "vr_base, vi_base, rho, A_cut, b_cut = model(X_in)")
            
            # Make sure Cell 7 evaluation loop also gets the fix!
            line = line.replace("_, vr_base, vi_base, rho, A_cut, b_cut = model(x_test)", "vr_base, vi_base, rho, A_cut, b_cut = model(x_test)")
            new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w') as f: json.dump(nb, f, indent=1)

print("Fixed Cell 6 & 7 unpacking in DF_model_unsupervised.ipynb")
