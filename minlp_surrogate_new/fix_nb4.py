import json
import re

file_path = "DF_model_unsupervised.ipynb"

with open(file_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Fix the unpacking assignment (it was using 6 variables instead of 9)
            line = re.sub(
                r'u_relax,\s*pg,\s*qg,\s*vr,\s*vi,\s*s_cut\s*=\s*cvx_layer',
                r'u_relax, pg, qg, vr, vi, s_cut, xi_c, xij_c, xij_s = cvx_layer',
                line
            )
            line = re.sub(
                r'u_relax,\s*_,\s*_,\s*_,\s*_,\s*s_cut\s*=\s*cvx_layer',
                r'u_relax, _, _, _, _, s_cut, xi_c, xij_c, xij_s = cvx_layer',
                line
            )
            new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("CVXPY 9-variable unpack logic fixed.")
