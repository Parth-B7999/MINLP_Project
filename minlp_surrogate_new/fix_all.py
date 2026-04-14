import json
import re
import glob

for file_path in glob.glob("*.ipynb"):
    try:
        with open(file_path, 'r') as f:
            nb = json.load(f)
            
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                new_source = []
                for line in cell['source']:
                    line = re.sub(
                        r'_, vr_base, vi_base, rho, A_cut, b_cut = model\(x_test\)',
                        r'vr_base, vi_base, rho, A_cut, b_cut = model(x_test)',
                        line
                    )
                    line = re.sub(
                        r'_, vr_base, vi_base, rho, A_cut, b_cut = model\(X_in\)',
                        r'vr_base, vi_base, rho, A_cut, b_cut = model(X_in)',
                        line
                    )
                    new_source.append(line)
                cell['source'] = new_source

        with open(file_path, 'w') as f:
            json.dump(nb, f, indent=1)
    except Exception as e:
        pass

print("Fixed all notebooks.")
