import json

found = False
for file_path in ["DF_model_unsupervised.ipynb", "DF_model_5.ipynb"]:
    try:
        with open(file_path, 'r') as f:
            nb = json.load(f)
            
        for idx, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                for l_idx, line in enumerate(cell['source']):
                    if "model(" in line and "vr_base" in line:
                        print(f"[{file_path}] Cell {idx} Line {l_idx}: {repr(line)}")
                    
                    if "cvx_layer(" in line and "=" in line:
                        print(f"[{file_path}] Cell {idx} Line {l_idx}: {repr(line)}")
    except Exception as e:
        pass
