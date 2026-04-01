import os
import shutil

# Move PDF
src = 'rag/knowledge_base.pdf'
dest_dir = 'data/raw/pdfs'
dest = f'{dest_dir}/knowledge_base.pdf'

os.makedirs(dest_dir, exist_ok=True)
if os.path.exists(src):
    shutil.move(src, dest)
    print(f"Moved PDF to {dest}")
elif os.path.exists(dest):
    print(f"PDF already at {dest}")
else:
    print(f"Cannot find PDF at {src} or {dest}")
