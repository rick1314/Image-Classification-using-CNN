import os
import distutils
from distutils import dir_util
from fnmatch import fnmatch


def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

root = 'train_val_images_mini'
pattern = "*.jpg"
folders = []
dest = 'train/'
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            if os.path.dirname(path) not in folders:
            	folders.append(os.path.dirname(path))

for path in folders:
	print(os.path.abspath(path), " and ", os.path.abspath(dest))
	dir_util.copy_tree(path, dest)


