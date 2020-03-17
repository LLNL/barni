# -*- mode: python ; coding: utf-8 -*-

import scipy
import os

# we need to tell explicitly where the dlls are for scipy
scipy_path = os.path.dirname(scipy.__file__)
scipy_path = os.path.join(scipy_path,".libs")


block_cipher = None


a = Analysis(['barni_cli.py'],
             pathex=[scipy_path],
             binaries=[],
             datas=[],
             hiddenimports=['Cython', 'yaml', 'sklearn', 'sklearn.ensemble', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree._utils',
             'sklearn.utils._cython_blas'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['IPython','Sphinx'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          exclude_binaries=False,
          name='barni',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          console=True )
# coll = COLLECT(exe,
#               a.binaries,
#               a.zipfiles,
#               a.datas,
#               strip=False,
#               upx=True,
#               upx_exclude=[],
#               name='_cli')
