import PyInstaller.__main__

PyInstaller.__main__.run([
    'barni_cli.spec',
    '--onefile',
    '--windowed',
#    '--add-binary=%s' % os.path.join('resource', 'path', '*.png'),
#    '--add-data=%s' % os.path.join('resource', 'path', '*.txt'),
#    '--icon=%s' % os.path.join('resource', 'path', 'icon.ico'),
])
