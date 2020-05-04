Create a virtual environment :

```
python -m venv forjupyter
cd forjupyter
cd Scripts
activate
python -p pip install --upgrade pip
pip install jupyter
pip install jsonschema
```
Download pyrsistent 0.15.7 source file from Download Files section

https://pypi.org/project/pyrsistent/0.15.7/#files

Extract the archive into the forjupyter folder

open setup.py

change line 54 from this:
```
version=__version__,
```
To this:
```
version='0.15.7',
```
Jupyter should work

```
jupyter notebook
```
