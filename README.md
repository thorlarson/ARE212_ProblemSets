## Getting Set Up to Collaborate: 
All package dependencies live in the requirements.txt file. To utilize this file effectively, follow the instructions below


1. create your own virtual environment (I personally prefer venv to conda or other options, but you can use whatever)
2. activate your virtual environment
3. load the requirements.txt file 
```
pip install -r requirements.txt
```
4. If you add any package dependencies through a pip install, update the requirements.txt file
```
pip freeze > requirements.txt
```
5. When pushing changes, make sure your virtual environment is included in the .gitignore (notice the only thing in there right now is "venv/", which is the name of my local virtual environment). If your venv is called venv, you're good to go. 
