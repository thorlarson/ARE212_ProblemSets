### Entering Virtual Environment (venv)
Starting from ARE212_Materials

#### Windows:
Note the backslashes (not forward slashes) 
```
cd venv\Scripts\activate.bat 
```

#### Mac/Linux (I think this is right but I don't have a Mac to test)
```
source venv/bin/activate
```

### Exiting the venv

```
deactivate
```

### Installing required packages 
After entering the virtual environment and navigating back to main directory (ARE212_Materials): 

```
pip install -r requirements.txt
```

### Adding packages 
If you add a package, be sure to generate a new requirements.txt file, and push your changes. 

```
pip freeze > requirements.txt
```
