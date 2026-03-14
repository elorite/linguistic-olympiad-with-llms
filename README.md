<div align="center">
    <h1>Solving linguistic olympiad problems with LLMs</h1>
    <h3>Authors: Christian Faccio, Elena Lorite Acosta</h3>
    <h5>Emails: christianfaccio@outlook.it, elenalorite@gmail.com</h4>
    <h5>Github: <a href="https://github.com/christianfaccio" target="_blank">christianfaccio</a>, <a href="https://github.com/elorite" target"_blank">elorite</a></h5>
    <h6></h6>
</div>

---

# How to run the code

Create a virtual environment and install the dependencies:
```
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Then, enter the `src/` folder and run the `main.py` script:
```
cd src
uv run main.py
```

>[!NOTE]
> You can specify several flags to personalize the analysis:
> - `--task` - refers to which task to run (default -> baseline)
> - `--language` - choose a language from the dataset (default -> All)
> - `--difficulty` - choose a difficulty level from the dataset (default -> All)
> - `--type` - choose a problem type from the dataset (default -> All)