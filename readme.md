## Beskrivning
Ett Python-program som automatiskt läser försäljningsdata från en CSV-fil och uppdaterar en SQLite-databas. Programmet kan schemaläggas för att köras automatiskt.

**Kurs:** Fördjupning i Pythonprogrammering  
**Program:** Data Science, EC Utbildning

---

## Filer i projektet
- `main.py` - Huvudprogrammet som läser CSV och uppdaterar databasen
- `test.py` - Testprogram med automatiska tester
- `forsaljningsdata.csv` - Exempeldata med försäljningsinformation
- `logg.txt` - Loggfil som skapas automatiskt när programmet körs
- `forsaljning.db` - SQLite-databas som skapas automatiskt

---

## Installation

### Krav
- Python 3.x
- pandas biblioteket

### Installera pandas
Öppna kommandotolken (CMD) och kör:
```bash
pip install pandas
```

---

## Användning

### Köra programmet manuellt
```bash
python main.py
```

### Köra tester
```bash
python test.py
```

---

## Kodstandarder
Projektet följer:
- **PEP 8** - Pythons officiella stilguide
- **Docstrings** - All kod är dokumenterad
- **Exception handling** - Felhantering med try-except
- **Logging** - Alla händelser loggas
- **Unittest** - Automatiska tester för alla funktioner
