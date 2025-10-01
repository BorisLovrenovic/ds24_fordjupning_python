"""
Automatiserad pipeline för försäljningsdata

Detta program läser försäljningsdata från en CSV-fil och uppdaterar
en SQLite-databas. Programmet kan schemaläggas för att köras automatiskt.

"""

import pandas as pd
import sqlite3
import logging
import os


# Konfigurera logging
logging.basicConfig(
    filename='logg.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def las_csv_fil(filnamn):
    """
    Läser data från CSV-fil.
    """
    try:
        # Kontrollera att filen finns
        if not os.path.exists(filnamn):
            logging.error(f"Filen {filnamn} hittades inte")
            return None
        
        # Läs CSV-filen
        df = pd.read_csv(filnamn)
        logging.info(f"Läste in {len(df)} rader från {filnamn}")
        return df
        
    except pd.errors.EmptyDataError:
        logging.error("CSV-filen är tom")
        return None
        
    except Exception as e:
        logging.error(f"Fel vid läsning av CSV: {e}")
        return None


def validera_data(df):
    """
    Bekräfta att DataFrame har rätt struktur och innehåll.
    """
    # Kontrollera att DataFrame inte är tom
    if df is None or df.empty:
        logging.error("DataFrame är tom")
        return False
    
    # Kontrollera att nödvändiga kolumner finns
    required_columns = ['datum', 'produkt', 'antal', 'pris']
    for kolumn in required_columns:
        if kolumn not in df.columns:
            logging.error(f"Kolumn '{kolumn}' saknas")
            return False
    
    logging.info("Datan bekräftad")
    return True


def spara_till_databas(df, databas_namn, tabell_namn):
    """
    Sparar DataFrame till SQLite-databas.
    """
    try:
        # Anslut till databas (skapas automatiskt om den inte finns)
        conn = sqlite3.connect(databas_namn)
        
        # Lägg till data i tabellen
        df.to_sql(tabell_namn, conn, if_exists='append', index=False)
        
        # Stäng anslutningen
        conn.close()
        
        logging.info(f"Sparade {len(df)} rader till databas")
        return True
        
    except sqlite3.Error as e:
        logging.error(f"Databasfel: {e}")
        return False
        
    except Exception as e:
        logging.error(f"Oväntat fel: {e}")
        return False


def main():
    """
    Huvudfunktion som kör hela processen.
    """
    logging.info("Startar automatiserad process")
    
    try:
        # Steg 1 - Läs CSV-filen
        csv_fil = 'forsaljningsdata.csv'
        df = las_csv_fil(csv_fil)
        
        if df is None:
            logging.error("Kunde inte läsa CSV-fil")
            return
        
        # Steg 2 - Validera data
        if not validera_data(df):
            logging.error("Datavalidering misslyckades.. avslutar..")
            return
        
        # Steg 3 - Spara till databas
        if spara_till_databas(df, 'forsaljning.db', 'forsaljning'):
            logging.info("Process slutförd")
        else:
            logging.error("Process misslyckades")
            
    except Exception as e:
        logging.error(f"Kritiskt fel: {e}")


if __name__ == "__main__":
    main()