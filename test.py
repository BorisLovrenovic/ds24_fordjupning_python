"""
Testskript för main.py

Detta skript innehåller automatiska tester för att verifiera att
försäljningsdatan fungerar korrekt.
"""

import unittest
import pandas as pd
import sqlite3
import os
from main import las_csv_fil, validera_data, spara_till_databas


class TestForsaljningsdata(unittest.TestCase):
    """
    Testklass för att testa försäljningsdata-pipeline.
    """
    
    def setUp(self):
        """
        Körs före varje test.
        Skapar testfiler som behövs för testerna.
        """
        # Skapa en test CSV-fil
        self.test_csv = 'test_data.csv'
        test_data = {
            'datum': ['2025-09-01', '2025-09-02'],
            'produkt': ['Produkt A', 'Produkt B'],
            'antal': [5, 10],
            'pris': [99.99, 149.99]
        }
        df = pd.DataFrame(test_data)
        df.to_csv(self.test_csv, index=False)
        
        # Definiera testdatabasnamn
        self.test_db = 'test_forsaljning.db'
    
    def tearDown(self):
        """
        Körs efter varje test.
        Städar upp testfiler som skapats.
        """
        # Ta bort testfiler om de finns
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)
        
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
    
    def test_csv_fil_finns(self):
        """
        Testar att CSV-filen finns på rätt plats.
        """
        resultat = os.path.exists('forsaljningsdata.csv')
        self.assertTrue(resultat, "CSV-filen kunde inte hittas")
        print("Test 1 - CSV-filen finns")
    
    def test_las_csv_fil_success(self):
        """
        Testar att las_csv_fil() kan läsa CSV-fil.
        """
        df = las_csv_fil(self.test_csv)
        
        # Kontrollera att vi fick en DataFrame
        self.assertIsNotNone(df, "DataFrame är None")
        
        # Kontrollera att vi fick data
        self.assertGreater(len(df), 0, "DataFrame är tom")
        
        print(f"Test 2 - Kunde läsa CSV-fil - ({len(df)} rader)")
    
    def test_las_csv_fil_finns_inte(self):
        """
        Testar att las_csv_fil() hanterar saknad fil.
        """
        df = las_csv_fil('finns_inte.csv')
        
        # Funktionen ska returnera None om filen inte finns
        self.assertIsNone(df, "Funktionen returnerade inte None för saknad fil")
        
        print("Test 3 - Hanterar saknad fil korrekt")
    
    def test_validera_data_success(self):
        """
        Testar att validera_data() godkänner giltig data.
        """
        df = las_csv_fil(self.test_csv)
        resultat = validera_data(df)
        
        # Valideringen ska returnera True
        self.assertTrue(resultat, "Validering misslyckades för giltig data")
        
        print("Test 4 - Validerar korrekt data")
    
    def test_validera_data_saknar_kolumn(self):
        """
        Testar att validera_data() upptäcker saknade kolumner.
        """
        # Skapa DataFrame utan obligatorisk kolumn
        df = pd.DataFrame({
            'datum': ['2025-09-01'],
            'produkt': ['Test']
            # 'antal' och 'pris' saknas
        })
        
        resultat = validera_data(df)
        
        # Valideringen ska misslyckas
        self.assertFalse(resultat, "Validering godkände data med saknade kolumner")
        
        print("Test 5 - Upptäcker saknade kolumner")
    
    def test_spara_till_databas(self):
        """
        Testar att spara_till_databas() sparar data korrekt.
        """
        df = las_csv_fil(self.test_csv)
        
        # Spara till testdatabas
        resultat = spara_till_databas(df, self.test_db, 'forsaljning')
        
        # Funktionen ska returnera True
        self.assertTrue(resultat, "Kunde inte spara till databas")
        
        # Kontrollera att databasen skapades
        self.assertTrue(os.path.exists(self.test_db), "Databasfilen skapades inte")
        
        # Kontrollera att data finns i databasen
        conn = sqlite3.connect(self.test_db)
        df_fran_db = pd.read_sql_query("SELECT * FROM forsaljning", conn)
        conn.close()
        
        # Kontrollera antal rader
        self.assertEqual(len(df), len(df_fran_db), "Antal rader matchar inte")
        
        print(f"Test 6 - Sparade {len(df_fran_db)} rader till databas")


def run_tests():
    """
    Kör alla tester och visar resultat.
    """
    print("\n" + "="*60)
    print("Kör automatiska tester")
    print("="*60 + "\n")
    
    # Skapa test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestForsaljningsdata)
    
    # Kör testerna med detaljerad output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Visa sammanfattning
    print("\n" + "="*60)
    print("Test sammanfattning")
    print("="*60)
    print(f"Antal tester: {result.testsRun}")
    print(f"Lyckade: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Misslyckade: {len(result.failures)}")
    print(f"Fel: {len(result.errors)}")
    print("="*60 + "\n")


if __name__ == '__main__':
    run_tests()