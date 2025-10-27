"""
Innehåller alla testqueries som används för att jämföra embedding modeller.

Queries är uppdelade i 4 kategorier för att testa hur väl modellerna kan:
1. Gruppera liknande frågor (intra-category similarity)
2. Separera olika ämnen (inter-category separation)
"""

TECH_QUERIES = [
    "Hur fungerar maskininlärning?",
    "Vad är ett neural network?",
    "Förklara deep learning för mig",
    "Skillnad mellan AI och ML",
    "Vad är en algoritm?",
]


HEALTH_QUERIES = [
    "Hur tränar jag effektivt?",
    "Vad anses vara nyttig kost?",
    "Tips för bättre sömn",
    "Kan jag träna när jag är förkyld?",
    "Bra HIIT övningar som inte kräver utrustning",
]


TRAVEL_QUERIES = [
    "Bästa resemålet i Europa",
    "Tips för backpacking i Sydasien",
    "Hur packar jag smart?",
    "Vilket land i Sydasien har varmast klimat?",
    "Billigaste sättet att flyga",
]


COOKING_QUERIES = [
    "Enkla recept för styrketräning",
    "Hur lagar jag hälsosam mat?",
    "Vegetariska matidéer",
    "Bästa frukostrecept med mycket protein",
    "Hälsosamma snacks som är lätta att göra",
]


ALL_QUERIES = (
    TECH_QUERIES + 
    HEALTH_QUERIES + 
    TRAVEL_QUERIES + 
    COOKING_QUERIES
)


LABELS = (
    ['tech'] * len(TECH_QUERIES) +
    ['health'] * len(HEALTH_QUERIES) +
    ['travel'] * len(TRAVEL_QUERIES) +
    ['cooking'] * len(COOKING_QUERIES)
)


def print_dataset_info():
    print("\n" + "=" * 60)
    print("Testdata info")
    print("=" * 60)
    print(f"Totalt antal queries: {len(ALL_QUERIES)}")
    print(f"Antal kategorier: {len(set(LABELS))}")
    print(f"\nKategorier:")
    print(f"  - Teknologi: {len(TECH_QUERIES)} queries")
    print(f"  - Hälsa: {len(HEALTH_QUERIES)} queries")
    print(f"  - Resor: {len(TRAVEL_QUERIES)} queries")
    print(f"  - Matlagning: {len(COOKING_QUERIES)} queries")
    print("=" * 60 + "\n")

