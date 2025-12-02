"""
Convert country column to continent column in train_metadata_cleaned.csv
Uses offline mapping to avoid API rate limiting issues.
"""

import pandas as pd

# Comprehensive country-to-continent mapping
# Handles country names in various languages
COUNTRY_TO_CONTINENT = {
    # Europe
    'France': 'Europe',
    'España': 'Europe',  # Spain
    'Polska': 'Europe',  # Poland
    'Россия': 'Europe',  # Russia (primary continent - spans Europe and Asia)
    'Deutschland': 'Europe',  # Germany
    'United Kingdom': 'Europe',
    'Portugal': 'Europe',
    'Nederland': 'Europe',  # Netherlands
    'Suomi / Finland': 'Europe',  # Finland
    'Sverige': 'Europe',  # Sweden
    'Norge': 'Europe',  # Norway
    'Danmark': 'Europe',  # Denmark
    'Italia': 'Europe',  # Italy
    'Schweiz/Suisse/Svizzera/Svizra': 'Europe',  # Switzerland
    'België / Belgique / Belgien': 'Europe',  # Belgium
    'Österreich': 'Europe',  # Austria
    'Česko': 'Europe',  # Czech Republic
    'Slovensko': 'Europe',  # Slovakia
    'Slovenija': 'Europe',  # Slovenia
    'Hrvatska': 'Europe',  # Croatia
    'Magyarország': 'Europe',  # Hungary
    'România': 'Europe',  # Romania
    'България': 'Europe',  # Bulgaria
    'Ελλάς': 'Europe',  # Greece
    'Κύπρος - Kıbrıs': 'Europe',  # Cyprus
    'Eesti': 'Europe',  # Estonia
    'Latvija': 'Europe',  # Latvia
    'Lietuva': 'Europe',  # Lithuania
    'Беларусь': 'Europe',  # Belarus
    'Україна': 'Europe',  # Ukraine
    'Србија': 'Europe',  # Serbia
    'Crna Gora / Црна Гора': 'Europe',  # Montenegro
    'Bosna i Hercegovina / Босна и Херцеговина': 'Europe',  # Bosnia and Herzegovina
    'Türkiye': 'Europe',  # Turkey (primary continent - spans Europe and Asia)
    'Éire / Ireland': 'Europe',  # Ireland

    # Asia
    'India': 'Asia',
    'Indonesia': 'Asia',
    'Malaysia': 'Asia',
    'Singapore': 'Asia',
    'Thailand': 'Asia',
    'ประเทศไทย': 'Asia',  # Thailand
    'Việt Nam': 'Asia',  # Vietnam
    'ປະເທດລາວ': 'Asia',  # Laos
    'ព្រះរាជាណាចក្រ​កម្ពុជា': 'Asia',  # Cambodia
    '中国': 'Asia',  # China
    '日本': 'Asia',  # Japan
    '대한민국': 'Asia',  # South Korea
    '臺灣': 'Asia',  # Taiwan
    'Sri Lanka': 'Asia',
    'Azərbaycan': 'Asia',  # Azerbaijan
    'Oʻzbekiston': 'Asia',  # Uzbekistan
    'Қазақстан': 'Asia',  # Kazakhstan
    'Монгол улс ᠮᠤᠩᠭᠤᠯ ᠤᠯᠤᠰ': 'Asia',  # Mongolia
    # Abkhazia (disputed, but geographically in Asia)
    'Абхазия - Аԥсны': 'Asia',
    # Georgia (spans Europe and Asia, but primarily Asia)
    'საქართველო': 'Asia',
    'ایران': 'Asia',  # Iran
    'الإمارات العربية المتحدة': 'Asia',  # UAE
    'السعودية': 'Asia',  # Saudi Arabia
    'الكويت': 'Asia',  # Kuwait
    'عمان': 'Asia',  # Oman
    'ישראל': 'Asia',  # Israel

    # Africa
    'South Africa': 'Africa',
    'Kenya': 'Africa',
    'Nigeria': 'Africa',
    'Tanzania': 'Africa',
    'Uganda': 'Africa',
    'Ghana': 'Africa',
    'Botswana': 'Africa',
    'Namibia': 'Africa',
    'Zambia': 'Africa',
    'Sénégal': 'Africa',  # Senegal
    'Bénin': 'Africa',  # Benin
    'Cameroun': 'Africa',  # Cameroon
    'Maroc ⵍⵎⵖⵔⵉⴱ المغرب': 'Africa',  # Morocco
    'تونس': 'Africa',  # Tunisia
    'مصر': 'Africa',  # Egypt
    'République démocratique du Congo': 'Africa',  # DRC
    'Cabo Verde': 'Africa',  # Cape Verde
    'Soomaaliland أرض الصومال': 'Africa',  # Somaliland

    # Americas
    'United States of America': 'Americas',
    'Canada': 'Americas',
    'México': 'Americas',  # Mexico
    'Brasil': 'Americas',  # Brazil
    'Argentina': 'Americas',
    'Bolivia': 'Americas',
    'Honduras': 'Americas',
    'Panamá': 'Americas',  # Panama

    # Oceania (none in current dataset, but included for completeness)
}


def map_country_to_continent(country):
    """
    Map country name to continent.
    Handles NaN values and missing countries.
    """
    if pd.isna(country) or country == '':
        return None

    # Direct lookup
    if country in COUNTRY_TO_CONTINENT:
        return COUNTRY_TO_CONTINENT[country]

    # If not found, return None (shouldn't happen with comprehensive mapping)
    return None


def convert_country_to_continent(input_file, output_file=None):
    """
    Convert country column to continent column in the CSV file.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (defaults to overwriting input_file)
    """
    # Load the CSV
    df = pd.read_csv(input_file)

    print(f"Loaded {len(df)} records from {input_file}")
    print(f"Countries with values: {df['country'].notna().sum()}")
    print(f"Continents with values (before): {df['continent'].notna().sum()}")

    # Apply mapping to create continent from country
    df['continent'] = df['country'].apply(map_country_to_continent)

    # Drop the country column
    df = df.drop(columns=['country'])

    # Statistics
    print(f"\nContinents with values (after): {df['continent'].notna().sum()}")
    print(f"\nContinent distribution:")
    print(df['continent'].value_counts())

    # Save the updated CSV
    if output_file is None:
        output_file = input_file

    df.to_csv(output_file, index=False)
    print(f"\nUpdated CSV saved to: {output_file}")

    return df


if __name__ == "__main__":
    input_csv = "birdclef-2023/train_metadata_cleaned.csv"
    output_csv = "birdclef-2023/train_metadata_cleaned_continent.csv"
    convert_country_to_continent(input_csv, output_csv)
