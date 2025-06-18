class Language:
    """
    A class responsible for handling multilingual translations (German/English).
    """
    def __init__(self, lang="DE"):
        """
        Initialize the Language class with a dictionary of translations for both languages.
        By default, German (DE) is used if no language is specified.
        """
        self.translations = {
            "DE": {
                # Menu entries
                "menu_settings": "Einstellungen",
                "menu_language": "Sprache",
                "menu_german": "Deutsch",
                "menu_english": "Englisch",

                # Buttons
                "start": "Starten",
                "stop": "Stoppen",
                "drinking": "Trinkverhalten",

                # LED texts
                "Standing": "Stehend",
                "Sitting": "Sitzend",
                "Lying (Left side)": "Liegend         (Linke Seite)",
                "Lying (Right side)": "Liegend        (Rechte Seite)",
                "Lying (Back)": "Liegend (Rücken)",
                "Lying (Belly)": "Liegend (Bauch)",
                "Sleeping": "Schlafend",
                "Awake": "Wach",

                # Window title
                "app_title": "Positions- und Trinkerkennung",

                # Label for drinking
                "drink_label": "Getrunken: ",
            },
            "EN": {
                # Menu entries
                "menu_settings": "Settings",
                "menu_language": "Language",
                "menu_german": "German",
                "menu_english": "English",

                # Buttons
                "start": "Start",
                "stop": "Stop",
                "drinking": "Drinking",

                # LED texts
                "Standing": "Standing",
                "Sitting": "Sitting",
                "Lying (Left side)": "Lying             (Left side)",
                "Lying (Right side)": "Lying            (Right side)",
                "Lying (Back)": "Lying (Back)",
                "Lying (Belly)": "Lying (Belly)",
                "Sleeping": "Sleeping",
                "Awake": "Awake",

                # Window title
                "app_title": "Posture & Drink Recognition",

                # Label for drinking
                "drink_label": "Drinks: ",
            }
        }
        self.set_language(lang)

    def set_language(self, lang):
        """
        Set the active language (e.g., 'DE' or 'EN').
        If the provided language is not recognized, defaults to German (DE).
        """
        if lang in self.translations:
            self.lang = lang
        else:
            self.lang = "DE"  # Fallback

    def t(self, key):
        """
        Return the translated string corresponding to the given key.
        If the key is not found, return the key itself.
        """
        return self.translations[self.lang].get(key, key)
