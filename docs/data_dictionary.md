# Data Dictionary

| Column             | Type                      | Description                                                                 |
|--------------------|---------------------------|-----------------------------------------------------------------------------|
| **Release Year**   | integer                   | Year the movie was released                                                 |
| **Title**          | string                    | Official title of the movie                                                 |
| **Origin/Ethnicity** | string                  | Country or film industry (e.g., American, Bollywood, Tamil, etc.)            |
| **Director**       | string                    | Director name(s). If multiple, separated by commas.                         |
| **Cast**           | string                    | Principal cast members. If multiple, separated by commas.                   |
| **Genre**          | string (pipe-separated)   | One or more genres delimited by “\|” (e.g., “Drama\|Romance\|Comedy”).       |
| **Wiki Page**      | string                    | URL of the Wikipedia page from which the plot was scraped                    |
| **Plot**           | string                    | Full textual synopsis of the movie’s storyline                               |