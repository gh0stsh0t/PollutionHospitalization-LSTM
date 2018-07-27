import tabula
import pandas
import os
for x in os.listdir("pdf_data"):
    tabula.convert_into("pdf_data/"+x, x+".json", multiple_tables = True)
