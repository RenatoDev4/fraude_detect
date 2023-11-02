from functions.func import data_acquisition

URL = "https://github.com/RenatoDev4/fraude_detect/raw/main/data/card_transdata.csv"
folder = "data"
final_file_name = "raw_data.csv"

data_acquisition(URL, folder, final_file_name)
