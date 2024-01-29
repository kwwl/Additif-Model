from data_preparation import DataPreparation
from additif import Additif


csv_path = (
    "/Users/kemillamouri/Desktop/projet_Hakim/Additif-Model/vente_maillots_de_bain.csv"
)

data_preparation_object = DataPreparation(csv_path)
regression_object = Additif(data_preparation_object)

# data_preparation_object.show_graph()
