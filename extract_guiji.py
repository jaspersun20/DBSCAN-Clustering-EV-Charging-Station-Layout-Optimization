import pandas as pd

data = pd.read_csv('result_1116.txt', sep=';')

beifen_data = data[['id', 'ogeom']]
beifen_data.to_csv('location.txt', sep=';',index=False)