import numpy as np
import pandas as pd
# from geopandas.io.file import fiona
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
import os
from geopandas import GeoDataFrame
from shapely.geometry import Point


# from dbscan import read_data
def str2xylst(loc_str):
    x, y = loc_str.split(' ')
    return float(x), float(y)


def read_data(line):
    id, loc = line.strip().split(';')[0], line.strip().split(';')[1][11:].strip('()').split(',')

    lst_x, lst_y, zuobiao = [], [], []
    for i in range(len(loc)):
        x, y = str2xylst(loc[i])
        lst_x.append(x)
        lst_y.append(y)
        zuobiao.append((x, y))
    return id, lst_x, lst_y, zuobiao


if __name__ == '__main__':
    # os.environ['USE_PYGEOS'] = '0'
    # Load the road network from the .shp file using geopandas
    road_network = gpd.read_file("deal_sc2018.shp")  # Replace "sc2018.shp" with the path to your .shp file.
    print(road_network)

    # f=open('sc2018.txt', encoding='gbk')
    # total_x, total_y, total_zuobiao =[], [], []
    # for line in f:
    #     try:
    #         id, lst_x, lst_y, zuobiao = read_data(line)
    #         total_x.extend(lst_x)
    #         total_y.extend(lst_y)
    #         total_zuobiao.extend(zuobiao)
    #     except:
    #         continue

    # location = pd.DataFrame({ 'lng': total_x, 'lat': total_y})  # Create DataFrame using dictionaries

    # geometry = [Point(xy) for xy in zip(location.lng, location.lat)]

    # # location = location.drop(['lng', 'lat'], axis=1)
    # crs = {'init': 'epsg:4326'}
    # gdf = GeoDataFrame(location, crs=crs, geometry=geometry)
    # gdf.to_file('deal_sc2018.shp', driver='ESRI Shapefile')
    point = Point(27.70214, 100.82924)
    matched_point = road_network.distance(point).sort_values().index[0]
    print("matchedpoint: ", matched_point)
    res = road_network.loc[0, ['lng', 'lat']]
    print(res)
