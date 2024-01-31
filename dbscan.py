import numpy as np
import pandas as pd
from geopandas.io.file import fiona
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
from shapely.geometry import Point

from scipy.spatial import distance
from haversine import haversine

import os

RADIUS = 1.0  # Change this based on your requirement. This is in kilometers.


def str2xylst(loc_str):
    x, y = loc_str.split(' ')
    return float(x), float(y)


def read_data(line):
    id, loc = line.strip().split(';')[0], line.strip().split(';')[1][11:].strip('()').split(',')

    lst_x, lst_y = [], []
    for i in range(len(loc)):
        x, y = str2xylst(loc[i])
        lst_x.append(x)
        lst_y.append(y)
    return id, lst_x, lst_y


def myScore(estimator, X):  # 轮廓系数
    labels = estimator.fit_predict(X)
    score = silhouette_score(X, labels, metric='euclidean')
    # return score
    if labels.max() > 3:
        return score
    else:
        return -0.99


def roadCluster(trajectory):  # 使用DBSCAN聚类算法进行路段划分
    sample_num = int(0.002 * len(trajectory))
    print(sample_num)
    trajectory_sample = trajectory.sample(sample_num)  # 随机抽样60%样本点
    locations = np.array(trajectory_sample[['lat', 'lng']])  # 位置数据
    param_grid = {"eps": [0.0002, 0.0003, 0.0005, 0.0006, 0.0008, 0.0009, 0.001],
                  "min_samples": [3, 4, 6, 7, 8, 9, 15, 20, 30, 50, 70]
                  }  # epsilon控制聚类的距离阈值，min_samples控制形成簇的最小样本数
    dbscan = DBSCAN()
    grid_search = GridSearchCV(estimator=dbscan, param_grid=param_grid, scoring=myScore, n_jobs=-1)
    grid_search.fit(locations)
    print("best parameters:{}".format(grid_search.best_params_))
    print("label:{}".format(grid_search.best_estimator_.labels_))
    labels = grid_search.best_estimator_.labels_  # -1表示离群点
    score = silhouette_score(locations, labels, metric='euclidean')  # 轮廓系数
    total_cluster = labels.max() - labels.min()
    print("一共聚了{}类, 轮廓系数为{}".format(total_cluster, score))

    center_loc = np.ndarray((total_cluster, 2))
    center_loc_pd = pd.DataFrame(columns=['lng', 'lat', 'road_label'])

    cluster_loc_dict = {i: [] for i in range(total_cluster)}
    for label in range(total_cluster):
        temp = locations[labels == label]
        count = temp.shape[0]
        temp_label = np.ones(temp.shape[0]) * count
        temp = np.column_stack((temp, temp_label))
        cluster_loc_dict[label] = pd.DataFrame(temp, columns=['lng', 'lat', 'road_label'])
        cluster_loc_dict[label]['road_label'] = cluster_loc_dict[label]['road_label'].astype(int)

        center_loc[label] = np.array((np.median(temp[:, 0]), np.median(temp[:, 1])))  # 用每一个类别的中位数经纬度作为簇的代表
        center_loc_pd.loc[label] = [np.median(temp[:, 0]), np.median(temp[:, 1]), str(label)]

    plt.plot(center_loc[:, 0], center_loc[:, 1], 'o')
    plt.title("Road Cluster")
    plt.show()
    # import pdb;pdb.set_trace()
    # 需要全部轨迹点：
    # road_label = pd.DataFrame({"road_label": labels})
    # trajectory_sample.reset_index(drop=True, inplace=True)
    # road_label.reset_index(drop=True, inplace=True)
    # cluster_data = pd.concat([trajectory_sample, road_label], axis=1, ignore_index=True)  # 带标签的行驶记录
    # cluster_data.columns = ['lng', 'lat', 'road_label']
    # cluster_data['road_label'] = [str(i) for i in cluster_data['road_label']]
    print(center_loc_pd)
    return center_loc_pd, cluster_loc_dict


def roadRaster(road_data, unit_gap):  # 轨迹栅格化
    min_lng, max_lng = np.min(road_data['lng']), np.max(road_data['lng'])  # 经度范围
    min_lat, max_lat = np.min(road_data['lat']), np.max(road_data['lat'])  # 纬度范围
    lng_gap = max_lng - min_lng
    lat_gap = max_lat - min_lat
    m = int(lng_gap / unit_gap)
    n = int(lat_gap / unit_gap)
    print(min_lng, max_lng, min_lat, max_lat, m, n, (m - 1) * (n - 1))
    slice_lng = np.linspace(min_lng, max_lng, m)  # 对经度等间距划分
    slice_lat = np.linspace(min_lat, max_lat, n)  # 对纬度等间距划分
    idx = 0

    for i in range(len(slice_lng) - 1):
        for j in range(len(slice_lat) - 1):
            raster_a_lng = slice_lng[i]
            raster_a_lat = slice_lat[j]
            raster_b_lng = slice_lng[i + 1]
            raster_b_lat = slice_lat[j + 1]
            idx += 1

            # Plot the grid cell
            plt.plot([raster_a_lng, raster_b_lng], [raster_a_lat, raster_a_lat], color='k')  # Horizontal line
            plt.plot([raster_a_lng, raster_b_lng], [raster_b_lat, raster_b_lat], color='k')  # Horizontal line
            plt.plot([raster_a_lng, raster_a_lng], [raster_a_lat, raster_b_lat], color='k')  # Vertical line
            plt.plot([raster_b_lng, raster_b_lng], [raster_a_lat, raster_b_lat], color='k')  # Vertical line

    # Plot the road data points on top of the grid
    plt.scatter(road_data['lng'], road_data['lat'], color='red', label='$s=n$')

    # Set plot labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Road Raster Grid')
    plt.legend()

    # Show the plot
    plt.show()


# def plot_on_map(cluster_data):
#     # 创建Cartopy绘图
#     fig = plt.figure(figsize=(10, 8))
#     ax = plt.axes(projection=ccrs.PlateCarree())
#
#     # 绘制地图
#     ax.coastlines()
#
#     # 绘制栅格线
#     for i in range(len(cluster_data)):
#         lng = cluster_data['lng'].iloc[i]
#         lat = cluster_data['lat'].iloc[i]
#         road_label = cluster_data['road_label'].iloc[i]
#
#         # 绘制栅格点，根据不同的road_label使用不同的颜色表示
#         if road_label == '-1':
#             color = 'gray'  # 离群点使用灰色
#         else:
#             color = 'red'  # 其他路段使用红色
#
#         ax.plot(lng, lat, 'o', color=color, markersize=5, transform=ccrs.PlateCarree())
#
#     # 设置地图范围
#     min_lng, max_lng = min(cluster_data['lng']), max(cluster_data['lng'])
#     min_lat, max_lat = min(cluster_data['lat']), max(cluster_data['lat'])
#     ax.set_extent([min_lng - 0.1, max_lng + 0.1, min_lat - 0.1, max_lat + 0.1], crs=ccrs.PlateCarree())
#
#     # 显示地图
#     plt.title('Road Raster Grid on Map')
#     plt.show()


def map_match_to_road_network(trajectory, road_network):
    matched_trajectories = []
    # for index, row in trajectory.iterrows():
    # points = [Point(lon, lat) for lat, lon in zip(row['lat'], row['lng'])]
    points = [Point(row['lng'], row['lat']) for _, row in trajectory.iterrows()]
    labels = [row['road_label'] for _, row in trajectory.iterrows()]
    matched_points = []
    for point in points:
        matched_point = road_network.distance(point).sort_values().index[0]
        matched_points.append(road_network.loc[matched_point, ['lat', 'lng']])
    matched_lat, matched_lon = zip(*matched_points)
    matched_trajectory = pd.DataFrame({'lat': matched_lat, 'lng': matched_lon, 'road_label': labels})
    matched_trajectories.append(matched_trajectory)

    return matched_trajectories


def plot_on_map(charging_stations):
    # Load the road network from the .shp file using geopandas
    road_network = gpd.read_file("deal_sc2018.shx").sample(frac=1)

    # Visualize the road network and charging stations on a map
    fig, ax = plt.subplots(figsize=(60, 48))

    # Plot the road network
    road_network.plot(ax=ax, color='blue', label='Road Network')

    # Convert charging stations to DataFrame for map matching
    charging_stations_df = pd.DataFrame(charging_stations, columns=['lat', 'lng'])

    # Add a dummy 'road_label' column to charging_stations_df for compatibility
    charging_stations_df['road_label'] = -1  # Assigning a dummy label

    # Perform map matching for charging stations
    temp = []
    for _, station in charging_stations_df.iterrows():
        temp.append(map_match_to_road_network(pd.DataFrame([station]), road_network)[0])

    temp = pd.concat(temp)

    # Plot the map-matched charging stations
    ax.scatter(temp['lng'], temp['lat'], color='red', marker='^', label='Charging Station', s=100)

    ax.set_title('Road Network with Charging Stations')
    plt.legend(loc="upper right")
    plt.show()

def read_charging_stations(filename):
    stations = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # skip empty lines
                continue
            if ';' not in line:  # if no semicolon in the line
                print(f"Warning: Skipping line without semicolon: {line}")
                continue

            _, loc_str = line.split(';')
            lat, lng = map(float, loc_str.strip('()').split(','))
            stations.append((lat, lng))
    return stations


def is_within_radius(point, station, radius=RADIUS):
    return haversine(point, station) <= radius


def calculate_coverage(test_points, stations):
    covered_points = sum(any(is_within_radius(point, station) for station in stations) for point in test_points)
    return (covered_points / len(test_points)) * 100


if __name__ == '__main__':
    # os.environ['SHAPE_RESTORE_SHX'] = 'YES'
    f = open('location.txt', encoding='gbk')
    total_x, total_y = [], []
    for line in f:
        id, lst_x, lst_y = read_data(line)
        del lst_x[10:-10]
        del lst_y[10:-10]
        total_x.extend(lst_x)
        total_y.extend(lst_y)
    trajectory = pd.DataFrame({'lat': total_y, 'lng': total_x})  # Create DataFrame using dictionaries

    # cluster_data, cluster_loc_dict = roadCluster(trajectory)
    # roadRaster(cluster_data, 0.01)
    # # Load the road network from the .shp file using geopandas
    # road_network = gpd.read_file("deal_sc2018.shx").sample(frac=1)
    # # Perform map matching and get matched trajectories
    # # import pdb;pdb.set_trace()
    # temp = []
    # for i in range(len(cluster_loc_dict)):
    #     temp.append(map_match_to_road_network(cluster_loc_dict[i], road_network)[0])  # cluster 每个类别的每个样本的经纬度
    #
    # temp = pd.concat(temp)
    # # Visualize the matched trajectories on a map
    # fig = plt.figure(figsize=(60, 48))
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.scatter(trajectory['lng'], trajectory['lat'], color='blue', label='Original GPS Points', s=1)
    # # 分颜色取车流量
    # scatter = ax.scatter(temp['lng'], temp['lat'], c=temp['road_label'], cmap='Spectral', label='Map Matched Points', s=5)
    # legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    # ax.add_artist(legend)
    # for i in range(temp['lng'].shape[0]):
    #     ax.annotate(str(int(temp['road_label'].to_numpy().tolist()[i])), (temp['lng'].to_numpy().tolist()[i], temp['lat'].to_numpy().tolist()[i]))
    # ax.set_title('GPS Map Matching')
    # plt.show()

    # # print charging station on map
    # # Read charging stations
    # charging_stations = read_charging_stations('chargingstationlocation.txt')
    # # Load the road net   work data
    # road_network = gpd.read_file("deal_sc2018.shx").sample(frac=1)
    # # Plot the charging stations and the road network on the map
    # plot_on_map(charging_stations)

    # # testEvaluations
    charging_stations = read_charging_stations('chargingstationlocation.txt')
    # Get test data points
    test_data = [(row['lat'], row['lng']) for _, row in trajectory.iterrows()]
    print(len(test_data))
    # Calculate coverage percentage
    coverage = calculate_coverage(test_data, charging_stations)
    print(f"The coverage of charging stations is {coverage:.2f}%")
