import pandas as pd
import numpy as np
from ast import literal_eval
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
from shapely.geometry import Point

rest_df = pd.read_csv('data/zomato_df_final_data.csv')
rest_df.head()

rest_df = rest_df.dropna()

rest_df.cuisine = rest_df.cuisine.apply(literal_eval)
rest_df.type = rest_df.type.apply(literal_eval)

rest_df = rest_df.dropna(subset=['lat', 'lng'])

# combine lat and long to Point
rest_df['geometry'] = rest_df.apply(lambda x: Point((float(x.lng), float(x.lat))), axis=1)

sydney_geo = gpd.read_file("data/sydney.geojson")

sydney_geo['rest_count'] = 0
# reset index of rest_df
rest_df = rest_df.reset_index(drop=True)


for i in range(len(rest_df)):
    print(i)
    matched_row_list = sydney_geo.index[sydney_geo.contains(rest_df.geometry[i])].tolist()
    if len(matched_row_list) > 0:
        print('matched: ', matched_row_list[0])
        sydney_geo.rest_count[matched_row_list[0]] += 1
    else:
        print('no match')
        continue

