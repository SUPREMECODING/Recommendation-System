# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:34:26 2019

@author: Vivek
"""

from loadFoodData import LoadFoods
import heapq

#load food data 
lf = LoadFoods()
data = lf.loadFoodData()

k = 10  #for 10 popular movies at output

#get top k popular movies
ur = lf.getPopularityRanks()
top_popular = heapq.nlargest(k,ur.items(), key = lambda t: t[1])

#get most popular food items from above list
for foodID,ranking in top_popular:
    foodName = lf.getFoodName(foodID)
    print(foodName)