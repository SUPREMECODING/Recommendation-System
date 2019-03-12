# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 18:33:26 2019

@author: Asus
"""

from loadFoodData import LoadFoods
from surprise import KNNBasic

import heapq
from collections import defaultdict
from operator import itemgetter

testSubject = '85'
k = 10

# loading the dataset
lf = LoadFoods()
dataset = lf.loadFoodData()

trainSet = dataset.build_full_trainset()

#building the similarity matrix
sim_options = {'name': 'cosine',
              'user_based': False
              }

model = KNNBasic(sim_options=sim_options)
sm = model.fit(trainSet)
simsMatrix = model.compute_similarities()

testUserInnerID = trainSet.to_inner_uid(testSubject)

# Get the top K items we rated
testUserRatings = trainSet.ur[testUserInnerID]
kNeighbors = heapq.nlargest(k, testUserRatings, key=lambda t: t[1])

# Get similar items to stuff we liked (weighted by rating)
candidates = defaultdict(float)
for itemID, rating in kNeighbors:
    similarityRow = simsMatrix[itemID]
    for innerID, score in enumerate(similarityRow):     #enumerate it to get the index , score of each cell in similarityRow
        candidates[innerID] += score * (rating / 5.0)
    
# Build a dictionary of stuff the user has already seen
watched = {}
for itemID, rating in trainSet.ur[testUserInnerID]:
    watched[itemID] = 1
    
# Get top-rated items from similar users:
pos = 0
for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not itemID in watched:
        foodID = trainSet.to_raw_iid(itemID)
        print(lf.getFoodName(int(foodID)), ratingSum)
        pos += 1
        if (pos > 10):
            break
