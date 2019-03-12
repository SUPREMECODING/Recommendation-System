import os
import csv
import sys
import re

from surprise import Dataset
from surprise import Reader
from collections import defaultdict

#import numpy as np

class LoadFoods:
    
    foodItemsPath = './dataset/foodItems.csv'
    ratingsPath = './dataset/ratings.csv'
    
    def loadFoodData(self):
        # dictionaries for storing the mapping of foodID to foodName and vice versa
        self.foodID_to_name = {}
        self.name_to_foodID = {}
        
        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))
        
        # using reader class to read or parse the ratings file AND store result(ratings data) in "ratingsDataset"
        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
        ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)
        
        #doing mapping of foodID to foodName and vice versa
        
        with open(self.foodItemsPath, newline='', encoding='ISO-8859-1') as csvfile:
            foodReader = csv.reader(csvfile)
            next(foodReader)  #Skip header line
            for row in foodReader:  #for each item in the foodItems.csv we loop through
                foodID = int(row[0])
                foodName = row[1]
                self.foodID_to_name[foodID] = foodName
                self.name_to_foodID[foodName] = foodID
                
        return ratingsDataset
    
    #get ratings of a specified user
    def getUserRatings(self, user):
        userRatings = []    
        hitUser = False
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                userID = int(row[0])
                if (user == userID):
                    foodID = int(row[1])
                    rating = float(row[2])
                    userRatings.append((foodID, rating))
                    hitUser = True
                if (hitUser and (user != userID)):
                    break

        return userRatings
    
    def getPopularityRanks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                foodID = int(row[1])
                ratings[foodID] += 1
        rank = 1
        
        for foodID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[foodID] = rank
            rank += 1
        return rankings
    
    #get cuisines list for all the     
    def getCuisines(self):
        cuisines = defaultdict(list)
        cuisineIDs = {}
        maxCuisineID = 0
        with open(self.foodItemsPath, newline='', encoding='ISO-8859-1') as csvfile:
            foodReader = csv.reader(csvfile)
            next(foodReader)
            for row in foodReader:
                foodID = row[0]
                cuisineList = row[2].split('|')
                cuisineIDList = []
                #for each unique cuisine we have unique ID
                for cuisine in cuisineList:
                    if cuisine in cuisineIDs:
                        cuisineID = cuisineIDs[cuisine]
                    else:
                        cuisineID = maxCuisineID
                        maxCuisineID += 1
                        cuisineIDs[cuisine] = cuisineID
                    cuisineIDList.append(cuisineID)
                cuisines[foodID] = cuisineIDList        #{foodID:cuisineList}
        # Convert integer-encoded genre lists to bitfields that we can treat as vectors
        for (foodID, cuisineIDList) in cuisines.items():
            bitfield = [0] * maxCuisineID               #create a list of size = maxGenreID
            for cuisineID in cuisineIDList:
                bitfield[cuisineID] = 1
            cuisines[foodID] = bitfield   
        return cuisines
    
    
    def getOrderTime(self):
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)
        with open(self.foodItemsPath, newline='', encoding='ISO-8859-1') as csvfile:
            foodReader = csv.reader(csvfile)
            next(foodReader)
            for row in foodReader:
                foodID = int(row[0])
                title = row[1]
                m = p.search(title)
                year = m.group(1)
                if year:
                    years[foodID] = int(year)
        return years
    
    def getFoodName(self,foodID):
        if foodID in self.foodID_to_name:
            foodName = self.foodID_to_name[foodID]
        else:
            foodName = ''
        return foodName
    
    def getFoodID(self,foodName):
        if foodName in self.name_to_foodID:
            foodID = self.name_to_foodID[foodName]
        else:
            foodID = 0      # 0 for not found
        return foodID