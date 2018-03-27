
import missingno as msno
from pandas import read_csv
import pandas as pd

df = pd.read_csv('C://Users/Ahmed/Desktop/backup/MyProject/ML-EXP/Regresstion/dataset (1).csv')

# normalize and replace missing values with mean of the data
df.fillna(0, inplace=True)
# df.fillna(df.mean(), inplace=True)

print(df.describe())

# see how diff features correlate to the target value

print(df['air_conditioning_type'].corr(df['target']))
print(df['year_built'].corr(df['target']))
print(df['county'].corr(df['target']))
print(df['city'].corr(df['target']))
print(df['neighborhood'].corr(df['target']))
print(df['number_of_units'].corr(df['target']))
print(df['heating_system_type'].corr(df['target']))
print(df['architectural_style'].corr(df['target']))
print(df['building_quality'].corr(df['target']))
print(df['building_class'].corr(df['target']))
print(df['bathrooms'].corr(df['target']))
print(df['bedrooms'].corr(df['target']))
print(df['rooms'].corr(df['target']))
print(df['pools'].corr(df['target']))
print(df['total_area'].corr(df['target']))
print(df['living_area'].corr(df['target']))
print(df['basement_area'].corr(df['target']))
print(df['garages'].corr(df['target']))
print(df['number_of_levels'].corr(df['target']))
print(df['property_tax'].corr(df['target']))
print(df['land_value'].corr(df['target']))
print(df['structure_value'].corr(df['target']))




"""
ID          Feature                          Correlation  With Target Prediction "no normalization"             replacing missing values with the mean of the cloumn      replacing missing values with Zero
1        air_conditioning_type   |         0.008898397685418639                                          |            0.0041393682123215345                             |    0.005623352243664612
2        year_built              |         0.0061019435395290944                                         |            0.006056722344805414                              |   -0.0002039286707611731
3        county                  |        -0.01621881114888017                                           |           -0.016200833703232893                            |   -0.015239817910152305
4        city                    |         0.0014079890601008708                                         |            0.0013986554706580398                           |    0.0011961344860851241
5        neighborhood            |        -0.006178504960148192                                          |           -0.0039659702103222545                           |   -0.01236237462299073
6        number_of_units         |         0.00070255637537411                                           |            0.0005471930646546582                           |   -0.006647354094642568
7        heating_system_type     |        -0.006227790369063797                                          |           -0.004653275995842127                            |   -0.0141767578934842
8        architectural_style     |        -0.011528214840648074                                          |           -0.000338393519666977                            |    0.0016276042039946852
9        building_quality        |        -0.013019751787306272                                          |           -0.009849110844103955                            |   -0.017966489822454034
10        building_class          |         0.3153718832198066                                            |            0.008868160264431777                            |   -0.0002856422368757223
11        bathrooms               |         0.02314717713360434                                           |            0.023121520066943436                            |   0.0237721553773093
12        bedrooms                |         0.026892100110786736                                          |            0.02686229205249904                             |   0.027607410323045922
13        rooms                   |         0.014273019972380964                                          |            0.014257199303503327                            |   0.014409957144304696
14        pools                   |         Nan                                                           |            nan                                             |   -0.008320369655465206
15        total_area              |        -0.04126754822539879                                           |           -0.010203635583151417                            |   -0.005022364418884382
16        living_area             |         0.041941921407546205                                          |            0.039815316705200714                            |   0.035983441815555635
17        basement_area           |         0.37206746957696224                                           |            0.010087950692524626                            |   0.004512993698743189
18        garages                 |         0.028727234011940033                                          |            0.017101610027062272                            |   0.019141245728212648
19        number_of_levels        |         0.007842491392854471                                          |            0.003934392210018791                            |   0.013712669138441753
20        property_tax            |        -0.001201939372776601                                          |           -0.001200478068637723                            |   -0.000997155528927536
21        land_value              |        -0.0027245942314685027                                         |           -0.0027207377684006382                           |   -0.002484132995613272
22        structure_value         |         0.005646719568609491                                          |           0.005623352243664612                             |   0.005941049912425154
"""


"""
0         1                         | This is just the ID of the data Integer in decreasing order
1     52461   Air Conditioning Type | Type of cooling system present in the home (if any) | integers 1, 2, 3  																							
2       296   year_built        	| Architectural style of the home (i.e. ranch, colonial, split-level, etc…) | years 1923, 1978, 1949,1967																							
3        33   county	            | Finished living area below or partially below ground level.  |	1286,  1286, 3101, 2061, 1286																							
4      1504   city	                | Number of bathrooms in home including fractional bathrooms.	12447,  33612, 12447, 53571																						
5     46524   neighborhood	        | Number of bedrooms in home.	|  115657, 275078, 54300, 274359																							
6     26851   number_of_units	    | Overall assessment of condition of the building from best (lowest) to worst (highest). |	1.0, 2.0, NaN 																				
7     27974   heating_system_type   | The building framing type (steel frame, wood frame, concrete/brick) 	2.0, 7.0																							
8     77208   architectural_styl    | Finished living area	|	NaN																						
9     27742   building_quality	    | Total area			10.0, 8.0, 5.0																				
10    77399   building_class	    | Total number of garages on the lot including an attached garage	| NaN																							
11       33   bathrooms	            | Type of home heating system |	 3.5, 1.0, 2.0, 2.5	,4.0																				
12       33   bedrooms	            | Number of stories or levels the home has | 3.5, 1.0, 2.0, 2.5, 4.0																								
13       33   rooms	                | Number of pools on the lot (if any)|	 	11.0, 0.0, 7.0, 8.0																						
14    61277   pools	                | County in which the property is located		1.0, NaN																						
15    74405   total_area	        | City in which the property is located (if any) Nan																								
16     3665   living_area	        | Neighborhood in which the property is located. 3539.0, 1465.0, 2540.0	,1243.0																						
17    77364   basement_area	        | Number of units the structure is built into (i.e. 2 = duplex, 3 = triplex, etc...) NaN, NaN																							
18    51940   garages	            | The Year the principal residence was built  3.0,NaN, 2.0																								
19    59854   number_of_levels	    | The assessed value of the built structure on the parcel NaN, 1.0 NaN																							
20       38  property_tax	        | The assessed value of the land area of the parcel. | doubles 11013.72, 5672.48, 21758.26, 16388.30																							
21       35   land_value	        | The total property tax assessed for that assessment year | 382000.0, 	1188359.0, 293396.0, 526000.0, 1741737.0																						
22      145   structure_value       | 211040.0, 226351.0, 265000.0, 140000.0, 790000.0
23        0   Target	            | Target variable to predict: ln(predicted) - ln(actual)  | -0.062404, 0.013268,-0.232215, 0.094750, 0.008669, 0.025595																
"""

